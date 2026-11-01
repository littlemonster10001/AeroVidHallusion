"""Resolution gain R_t: target detail that a high-resolution input would add (Eq. 4-5, Appendix C.2).

Each frame is resized twice from the same source frame, as the MLLM resizes
its inputs: LR at the Probing resolution, HR at a short side of 1024. The
luminance residual between HR and the upsampled LR, band-passed by a
difference of Gaussians, holds the detail that LR loses; its local energy
e_t(x) depends only on the frame (``band_energy``, once per video). Per
question, the DINOv3 similarity maps weight it towards the targets
(``information_gain``): the energy above the background median counts as
signal, and R_t is the target-weighted mean of ½ log2(1 + SNR) bits.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from aerovera.evidence.target_relevance import combine_targets

SIGMA_LOW = 0.5  # σ_lo of the difference of Gaussians, in HR pixels


def _gaussian_blur(image: Any, sigma: float) -> Any:
    """Separable Gaussian blur of [N, 1, H, W], radius 4σ, mirrored borders (OpenCV's default)."""
    import torch
    import torch.nn.functional as F

    radius = max(1, math.ceil(4 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(image, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))


def band_energy(hr: Any, lr: Any, device: str = "cpu") -> Any:
    """Local energy e_t(x) [H, W] of the detail that the LR image loses (Eq. 16-18).

    ``hr`` [H, W, 3] and ``lr`` [h, w, 3] are RGB uint8 images resized from the
    same source frame. With η the HR-to-LR scale factor, the luminance residual
    Y(HR) - Y(Up(LR)) (bicubic upsampling) is band-passed by G_0.5 - G_{2 max(η, 1)}
    and its square averaged by G_{max(η, 1)}.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    luma = torch.tensor([0.299, 0.587, 0.114], device=device)

    def luminance(image):
        return (torch.from_numpy(np.ascontiguousarray(image)).to(device).float() / 255) @ luma

    y_hr, y_lr = luminance(hr), luminance(lr)
    height, width = y_hr.shape
    eta = min(height / y_lr.shape[0], width / y_lr.shape[1])
    upsampled = F.interpolate(y_lr[None, None], size=(height, width), mode="bicubic", align_corners=False)
    residual = y_hr[None, None] - upsampled
    band = _gaussian_blur(residual, SIGMA_LOW) - _gaussian_blur(residual, 2 * max(eta, 1.0))
    return _gaussian_blur(band.square(), max(eta, 1.0))[0, 0]


def target_maps(sim: Any, lower: float = 0.5, upper: float = 0.95) -> Any:
    """m_{t,k}: patch similarities [T, K, h, w] scaled per target between its ``lower`` and
    ``upper`` quantiles over all frames of the video, clipped to [0, 1] (0 where flat).
    Video-level statistics keep frames without the target from being amplified."""
    import torch

    values = sim.float().transpose(0, 1).flatten(1)  # [K, T*h*w]
    low = values.quantile(lower, dim=1)[None, :, None, None]
    span = (values.quantile(upper, dim=1) - values.quantile(lower, dim=1))[None, :, None, None]
    maps = ((sim.float() - low) / span.clamp(min=1e-6)).clamp(0, 1)
    return torch.where(span > 1e-6, maps, 0.0)


def information_gain(energy: Any, maps: Any, floor: float = 0.1, noise_floor: float = 2 / 255) -> Any:
    """R_{t,k} [K] in bits for one frame: ``energy`` e_t [H, W], ``maps`` m_{t,k} [K, h, w].

    Each map is upsampled bilinearly to [H, W]; pixels below ``floor`` (μ) form
    the target's background, whose median energy (at least ``noise_floor``²)
    is the noise σ0² (Eq. 19). Pixel gain ½ log2(1 + max(e - σ0², 0) / σ0²)
    (Eq. 4, 21) is averaged with weights clip((m - μ) / (1 - μ), 0, 1) (Eq. 22, 5).
    """
    import torch
    import torch.nn.functional as F

    maps = F.interpolate(maps[None].float().to(energy.device), size=energy.shape, mode="bilinear",
                         align_corners=False)[0].clamp(0, 1)
    gains = []
    for target in maps:
        background = energy[target < floor]
        noise = max((background if background.numel() else energy).median().item(), noise_floor ** 2)
        bits = 0.5 * torch.log2(1 + (energy - noise).clamp(min=0) / noise)
        weights = ((target - floor) / (1 - floor)).clamp(0, 1)
        total = weights.sum()
        gains.append((weights * bits).sum() / total if total > 0 else total)
    return torch.stack(gains)


def resolution_gains(
    energies: Any,
    sim: Any,
    option_targets: dict[str, list[int]] | None,
    positions: Sequence[int],
    lower: float = 0.5,
    upper: float = 0.95,
    floor: float = 0.1,
    noise_floor: float = 2 / 255,
) -> Any:
    """R_t [len(positions)] on the CPU for the frames at ``positions`` among the scanned frames.

    ``energies`` [T, H, W] and ``sim`` [T, K, h, w] cover all scanned frames
    (the map quantiles use them all); targets are combined like Q_t.
    """
    import torch

    maps = target_maps(sim, lower, upper)
    per_target = torch.stack([information_gain(energies[t], maps[t], floor, noise_floor) for t in positions])
    return combine_targets(per_target.cpu(), option_targets)


def scaled_size(lr_size: Sequence[int], source_size: Sequence[int], scale: float, factor: int = 28) -> tuple[int, int]:
    """HR size ``scale`` times the LR size per side, rounded to multiples of
    ``factor``, never beyond the source (floored); it may thus not exceed LR."""
    height, width = (min(round(lr * scale / factor), source // factor) * factor
                     for lr, source in zip(lr_size, source_size))
    return height, width


def top_frames(gains: Any, count: int) -> list[int]:
    """Indices of the ``count`` largest gains, ties to the earlier frame, in ascending order."""
    order = sorted(range(len(gains)), key=lambda i: (-float(gains[i]), i))
    return sorted(order[:count])
