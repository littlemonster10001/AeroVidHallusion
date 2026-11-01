"""Uniqueness U_t from appearance and motion similarity to the frames already chosen (Eq. 12-15, Appendix C.1).

Two frames are redundant only if they look alike in both scene and target
(DINOv3 scene feature g_t and target features h_{t,k}) and also move alike
(residual speed V_t). Frames of one video all have high cosines, so each
cosine is calibrated against its median over the candidate pairs. The
pairwise similarity S of all scanned frames is computed once per question;
the greedy selection then recomputes U_t = 1 - max_{s in S} S(t, s) as frames
are added.
"""

from __future__ import annotations

from typing import Any, Sequence


def calibrated(cosine: Any, candidates: Sequence[int]) -> Any:
    """Eq. 12: clip((cos - c0) / (1 - c0), 0, 1) of cosine matrices [..., T, T].

    c0 is the median of each matrix over the distinct pairs of candidate
    frames (positions among the T frames); 0 with fewer than two candidates.
    """
    import torch

    index = torch.as_tensor(list(candidates), dtype=torch.long)
    if len(index) < 2:
        c0 = cosine.new_zeros(cosine.shape[:-2])
    else:
        rows, cols = torch.triu_indices(len(index), len(index), offset=1)
        c0 = cosine[..., index[rows], index[cols]].quantile(0.5, dim=-1)
    c0 = c0.clamp(max=1 - 1e-6)[..., None, None]  # all candidates alike: identical frames still give 1
    return ((cosine - c0) / (1 - c0)).clamp(0, 1)


def motion_similarity(speed: Any, valid: Any, candidates: Sequence[int]) -> Any:
    """Eq. 13 with the residual speed V_t [T]: exp(-(V_t - V_s)^2 / σ_v^2), σ_v the
    median speed of the valid candidates; 1 (neutral) for a pair with an invalid
    frame, or for all pairs if no candidate is valid."""
    import torch

    index = torch.as_tensor(list(candidates), dtype=torch.long)
    reference = speed[index][valid[index]]
    count = len(speed)
    if reference.numel() == 0:
        return torch.ones(count, count)
    sigma = reference.quantile(0.5)
    difference = (speed[:, None] - speed[None, :]).square()
    similarity = torch.exp(-difference / sigma.square()) if sigma > 0 else (difference == 0).float()
    return torch.where(valid[:, None] & valid[None, :], similarity, 1.0)


def pair_similarity(
    scene: Any,
    target_features: Any,
    speed: Any,
    valid: Any,
    candidates: Sequence[int],
) -> Any:
    """Eq. 14: S(t, s) = (1/K) Σ_k min(S_app^k, S_mot) for all T scanned frames, [T, T] on the CPU.

    S_app^k = min(ĉ_g, ĉ_{h,k}): scene features ``scene`` [T, D] and target
    features ``target_features`` [T, K, D] (each target calibrated on its own);
    S_mot from ``speed`` and ``valid`` [T]. ``candidates`` are the positions
    of the candidate frames among the T frames.
    """
    import torch
    import torch.nn.functional as F

    if target_features.shape[1] == 0:
        raise ValueError("Uniqueness needs at least one target")
    scene = F.normalize(scene.detach().float().cpu(), dim=-1)
    target = F.normalize(target_features.detach().float().cpu(), dim=-1)
    scene_sim = calibrated(scene @ scene.T, candidates)  # [T, T]
    target_sim = calibrated(torch.einsum("tkd,skd->kts", target, target), candidates)  # [K, T, T]
    motion = motion_similarity(speed.float().cpu(), valid.cpu(), candidates)
    return torch.minimum(torch.minimum(scene_sim, target_sim), motion).mean(dim=0)


def uniqueness(similarity: Any, candidates: Sequence[int], selected: Sequence[int]) -> Any:
    """Eq. 15: U_t = 1 - max_{s in selected} S(t, s) for each candidate, in the order
    of ``candidates`` (all 1 if nothing is selected); positions among the T frames."""
    import torch

    rows = similarity[list(candidates)]
    if not selected:
        return torch.ones(len(rows))
    return 1 - rows[:, list(selected)].max(dim=1).values
