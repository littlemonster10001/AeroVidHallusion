"""Motion change M_t from RAFT optical flow with camera-motion compensation (Eq. 10-11, Appendix C.1).

Each scanned frame t gets a triplet of source frames t - Δ, t, t + Δ
(Δ = 0.3 s; at the first and last frame of the video the triplet lies on one
side). RAFT-Small estimates the flows from the triplet's middle frame to the
other two, on frames decoded at a maximum side of 640 pixels, so that both
intervals refer to the same pixels. The partial affine transform that RANSAC
fits to a grid of flow correspondences is taken as camera motion and
subtracted, which leaves per pixel the velocity of scene objects after the
middle frame, v_t(p), and before it, v_{t-1}(p). V_t and A_t are the 99th
percentiles of ||v_t(p)|| and ||v_t(p) - v_{t-1}(p)|| over the pixels that
stay in the frame, divided by the image diagonal: a high percentile keeps
small moving objects that a mean over the whole frame would dilute. A frame
whose camera fit fails in either interval is invalid and gets M_t = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from aerovera.evidence.candidates import frame_size, read_frames

MIN_CAMERA_POINTS = 12  # grid correspondences needed to fit the camera motion at all


@dataclass
class FrameMotion:
    """Motion evidence of the scanned frames of one video, on the CPU."""

    motion: Any  # [T] M_t in [0, 1); 0 for invalid frames
    speed: Any  # [T] V_t: residual speed in image diagonals per second (for uniqueness); 0 for invalid frames
    valid: Any  # [T] bool: the camera fit of both intervals was reliable


def motion_triplet(frame_id: int, total_frames: int, fps: float, interval: float = 0.3) -> tuple[int, int, int]:
    """Source frames about ``interval`` seconds before and after ``frame_id``, in time order.

    The first frame takes (0, Δ, 2Δ) and the last frame (last - 2Δ, last - Δ,
    last), so that every frame has two consecutive intervals; frames close to
    either end get a shorter interval on that side.
    """
    if total_frames < 3:
        raise ValueError("A motion triplet needs a video of at least 3 frames")
    if not 0 <= frame_id < total_frames:
        raise ValueError(f"frame_id {frame_id} outside [0, {total_frames - 1}]")
    step = max(1, round(interval * fps))
    last = total_frames - 1
    if frame_id == 0:
        return 0, min(step, last - 1), min(2 * step, last)
    if frame_id == last:
        return max(0, last - 2 * step), max(1, last - step), last
    return max(0, frame_id - step), frame_id, min(last, frame_id + step)


def flow_size(height: int, width: int, max_side: int = 640) -> tuple[int, int]:
    """(height, width) of the RAFT input: scaled to a maximum side of ``max_side``
    (never enlarged), both sides rounded to multiples of 8 and at least 128."""
    scale = min(1.0, max_side / max(height, width))
    return max(128, 8 * round(height * scale / 8)), max(128, 8 * round(width * scale / 8))


def camera_residual(
    flow: Any,
    grid_step: int = 16,
    ransac_threshold: float = 2.0,
    min_inlier_ratio: float = 0.5,
) -> tuple[Any, Any] | None:
    """One flow [H, W, 2] minus the camera motion, and which pixels stay in the frame [H, W].

    Correspondences p -> p + F(p) on a grid with a stride of ``grid_step``
    pixels (those leaving the frame dropped) fit a partial affine transform
    (rotation, uniform scale, translation) with RANSAC. Returns None when the
    flow is not finite, too few points remain, or fewer than
    ``min_inlier_ratio`` of them are inliers.
    """
    import cv2
    import numpy as np

    flow = np.asarray(flow, dtype=np.float32)
    if not np.isfinite(flow).all():
        return None
    height, width = flow.shape[:2]
    ys, xs = np.mgrid[:height, :width].astype(np.float32)
    moved_x, moved_y = xs + flow[..., 0], ys + flow[..., 1]
    inside = (moved_x >= 0) & (moved_x < width) & (moved_y >= 0) & (moved_y < height)
    grid = (slice(grid_step // 2, None, grid_step),) * 2
    use = inside[grid]
    if use.sum() < MIN_CAMERA_POINTS:
        return None
    points = np.stack((xs[grid][use], ys[grid][use]), axis=1)
    moved = np.stack((moved_x[grid][use], moved_y[grid][use]), axis=1)
    matrix, inliers = cv2.estimateAffinePartial2D(
        points, moved, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
    if matrix is None or inliers.mean() < min_inlier_ratio:
        return None
    camera = np.stack((
        (matrix[0, 0] - 1) * xs + matrix[0, 1] * ys + matrix[0, 2],
        matrix[1, 0] * xs + (matrix[1, 1] - 1) * ys + matrix[1, 2],
    ), axis=-1)
    return flow - camera, inside


def speed_and_change(
    backward: tuple[Any, Any],
    forward: tuple[Any, Any],
    before: float,
    after: float,
    diagonal: float,
    percentile: float = 99.0,
) -> tuple[float, float] | None:
    """Eq. 10 from the camera residuals of the flows from the middle frame back
    (``before`` seconds) and forward (``after`` seconds): V_t and A_t, the
    ``percentile`` of ||v_t(p)|| and ||v_t(p) - v_{t-1}(p)|| over the pixels
    that stay in the frame in both flows, in image diagonals per second.
    None if no pixel does."""
    import numpy as np

    (back, back_inside), (ahead, ahead_inside) = backward, forward
    keep = back_inside & ahead_inside
    if not keep.any():
        return None
    current = ahead[keep] / after  # v_t(p), pixels per second
    previous = -back[keep] / before  # v_{t-1}(p): the backward flow points the other way
    speed = np.percentile(np.linalg.norm(current, axis=1), percentile) / diagonal
    change = np.percentile(np.linalg.norm(current - previous, axis=1), percentile) / diagonal
    return float(speed), float(change)


def motion_change(
    speed: float,
    change: float,
    gamma: float = 0.3,
    speed_scale: float = 0.03,
    change_scale: float = 0.1,
) -> float:
    """Eq. 11: γ g(V_t; κV) + (1 - γ) g(A_t; κA) with g(x; κ) = x / (x + κ)."""
    return gamma * speed / (speed + speed_scale) + (1 - gamma) * change / (change + change_scale)


class MotionChange:
    """RAFT-Small from local torchvision weights, and M_t of the scanned frames of a video."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda:1",
        interval: float = 0.3,
        max_side: int = 640,
        num_flow_updates: int = 12,
        grid_step: int = 16,
        ransac_threshold: float = 2.0,
        min_inlier_ratio: float = 0.5,
        percentile: float = 99.0,
        gamma: float = 0.3,
        speed_scale: float = 0.03,
        change_scale: float = 0.1,
        batch_size: int = 16,
    ):
        if max_side < 128:
            raise ValueError("max_side must be at least 128 (RAFT's smallest input)")
        if batch_size < 2:
            raise ValueError("batch_size must hold the two flows of a frame")
        self.weights = Path(weights)
        self.device = device
        self.interval, self.max_side, self.num_flow_updates = interval, max_side, num_flow_updates
        self.grid_step, self.ransac_threshold, self.min_inlier_ratio = grid_step, ransac_threshold, min_inlier_ratio
        self.percentile, self.gamma, self.speed_scale, self.change_scale = percentile, gamma, speed_scale, change_scale
        self.batch_size = batch_size
        self.model: Any = None

    def load(self) -> None:
        import torch
        from torchvision.models.optical_flow import raft_small

        if not self.weights.is_file():
            raise FileNotFoundError(f"RAFT weights not found: {self.weights}")
        model = raft_small(weights=None)
        model.load_state_dict(torch.load(self.weights, map_location="cpu", weights_only=True))
        self.model = model.to(self.device).eval()

    def score(self, path: str | Path, frame_ids: Sequence[int], total_frames: int, fps: float) -> FrameMotion:
        """M_t, V_t and validity for each of ``frame_ids``, in that order.

        All triplet frames are decoded once at the RAFT input size; frames
        that share a triplet (only possible at the ends) share its flows.
        """
        import numpy as np
        import torch

        triplets = [motion_triplet(frame_id, total_frames, fps, self.interval) for frame_id in frame_ids]
        distinct = sorted(set(triplets))
        needed = sorted({frame_id for triplet in distinct for frame_id in triplet})
        height, width = flow_size(*frame_size(path), self.max_side)
        frames = torch.from_numpy(np.stack(read_frames(path, needed, (height, width))))
        images = dict(zip(needed, frames.to(self.device).permute(0, 3, 1, 2).float() / 127.5 - 1))  # RAFT input in [-1, 1]
        diagonal = math.hypot(height, width)

        results = {}  # triplet -> (V_t, A_t), or None if invalid
        step = self.batch_size // 2  # triplets per RAFT pass: two flows each
        with torch.inference_mode():
            for begin in range(0, len(distinct), step):
                chunk = distinct[begin:begin + step]
                first = torch.stack([images[b] for a, b, c in chunk for _ in range(2)])
                second = torch.stack([images[other] for a, b, c in chunk for other in (a, c)])
                flows = self.model(first, second, num_flow_updates=self.num_flow_updates)[-1]
                flows = flows.permute(0, 2, 3, 1).float().cpu().numpy()
                for (a, b, c), back, ahead in zip(chunk, flows[0::2], flows[1::2]):
                    residuals = [camera_residual(flow, self.grid_step, self.ransac_threshold, self.min_inlier_ratio)
                                 for flow in (back, ahead)]
                    results[a, b, c] = None if any(r is None for r in residuals) else speed_and_change(
                        *residuals, (b - a) / fps, (c - b) / fps, diagonal, self.percentile)

        motion = torch.zeros(len(triplets))
        speed = torch.zeros(len(triplets))
        valid = torch.zeros(len(triplets), dtype=torch.bool)
        for t, triplet in enumerate(triplets):
            if results[triplet] is None:
                continue
            speed_t, change_t = results[triplet]
            motion[t] = motion_change(speed_t, change_t, self.gamma, self.speed_scale, self.change_scale)
            speed[t] = speed_t
            valid[t] = True
        return FrameMotion(motion, speed, valid)
