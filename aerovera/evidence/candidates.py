"""Candidate frames sampled in the time gaps between the uniform Probing frames."""

from __future__ import annotations

from bisect import insort
from pathlib import Path
from typing import Any, Sequence


def candidate_frame_ids(original_ids: Sequence[int], total_frames: int, count: int) -> list[int]:
    """Up to ``count`` new source frames that fill the largest time gaps, in ascending order.

    Each candidate goes to the gap that is currently largest: the midpoint of
    two chosen frames, or the first / last frame of the video for a gap at
    either end. Equal gaps go to the one nearest to an unused position of a
    uniform grid, so candidates do not crowd one end of the video. A short
    video gets fewer candidates when it runs out of unused frames.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    chosen = sorted(set(original_ids))
    if not chosen or chosen[0] < 0 or chosen[-1] >= total_frames:
        raise ValueError("original_ids must be source frames of the video")
    count = min(count, total_frames - len(chosen))
    if count <= 0:
        return []
    last = total_frames - 1
    targets = [last / 2] if count == 1 else [i * last / (count - 1) for i in range(count)]

    candidates = []
    for _ in range(count):
        gaps = []  # (frames to the nearest chosen frame, candidate)
        for left, right in zip([-1, *chosen], [*chosen, total_frames]):
            if right - left <= 1:
                continue
            if left == -1:
                gaps.append((right, 0))
            elif right == total_frames:
                gaps.append((last - left, last))
            else:
                middle = (left + right) // 2
                gaps.append((min(middle - left, right - middle), middle))
        _, frame = min(gaps, key=lambda gap: (-gap[0], min(abs(gap[1] - t) for t in targets), gap[1]))
        candidates.append(frame)
        insort(chosen, frame)
        targets.pop(min(range(len(targets)), key=lambda i: abs(frame - targets[i])))
    return sorted(candidates)


def read_frames(path: str | Path, frame_ids: Sequence[int], size: tuple[int, int] | None = None) -> list[Any]:
    """Source frames as RGB uint8 arrays [H, W, 3], in the order of ``frame_ids``.

    Read with decord, the reader that the Probing Pass uses through
    qwen-vl-utils, so that frame indices refer to the same frames. Full
    resolution by default; ``size`` = (height, width) lets the decoder scale
    the frames (antialiased), which is much faster for 4K video.
    """
    from decord import VideoReader, cpu

    ids = list(frame_ids)
    if not ids:
        return []
    scale = {} if size is None else {"height": size[0], "width": size[1]}
    reader = VideoReader(str(path), ctx=cpu(0), **scale)
    if min(ids) < 0 or max(ids) >= len(reader):
        raise ValueError(f"{path}: frame ids outside [0, {len(reader) - 1}]")
    return list(reader.get_batch(ids).asnumpy())


def frame_size(path: str | Path) -> tuple[int, int]:
    """(height, width) of the source frames."""
    from decord import VideoReader, cpu

    height, width = VideoReader(str(path), ctx=cpu(0))[0].shape[:2]
    return int(height), int(width)
