"""Evidence-aware frame selection (Eq. 3, Sec. 4.2, Appendix C.1).

Starting from the Probing frames, candidates are added one at a time: each
round takes the candidate that maximizes λq Q_t + λm M_t + λu U_t, where U_t
is recomputed against the frames chosen so far while Q_t and M_t stay fixed.
Ties go to the higher Q_t, then to the earlier frame. Frames are identified by
their positions among the scanned frames (Probing frames and candidates in
time order), as in the evidence tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from aerovera.evidence.uniqueness import uniqueness


@dataclass
class Pick:
    """One added frame and the scores it was chosen with."""

    position: int  # among the scanned frames
    relevance: float  # Q_t
    motion: float  # M_t
    uniqueness: float  # U_t against the frames chosen before it
    score: float  # λq Q_t + λm M_t + λu U_t


def select_frames(
    relevance: Any,
    motion: Any,
    similarity: Any,
    candidates: Sequence[int],
    selected: Sequence[int],
    count: int,
    weights: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> list[Pick]:
    """Greedily add up to ``count`` of ``candidates`` to ``selected`` (the Probing frames).

    ``relevance`` (Q_t) and ``motion`` (M_t) are [T] and ``similarity`` (S of
    Eq. 14) is [T, T] over the scanned frames; ``weights`` are (λq, λm, λu).
    Returns the picks in the order they were added; all candidates when there
    are no more than ``count``.
    """
    import torch

    q_weight, m_weight, u_weight = weights
    remaining = list(candidates)
    chosen = list(selected)
    picks = []
    for _ in range(min(count, len(remaining))):
        index = torch.tensor(remaining)
        q, m = relevance[index].float(), motion[index].float()
        u = uniqueness(similarity, remaining, chosen)
        scores = q_weight * q + m_weight * m + u_weight * u
        best = max(range(len(remaining)), key=lambda i: (scores[i].item(), q[i].item(), -remaining[i]))
        position = remaining.pop(best)
        chosen.append(position)
        picks.append(Pick(position, q[best].item(), m[best].item(), u[best].item(), scores[best].item()))
    return picks
