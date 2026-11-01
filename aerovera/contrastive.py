"""Model-independent parts of the contrastive decision (Eq. 2): masking, q = q+ + alpha (q+ - q-), label choice."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence


def top_percent(scores: Sequence[float], percent: float) -> list[int]:
    """Indices of the ceil(M * percent / 100) highest scores, in ascending order.

    The negative branch masks these visual tokens (Appendix C.3, rho = 20).
    Scores are ranked by their signed value, not their magnitude: a negative
    attribution means the token speaks against the answer. Ties keep the
    lower index.
    """
    if not 0 <= percent <= 100:
        raise ValueError(f"percent must be in [0, 100], got {percent}")
    values = [float(score) for score in scores]
    count = math.ceil(len(values) * percent / 100)
    ranked = sorted(range(len(values)), key=lambda index: -values[index])
    return sorted(ranked[:count])


def contrast(positive: Any, negative: Any, alpha: float) -> Any:
    """q = q+ + alpha (q+ - q-) on raw logits (Eq. 2); a per-branch constant does not change the choice."""
    return positive + alpha * (positive - negative)


def best_label(labels: Sequence[str], scores: Any) -> str:
    """The label with the highest score; ties go to the earlier label."""
    values = [float(score) for score in scores]
    return labels[max(range(len(values)), key=lambda index: (values[index], -index))]


def top_candidates(
    positive: Any, negative: Any, combined: Any, decode: Callable[[int], str], count: int = 20
) -> list[dict[str, Any]]:
    """First-position candidates of an open-ended answer: the union of the top
    ``count`` tokens of q+, q- and q, each with its three scores, best q first."""
    import torch

    ids = torch.cat([scores.topk(count).indices for scores in (positive, negative, combined)]).unique().tolist()
    ids.sort(key=lambda token: -float(combined[token]))
    return [
        {"token_id": token, "text": decode(token), "positive": round(float(positive[token]), 4),
         "negative": round(float(negative[token]), 4), "contrastive": round(float(combined[token]), 4)}
        for token in ids
    ]
