"""Model-independent parts of the contrastive decision (Eq. 2): masking, q = q+ + alpha (q+ - q-), label and option choice."""

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


def confident(scores: Any, threshold: float) -> bool:
    """Whether the softmax of the option scores gives the best option at least ``threshold``.

    Below it AeroVera keeps the Probing answer (A21). With two options the best
    one always has at least 0.5, so a threshold up to 0.5 never applies to them.
    """
    values = [float(score) for score in scores]
    top = max(values)
    return 1 / sum(math.exp(value - top) for value in values) >= threshold


# Not "yes": YAML reads an unquoted yes as true.
OPTION_RULES = ("above_mean", "largest_gap", "sign")


def chosen_options(margins: dict[str, float], rule: str) -> list[str]:
    """Options of a multiple-choice question chosen from their yes-minus-no margins, in option order.

    ``sign``: every option whose verification is answered A (是), ties included.
    ``largest_gap``: sorted from high to low (ties keep the earlier option),
    the options above the largest drop between neighbouring margins; at least
    one, never all of them.
    ``above_mean``: the options whose margin is above the mean margin of the
    question's options; if none is (all equal), the first option. At least
    one, never all of them.
    A shift of every margin, such as a branch leaning towards 是, moves the
    sign cut only.
    """
    if rule not in OPTION_RULES:
        raise ValueError(f"option rule must be one of {OPTION_RULES}, got {rule!r}")
    labels = list(margins)
    if rule == "sign":
        return [label for label in labels if margins[label] >= 0]
    if len(labels) <= 1:
        return labels
    if rule == "above_mean":
        mean = sum(margins.values()) / len(labels)
        return [label for label in labels if margins[label] > mean] or [max(labels, key=margins.get)]
    ranked = sorted(labels, key=lambda label: (-margins[label], labels.index(label)))
    drops = [margins[high] - margins[low] for high, low in zip(ranked, ranked[1:])]
    kept = set(ranked[:drops.index(max(drops)) + 1])  # the first largest drop: the smaller set
    return [label for label in labels if label in kept]


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
