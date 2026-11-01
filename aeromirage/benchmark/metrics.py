"""Benchmark metrics over scored rows.

A row is one question: ``question_type``, ``status`` (ok / error / missing),
``prediction``, ``reference``, ``correct``, ``answerability`` and, for judged
open-ended rows, ``judge_label``. Every question counts in the denominator:
missing or failed answers are wrong, and ``coverage`` reports how many were
answered at all.
"""

from __future__ import annotations

from typing import Any

from aeromirage.benchmark.parser import REFUSAL
from aeromirage.types import TASK_TYPES


JUDGE_LABELS = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def compute_metrics(rows: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    ok = [row for row in rows if row["status"] == "ok"]
    metrics: list[tuple[str, Any]] = [
        ("questions", len(rows)),
        ("answered", len(ok)),
        ("errors", sum(row["status"] == "error" for row in rows)),
        ("missing", sum(row["status"] == "missing" for row in rows)),
        ("coverage", ratio(len(ok), len(rows))),
    ]
    objective = [row for row in rows if row["question_type"] != "open_ended"]
    metrics.append(("objective/accuracy", ratio(sum(row["correct"] for row in objective), len(objective))))

    for task in TASK_TYPES:
        task_rows = [row for row in rows if row["question_type"] == task]
        metrics += [
            (f"{task}/questions", len(task_rows)),
            (f"{task}/coverage", ratio(sum(row["status"] == "ok" for row in task_rows), len(task_rows))),
            (f"{task}/parsed", sum(bool(row["prediction"]) for row in task_rows)),
            (f"{task}/accuracy", ratio(sum(row["correct"] for row in task_rows), len(task_rows))),
        ]

    # Hallucination: saying Yes to a statement the video does not support.
    negatives = [r for r in rows if r["question_type"] == "binary_verification" and r["reference"] == "No"]
    false_yes = sum(row["prediction"] == "Yes" for row in negatives)
    metrics += [
        ("binary_verification/negatives", len(negatives)),
        ("binary_verification/hallucination_rate", ratio(false_yes, len(negatives))),
    ]

    true_pos = false_pos = false_neg = 0
    for row in rows:
        if row["question_type"] != "multiple_choice":
            continue
        predicted = set(filter(None, row["prediction"].split(",")))
        expected = set(filter(None, row["reference"].split(",")))
        true_pos += len(predicted & expected)
        false_pos += len(predicted - expected)
        false_neg += len(expected - predicted)
    precision = ratio(true_pos, true_pos + false_pos)
    recall = ratio(true_pos, true_pos + false_neg)
    f1 = (
        round(2 * precision * recall / (precision + recall), 6)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    metrics += [
        ("multiple_choice/option_precision", precision),
        ("multiple_choice/option_recall", recall),
        ("multiple_choice/option_f1", f1),
    ]

    open_rows = [row for row in rows if row["question_type"] == "open_ended"]
    answerable = [row for row in open_rows if row["answerability"] != "unanswerable"]
    unanswerable = [row for row in open_rows if row["answerability"] == "unanswerable"]
    metrics += [
        ("open_ended/refusal_rate", ratio(sum(r["prediction"] == REFUSAL for r in open_rows), len(open_rows))),
        ("open_ended/answerable_accuracy", ratio(sum(r["correct"] for r in answerable), len(answerable))),
        ("open_ended/unanswerable_refusal_rate", ratio(sum(r["correct"] for r in unanswerable), len(unanswerable))),
    ]
    return metrics + compute_judge_metrics(open_rows)


def compute_judge_metrics(open_rows: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """Video SimpleQA-style rates; unjudged rows count as NOT_ATTEMPTED."""
    if not any(row.get("judge_label") for row in open_rows):
        return []
    labels = [row.get("judge_label") or "NOT_ATTEMPTED" for row in open_rows]
    counts = {label: labels.count(label) for label in JUDGE_LABELS}
    total = len(labels)
    attempted = counts["CORRECT"] + counts["INCORRECT"]
    correct = ratio(counts["CORRECT"], total)
    given_attempted = ratio(counts["CORRECT"], attempted)
    if correct is None or given_attempted is None:
        f1 = None
    elif correct + given_attempted == 0:
        f1 = 0.0
    else:
        f1 = round(2 * correct * given_attempted / (correct + given_attempted), 6)
    return [
        ("open_ended/judged", sum(bool(row.get("judge_label")) for row in open_rows)),
        ("open_ended/judge_correct", correct),
        ("open_ended/judge_incorrect", ratio(counts["INCORRECT"], total)),
        ("open_ended/judge_not_attempted", ratio(counts["NOT_ATTEMPTED"], total)),
        ("open_ended/judge_correct_given_attempted", given_attempted),
        ("open_ended/judge_f1", f1),
    ]
