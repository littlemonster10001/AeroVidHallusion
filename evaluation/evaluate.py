"""Score prediction files offline: parse answers, compute metrics, write XLSX.

    python -m evaluation.evaluate outputs/predictions/qwen35_9b_vllm.jsonl
    python -m evaluation.evaluate outputs/predictions/*.jsonl          # also writes a leaderboard

Questions and references come from the QA files: the dataset root is taken
from the prediction file's meta line, or from --dataset-root for files
written by other methods. Every question of the task counts; missing or
failed answers are wrong. Judge labels are read from
``<judgments-dir>/<same file name>`` when it exists (see evaluation.judge).
Run from the AeroVidHallusion directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aeromirage.benchmark.loader import load_samples
from aeromirage.benchmark.metrics import compute_metrics
from aeromirage.benchmark.parser import is_correct, normalize_reference, parse_response
from aeromirage.config import DatasetConfig
from aeromirage.inference.predictions import read_predictions
from aeromirage.types import BenchmarkSample
from evaluation.judge import read_judgments


RESULT_COLUMNS = [
    "question_type", "question_id", "original_question_id", "video_id", "clue", "answerability",
    "question", "options", "reference_answer", "reference", "model_answer", "prediction",
    "correct", "judge_label", "status", "error", "raw_response",
]
# Columns of the cross-model table, in order.
LEADERBOARD_METRICS = [
    "coverage",
    "objective/accuracy",
    "binary_verification/accuracy",
    "binary_verification/hallucination_rate",
    "single_choice/accuracy",
    "multiple_choice/accuracy",
    "multiple_choice/option_f1",
    "open_ended/answerable_accuracy",
    "open_ended/judge_correct",
    "open_ended/judge_f1",
]
EXCEL_CELL_LIMIT = 32_000  # Excel holds 32,767 characters per cell; the JSONL keeps the full text


def score_rows(
    samples: list[BenchmarkSample],
    predictions: dict[tuple[str, str], dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        record = predictions.get(sample.key)
        status = record.get("status", "ok") if record else "missing"
        raw = str(record.get("raw_response") or "") if record else ""
        model_answer, prediction = parse_response(raw, sample) if status == "ok" else ("", "")
        judgment = judgments.get(sample.question_id) if sample.question_type == "open_ended" else None
        rows.append({
            "question_type": sample.question_type,
            "question_id": sample.question_id,
            "original_question_id": sample.original_question_id,
            "video_id": sample.video_id,
            "clue": sample.clue,
            "answerability": sample.answerability if sample.question_type == "open_ended" else "",
            "question": sample.question,
            "options": json.dumps(sample.options, ensure_ascii=False) if sample.options else "",
            "reference_answer": json.dumps(sample.reference_answer, ensure_ascii=False),
            "reference": normalize_reference(sample),
            "model_answer": model_answer,
            "prediction": prediction,
            "correct": status == "ok" and is_correct(sample, prediction),
            "judge_label": (judgment or {}).get("label") or "",
            "status": status,
            "error": str(record.get("error") or "") if record else "",
            "raw_response": raw,
        })
    return rows


def evaluate_file(
    path: Path,
    *,
    dataset_root: Path | None = None,
    task: str | None = None,
    judgments_dir: Path | None = None,
    answered_only: bool = False,
) -> tuple[list[dict[str, Any]], list[tuple[str, Any]], dict[str, Any]]:
    """(rows, metrics, info) for one prediction file."""
    meta, predictions = read_predictions(path)
    root = dataset_root or meta.get("dataset_root")
    if not root:
        raise ValueError(f"{path} has no meta line; pass --dataset-root")
    dataset = DatasetConfig(root=Path(root), task=task or meta.get("task", "all"))
    samples = load_samples(dataset, check_videos=False)
    if answered_only:  # quick checks on partial runs
        samples = [sample for sample in samples if sample.key in predictions]
    judgment_path = (judgments_dir or path.parent.parent / "judgments") / path.name
    judgments = read_judgments(judgment_path) if judgment_path.exists() else {}
    rows = score_rows(samples, predictions, judgments)
    info = {
        "predictions": str(path),
        "judgments": str(judgment_path) if judgments else "",
        "dataset_root": str(root),
        "task": dataset.task,
        "answered_only": answered_only,
        **{f"fingerprint/{k}": v for k, v in meta.get("fingerprint", {}).items()},
    }
    return rows, compute_metrics(rows), info


def _sheet(workbook: Any, title: str, header: list[str], rows: list[list[Any]], widths: dict[str, int]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    sheet = workbook.create_sheet(title)
    sheet.append(header)
    for row in rows:
        sheet.append([
            value[:EXCEL_CELL_LIMIT] if isinstance(value, str) else value for value in row
        ])
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center")
    for index, name in enumerate(header, 1):
        sheet.column_dimensions[get_column_letter(index)].width = widths.get(name, 16)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def write_report(
    path: Path, rows: list[dict[str, Any]], metrics: list[tuple[str, Any]], info: dict[str, Any]
) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)
    _sheet(workbook, "Summary", ["metric", "value"], [list(m) for m in metrics], {"metric": 44, "value": 14})
    _sheet(
        workbook, "Results", RESULT_COLUMNS, [[row[c] for c in RESULT_COLUMNS] for row in rows],
        {"question": 50, "options": 40, "model_answer": 30, "raw_response": 60, "error": 40},
    )
    _sheet(workbook, "Info", ["key", "value"], [list(item) for item in info.items()], {"key": 28, "value": 80})
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def write_leaderboard(path: Path, reports: dict[str, list[tuple[str, Any]]]) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)
    rows = [[name, *[dict(metrics).get(m) for m in LEADERBOARD_METRICS]] for name, metrics in reports.items()]
    _sheet(workbook, "Leaderboard", ["model", *LEADERBOARD_METRICS], rows, {"model": 36})
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score AeroMirage prediction files.")
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--dataset-root", type=Path, help="for files without a meta line")
    parser.add_argument("--task", choices=["all", "objective", "binary_verification", "single_choice",
                                           "multiple_choice", "open_ended"])
    parser.add_argument("--judgments-dir", type=Path, help="default: outputs/judgments next to predictions/")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--answered-only", action="store_true",
                        help="score only questions that have a prediction (for partial test runs)")
    args = parser.parse_args(argv)

    reports = {}
    for path in args.predictions:
        rows, metrics, info = evaluate_file(
            path, dataset_root=args.dataset_root, task=args.task,
            judgments_dir=args.judgments_dir, answered_only=args.answered_only,
        )
        out = args.output_dir / f"{path.stem}.xlsx"
        write_report(out, rows, metrics, info)
        summary = dict(metrics)
        print(f"{path.name}: {summary['answered']}/{summary['questions']} answered, "
              f"objective accuracy {summary['objective/accuracy']} -> {out}")
        reports[path.stem] = metrics
    if len(reports) > 1:
        board = args.output_dir / "leaderboard.xlsx"
        write_leaderboard(board, reports)
        print(f"Leaderboard -> {board}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
