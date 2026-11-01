"""Read and validate the AeroMirage QA files (``<root>/QA/<task>.jsonl``).

Every record is checked; all problems are reported together. An open-ended
record expands into one sample per cloze item in its ``eval`` list.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from aeromirage.benchmark.prompts import CLUE_NAMES, option_letters
from aeromirage.config import DatasetConfig
from aeromirage.types import TASK_TYPES, BenchmarkSample


TASK_FILES = {task: f"{task}.jsonl" for task in TASK_TYPES}
OBJECTIVE_TASKS = tuple(task for task in TASK_TYPES if task != "open_ended")
CHINESE = re.compile(r"[㐀-鿿]")
# The QA files store binary answers in Chinese; samples use Yes / No.
BINARY_ANSWERS = {"是": "Yes", "否": "No", "Yes": "Yes", "No": "No"}


class DatasetValidationError(ValueError):
    """Raised when one or more QA records are invalid."""


def selected_tasks(task: str) -> tuple[str, ...]:
    if task == "all":
        return TASK_TYPES
    if task == "objective":
        return OBJECTIVE_TASKS
    if task in TASK_TYPES:
        return (task,)
    raise ValueError(f"unknown task: {task!r}")


def qa_sha256(root: Path) -> str:
    """Hash of the four QA files, for the experiment fingerprint."""
    digest = hashlib.sha256()
    for filename in sorted(TASK_FILES.values()):
        path = root / "QA" / filename
        digest.update(filename.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def redact_reference(question: str, reference: str | None, clue: str) -> str:
    """Hide a cloze's answer if the original question already states it."""
    token = (reference or "").strip()
    if not token:
        return question
    if clue == "count" and token.isdecimal() and len(token) == 1:
        # A bare digit in a time anchor (e.g. "2秒") is not a count answer.
        pattern = rf"(?<!\d){re.escape(token)}(?=(?:个|辆|人|只|台|艘|架|名|条|棵))"
    elif clue == "color" and re.fullmatch(r"[㐀-鿿]+", token) and not token.endswith("色"):
        pattern = re.escape(token) + r"色?"
    else:
        pattern = re.escape(token)
    return re.sub(pattern, "〔省略〕", question, flags=re.IGNORECASE)


def _text(record: dict[str, Any], key: str, errors: list[str]) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")
        return ""
    return value.strip()


def _check_options(options: Any, errors: list[str]) -> list[str]:
    if not isinstance(options, list) or any(not isinstance(item, str) for item in options):
        errors.append("options must be a list of strings")
        return []
    options = [item.strip() for item in options]
    if not 2 <= len(options) <= 8:
        errors.append(f"choice question must have 2-8 options, got {len(options)}")
    if len(options) != len(set(options)):
        errors.append("choice options are not unique")
    expected = [chr(ord("A") + index) for index in range(len(options))]
    if option_letters(options) != expected:
        errors.append(f"option labels must be {expected}")
    return options


def _check_answer(task: str, answer: Any, options: list[str], errors: list[str]) -> Any:
    if task == "binary_verification":
        if answer not in BINARY_ANSWERS:
            errors.append("binary_verification answer must be 是 or 否")
            return None
        return BINARY_ANSWERS[answer]
    labels = option_letters(options)
    if task == "single_choice":
        if not isinstance(answer, str) or answer.strip().upper() not in labels:
            errors.append("single_choice answer must be one option label")
            return None
        return answer.strip().upper()
    if task == "multiple_choice":
        if not isinstance(answer, list) or not answer:
            errors.append("multiple_choice answer must be a non-empty list")
            return None
        # One source row spells its second label ",B"; fix that delimiter
        # typo in memory only, so the QA file and its hash stay unchanged.
        normalized = [re.sub(r"^,\s*", "", str(item).strip()).upper() for item in answer]
        if any(item not in labels for item in normalized) or len(set(normalized)) != len(normalized):
            errors.append(f"multiple_choice answer has invalid or repeated labels: {answer!r}")
            return None
        return normalized
    if not isinstance(answer, str) or not answer.strip():
        errors.append("open_ended answer must be a non-empty string")
    return answer


def _check_eval_items(record: dict[str, Any], errors: list[str]) -> list[dict[str, Any]]:
    items = record.get("eval")
    if not isinstance(items, list) or not 1 <= len(items) <= 5:
        errors.append("open_ended eval must contain 1-5 cloze items")
        return []
    checked = []
    for index, item in enumerate(items, 1):
        label = f"eval[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label} must be a JSON object")
            continue
        cloze = item.get("cloze")
        if not isinstance(cloze, str) or len(re.findall(r"(?<!_)____(?!_)", cloze)) != 1 or "_____" in cloze:
            errors.append(f"{label}.cloze must contain exactly one ____ blank")
        if item.get("clue") not in CLUE_NAMES:
            errors.append(f"{label}.clue must be one of {sorted(CLUE_NAMES)}")
        answerability = item.get("answerability", record.get("answerability", "answerable"))
        if answerability not in {"answerable", "unanswerable"}:
            errors.append(f"{label}.answerability must be answerable or unanswerable")
        reference = item.get("reference")
        if answerability == "unanswerable" and reference is not None:
            errors.append(f"{label}.reference must be null when unanswerable")
        if answerability == "answerable" and (not isinstance(reference, str) or not reference.strip()):
            errors.append(f"{label}.reference must be a non-empty string")
        equivalents = item.get("accepted_equivalents", [])
        if not isinstance(equivalents, list) or any(
            not isinstance(value, str) or not value.strip() for value in equivalents
        ):
            errors.append(f"{label}.accepted_equivalents must be a list of non-empty strings")
        checked.append({**item, "answerability": answerability})
    return checked


def parse_record(
    record: Any,
    *,
    task: str,
    source: str,
    line: int,
    dataset: DatasetConfig,
    check_videos: bool,
) -> tuple[list[BenchmarkSample], list[str]]:
    """Validate one JSONL record and turn it into samples."""
    if not isinstance(record, dict):
        return [], ["record must be a JSON object"]
    errors: list[str] = []
    video_id = _text(record, "video_id", errors)
    question_id = _text(record, "question_id", errors)
    question = _text(record, "question", errors)
    if video_id and not re.fullmatch(r"\d{4}", video_id):
        errors.append(f"invalid video_id: {video_id!r}")
    if question_id and not re.fullmatch(r"\d{4}q\d{6}", question_id):
        errors.append(f"invalid question_id: {question_id!r}")
    if record.get("question_type") != task:
        errors.append(f"question_type {record.get('question_type')!r} does not match file {task!r}")
    if question and not CHINESE.search(question):
        errors.append("question must be Chinese (the prompts are Chinese)")

    choice = task in {"single_choice", "multiple_choice"}
    options = _check_options(record.get("options"), errors) if choice else []
    answer = _check_answer(task, record.get("answer"), options, errors)
    items = _check_eval_items(record, errors) if task == "open_ended" else []

    video_path = Path(dataset.root) / dataset.video_dir / f"{video_id}.mp4"
    if check_videos and video_id and not video_path.is_file():
        errors.append(f"video not found: {video_path}")
    if errors:
        return [], errors

    common = dict(
        video_id=video_id, question_type=task, video_path=video_path,
        source_file=source, source_line=line,
    )
    if task != "open_ended":
        return [BenchmarkSample(
            question_id=question_id, question=question, options=options,
            reference_answer=answer, **common,
        )], []
    return [
        BenchmarkSample(
            question_id=f"{question_id}_e{index:02d}",
            question=item["cloze"].strip(),
            options=[],
            reference_answer=item.get("reference"),
            original_question_id=question_id,
            eval_index=index,
            clue=item["clue"],
            answerability=item["answerability"],
            accepted_equivalents=tuple(item.get("accepted_equivalents", [])),
            original_question=question,
            prompt_question=redact_reference(question, item.get("reference"), item["clue"]),
            **common,
        )
        for index, item in enumerate(items, 1)
    ], []


def load_samples(
    dataset: DatasetConfig,
    *,
    limit: int = 0,
    limit_per_task: int = 0,
    check_videos: bool = True,
) -> list[BenchmarkSample]:
    """All samples of ``dataset.task``, in file order; raises on any invalid record."""
    samples: list[BenchmarkSample] = []
    problems: list[str] = []
    for task in selected_tasks(dataset.task):
        path = Path(dataset.root) / "QA" / TASK_FILES[task]
        if not path.is_file():
            raise FileNotFoundError(f"QA file not found: {path}")
        task_samples: list[BenchmarkSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    problems.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
                    continue
                parsed, errors = parse_record(
                    record, task=task, source=path.name, line=line_number,
                    dataset=dataset, check_videos=check_videos,
                )
                if errors:
                    problems.append(f"{path.name}:{line_number}: " + "; ".join(errors))
                task_samples.extend(parsed)
        samples.extend(task_samples[:limit_per_task] if limit_per_task else task_samples)

    if problems:
        preview = "\n".join(f"  {problem}" for problem in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise DatasetValidationError(
            f"{len(problems)} invalid QA records:\n{preview}{more}"
        )
    duplicates = [key for key, count in Counter(s.key for s in samples).items() if count > 1]
    if duplicates:
        raise DatasetValidationError(f"duplicate (question_type, question_id): {duplicates[:10]}")
    return samples[:limit] if limit else samples
