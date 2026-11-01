"""Prediction JSONL: one raw answer per line, appended as soon as it exists.

Line 1 is a meta record (``"type": "meta"``) holding the dataset location and
the experiment fingerprint; every other line is a prediction. Readers keep the
last line per (question_type, question_id), so a retried question simply
appends a newer line. Files written by other methods may omit the meta line
and need only ``question_type``, ``question_id`` and ``raw_response``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from aeromirage.types import BenchmarkSample


FORMAT_VERSION = 4


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_predictions(path: str | Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    """(meta, latest prediction per key). Tolerates a torn last line."""
    meta: dict[str, Any] = {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):  # interrupted while writing the last line
                break
            raise ValueError(f"{path}:{number}: invalid JSON")
        if record.get("type") == "meta":
            meta = record
            continue
        try:
            key = (str(record["question_type"]), str(record["question_id"]))
        except KeyError as exc:
            raise ValueError(f"{path}:{number}: prediction lacks {exc}") from exc
        records[key] = record
    return meta, records


class PredictionStore:
    """Append-only writer with resume, overwrite and fingerprint checks."""

    def __init__(
        self,
        path: str | Path,
        meta: dict[str, Any],
        *,
        resume: bool = True,
        overwrite: bool = False,
    ):
        self.path = Path(path)
        self.meta = {"type": "meta", "format_version": FORMAT_VERSION, **meta}
        self.records: dict[tuple[str, str], dict[str, Any]] = {}
        if self.path.exists() and not overwrite:
            if not resume:
                raise FileExistsError(f"{self.path} exists; set output.resume or output.overwrite")
            stored, self.records = read_predictions(self.path)
            self._check_fingerprint(stored)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({**self.meta, "created_at": now()}, ensure_ascii=False) + "\n", encoding="utf-8")

    def _check_fingerprint(self, stored: dict[str, Any]) -> None:
        old = stored.get("fingerprint", {})
        new = self.meta.get("fingerprint", {})
        changed = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
        if changed:
            details = "\n".join(f"  {key}: file={old.get(key)!r} now={new.get(key)!r}" for key in changed)
            raise ValueError(
                f"{self.path} was produced by a different setup; resume refused:\n{details}\n"
                "Use a new output.predictions or set output.overwrite=true."
            )

    def pending(self, samples: list[BenchmarkSample], retry_errors: bool = True) -> list[BenchmarkSample]:
        def done(sample: BenchmarkSample) -> bool:
            record = self.records.get(sample.key)
            if record is None:
                return False
            return record.get("status") == "ok" or not retry_errors

        return [sample for sample in samples if not done(sample)]

    def append(
        self,
        sample: BenchmarkSample,
        *,
        raw_response: str = "",
        error: str = "",
        prompt: str = "",
        elapsed: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write one prediction line; ``extra`` adds method-specific fields
        (AeroVera's intermediate answers) after the standard ones."""
        record = {
            "type": "prediction",
            "question_type": sample.question_type,
            "question_id": sample.question_id,
            "video_id": sample.video_id,
            "raw_response": raw_response,
            "status": "error" if error else "ok",
            "error": error,
            "prompt": prompt,
            "elapsed": round(elapsed, 3),
            "run_timestamp": now(),
        }
        clash = set(extra or {}) & set(record)
        if clash:
            raise ValueError(f"extra fields would overwrite standard ones: {sorted(clash)}")
        record.update(extra or {})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[sample.key] = record
        return record
