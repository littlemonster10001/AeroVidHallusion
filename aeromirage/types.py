"""Data objects shared by the benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# The four AeroMirage question types; each has its own QA/<type>.jsonl file.
TASK_TYPES = ("binary_verification", "single_choice", "multiple_choice", "open_ended")


@dataclass(frozen=True)
class BenchmarkSample:
    """One evaluation item.

    An open-ended record expands into one sample per cloze item in its
    ``eval`` list; the open-ended-only fields below describe that item.
    """

    video_id: str
    question_id: str  # open_ended: f"{original_question_id}_e{eval_index:02d}"
    question_type: str
    question: str  # open_ended: the cloze sentence with exactly one ____
    options: list[str]
    reference_answer: str | list[str] | None
    video_path: Path
    source_file: str
    source_line: int
    original_question_id: str = ""
    eval_index: int = 0
    clue: str = ""
    answerability: str = "answerable"
    accepted_equivalents: tuple[str, ...] = ()
    original_question: str = ""
    prompt_question: str = ""  # original question with the reference fill redacted

    @property
    def key(self) -> tuple[str, str]:
        """Question IDs repeat across tasks, so results are keyed by both."""
        return self.question_type, self.question_id
