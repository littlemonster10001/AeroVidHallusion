"""AeroMirage-specific rules: what a question looks like and how it is asked."""

from aeromirage.benchmark.prompts import (
    CLUE_NAMES,
    SYSTEM_PROMPT,
    build_task_prompt,
    option_letters,
)

__all__ = ["CLUE_NAMES", "SYSTEM_PROMPT", "build_task_prompt", "option_letters"]
