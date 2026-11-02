"""AeroMirage-specific rules: what a question looks like and how it is asked."""

from aeromirage.benchmark.prompts import (
    CLUE_NAMES,
    META_OPTIONS,
    SYSTEM_PROMPT,
    VERIFICATION_STATEMENT,
    build_task_prompt,
    option_letters,
    option_text,
    ordinary_options,
    verification_prompt,
)

__all__ = [
    "CLUE_NAMES", "META_OPTIONS", "SYSTEM_PROMPT", "VERIFICATION_STATEMENT", "build_task_prompt",
    "option_letters", "option_text", "ordinary_options", "verification_prompt",
]
