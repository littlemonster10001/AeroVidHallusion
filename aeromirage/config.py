"""Benchmark configuration: YAML validated by an OmegaConf structured schema.

The dataclasses below are the schema: unknown keys and wrong types fail when
the YAML is merged into them. Any field can be overridden from the command
line with OmegaConf's dot-list syntax, e.g. ``runtime.limit=5``.
``load_config`` returns plain dataclasses, so no other module depends on
OmegaConf.

Note: no ``from __future__ import annotations`` here; OmegaConf reads these
annotations at runtime and only understands ``typing`` generics
(``Dict``, ``Optional``), not ``dict[...]`` or ``X | None``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

from omegaconf import MISSING, OmegaConf

from aeromirage.types import TASK_TYPES

MULTIPLE_CHOICE_PROTOCOLS = ("per_option", "whole")


@dataclass
class ModelConfig:
    family: str = MISSING
    path: str = MISSING
    trust_remote_code: bool = False
    local_files_only: bool = False
    # Loading and generation options; each adapter validates its own keys.
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    root: Path = MISSING
    video_dir: str = "video_reencoded"
    task: str = "all"  # all | objective | one of TASK_TYPES


@dataclass
class GenerationConfig:
    limits: Dict[str, int] = MISSING  # max_new_tokens per question type
    # per_option: each ordinary option of a multiple-choice question asked as a
    # yes/no question, the options answered 是 form the answer; whole: one
    # answer listing the options (meta options shown).
    multiple_choice: str = "per_option"

    def limit_for(self, question_type: str) -> int:
        return self.limits[question_type]


@dataclass
class OutputConfig:
    predictions: Path = MISSING  # prediction JSONL written by inference
    resume: bool = True
    retry_errors: bool = True
    overwrite: bool = False


@dataclass
class RuntimeConfig:
    limit: int = 0
    limit_per_task: int = 0
    seed: int = 42
    fail_fast: bool = False


@dataclass
class BenchmarkConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    # Keys differ per model; the adapter's video_defaults validate them.
    video: Dict[str, Any] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        """Value checks the schema cannot express."""
        tasks = ("all", "objective", *TASK_TYPES)
        if self.dataset.task not in tasks:
            raise ValueError(f"dataset.task must be one of {tasks}, got {self.dataset.task!r}")
        limits = self.generation.limits
        if sorted(limits) != sorted(TASK_TYPES):
            raise ValueError(f"generation.limits must set exactly {TASK_TYPES}, got {sorted(limits)}")
        if any(value <= 0 for value in limits.values()):
            raise ValueError("generation.limits must be positive")
        if self.generation.multiple_choice not in MULTIPLE_CHOICE_PROTOCOLS:
            raise ValueError(f"generation.multiple_choice must be one of {MULTIPLE_CHOICE_PROTOCOLS}, "
                             f"got {self.generation.multiple_choice!r}")
        if self.runtime.limit < 0 or self.runtime.limit_per_task < 0:
            raise ValueError("runtime.limit and runtime.limit_per_task cannot be negative")
        if self.runtime.limit and self.runtime.limit_per_task:
            raise ValueError("runtime.limit and runtime.limit_per_task are mutually exclusive")


def load_config(path: "str | Path", overrides: Sequence[str] = ()) -> BenchmarkConfig:
    """Load a YAML file, apply ``key=value`` overrides, and validate."""
    merged = OmegaConf.merge(
        OmegaConf.structured(BenchmarkConfig),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(list(overrides)),
    )
    config = OmegaConf.to_object(merged)
    config.validate()
    return config
