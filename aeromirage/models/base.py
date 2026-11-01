"""Interface implemented by every multimodal model family.

Adapters import heavy dependencies (torch, transformers, vllm) inside
``load()`` and ``prepare_video()`` so that every family can be registered
from any Conda environment.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from aeromirage.config import ModelConfig


DTYPES = ("bfloat16", "float16", "float32")


class BaseModelAdapter(ABC):
    # Defaults for the YAML ``video`` section. Keys follow the model's official
    # video-reading code; ``num_frames`` is shared by every model so that all
    # of them see the same number of uniformly sampled frames.
    video_defaults: ClassVar[dict[str, Any]] = {"num_frames": 32}

    def __init__(self, config: ModelConfig, video: dict[str, Any] | None = None):
        self.config = config
        self.video = self._video_options(video or {})
        # The most recent prepare_video() result; the runner orders questions
        # by video, so consecutive questions on one video decode it once.
        self._video_path: str | None = None
        self._visual: Any = None
        self.last_prompt = ""  # inputs["prompt"] of the latest generate(), for the prediction file

    def _video_options(self, video: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(video) - set(self.video_defaults))
        if unknown:
            raise ValueError(
                f"Unknown video options for {self.config.family}: {unknown}; "
                f"allowed: {sorted(self.video_defaults)}"
            )
        options = {**self.video_defaults, **video}
        num_frames = options["num_frames"]
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError("video.num_frames must be a positive integer")
        return options

    @abstractmethod
    def load(self) -> None:
        """Load model-specific weights and processors."""

    @abstractmethod
    def prepare_video(self, video_path: str | Path) -> Any:
        """Read one video with the model's official reader.

        Returns whatever ``build_inputs()`` of the same adapter consumes:
        decoded frames, preprocessed tensors, or arguments for an official
        function that decodes the video itself.
        """

    @abstractmethod
    def build_inputs(self, visual: Any, system_prompt: str, task_prompt: str) -> dict[str, Any]:
        """Put the prompts and the prepared video into this model's input format.

        Must not modify ``visual``: questions on the same video reuse it.
        The result always has ``"prompt"``, the text the model sees (chat
        template and placeholders included), for inspection; the other keys
        are what ``run()`` consumes.
        """

    @abstractmethod
    def run(self, inputs: dict[str, Any], max_new_tokens: int) -> str:
        """Decode greedily and return the raw text, thinking included."""

    def generate(
        self, video_path: str | Path, system_prompt: str, task_prompt: str, max_new_tokens: int
    ) -> str:
        """Answer one question: read the video, build the inputs, decode."""
        visual = self._prepared_video(video_path)
        inputs = self.build_inputs(visual, system_prompt, task_prompt)
        self.last_prompt = inputs["prompt"]
        return self.run(inputs, max_new_tokens)

    def _prepared_video(self, video_path: str | Path) -> Any:
        if str(video_path) != self._video_path:
            self._visual = None  # free the previous video before decoding the next
            self._visual = self.prepare_video(video_path)
            self._video_path = str(video_path)
        return self._visual

    def set_seed(self, seed: int) -> None:
        """Decoding is greedy, so most adapters ignore the seed."""

    def clear_cache(self) -> None:
        """Free cached GPU memory after a failed question."""
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        """Release cached GPU memory.

        Subclasses drop their own model references first, then call this.
        """
        self._video_path = None
        self._visual = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _check_model_path(self) -> None:
        if self.config.local_files_only and not Path(self.config.path).is_dir():
            raise FileNotFoundError(
                "local_files_only is enabled, but model.path is not a local directory: "
                f"{self.config.path}"
            )

    def _dtype_name(
        self, default: str = "bfloat16", allowed: tuple[str, ...] = DTYPES
    ) -> str:
        name = str(self.config.options.get("dtype", default))
        if name not in allowed:
            raise ValueError(f"model.options.dtype must be one of: {', '.join(allowed)}")
        return name

    def _require_loaded(self, *attributes: str) -> None:
        if any(getattr(self, name) is None for name in attributes):
            raise RuntimeError(f"{type(self).__name__}.load() must be called first")
