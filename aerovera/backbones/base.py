"""What the pipeline needs from a backbone, and the types every backbone shares.

``pipeline.py`` calls only the methods of ``Backbone``; each backbone module
(``qwen25_vl.py``, ``videollama3.py``) implements them for its own model and
input format. The model inputs a backbone returns are opaque to the pipeline:
it only hands them back to the same backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


# Greedy decoding, as in the benchmark baseline (temperature 0, repetition
# penalty 1.0). Checkpoints' generation_config.json often asks for sampling
# and a repetition penalty, which generate() would otherwise apply.
GREEDY: dict[str, Any] = {
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}


@dataclass
class Generation:
    """A greedy answer: its text and the new tokens without the closing end-of-turn token."""

    text: str
    token_ids: list[int]


@dataclass
class Attribution:
    """Gradient x input attribution A_i of each visual token to an answer (Appendix C.3)."""

    scores: Any  # float32 tensor [M] on the CPU, one per visual token that reaches the language model
    grid: tuple[int, ...]  # Qwen2.5-VL: (frame pairs, rows, columns); VideoLLaMA3: (tokens kept after compression,)
    log_likelihood: float  # s = sum of log p(answer token) under teacher forcing


def cut_cache(cache: Any, length: int) -> None:
    """Drop the cached tokens after the first ``length`` (``crop`` takes the number to drop)."""
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)


class Video(Protocol):
    """The Probing frames of one video; what the pipeline reads of a backbone's video."""

    path: str
    frame_ids: list[int]  # source frame indices (decord), in time order
    fps: float  # source frame rate
    total_frames: int
    size: tuple[int, int]  # (height, width) the Probing frames are fed at: the LR size


class Backbone(Protocol):
    def load(self) -> None: ...

    def read_video(self, path: str, **probing: Any) -> Video:
        """The Probing frames, exactly as the benchmark baseline samples them (``probing`` config section)."""

    def source_frames(self, video: Video) -> list[Any] | None:
        """The Probing frames at the source resolution (RGB uint8 [H, W, 3]) as the backbone read them,
        to replace the decord frames at those positions; None if decord's are the same frames."""

    @staticmethod
    def resize(frame: Any, size: Sequence[int]) -> Any:
        """A source frame resized to ``size`` the way the backbone's processor resizes its inputs."""

    @staticmethod
    def hr_size(height: int, width: int, short_side: int = 1024) -> tuple[int, int]: ...

    def video_inputs(self, video: Video, system_prompt: str, task_prompt: str) -> Any: ...

    def image_inputs(self, frames: Sequence[Any], timestamps: Sequence[float], system_prompt: str,
                     task_prompt: str) -> Any: ...

    def text_inputs(self, system_prompt: str, user_prompt: str) -> Any: ...

    def generate(self, inputs: Any, max_new_tokens: int, mask: Sequence[int] | None = None) -> Generation: ...

    def first_logits(self, inputs: Any, mask: Sequence[int] | None = None) -> Any: ...

    def first_logits_many(self, frames: Sequence[Any], timestamps: Sequence[float], system_prompt: str,
                          task_prompts: Sequence[str]) -> list[Any]: ...

    def contrastive_generate(self, positive: Any, negative: Any, alpha: float, max_new_tokens: int,
                             mask: Sequence[int] | None = None) -> tuple[Generation, Generation, tuple[Any, Any, Any]]: ...

    def attribute(self, inputs: Any, *, max_new_tokens: int | None = None,
                  answer_ids: Sequence[int] | None = None) -> tuple[Generation, Attribution | None]: ...

    def label_ids(self, labels: Sequence[str]) -> list[int]: ...

    def token_text(self, token_id: int) -> str: ...
