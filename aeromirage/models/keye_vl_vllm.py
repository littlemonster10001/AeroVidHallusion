"""Keye-VL 1.5 through vLLM.

Video: official ``keye_vl_utils.process_vision_info`` (keye-vl-utils 1.5.2).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from aeromirage.models.registry import register_model
from aeromirage.models.vllm_base import VLLMAdapter


@register_model("keye_vl_vllm")
class KeyeVLVLLMAdapter(VLLMAdapter):
    model_label = "Keye-VL 1.5"
    # keye-vl-utils defaults: VIDEO_MIN_PIXELS / VIDEO_MAX_PIXELS (48 / 768 tokens of 28x28).
    video_defaults = {
        "num_frames": 32,
        "min_pixels": 48 * 28 * 28,
        "max_pixels": 768 * 28 * 28,
    }

    def _engine_kwargs(self) -> dict[str, Any]:
        kwargs = super()._engine_kwargs()
        # vLLM profiles Keye with at most 16 dummy frames, but keye-vl-utils
        # can produce 32-frame inputs, so the default encoder cache is too
        # small. max_num_batched_tokens is the lower bound of that cache.
        encoder_cache_floor = min(int(kwargs["max_model_len"]), 73_728)
        kwargs["max_num_batched_tokens"] = max(
            int(kwargs.get("max_num_batched_tokens", 0)), encoder_cache_floor
        )
        return kwargs

    def _messages(self, system_prompt: str, task_prompt: str) -> list[dict[str, Any]]:
        # Keye switches thinking with a trailing /think or /no_think tag.
        tag = "/think" if self.config.options.get("enable_thinking", False) else "/no_think"
        return super()._messages(system_prompt, f"{task_prompt}\n{tag}")

    def prepare_video(self, video_path: str | Path) -> dict[str, Any]:
        from keye_vl_utils import process_vision_info

        # keye-vl-utils has no ``nframes`` option: it samples
        # min(fps, source_fps) frames per second, capped by ``max_frames``,
        # uniformly over the whole video. An unbounded fps makes it consider
        # every source frame, so the cap yields exactly num_frames frames.
        element = {
            "type": "video",
            "video": str(video_path),
            "fps": math.inf,
            "max_frames": self.video["num_frames"],
            "min_pixels": self.video["min_pixels"],
            "max_pixels": self.video["max_pixels"],
        }
        _, videos, video_kwargs = process_vision_info([{"role": "user", "content": [element]}])
        return {"video": videos, "mm_processor_kwargs": dict(video_kwargs)}
