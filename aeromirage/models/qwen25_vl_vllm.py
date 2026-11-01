"""Qwen2.5-VL through vLLM (3B / 7B / 32B).

Video: official ``qwen_vl_utils.process_vision_info`` (qwen-vl-utils 0.0.14).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from aeromirage.models.registry import register_model
from aeromirage.models.vllm_base import VLLMAdapter


@register_model("qwen25_vl_vllm")
class Qwen25VLVLLMAdapter(VLLMAdapter):
    model_label = "Qwen2.5-VL"
    # qwen-vl-utils defaults for patch size 14: 128 / 768 tokens of 28x28 per frame.
    video_defaults = {
        "num_frames": 32,
        "min_pixels": 128 * 28 * 28,
        "max_pixels": 768 * 28 * 28,
        "video_reader_backend": "decord",
    }
    # vLLM's Qwen2.5-VL parser drops the metadata of (frames, metadata) video
    # items, and the HF processor then assumes 24 fps when it sets the
    # temporal positions (second_per_grid_ts), which collapses them to about
    # zero. Passing the metadata as a processor kwarg restores them. Qwen3-VL
    # keeps the metadata itself.
    metadata_as_processor_kwarg = True

    def _vision_info_kwargs(self) -> dict[str, Any]:
        return {}

    def prepare_video(self, video_path: str | Path) -> dict[str, Any]:
        num_frames = self.video["num_frames"]
        if num_frames % 2:
            raise ValueError("Qwen video.num_frames must be even (temporal patch size 2)")
        backend = str(self.video["video_reader_backend"])
        if backend not in {"decord", "torchcodec", "torchvision"}:
            raise ValueError("video.video_reader_backend must be decord, torchcodec, or torchvision")
        # qwen-vl-utils reads this once at import time; without it, an
        # importable but mismatched torchcodec is chosen and floods the log.
        os.environ["FORCE_QWENVL_VIDEO_READER"] = backend
        from qwen_vl_utils import process_vision_info

        element = {
            "type": "video",
            "video": str(video_path),
            "nframes": num_frames,
            "min_pixels": self.video["min_pixels"],
            "max_pixels": self.video["max_pixels"],
        }
        _, videos, video_kwargs = process_vision_info(
            [{"role": "user", "content": [element]}],
            return_video_kwargs=True,
            return_video_metadata=True,
            **self._vision_info_kwargs(),
        )
        # vLLM inputs: frames already sampled, so the processor must not resample.
        processor_kwargs = dict(video_kwargs)
        if self.metadata_as_processor_kwarg:
            processor_kwargs["video_metadata"] = [metadata for _, metadata in videos]
        return {"video": videos, "mm_processor_kwargs": processor_kwargs}
