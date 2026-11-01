"""Qwen3-VL through vLLM (4B / 8B / 32B).

Video: official ``qwen_vl_utils.process_vision_info`` with the processor's patch size.
"""

from __future__ import annotations

from typing import Any

from aeromirage.models.qwen25_vl_vllm import Qwen25VLVLLMAdapter
from aeromirage.models.registry import register_model


@register_model("qwen3_vl_vllm")
class Qwen3VLVLLMAdapter(Qwen25VLVLLMAdapter):
    model_label = "Qwen3-VL"
    # qwen-vl-utils defaults for patch size 16: 128 / 768 tokens of 32x32 per frame.
    video_defaults = {
        **Qwen25VLVLLMAdapter.video_defaults,
        "min_pixels": 128 * 32 * 32,
        "max_pixels": 768 * 32 * 32,
    }
    metadata_as_processor_kwarg = False  # vLLM's Qwen3-VL parser keeps the metadata

    def _vision_info_kwargs(self) -> dict[str, Any]:
        self._require_loaded("processor")
        patch_size = getattr(self.processor.image_processor, "patch_size", None)
        if patch_size is None:
            raise RuntimeError(f"{self.model_label} processor does not expose image patch_size")
        return {"image_patch_size": patch_size}
