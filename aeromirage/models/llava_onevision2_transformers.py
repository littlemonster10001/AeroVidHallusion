"""LLaVA-OneVision-2 (8B) with its checkpoint-provided video processor.

Conda environment: ``dronevidhallubench`` (needs transformers>=5.7).
Video: official ``LlavaOnevision2VideoProcessor`` (video_processing_llava_onevision2.py),
which decodes the file itself when the processor is called with ``num_frames``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeromirage.models.hf_base import HFAdapter
from aeromirage.models.registry import register_model


@register_model("llava_onevision2_transformers")
class LlavaOneVision2TransformersAdapter(HFAdapter):
    # Official README video example: max_pixels=200704 (its num_frames=16; we use the shared 32).
    video_defaults = {"num_frames": 32, "max_pixels": 200704}

    def load(self) -> None:
        self._check_model_path()
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "LLaVA-OneVision-2 requires PyTorch, transformers>=5.7, Pillow, "
                "and decord; activate the `dronevidhallubench` Conda environment."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("LLaVA-OneVision-2 requires CUDA")

        options = self.config.options
        dtype_name = self._dtype_name()
        attention = str(options.get("attn_implementation", "sdpa"))
        if attention not in {"sdpa", "flash_attention_2"}:
            raise ValueError("model.options.attn_implementation must be sdpa or flash_attention_2")
        device = str(options.get("device", "cuda:0"))

        print(f"Loading LLaVA-OneVision-2 with Transformers: {self.config.path}")
        print(
            f"dtype={dtype_name}; device={device}; attention={attention}; "
            f"video max_pixels={self.video['max_pixels']}"
        )
        common_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        self.processor = AutoProcessor.from_pretrained(self.config.path, **common_kwargs)
        self.processor.video_processor.max_pixels = int(self.video["max_pixels"])
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.path,
            dtype=getattr(torch, dtype_name),
            low_cpu_mem_usage=True,
            attn_implementation=attention,
            **common_kwargs,
        ).eval()
        self.input_device = torch.device(device)
        self.model = self.model.to(self.input_device)
        self.torch = torch

    def prepare_video(self, video_path: str | Path) -> dict[str, Any]:
        """Arguments for ``processor(text=..., videos=..., num_frames=...)``.

        The official processor decodes and tokenizes the video together with
        the prompt, so decoding happens in build_inputs.
        """
        return {"videos": [str(video_path)], "num_frames": self.video["num_frames"]}

    def build_inputs(self, visual: Any, system_prompt: str, task_prompt: str) -> dict[str, Any]:
        """Official README video call; the processor decodes the video here."""
        self._require_loaded("processor", "input_device")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": task_prompt}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(
            text=[prompt],
            videos=visual["videos"],
            num_frames=visual["num_frames"],
            return_tensors="pt",
            padding=True,
        )
        return {"prompt": prompt, "model_inputs": self._to_device(batch)}

    def _generate_kwargs(self) -> dict[str, Any]:
        # As in the official README: end at EOS, pad with EOS when unset.
        tokenizer = self.processor.tokenizer
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        return {"eos_token_id": tokenizer.eos_token_id, "pad_token_id": pad}
