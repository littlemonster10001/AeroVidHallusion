"""Kimi-VL-A3B (16B MoE) with its checkpoint-provided Transformers code.

Conda environment: ``kimi_vl``.
Video: Kimi-VL publishes no video reader or video example, so frames are
sampled uniformly here and passed to the model as images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeromirage.models.hf_base import HFAdapter
from aeromirage.models.registry import register_model


@register_model("kimi_vl_transformers")
class KimiVLTransformersAdapter(HFAdapter):
    # image_max_side bounds MoonViT's native-resolution tokens per frame.
    video_defaults = {"num_frames": 32, "image_max_side": 448}
    # As in the official README's batch_decode call.
    decode_kwargs = {"clean_up_tokenization_spaces": False}

    def load(self) -> None:
        self._check_model_path()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Kimi-VL requires PyTorch, Transformers, Accelerate, Pillow, and "
                "decord; activate the `kimi_vl` Conda environment."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Kimi-VL requires CUDA")

        options = self.config.options
        dtype_name = self._dtype_name()
        attention = str(options.get("attn_implementation", "flash_attention_2"))
        if attention not in {"sdpa", "flash_attention_2"}:
            raise ValueError("model.options.attn_implementation must be sdpa or flash_attention_2")
        device_map = options.get("device_map", "balanced")
        if not isinstance(device_map, (str, dict)):
            raise TypeError("model.options.device_map must be a string or mapping")
        if int(self.video["image_max_side"]) <= 0:
            raise ValueError("video.image_max_side must be positive")

        print(f"Loading Kimi-VL with Transformers: {self.config.path}")
        print(
            f"dtype={dtype_name}; device_map={device_map}; attention={attention}; "
            f"image_max_side={self.video['image_max_side']}"
        )
        common_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        self.processor = AutoProcessor.from_pretrained(self.config.path, **common_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.path,
            torch_dtype=getattr(torch, dtype_name),
            device_map=device_map,
            low_cpu_mem_usage=True,
            attn_implementation=attention,
            **common_kwargs,
        ).eval()
        self.torch = torch
        self.input_device = next(self.model.parameters()).device

    def prepare_video(self, video_path: str | Path) -> dict[str, Any]:
        """Uniformly sampled PIL frames and their timestamps in seconds."""
        import numpy as np
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        indices = np.linspace(0, len(vr) - 1, self.video["num_frames"]).round().astype(int)
        max_side = int(self.video["image_max_side"])
        frames = []
        for frame in vr.get_batch(indices.tolist()).asnumpy():
            image = Image.fromarray(frame).convert("RGB")
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side), resample=Image.Resampling.LANCZOS)
            frames.append(image)
        fps = float(vr.get_avg_fps())
        return {"frames": frames, "timestamps": [round(int(i) / fps, 3) for i in indices]}

    def build_inputs(self, visual: Any, system_prompt: str, task_prompt: str) -> dict[str, Any]:
        """Frames as images, each labelled with its position and time.

        Kimi-VL has no official video format; the labels are ours and in
        Chinese like the rest of the prompt.
        """
        self._require_loaded("processor", "input_device")
        content: list[dict[str, Any]] = []
        for index, seconds in enumerate(visual["timestamps"], start=1):
            content.append({"type": "text", "text": f"第{index}帧（约{seconds:.2f}秒）："})
            content.append({"type": "image", "image": f"frame_{index:02d}"})
        content.append({"type": "text", "text": task_prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        # Official README call: the template returns text despite return_tensors.
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        batch = self.processor(
            images=visual["frames"], text=prompt, return_tensors="pt", padding=True, truncation=True
        )
        return {"prompt": prompt, "model_inputs": self._to_device(batch)}
