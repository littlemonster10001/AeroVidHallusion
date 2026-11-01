"""VideoLLaMA3 (2B / 7B) with its checkpoint-provided Transformers code.

Conda environment: ``videollama3``.
Video: official ``processor.load_video`` (processing_videollama3.py, ffmpeg).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeromirage.models.hf_base import HFAdapter
from aeromirage.models.registry import register_model


@register_model("videollama3_transformers")
class VideoLLaMA3TransformersAdapter(HFAdapter):
    # Official defaults: fps=1 (processor), max_tokens=16384 (preprocessor_config.json).
    video_defaults = {"num_frames": 32, "fps": 1, "max_visual_tokens": 16384}

    def load(self) -> None:
        self._check_model_path()
        if not self.config.trust_remote_code:
            raise ValueError(
                "VideoLLaMA3 requires model.trust_remote_code=true because the "
                "checkpoint supplies its model and processor implementations"
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "VideoLLaMA3 requires PyTorch and Transformers; activate the "
                "`videollama3` Conda environment."
            ) from exc

        options = self.config.options
        dtype_name = self._dtype_name()
        attention = str(options.get("attention_implementation", "flash_attention_2"))
        device = str(options.get("device", "cuda:0"))
        max_visual_tokens = int(self.video["max_visual_tokens"])
        if max_visual_tokens <= 0:
            raise ValueError("video.max_visual_tokens must be positive")
        device_map = options.get("device_map", {"": device})
        if not isinstance(device_map, (str, dict)):
            raise TypeError("model.options.device_map must be a string or mapping")
        input_device = str(options.get("input_device", device))
        if input_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("VideoLLaMA3 is configured for CUDA, but CUDA is unavailable")

        print(f"Loading VideoLLaMA3 with Transformers: {self.config.path}")
        print(
            f"device={input_device}; dtype={dtype_name}; attention={attention}; "
            f"max_visual_tokens={max_visual_tokens}"
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.path,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None or not hasattr(image_processor, "max_tokens"):
            raise RuntimeError("VideoLLaMA3 processor does not expose image_processor.max_tokens")
        min_visual_tokens = int(getattr(image_processor, "min_tokens", 1))
        if max_visual_tokens < min_visual_tokens:
            raise ValueError(f"video.max_visual_tokens must be at least {min_visual_tokens}")
        image_processor.max_tokens = max_visual_tokens

        self.dtype = getattr(torch, dtype_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.path,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
            device_map=device_map,
            torch_dtype=self.dtype,
            attn_implementation=attention,
        ).eval()
        self.torch = torch
        self.input_device = torch.device(input_device)

    def prepare_video(self, video_path: str | Path) -> dict[str, Any]:
        self._require_loaded("processor")
        import ffmpeg

        num_frames = self.video["num_frames"]
        # load_video keeps one frame per 1/fps seconds and thins to max_frames
        # with np.linspace. Raise fps for short clips so that at least
        # num_frames candidates exist; the official fps applies otherwise.
        duration = float(ffmpeg.probe(str(video_path))["format"]["duration"])
        fps = max(float(self.video["fps"]), (num_frames + 2) / duration)
        frames, timestamps = self.processor.load_video(
            str(video_path), fps=fps, max_frames=num_frames
        )
        if len(frames) != num_frames:
            raise RuntimeError(
                f"VideoLLaMA3 load_video returned {len(frames)} frames, expected {num_frames}"
            )
        return {"frames": list(frames), "timestamps": list(timestamps)}

    def build_inputs(self, visual: Any, system_prompt: str, task_prompt: str) -> dict[str, Any]:
        """The official conversation, with frames already loaded by prepare_video."""
        self._require_loaded("processor", "input_device")
        frames, timestamps = visual["frames"], visual["timestamps"]
        conversation = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "num_frames": len(frames),
                        "timestamps": timestamps,
                    },
                    {"type": "text", "text": task_prompt},
                ],
            },
        ]
        template_kwargs = {"add_system_prompt": True, "add_generation_prompt": True}
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, **template_kwargs)
        # Passing images skips the processor's own loading; the rest of the
        # call matches the official conversation path.
        batch = self.processor(
            conversation=conversation,
            images=[("video", frames)],
            return_tensors="pt",
            **template_kwargs,
        )
        return {"prompt": prompt, "model_inputs": self._to_device(batch, cast=("pixel_values",))}
