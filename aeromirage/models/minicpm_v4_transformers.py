"""MiniCPM-V 4.0 with its checkpoint-provided chat implementation.

Conda environment: ``minicpm_v4``.
Video: ``encode_video`` from the official "Chat with video" example
(MiniCPM-V model card / GitHub); the checkpoint ships no video reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeromirage.models.hf_base import HFAdapter
from aeromirage.models.registry import register_model


def encode_video(video_path: str, num_frames: int, fps: float = 1) -> list[Any]:
    """Official MiniCPM-V example: sample at ``fps``, thin to ``num_frames``."""
    from decord import VideoReader, cpu
    from PIL import Image

    def uniform_sample(items, n):
        gap = len(items) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [items[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = max(1, round(vr.get_avg_fps() / fps))
    frame_idx = list(range(0, len(vr), sample_fps))
    # Deviation from the example: a clip shorter than num_frames / fps
    # seconds samples from every source frame so it still gets num_frames.
    if len(frame_idx) < num_frames:
        frame_idx = list(range(len(vr)))
    if len(frame_idx) > num_frames:
        frame_idx = uniform_sample(frame_idx, num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    return [Image.fromarray(frame.astype("uint8")) for frame in frames]


@register_model("minicpm_v4_transformers")
class MiniCPMV4TransformersAdapter(HFAdapter):
    # Official video decode params: use_image_id=False, max_slice_nums=2.
    video_defaults = {"num_frames": 32, "fps": 1, "max_slice_nums": 2, "use_image_id": False}

    def __init__(self, config, video=None):
        super().__init__(config, video)
        self.max_inp_length = 32768
        self.num_beams = 1
        self.repetition_penalty = 1.0

    def load(self) -> None:
        self._check_model_path()
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "MiniCPM-V 4.0 requires PyTorch, Transformers, Pillow, and Accelerate; "
                "activate the `minicpm_v4` Conda environment."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("MiniCPM-V 4.0 requires CUDA")

        options = self.config.options
        dtype_name = self._dtype_name()
        attention = str(options.get("attn_implementation", "sdpa"))
        if attention not in {"sdpa", "flash_attention_2"}:
            raise ValueError("model.options.attn_implementation must be sdpa or flash_attention_2")
        device = str(options.get("device", "cuda:0"))
        self.max_inp_length = int(options.get("max_inp_length", 32768))
        self.num_beams = int(options.get("num_beams", 1))
        self.repetition_penalty = float(options.get("repetition_penalty", 1.0))
        if self.max_inp_length <= 0 or self.num_beams <= 0:
            raise ValueError("max_inp_length and num_beams must be positive")
        if int(self.video["max_slice_nums"]) <= 0:
            raise ValueError("video.max_slice_nums must be positive")

        print(f"Loading MiniCPM-V 4.0 with Transformers: {self.config.path}")
        print(
            f"dtype={dtype_name}; device={device}; attention={attention}; "
            f"max_slice_nums={self.video['max_slice_nums']}"
        )
        common_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.path, use_fast=True, **common_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.path, use_fast=False, **common_kwargs
        )
        self.model = AutoModel.from_pretrained(
            self.config.path,
            torch_dtype=getattr(torch, dtype_name),
            low_cpu_mem_usage=True,
            attn_implementation=attention,
            **common_kwargs,
        ).eval()
        self.input_device = torch.device(device)
        self.model = self.model.to(self.input_device)
        self.torch = torch

    def prepare_video(self, video_path: str | Path) -> list[Any]:
        """PIL frames, placed before the question in the user message."""
        return encode_video(str(video_path), self.video["num_frames"], self.video["fps"])

    def build_inputs(self, visual: Any, system_prompt: str, task_prompt: str) -> dict[str, Any]:
        """Arguments for the official model.chat(), as in its video example."""
        self._require_loaded("processor")
        # chat() turns each image into "(<image>./</image>)", joins the user
        # content with newlines, prepends the system message, and applies the
        # tokenizer's chat template; render the same text for inspection.
        user_text = "\n".join(["(<image>./</image>)"] * len(visual) + [task_prompt])
        prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "msgs": [{"role": "user", "content": [*visual, task_prompt]}],
            "system_prompt": system_prompt,
            "max_slice_nums": int(self.video["max_slice_nums"]),
            "use_image_id": bool(self.video["use_image_id"]),
        }

    def run(self, inputs: dict[str, Any], max_new_tokens: int) -> str:
        """Official model.chat() without sampling.

        chat(sampling=False) defaults to beam search (num_beams=3,
        repetition_penalty=1.2) and only lets callers override those two
        keys; model.options.num_beams / repetition_penalty set them and
        default to greedy (1 / 1.0) like every other model.
        """
        self._require_loaded("model", "tokenizer", "processor", "torch")
        with self.torch.inference_mode():
            answer = self.model.chat(
                image=None,
                msgs=inputs["msgs"],
                tokenizer=self.tokenizer,
                processor=self.processor,
                system_prompt=inputs["system_prompt"],
                max_new_tokens=max_new_tokens,
                max_inp_length=self.max_inp_length,
                max_slice_nums=inputs["max_slice_nums"],
                use_image_id=inputs["use_image_id"],
                sampling=False,
                num_beams=self.num_beams,
                repetition_penalty=self.repetition_penalty,
            )
        return str(answer).strip()
