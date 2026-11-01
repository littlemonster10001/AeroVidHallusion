"""VideoChat-Flash (Qwen2-7B, res448) with its checkpoint-provided chat code.

Conda environment: ``videochat_flash`` (the remote code needs transformers 4.40.1).
Video: official ``read_frames_decord`` from the checkpoint's mm_utils.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeromirage.models.hf_base import HFAdapter
from aeromirage.models.registry import register_model


@register_model("videochat_flash_transformers")
class VideoChatFlashTransformersAdapter(HFAdapter):
    # The model encodes frames in groups of config.mm_local_num_frames (4).
    video_defaults = {"num_frames": 32}

    def load(self) -> None:
        self._check_model_path()
        if not self.config.trust_remote_code:
            raise ValueError(
                "VideoChat-Flash requires model.trust_remote_code=true because its "
                "chat and video encoder implementations ship with the checkpoint"
            )
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "VideoChat-Flash requires transformers 4.40.1; activate the "
                "`videochat_flash` Conda environment."
            ) from exc

        options = self.config.options
        dtype_name = self._dtype_name()
        device = str(options.get("device", "cuda:0"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("VideoChat-Flash is configured for CUDA, but CUDA is unavailable")
        mm_llm_compress = bool(options.get("mm_llm_compress", False))

        print(f"Loading VideoChat-Flash with Transformers: {self.config.path}")
        print(f"device={device}; dtype={dtype_name}; mm_llm_compress={mm_llm_compress}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.path,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
            use_fast=False,
        )
        if self.tokenizer.pad_token_id is None:
            # Same fallback as the official chat() for Qwen tokenizers.
            self.tokenizer.pad_token_id = 151643
        self.dtype = getattr(torch, dtype_name)
        self.input_device = torch.device(device)
        self.model = AutoModel.from_pretrained(
            self.config.path,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).eval()
        self.model.to(device=self.input_device, dtype=self.dtype)
        self.model.config.mm_llm_compress = mm_llm_compress
        self.torch = torch

    def prepare_video(self, video_path: str | Path) -> dict[str, Any]:
        """Decoded frames plus the time message chat() puts before the question."""
        self._require_loaded("model")
        num_frames = self.video["num_frames"]
        group = int(self.model.config.mm_local_num_frames)
        if num_frames % group:
            raise ValueError(f"VideoChat-Flash video.num_frames must be a multiple of {group}")
        mm_utils = self._remote_module("mm_utils")
        # Same reader and arguments as the official load_video(), except that
        # load_video() hard-codes min_num_frames=64; pin both bounds instead.
        # (local_num_frames=8 only matters for clips shorter than num_frames.)
        frames, frame_indices, fps, duration = mm_utils.read_frames_decord(
            video_path=str(video_path),
            num_frames=num_frames,
            sample="dynamic_fps1",
            fix_start=None,
            min_num_frames=num_frames,
            max_num_frames=num_frames,
            client=None,
            clip=None,
            local_num_frames=8,
        )
        sec = [str(round(f / fps, 1)) for f in frame_indices]
        time_msg = (
            f"\nThe video lasts for {duration:.2f} seconds, and {len(sec)} frames "
            "are uniformly sampled from it. "
        )
        return {"frames": frames, "time_msg": time_msg}

    def build_inputs(self, visual: Any, system_prompt: str, task_prompt: str) -> dict[str, Any]:
        """The official chat() prompt, built here so the system prompt can be set.

        chat() always keeps the qwen_2 template's default system message and
        offers no way to change it; the template does have a system slot, so
        ours goes there as for every other model.
        """
        self._require_loaded("model", "tokenizer")
        constants = self._remote_module("constants")
        conversation_module = self._remote_module("conversation")
        mm_utils = self._remote_module("mm_utils")

        frames = visual["frames"]
        pixel_values = self.model.get_vision_tower().image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"].to(self.input_device, dtype=self.dtype)
        conversation = conversation_module.conv_templates["qwen_2"].copy()
        conversation.system = f"<|im_start|>system\n{system_prompt}"
        conversation.append_message(
            conversation.roles[0],
            f"{constants.DEFAULT_IMAGE_TOKEN}\n{visual['time_msg'].strip()} {task_prompt}",
        )
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = mm_utils.tokenizer_image_token(
            prompt, self.tokenizer, constants.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.input_device)
        separator_style = conversation_module.SeparatorStyle
        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id).long(),
            "images": [pixel_values],
            "image_sizes": [frames[0].shape[:2]],
            "stop_str": conversation.sep
            if conversation.sep_style != separator_style.TWO
            else conversation.sep2,
        }

    def run(self, inputs: dict[str, Any], max_new_tokens: int) -> str:
        """The generate call of the official chat(), with greedy decoding."""
        return self._generate_until_stop(
            inputs,
            max_new_tokens,
            self._remote_module("mm_utils").KeywordsStoppingCriteria,
            modalities=["video"],
            image_sizes=inputs["image_sizes"],
        )
