"""LLaVA-NeXT-Video (7B) with the official LLaVA-NeXT code.

Conda environment: ``llava_next_video`` (provides the ``llava`` package).
Video: official ``llava.utils.process_video_with_decord`` and the CLIP image processor.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aeromirage.models.hf_base import HFAdapter
from aeromirage.models.registry import register_model


@register_model("llava_next_video")
class LlavaNextVideoAdapter(HFAdapter):
    video_defaults = {"num_frames": 32}

    def __init__(self, config, video=None):
        super().__init__(config, video)
        self.conv_mode = "vicuna_v1"
        self.context_length = 0

    def _load_options(self) -> dict[str, Any]:
        options = self.config.options
        device = str(options.get("device", "cuda:0"))
        spatial_pool_stride = int(options.get("mm_spatial_pool_stride", 2))
        spatial_pool_mode = str(options.get("mm_spatial_pool_mode", "average"))
        conv_mode = str(options.get("conv_mode", "vicuna_v1"))
        vision_tower_path = str(options.get("vision_tower_path", ""))

        if not device.startswith("cuda"):
            raise ValueError("LLaVA-NeXT-Video currently requires a CUDA device")
        if self.video["num_frames"] < 2:
            raise ValueError("video.num_frames must be at least 2")
        if spatial_pool_stride <= 0:
            raise ValueError("model.options.mm_spatial_pool_stride must be positive")
        if spatial_pool_mode not in {"average", "max", "bilinear"}:
            raise ValueError("model.options.mm_spatial_pool_mode must be average, max, or bilinear")
        if not conv_mode:
            raise ValueError("model.options.conv_mode cannot be empty")
        if self.config.local_files_only:
            if not vision_tower_path:
                raise ValueError(
                    "model.options.vision_tower_path is required with local_files_only=true"
                )
            if not Path(vision_tower_path).is_dir():
                raise FileNotFoundError(
                    f"Local vision tower directory does not exist: {vision_tower_path}"
                )

        return {
            "dtype": self._dtype_name("float16", ("float16", "bfloat16")),
            "device": device,
            "attention": str(options.get("attention_implementation", "flash_attention_2")),
            "conv_mode": conv_mode,
            "num_frames": self.video["num_frames"],
            "spatial_pool_stride": spatial_pool_stride,
            "spatial_pool_mode": spatial_pool_mode,
            "newline_position": str(options.get("mm_newline_position", "no_token")),
            "vision_tower_path": vision_tower_path,
        }

    @staticmethod
    def _context_overrides(options: dict[str, Any]) -> dict[str, Any]:
        """RoPE settings recommended by the official video demo."""
        stride = options["spatial_pool_stride"]
        frame_tokens = (24 // stride) ** 2
        least_tokens = options["num_frames"] * frame_tokens + 1000
        scaling_factor = max(1, math.ceil(least_tokens / 4096))
        overrides: dict[str, Any] = {
            "mm_spatial_pool_mode": options["spatial_pool_mode"],
            "mm_spatial_pool_stride": stride,
            "mm_newline_position": options["newline_position"],
        }
        if options["vision_tower_path"]:
            overrides["mm_vision_tower"] = options["vision_tower_path"]
        if scaling_factor >= 2:
            overrides["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            overrides["max_sequence_length"] = 4096 * scaling_factor
            overrides["tokenizer_model_max_length"] = 4096 * scaling_factor
        return overrides

    def load(self) -> None:
        self._check_model_path()
        try:
            import torch
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise ImportError(
                "LLaVA-NeXT-Video requires the official LLaVA-NeXT package; "
                "activate the `llava_next_video` Conda environment."
            ) from exc

        options = self._load_options()
        if not torch.cuda.is_available():
            raise RuntimeError("LLaVA-NeXT-Video is configured for CUDA, but CUDA is unavailable")
        if options["conv_mode"] not in conv_templates:
            available = ", ".join(sorted(conv_templates))
            raise ValueError(
                f"Unknown LLaVA conversation mode {options['conv_mode']!r}; "
                f"available modes: {available}"
            )

        print(f"Loading LLaVA-NeXT-Video: {self.config.path}")
        print(
            f"device={options['device']}; dtype={options['dtype']}; "
            f"frames={options['num_frames']}; conv={options['conv_mode']}"
        )
        previous_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            tokenizer, model, image_processor, context_length = load_pretrained_model(
                self.config.path,
                None,
                get_model_name_from_path(self.config.path),
                device_map={"": options["device"]},
                torch_dtype=options["dtype"],
                attn_implementation=options["attention"],
                overwrite_config=self._context_overrides(options),
                local_files_only=self.config.local_files_only,
            )
        finally:
            transformers_logging.set_verbosity(previous_verbosity)

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.processor = image_processor  # CLIP image processor of the vision tower
        self.torch = torch
        self.input_device = torch.device(options["device"])
        self.dtype = model.get_vision_tower().dtype
        self.conv_mode = options["conv_mode"]
        self.context_length = int(context_length)

    def prepare_video(self, video_path: str | Path) -> Any:
        """CLIP pixel values of num_frames uniformly sampled frames."""
        self._require_loaded("processor", "input_device", "dtype")
        from llava.utils import process_video_with_decord

        # force_sample=True makes the reader take np.linspace(frames_upbound)
        # over the whole video; video_fps is only used without it.
        data_args = SimpleNamespace(
            video_fps=1, frames_upbound=self.video["num_frames"], force_sample=True
        )
        frames, _, _, _ = process_video_with_decord(str(video_path), data_args)
        pixel_values = self.processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        return pixel_values.to(device=self.input_device, dtype=self.dtype)

    def build_inputs(self, visual: Any, system_prompt: str, task_prompt: str) -> dict[str, Any]:
        """Official LLaVA conversation template with one video token."""
        self._require_loaded("model", "tokenizer")
        from llava.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import tokenizer_image_token

        video_token = DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            video_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conversation = copy.deepcopy(conv_templates[self.conv_mode])
        conversation.system = system_prompt
        conversation.append_message(conversation.roles[0], f"{video_token}\n{task_prompt}")
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.input_device)
        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id).long(),
            "images": [visual],
            "stop_str": conversation.sep
            if conversation.sep_style != SeparatorStyle.TWO
            else conversation.sep2,
        }

    def run(self, inputs: dict[str, Any], max_new_tokens: int) -> str:
        from llava.mm_utils import KeywordsStoppingCriteria

        return self._generate_until_stop(
            inputs, max_new_tokens, KeywordsStoppingCriteria, modalities=["video"]
        )
