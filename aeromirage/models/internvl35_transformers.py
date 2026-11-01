"""InternVL3.5 (8B / 14B) with its checkpoint-provided Transformers code.

Conda environment: ``internvl35``.
Video: ``load_video`` and its helpers copied from the official model card
(README.md, "video multi-round conversation"); the checkpoint ships no video reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeromirage.models.hf_base import GREEDY, HFAdapter
from aeromirage.models.registry import register_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    import numpy as np

    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    import torch
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in img])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list), num_patches_list


@register_model("internvl35_transformers")
class InternVL35TransformersAdapter(HFAdapter):
    # Official video example: input_size=448, max_num=1 (one tile per frame).
    video_defaults = {"num_frames": 32, "input_size": 448, "max_num": 1}

    def load(self) -> None:
        self._check_model_path()
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "InternVL3.5 requires PyTorch and transformers>=4.52.1; activate "
                "the `internvl35` Conda environment."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("InternVL3.5 requires CUDA")

        options = self.config.options
        dtype_name = self._dtype_name()
        if int(self.video["input_size"]) <= 0 or int(self.video["max_num"]) <= 0:
            raise ValueError("video.input_size and video.max_num must be positive")
        device_map = options.get("device_map", "auto")
        if not isinstance(device_map, (str, dict)):
            raise TypeError("model.options.device_map must be a string or mapping")

        print(f"Loading InternVL3.5 with Transformers: {self.config.path}")
        print(
            f"dtype={dtype_name}; device_map={device_map}; "
            f"input_size={self.video['input_size']}; max_num={self.video['max_num']}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.path,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
            use_fast=False,
        )
        self.dtype = getattr(torch, dtype_name)
        model_kwargs: dict[str, Any] = {
            # The remote-code constructor rejects the newer ``dtype`` kwarg;
            # ``torch_dtype`` is consumed by from_pretrained instead.
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
            "device_map": device_map,
        }
        if bool(options.get("use_flash_attn", True)):
            model_kwargs["use_flash_attn"] = True
        self.model = AutoModel.from_pretrained(self.config.path, **model_kwargs).eval()
        self.torch = torch
        self.input_device = next(self.model.parameters()).device

    def prepare_video(self, video_path: str | Path) -> dict[str, Any]:
        """Pixel values on the model's input device, plus tiles per frame."""
        self._require_loaded("dtype", "input_device")
        pixel_values, num_patches_list = load_video(
            str(video_path),
            input_size=self.video["input_size"],
            max_num=self.video["max_num"],
            num_segments=self.video["num_frames"],
        )
        return {
            "pixel_values": pixel_values.to(device=self.input_device, dtype=self.dtype),
            "num_patches_list": num_patches_list,
        }

    def build_inputs(self, visual: Any, system_prompt: str, task_prompt: str) -> dict[str, Any]:
        """Arguments for the official model.chat(), as in its video example."""
        self._require_loaded("model")
        frame_count = len(visual["num_patches_list"])
        question = "".join(f"Frame{i + 1}: <image>\n" for i in range(frame_count)) + task_prompt
        # chat() fills the checkpoint's conversation template with
        # model.system_message and the question; render the same text for
        # inspection (each <image> later becomes 256 image tokens).
        template = self._remote_module("conversation").get_conv_template(self.model.template)
        template.system_message = system_prompt
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        return {
            "prompt": template.get_prompt(),
            "question": question,
            "system_message": system_prompt,
            "pixel_values": visual["pixel_values"],
            "num_patches_list": visual["num_patches_list"],
        }

    def run(self, inputs: dict[str, Any], max_new_tokens: int) -> str:
        """Official model.chat(); it reads the system prompt from model.system_message."""
        self._require_loaded("model", "tokenizer", "torch")
        self.model.system_message = inputs["system_message"]
        with self.torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                inputs["pixel_values"],
                inputs["question"],
                {"max_new_tokens": max_new_tokens, **GREEDY},
                num_patches_list=inputs["num_patches_list"],
                history=None,
                return_history=False,
            )
        return str(response).strip()
