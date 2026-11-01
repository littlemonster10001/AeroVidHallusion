"""Qwen2.5-VL backbone for AeroVera (transformers).

The official ``Qwen2_5_VLForConditionalGeneration`` is used as is. Its
language layers run ``aerovera_attention``: plain SDPA, except that inside
``Qwen25VL._visual_gate`` every visual token's Value is scaled by a gate
(Appendix C.3) in every layer and head. Heavy imports stay inside the
methods so the module imports in any environment.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from aerovera.backbones.base import GREEDY, Attribution, Generation
from aerovera.backbones.base import cut_cache as _cut

DTYPES = ("bfloat16", "float16", "float32")
ATTN_NAME = "aerovera_attention"
# Image sides must be multiples of 28: 14-pixel patches, merged 2 x 2.
PATCH_FACTOR = 28


@dataclass
class Video:
    """Uniformly sampled frames of one video, as qwen-vl-utils returns them."""

    path: str
    frames: Any  # float tensor [T, 3, H, W] holding 0-255 pixel values
    metadata: dict[str, Any]  # fps, frames_indices, total_num_frames, video_backend
    processor_kwargs: dict[str, Any]  # {"do_sample_frames": False}
    # Processed pixels on the device, shared by every question on this video
    # so that the vision encoder runs once per video (see Qwen25VL.load).
    pixels: Any = None

    @property
    def frame_ids(self) -> list[int]:
        return list(self.metadata["frames_indices"])

    @property
    def fps(self) -> float:
        return float(self.metadata["fps"])

    @property
    def total_frames(self) -> int:
        return int(self.metadata["total_num_frames"])

    @property
    def timestamps(self) -> list[float]:
        return [frame_id / self.fps for frame_id in self.frame_ids]

    @property
    def size(self) -> tuple[int, int]:
        """(height, width) after qwen-vl-utils resized the frames."""
        return int(self.frames.shape[-2]), int(self.frames.shape[-1])


# Set by Qwen25VL._visual_gate for one forward pass or one generation:
# {"gate": per-position Value scale over the prompt, "layers": layers reached}.
_gate_state: ContextVar[dict[str, Any] | None] = ContextVar("aerovera_visual_gate", default=None)


def aerovera_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """SDPA of the language layers; inside a visual gate, Values are scaled first.

    Only the Values change; queries, keys and attention weights do not. Tokens
    past the gate (the answer, or tokens generated after the prompt) keep
    their Values.
    """
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    state = _gate_state.get()
    if state is not None:
        gate = state["gate"]
        extra = value.shape[2] - gate.numel()
        if extra > 0:
            gate = torch.cat((gate, gate.new_ones(extra)))
        value = value * gate[: value.shape[2]][None, None, :, None]
        state["layers"].add(module.layer_idx)
    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
    )


class Qwen25VL:
    """Qwen2.5-VL with the inputs and decoding that AeroVera needs."""

    def __init__(self, path: str | Path, device: str = "cuda:0", dtype: str = "bfloat16"):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        self.path = str(path)
        self.device = device
        self.dtype = dtype
        self.model: Any = None
        self.processor: Any = None
        # Vision-encoder output for the pixels of the current video: (pixels, output).
        self._video_features: tuple[Any, Any] | None = None

    def load(self) -> None:
        if not Path(self.path).is_dir():
            raise FileNotFoundError(f"Qwen2.5-VL checkpoint not found: {self.path}")
        try:
            import torch
            from transformers import (
                AttentionInterface,
                AttentionMaskInterface,
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
            from transformers.masking_utils import sdpa_mask
        except ImportError as exc:
            raise ImportError(
                "The Qwen2.5-VL backbone requires PyTorch and Transformers; "
                "activate the `dronevidhallubench` Conda environment."
            ) from exc

        # The language layers use aerovera_attention with SDPA's causal mask;
        # the vision encoder keeps plain SDPA.
        AttentionInterface.register(ATTN_NAME, aerovera_attention)
        AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)
        model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.path,
            dtype=getattr(torch, self.dtype),
            device_map={"": self.device},
            attn_implementation={"vision_config": "sdpa", "text_config": ATTN_NAME},
            output_loading_info=True,
        )
        problems = {
            key: info[key]
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if info.get(key)
        }
        if problems:
            raise RuntimeError(f"Qwen2.5-VL checkpoint did not load completely: {problems}")
        model.generation_config.update(**GREEDY)
        # AeroVera never trains: gradients only ever flow to the visual gate.
        model.requires_grad_(False)
        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(self.path)
        self._cache_video_features()

    def _cache_video_features(self) -> None:
        """Run the vision encoder once per video instead of once per forward pass.

        Its output depends only on the pixels. Every question on a video gets
        the same pixel tensor (``Video.pixels``), so the output computed for
        the first forward pass is reused by the others. It is computed outside
        inference mode, so that the attribution's backward pass may use it.
        """
        import torch

        inner = self.model.model
        compute = inner.get_video_features

        def get_video_features(pixel_values_videos, video_grid_thw=None, **kwargs):
            cached = self._video_features
            if cached is not None and cached[0] is pixel_values_videos:
                return cached[1]
            with torch.inference_mode(False), torch.no_grad():
                output = compute(pixel_values_videos, video_grid_thw, **kwargs)
            self._video_features = (pixel_values_videos, output)
            return output

        inner.get_video_features = get_video_features

    def read_video(
        self, path: str | Path, num_frames: int, min_pixels: int, max_pixels: int, video_reader_backend: str = "decord"
    ) -> Video:
        """Uniformly sample ``num_frames`` frames with qwen-vl-utils, as the benchmark baseline does."""
        if num_frames % 2:
            raise ValueError("Qwen encodes frames in pairs; num_frames must be even")
        # qwen-vl-utils picks its reader once, at import time.
        os.environ["FORCE_QWENVL_VIDEO_READER"] = video_reader_backend
        from qwen_vl_utils import process_vision_info

        element = {
            "type": "video",
            "video": str(path),
            "nframes": num_frames,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
        _, videos, processor_kwargs = process_vision_info(
            [{"role": "user", "content": [element]}],
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        frames, metadata = videos[0]
        return Video(str(path), frames, metadata, dict(processor_kwargs))

    def source_frames(self, video: Video) -> None:
        """None: qwen-vl-utils reads the Probing frames with decord, so the decord
        source frames resized to ``Video.size`` already equal them."""
        return None

    @staticmethod
    def resize(frame: Any, size: Sequence[int]) -> Any:
        """RGB uint8 frame [H, W, 3] resized to ``size`` (height, width) the way
        qwen-vl-utils resizes video frames (torchvision bicubic with antialias on
        uint8), so that a source frame resized to ``Video.size`` equals its
        Probing frame; uint8 [h, w, 3]."""
        import numpy as np
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as F

        image = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
        resized = F.resize(image, list(size), interpolation=InterpolationMode.BICUBIC, antialias=True)
        return resized.permute(1, 2, 0).numpy()

    @staticmethod
    def hr_size(height: int, width: int, short_side: int = 1024) -> tuple[int, int]:
        """High-resolution input size of a source frame: short side at most
        ``short_side`` (never enlarged), both sides floored to multiples of 28."""
        scale = min(1.0, short_side / min(height, width))
        return (max(PATCH_FACTOR, int(height * scale) // PATCH_FACTOR * PATCH_FACTOR),
                max(PATCH_FACTOR, int(width * scale) // PATCH_FACTOR * PATCH_FACTOR))

    def video_inputs(self, video: Video, system_prompt: str, task_prompt: str) -> Any:
        """Model inputs for one video and one question, with the baseline's chat layout."""
        self._require_loaded()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": task_prompt}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # The metadata (source fps and frame indices) sets the temporal
        # positions; without it the processor assumes 24 fps.
        inputs = self.processor(
            text=[text],
            videos=[video.frames],
            video_metadata=[video.metadata],
            return_tensors="pt",
            **video.processor_kwargs,
        ).to(self.device)
        # One pixel tensor per video, so its vision-encoder output is reused.
        if video.pixels is None:
            video.pixels = inputs["pixel_values_videos"]
        elif video.pixels.shape != inputs["pixel_values_videos"].shape:
            raise RuntimeError(f"{video.path}: the processor returned different pixels for the same video")
        inputs["pixel_values_videos"] = video.pixels
        return inputs

    def text_inputs(self, system_prompt: str, user_prompt: str) -> Any:
        """Model inputs for a text-only request (target extraction): no video."""
        self._require_loaded()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], return_tensors="pt").to(self.device)

    def image_inputs(self, frames: Sequence[Any], timestamps: Sequence[float], system_prompt: str, task_prompt: str) -> Any:
        """Model inputs for frames as separate images, each preceded by its time
        ``<t seconds>`` (P3): the positive branch.

        ``frames`` are RGB uint8 arrays [h, w, 3] in time order, already at the
        size they are fed at (HR or LR, sides multiples of 28); the processor
        does not resize them again.
        """
        self._require_loaded()
        return self.processor(
            text=[self._image_chat(timestamps, system_prompt, task_prompt)],
            images=list(frames),
            do_resize=False,
            return_tensors="pt",
        ).to(self.device)

    def generate(self, inputs: Any, max_new_tokens: int, mask: Sequence[int] | None = None) -> Generation:
        """Greedy answer (the generation_config set in load()), as the baseline decodes.

        ``mask`` lists visual tokens (indices into the flattened grid) whose
        Values are zeroed in every layer while generating: the negative branch.
        """
        import torch

        self._require_loaded()
        gate = nullcontext() if not mask else self._visual_gate(self._visual_positions(inputs)[0], blocked=mask)
        with torch.inference_mode(), gate:
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._answer(output[0, inputs["input_ids"].shape[1]:].tolist())

    def first_logits(self, inputs: Any, mask: Sequence[int] | None = None) -> Any:
        """Raw logits [V] (float32) at the first answer position; ``mask`` as in ``generate``."""
        import torch

        self._require_loaded()
        gate = nullcontext() if not mask else self._visual_gate(self._visual_positions(inputs)[0], blocked=mask)
        with torch.inference_mode(), gate:
            return self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()

    def first_logits_many(
        self, frames: Sequence[Any], timestamps: Sequence[float], system_prompt: str, task_prompts: Sequence[str]
    ) -> list[Any]:
        """``first_logits(image_inputs(frames, timestamps, system_prompt, p))`` for each prompt p.

        The prompts differ only after the images, so the images and their time
        labels are encoded and prefilled once; each prompt then runs only its
        own text on that cache, which is cut back to the images afterwards.
        """
        import torch

        self._require_loaded()
        texts = [self._image_chat(timestamps, system_prompt, prompt) for prompt in task_prompts]
        cut = texts[0].rindex("<|vision_end|>") + len("<|vision_end|>")
        if any(text[:cut] != texts[0][:cut] for text in texts):
            raise ValueError("The prompts must share everything up to the last image")
        prefix = self.processor(text=[texts[0][:cut]], images=list(frames), do_resize=False,
                                return_tensors="pt").to(self.device)
        logits = []
        with torch.inference_mode():
            output = self.model(**prefix, use_cache=True, logits_to_keep=1)
            cache, deltas, length = output.past_key_values, self.model.model.rope_deltas, prefix["input_ids"].shape[1]
            for text in texts:
                # The suffix starts after a special token, so tokenizing it alone
                # gives the same ids as tokenizing the whole text.
                ids = self.processor.tokenizer(text[cut:], add_special_tokens=False, return_tensors="pt").input_ids
                self.model.model.rope_deltas = deltas  # positions continue after the images
                step = self.model(input_ids=ids.to(self.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits.append(step.logits[0, -1].float())
                _cut(cache, length)
        return logits

    def label_ids(self, labels: Sequence[str]) -> list[int]:
        """Token id of each answer label ("A", "B", ...); each must be a single token."""
        self._require_loaded()
        ids = [self.processor.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(token) != 1 for token in ids):
            raise ValueError(f"Every label must be one token: {dict(zip(labels, ids))}")
        return [token[0] for token in ids]

    def token_text(self, token_id: int) -> str:
        """Text of one token, for the record of open-ended candidates."""
        self._require_loaded()
        return self.processor.tokenizer.decode([token_id])

    def contrastive_generate(
        self, positive: Any, negative: Any, alpha: float, max_new_tokens: int, mask: Sequence[int] | None = None
    ) -> tuple[Generation, Generation, tuple[Any, Any, Any]]:
        """Open-ended contrastive decoding (P6): the first token maximizes
        q = q+ + alpha (q+ - q-) over the vocabulary, then the positive branch
        continues greedily on its own.

        ``positive``: image inputs; ``negative``: the Probing video inputs with
        ``mask`` blocked. Returns the contrastive answer, the positive branch's
        own greedy answer, and the first-position logits (q+, q-, q).
        """
        import torch

        self._require_loaded()
        gate = nullcontext() if not mask else self._visual_gate(self._visual_positions(negative)[0], blocked=mask)
        with torch.inference_mode():
            output = self.model(**positive, use_cache=True, logits_to_keep=1)
            cache, deltas, length = output.past_key_values, self.model.model.rope_deltas, positive["input_ids"].shape[1]
            with gate:
                negative_logits = self.model(**negative, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            positive_logits = output.logits[0, -1].float()
            combined = positive_logits + alpha * (positive_logits - negative_logits)
            answers = {}
            for first in dict.fromkeys((int(combined.argmax()), int(positive_logits.argmax()))):
                self.model.model.rope_deltas = deltas  # the negative prefill replaced them
                _cut(cache, length)
                answers[first] = self._answer(self._decode_greedy(first, cache, max_new_tokens))
        return (answers[int(combined.argmax())], answers[int(positive_logits.argmax())],
                (positive_logits, negative_logits, combined))

    def attribute(
        self, inputs: Any, *, max_new_tokens: int | None = None, answer_ids: Sequence[int] | None = None
    ) -> tuple[Generation, Attribution | None]:
        """A_i = g_i * ds/dg_i of every visual token of ``inputs`` to an answer (Eq. 7, Appendix C.3).

        One prefill with the gates g (all 1) on the autograd graph, then one
        backward pass to g; the model weights are frozen. The answer is either
        decoded greedily from that prefill, exactly as ``generate()`` would
        (``max_new_tokens``: the Probing Pass, which thus needs no second pass
        over the prompt), or given (``answer_ids``, without the end-of-turn
        token: the positive branch's judgment of one multiple-choice option).
        An empty answer has no attribution.
        """
        import torch
        from transformers import DynamicCache

        self._require_loaded()
        if (max_new_tokens is None) == (answer_ids is None):
            raise ValueError("Pass max_new_tokens to decode the answer, or answer_ids to attribute a given one")
        positions, grid = self._visual_positions(inputs)
        gate = torch.ones(positions.numel(), device=positions.device, dtype=torch.float32, requires_grad=True)
        with torch.enable_grad(), self._visual_gate(positions, gate=gate):
            prefill = self.model(**inputs, use_cache=True, logits_to_keep=1)
        first = prefill.logits[0, -1].float()
        if answer_ids is None:
            # A detached copy: the prefill's own cache still carries the graph for the backward pass.
            own = DynamicCache(ddp_cache_data=[(layer.keys.detach(), layer.values.detach())
                                               for layer in prefill.past_key_values.layers])
            answer = self._answer(self._decode_greedy(int(first.argmax()), own, max_new_tokens))
        else:
            answer = Generation(self._decode(list(answer_ids)), list(answer_ids))
        if not answer.token_ids:
            return answer, None
        with torch.enable_grad():
            score = first.log_softmax(-1)[answer.token_ids[0]]
            if len(answer.token_ids) > 1:
                # Teacher forcing on the prompt cache, which still carries the
                # graph: answer tokens 1..m-1 predict tokens 2..m.
                fed = torch.tensor([answer.token_ids[:-1]], device=positions.device)
                target = torch.tensor(answer.token_ids[1:], device=positions.device)[:, None]
                with self._visual_gate(positions, gate=gate):
                    later = self.model(
                        input_ids=fed,
                        past_key_values=prefill.past_key_values,
                        use_cache=True,
                        logits_to_keep=fed.shape[1],
                    )
                score = score + later.logits[0].float().log_softmax(-1).gather(1, target).sum()
            (gradient,) = torch.autograd.grad(score, gate)
        # Gradient x input; g = 1, kept to make the definition explicit. The
        # sign is kept: a negative A_i means the token speaks against the answer.
        return answer, Attribution((gate.detach() * gradient).cpu(), grid, float(score.detach()))

    def _visual_positions(self, inputs: Any) -> tuple[Any, tuple[int, int, int]]:
        """Sequence positions of the visual tokens of the one video in ``inputs``, and their grid."""
        grids = inputs.get("video_grid_thw")
        if grids is None or tuple(grids.shape) != (1, 3) or inputs["input_ids"].shape[0] != 1:
            raise ValueError("Visual gating needs one question with one video")
        merge = int(self.model.config.vision_config.spatial_merge_size)
        t, h, w = (int(value) for value in grids[0].tolist())
        grid = (t, h // merge, w // merge)
        positions = (inputs["input_ids"][0] == self.model.config.video_token_id).nonzero().flatten()
        if positions.numel() != grid[0] * grid[1] * grid[2]:
            raise ValueError("Video placeholders do not match the video grid")
        return positions, grid

    @contextmanager
    def _visual_gate(
        self, positions: Any, *, gate: Any = None, blocked: Sequence[int] | None = None
    ) -> Iterator[None]:
        """Scale visual Values in every language layer during the enclosed forward pass or generation.

        ``gate``: one differentiable value per visual token (attribution).
        ``blocked``: visual-token indices whose Values become 0 (negative branch).
        """
        import torch

        if (gate is None) == (blocked is None):
            raise ValueError("Pass exactly one of gate and blocked")
        # Built once per call over the prompt; index_copy keeps the gradient path to gate.
        scale = torch.ones(int(positions[-1]) + 1, device=positions.device, dtype=torch.float32)
        if gate is not None:
            scale = scale.index_copy(0, positions, gate)
        else:
            scale = scale.index_fill(0, positions[torch.as_tensor(list(blocked), device=positions.device)], 0.0)
        state = {"gate": scale.to(self.model.dtype), "layers": set()}
        token = _gate_state.set(state)
        try:
            yield
        finally:
            _gate_state.reset(token)
        layers = self.model.config.text_config.num_hidden_layers
        if len(state["layers"]) != layers:
            raise RuntimeError(f"The visual gate reached {len(state['layers'])} of {layers} language layers")

    def _stop_ids(self) -> set[int]:
        stop = self.model.generation_config.eos_token_id
        return set(stop) if isinstance(stop, (list, tuple)) else {stop}

    def _answer(self, token_ids: list[int]) -> Generation:
        """New tokens to an answer, without the closing end-of-turn token."""
        if token_ids and token_ids[-1] in self._stop_ids():
            token_ids = token_ids[:-1]
        return Generation(self._decode(token_ids), token_ids)

    def _decode_greedy(self, first_token: int, cache: Any, max_new_tokens: int) -> list[int]:
        """Greedy tokens after a prefill whose next token is ``first_token``, as
        generate() picks them, extending ``cache`` in place.

        Without an attention mask the model places each new token at the cached
        length plus the rope offset that the last prefill stored, as generate() does.
        """
        import torch

        stop = self._stop_ids()
        tokens = [first_token]
        with torch.inference_mode():
            while tokens[-1] not in stop and len(tokens) < max_new_tokens:
                step = self.model(
                    input_ids=torch.tensor([[tokens[-1]]], device=self.device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                tokens.append(int(step.logits[0, -1].float().argmax()))
        return tokens

    def _image_chat(self, timestamps: Sequence[float], system_prompt: str, task_prompt: str) -> str:
        """Chat text with one time label and one image placeholder per frame, then the task prompt."""
        content: list[dict[str, Any]] = []
        for seconds in timestamps:
            content += [{"type": "text", "text": f"<{seconds:.1f} seconds>"}, {"type": "image"}]
        content.append({"type": "text", "text": task_prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _decode(self, token_ids: list[int]) -> str:
        # vLLM's default detokenization, as in the baseline: no special tokens, stripped.
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def _require_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load() first")
