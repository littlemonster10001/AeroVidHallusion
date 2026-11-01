"""VideoLLaMA3 backbone for AeroVera (the checkpoint's own Transformers code; ``videollama3`` environment).

The language model is Qwen2 with ordinary 1D RoPE. A video's frames are
merged 2 x 2 and then compressed: patches that barely change from the
previous frame are dropped before the language model, so a video's visual
tokens are located after that compression. Inputs are therefore prepared
once into embeddings (``Prompt``); every forward pass starts from them.

Inside ``_visual_gate``, a forward hook on every layer's ``v_proj`` scales
the Values of the visual tokens (Appendix C.3) in any forward pass over the
whole gated sequence; decoding steps reuse the scaled Values from the cache.
A Probing prompt has about 15.7k tokens (the moving camera leaves little to
compress), too many to keep every activation for the attribution's backward
pass, so that pass recomputes the language layers one at a time (gradient
checkpointing). Heavy imports stay inside the methods.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Sequence

from aerovera.backbones.base import GREEDY, Attribution, Generation, cut_cache

DTYPES = ("bfloat16", "float16", "float32")
IMAGE_TOKEN = "<image>"  # the chat template's placeholder, expanded to one token per visual token
PATCH = 14
MERGE = 2  # video frames, and the positive branch's images, are merged 2 x 2: sides are multiples of 28


@dataclass
class Video:
    """Frames of one video as the official ``load_video`` samples them (the benchmark baseline's input)."""

    path: str
    frames: list[Any]  # [3, H, W] uint8 at the source resolution
    timestamps: list[float]  # the labels load_video gives them (shown to the model in the Probing Pass)
    frame_ids: list[int]  # the source frames they are (decord indices)
    fps: float  # source frame rate
    total_frames: int
    size: tuple[int, int]  # (height, width) the processor resizes them to: the LR size
    max_visual_tokens: int  # the processor's token budget for the video
    # Filled by the first video_inputs (main thread): the processor's pixel values
    # and grid, and those pixels on the device, shared by every question on the video.
    image_inputs: Any = None
    pixels: Any = None


@dataclass
class Prompt:
    """Model inputs after the multimodal preparation, ready for the language model."""

    embeds: Any  # [1, L, D] input embeddings, visual tokens included (after compression)
    ids: Any  # [L] token ids of that sequence (visual tokens as the image token)
    visual: Any  # [M] positions of the visual tokens in the sequence

    @property
    def length(self) -> int:
        return int(self.embeds.shape[1])


def source_frame_ids(timestamps: Sequence[float], sample_fps: float, fps: float, total: int) -> list[int]:
    """The source frames that ``load_video`` returns for ``timestamps``.

    Its ffmpeg ``fps`` filter (``round="down"``) hands out, as the frame
    labelled t, the source frame one sampling interval later: t + 1/sample_fps
    (checked pixel by pixel against decord).
    """
    return [min(total - 1, int((t + 1 / sample_fps) * fps + 1e-6)) for t in timestamps]


class VideoLLaMA3:
    """VideoLLaMA3 with the inputs and decoding that AeroVera needs."""

    def __init__(self, path: str | Path, device: str = "cuda:0", dtype: str = "bfloat16",
                 attention: str = "flash_attention_2"):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        self.path = str(path)
        self.device = device
        self.dtype = dtype
        self.attention = attention  # as the benchmark baseline
        self.model: Any = None
        self.processor: Any = None
        # Set by _visual_gate: {"scale": Value scale per position, "layers": layers scaled}. An
        # attribute, not a context variable: the backward pass's recomputation runs in
        # autograd's own threads. Only the main thread runs the language model.
        self._gate: dict[str, Any] | None = None
        self._video_pixels: Any = None  # the current video's pixels, whose vision-encoder output is kept
        self._features: tuple[Any, Any] | None = None  # (those pixels, the output)
        self._compressed: Any = None  # token ids after the last compression

    def load(self) -> None:
        if not Path(self.path).is_dir():
            raise FileNotFoundError(f"VideoLLaMA3 checkpoint not found: {self.path}")
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.path, trust_remote_code=True, local_files_only=True, device_map={"": self.device},
            torch_dtype=getattr(torch, self.dtype), attn_implementation=self.attention,
        )
        model.generation_config.update(**GREEDY)
        model.requires_grad_(False)
        self.model = model.eval()
        for index, layer in enumerate(model.model.layers):
            layer.self_attn.v_proj.register_forward_hook(partial(self._scale_values, index))
        self._wrap_model()

    def _scale_values(self, index: int, module: Any, args: Any, output: Any) -> Any:
        """Forward hook on layer ``index``'s ``v_proj``: inside a visual gate, scale the
        Values of a forward pass over the whole gated sequence (not decoding steps)."""
        state = self._gate
        if state is None or output.shape[1] != state["scale"].numel():
            return None
        state["layers"].add(index)
        return output * state["scale"][None, :, None]

    def _wrap_model(self) -> None:
        """Cache the vision encoder's output for the current video's pixels (every
        question on a video shares one pixel tensor), and record the token ids
        that survive compression, which locate the visual tokens."""
        import torch

        encode, compress = self.model.encode_images, self.model._compress_visual_tokens

        def encode_images(pixel_values, grid_sizes, merge_sizes):
            cached = self._features
            if cached is not None and cached[0] is pixel_values:
                return cached[1]
            with torch.no_grad():
                output = encode(pixel_values, grid_sizes, merge_sizes)
            if pixel_values is self._video_pixels:
                self._features = (pixel_values, output)
            return output

        def compress_visual_tokens(*args, **kwargs):
            result = compress(*args, **kwargs)
            self._compressed = result[1]
            return result

        self.model.encode_images = encode_images
        self.model._compress_visual_tokens = compress_visual_tokens

    # --- Inputs -------------------------------------------------------------

    def read_video(self, path: str | Path, num_frames: int, fps: float, max_visual_tokens: int) -> Video:
        """``load_video`` exactly as the benchmark baseline calls it: fps raised for
        short clips so that at least ``num_frames`` candidates exist.

        Runs in the prefetch thread, so it leaves the processor's settings alone:
        the LR size comes from the processor module's own sizing function, and
        the pixels are processed by the first ``video_inputs``."""
        import ffmpeg
        from decord import VideoReader, cpu

        self._require_loaded()
        duration = float(ffmpeg.probe(str(path))["format"]["duration"])
        sample_fps = max(float(fps), (num_frames + 2) / duration)
        frames, timestamps = self.processor.load_video(str(path), fps=sample_fps, max_frames=num_frames)
        if len(frames) != num_frames:
            raise RuntimeError(f"{path}: load_video returned {len(frames)} frames, expected {num_frames}")
        reader = VideoReader(str(path), ctx=cpu(0))
        source_fps, total = float(reader.get_avg_fps()), len(reader)
        sizing = sys.modules[type(self.processor.image_processor).__module__].simple_batched_resize
        size = sizing([list(frames)], factor=PATCH * MERGE, min_tokens=self.processor.image_processor.min_tokens,
                      max_tokens=max_visual_tokens)[0]
        return Video(
            str(path), list(frames), [float(t) for t in timestamps],
            source_frame_ids(timestamps, sample_fps, source_fps, total), source_fps, total,
            (int(size[0]), int(size[1])), max_visual_tokens,
        )

    def source_frames(self, video: Video) -> list[Any]:
        """The Probing frames as load_video decoded them (ffmpeg), [H, W, 3]."""
        return [frame.transpose(1, 2, 0) for frame in video.frames]

    @staticmethod
    def resize(frame: Any, size: Sequence[int]) -> Any:
        """RGB uint8 frame [H, W, 3] resized to ``size`` (height, width) as the
        processor resizes images and frames (PIL bicubic on uint8); a frame
        already at ``size`` passes through it unchanged."""
        from transformers.image_transforms import resize
        from transformers.image_utils import ChannelDimension, PILImageResampling

        return resize(frame, tuple(size), resample=PILImageResampling.BICUBIC,
                      input_data_format=ChannelDimension.LAST)

    @staticmethod
    def hr_size(height: int, width: int, short_side: int = 1024) -> tuple[int, int]:
        """Short side at most ``short_side`` (never enlarged), sides floored to multiples of 28."""
        factor = PATCH * MERGE
        scale = min(1.0, short_side / min(height, width))
        return (max(factor, int(height * scale) // factor * factor),
                max(factor, int(width * scale) // factor * factor))

    def video_inputs(self, video: Video, system_prompt: str, task_prompt: str) -> Prompt:
        """The baseline's conversation: the frames as a video with their timestamps, then the task prompt."""
        import torch

        self._require_loaded()
        if video.pixels is None:
            video.image_inputs = self._process_images([("video", list(video.frames))], video.max_visual_tokens)
            grid = video.image_inputs["grid_sizes"][0]
            if (int(grid[1]) * PATCH, int(grid[2]) * PATCH) != video.size:
                raise RuntimeError(f"{video.path}: the processor resized the frames to {grid}, not {video.size}")
            video.pixels = video.image_inputs["pixel_values"].to(self.device, getattr(torch, self.dtype))
        self._video_pixels = video.pixels
        content = [{"type": "video", "video": video.frames, "num_frames": len(video.frames),
                    "timestamps": video.timestamps}, {"type": "text", "text": task_prompt}]
        return self._prompt(self._chat(system_prompt, content), video.image_inputs, video.pixels)

    def image_inputs(self, frames: Sequence[Any], timestamps: Sequence[float], system_prompt: str,
                     task_prompt: str) -> Prompt:
        """Frames as separate images, each preceded by ``Time t s:`` (the template's
        timestamp field, P3), merged 2 x 2 like video frames and fed at their own size."""
        self._require_loaded()
        content = [{"type": "image", "timestamp": float(t)} for t in timestamps]
        content.append({"type": "text", "text": task_prompt})
        image_inputs = self._process_images([("image", frame) for frame in frames], None)
        return self._prompt(self._chat(system_prompt, content), image_inputs)

    def text_inputs(self, system_prompt: str, user_prompt: str) -> Prompt:
        """Text only (target extraction)."""
        import torch

        self._require_loaded()
        text = self._chat(system_prompt, [{"type": "text", "text": user_prompt}])
        ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            embeds = self.model.get_model().embed_tokens(ids)
        return Prompt(embeds, ids[0], ids.new_zeros(0))

    def _chat(self, system_prompt: str, content: list[dict[str, Any]]) -> str:
        conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
        return self.processor.apply_chat_template(conversation, tokenize=False, add_system_prompt=True,
                                                  add_generation_prompt=True)

    def _process_images(self, images: list[tuple[str, Any]], max_visual_tokens: int | None) -> Any:
        """The processor's image step (main thread only: it sets the processor's
        token budget for the call). Videos keep the official budget; the
        positive branch's images, already at their size and merged 2 x 2, get
        none, so that none is shrunk."""
        processor = self.processor.image_processor
        budget = processor.max_tokens
        processor.max_tokens = max_visual_tokens or 10 ** 9
        try:
            return self.processor.process_images(images, merge_size=[MERGE] * len(images), return_tensors="pt")
        finally:
            processor.max_tokens = budget

    def _prompt(self, text: str, image_inputs: Any, pixels: Any = None) -> Prompt:
        """Tokenize ``text`` around the images, then embed it with the model's own
        multimodal preparation (vision encoder, compression), as forward() would."""
        import torch

        batch = self.processor.process_text(text, image_inputs, padding=False, padding_side=None, return_tensors="pt")
        pixel_values = pixels if pixels is not None else image_inputs["pixel_values"].to(
            self.device, getattr(torch, self.dtype))
        with torch.no_grad():
            outputs = self.model.prepare_inputs_labels_for_multimodal(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=pixel_values,
                grid_sizes=image_inputs["grid_sizes"].to(self.device),
                merge_sizes=image_inputs["merge_sizes"].to(self.device),
                modals=image_inputs["modals"],
            )
        ids = self._compressed.flatten()
        visual = (ids == self.model.config.image_token_index).nonzero().flatten()
        return Prompt(outputs[4], ids, visual)

    # --- Decoding -----------------------------------------------------------

    def generate(self, inputs: Prompt, max_new_tokens: int, mask: Sequence[int] | None = None) -> Generation:
        """Greedy answer, as the baseline's generate() decodes; ``mask`` lists visual
        tokens (indices among ``inputs.visual``) whose Values are zeroed."""
        import torch

        self._require_loaded()
        gate = nullcontext() if not mask else self._visual_gate(inputs, blocked=mask)
        with torch.inference_mode(), gate:
            prefill = self._forward(inputs, use_cache=True)
        first = int(prefill.logits[0, -1].float().argmax())
        return self._answer(self._decode_greedy(first, prefill.past_key_values, inputs.length, max_new_tokens))

    def first_logits(self, inputs: Prompt, mask: Sequence[int] | None = None) -> Any:
        """Raw logits [V] (float32) at the first answer position; ``mask`` as in ``generate``."""
        import torch

        self._require_loaded()
        gate = nullcontext() if not mask else self._visual_gate(inputs, blocked=mask)
        with torch.inference_mode(), gate:
            return self._forward(inputs, use_cache=False).logits[0, -1].float()

    def first_logits_many(self, frames: Sequence[Any], timestamps: Sequence[float], system_prompt: str,
                          task_prompts: Sequence[str]) -> list[Any]:
        """``first_logits(image_inputs(frames, timestamps, system_prompt, p))`` for each prompt p.

        The prompts differ only after the last image, so the images are
        encoded and prefilled once; each prompt then runs only its own text on
        that cache, which is cut back to the images afterwards.
        """
        import torch

        full = self.image_inputs(frames, timestamps, system_prompt, task_prompts[0])
        cut = int(full.visual[-1]) + 1
        logits = []
        with torch.inference_mode():
            prefix = self._forward(Prompt(full.embeds[:, :cut], full.ids[:cut], full.visual), use_cache=True)
            cache = prefix.past_key_values
            for index, task_prompt in enumerate(task_prompts):
                text = self._chat(system_prompt, [{"type": "image", "timestamp": float(t)} for t in timestamps]
                                  + [{"type": "text", "text": task_prompt}])
                # The suffix starts after the image token, a special token, so
                # tokenizing it alone gives the same ids as the whole text.
                suffix = self.processor.tokenizer(text[text.rindex(IMAGE_TOKEN) + len(IMAGE_TOKEN):],
                                                  add_special_tokens=False, return_tensors="pt").input_ids
                if index == 0 and not torch.equal(suffix[0].to(full.ids.device), full.ids[cut:]):
                    raise RuntimeError("The text after the images does not tokenize as in the full prompt")
                logits.append(self._continue(suffix, cache, cut).logits[0, -1].float())
                cut_cache(cache, cut)
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
        self, positive: Prompt, negative: Prompt, alpha: float, max_new_tokens: int, mask: Sequence[int] | None = None
    ) -> tuple[Generation, Generation, tuple[Any, Any, Any]]:
        """Open-ended contrastive decoding (P6): the first token maximizes
        q = q+ + alpha (q+ - q-) over the vocabulary, then the positive branch
        continues greedily on its own. Returns the contrastive answer, the
        positive branch's own greedy answer, and the first-position logits."""
        import torch

        self._require_loaded()
        gate = nullcontext() if not mask else self._visual_gate(negative, blocked=mask)
        with torch.inference_mode():
            output = self._forward(positive, use_cache=True)
            cache = output.past_key_values
            with gate:
                negative_logits = self._forward(negative, use_cache=False).logits[0, -1].float()
            positive_logits = output.logits[0, -1].float()
            combined = positive_logits + alpha * (positive_logits - negative_logits)
            answers = {}
            for first in dict.fromkeys((int(combined.argmax()), int(positive_logits.argmax()))):
                cut_cache(cache, positive.length)
                answers[first] = self._answer(self._decode_greedy(first, cache, positive.length, max_new_tokens))
        return (answers[int(combined.argmax())], answers[int(positive_logits.argmax())],
                (positive_logits, negative_logits, combined))

    def attribute(
        self, inputs: Prompt, *, max_new_tokens: int | None = None, answer_ids: Sequence[int] | None = None
    ) -> tuple[Generation, Attribution | None]:
        """A_i = g_i * ds/dg_i of every visual token of ``inputs`` to an answer (Eq. 7, Appendix C.3).

        The answer is either decoded greedily, exactly as ``generate()``
        (``max_new_tokens``: the Probing Pass), or given (``answer_ids``, without
        the end-of-turn token). Then one forward pass over prompt + answer with
        the gates g (all 1) on the autograd graph gives s, and one backward pass
        gives ds/dg; the model weights are frozen and the language layers are
        recomputed during the backward pass instead of being stored. An empty
        answer has no attribution.
        """
        import torch

        self._require_loaded()
        if (max_new_tokens is None) == (answer_ids is None):
            raise ValueError("Pass max_new_tokens to decode the answer, or answer_ids to attribute a given one")
        if answer_ids is None:
            answer = self.generate(inputs, max_new_tokens)
        else:
            answer = Generation(self._decode(list(answer_ids)), list(answer_ids))
        if not answer.token_ids:
            return answer, None
        device = inputs.embeds.device
        ids = torch.tensor(answer.token_ids, device=device)
        with torch.no_grad():
            embeds = torch.cat((inputs.embeds, self.model.get_model().embed_tokens(ids[None, :-1])), dim=1)
        mask = torch.ones((1, embeds.shape[1]), dtype=torch.long, device=device)
        gate = torch.ones(inputs.visual.numel(), device=device, dtype=torch.float32, requires_grad=True)
        with torch.enable_grad(), self._visual_gate(inputs, gate=gate, length=embeds.shape[1]), self._checkpointing():
            logits = self.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False,
                                num_logits_to_keep=ids.numel()).logits[0].float()
            score = logits.log_softmax(-1).gather(1, ids[:, None]).sum()
            (gradient,) = torch.autograd.grad(score, gate)
        return answer, Attribution((gate.detach() * gradient).cpu(), (gate.numel(),), float(score.detach()))

    # --- Internals ----------------------------------------------------------

    def _forward(self, inputs: Prompt, use_cache: bool) -> Any:
        """The language model on a prepared prompt (with an all-ones attention mask, as generate() passes).

        With ``use_cache`` it gets a ``DynamicCache`` to fill: without one,
        Transformers 4.46 returns a legacy tuple that later steps would copy
        instead of extend, losing the generated tokens.
        """
        import torch
        from transformers import DynamicCache

        mask = torch.ones((1, inputs.length), dtype=torch.long, device=inputs.embeds.device)
        return self.model(inputs_embeds=inputs.embeds, attention_mask=mask, use_cache=use_cache,
                          past_key_values=DynamicCache() if use_cache else None, num_logits_to_keep=1)

    def _continue(self, ids: Any, cache: Any, length: int, keep: int = 1) -> Any:
        """Feed token ``ids`` [1, n] after ``length`` cached tokens."""
        import torch

        ids = ids.to(self.device)
        mask = torch.ones((1, length + ids.shape[1]), dtype=torch.long, device=ids.device)
        return self.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True,
                          num_logits_to_keep=keep)

    @contextmanager
    def _visual_gate(self, inputs: Prompt, *, gate: Any = None, blocked: Sequence[int] | None = None,
                     length: int | None = None) -> Iterator[None]:
        """Scale the visual tokens' Values in every language layer during forward passes over
        the whole gated sequence: ``inputs`` (default) or ``length`` tokens starting with it.

        ``gate``: one differentiable value per visual token (attribution).
        ``blocked``: visual-token indices whose Values become 0 (negative branch).
        """
        import torch

        if (gate is None) == (blocked is None):
            raise ValueError("Pass exactly one of gate and blocked")
        scale = torch.ones(length or inputs.length, device=inputs.embeds.device, dtype=torch.float32)
        if gate is not None:
            scale = scale.index_copy(0, inputs.visual, gate)
        else:
            scale = scale.index_fill(0, inputs.visual[torch.as_tensor(list(blocked), device=scale.device)], 0.0)
        self._gate = {"scale": scale.to(self.model.dtype), "layers": set()}
        try:
            yield
        finally:
            state, self._gate = self._gate, None
        layers = self.model.config.num_hidden_layers
        if len(state["layers"]) != layers:
            raise RuntimeError(f"The visual gate reached {len(state['layers'])} of {layers} language layers")

    @contextmanager
    def _checkpointing(self) -> Iterator[None]:
        """Gradient checkpointing of the language layers (Transformers applies it in training mode only)."""
        import torch.utils.checkpoint

        inner = self.model.model
        inner.gradient_checkpointing = True
        inner._gradient_checkpointing_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        inner.train()  # no dropout in this model: attention_dropout is 0
        try:
            yield
        finally:
            inner.gradient_checkpointing = False
            inner.eval()

    def _decode_greedy(self, first_token: int, cache: Any, length: int, max_new_tokens: int) -> list[int]:
        """Greedy tokens after a prefill of ``length`` tokens whose next token is
        ``first_token``, as generate() picks them, extending ``cache`` in place."""
        import torch

        stop = self._stop_ids()
        tokens = [first_token]
        with torch.inference_mode():
            while tokens[-1] not in stop and len(tokens) < max_new_tokens:
                step = self._continue(torch.tensor([[tokens[-1]]]), cache, length + len(tokens) - 1)
                tokens.append(int(step.logits[0, -1].float().argmax()))
        return tokens

    def _stop_ids(self) -> set[int]:
        stop = self.model.generation_config.eos_token_id
        return set(stop) if isinstance(stop, (list, tuple)) else {stop}

    def _answer(self, token_ids: list[int]) -> Generation:
        """New tokens to an answer, without the closing end-of-turn token."""
        if token_ids and token_ids[-1] in self._stop_ids():
            token_ids = token_ids[:-1]
        return Generation(self._decode(token_ids), token_ids)

    def _decode(self, token_ids: list[int]) -> str:
        # As the baseline: special tokens skipped, stripped.
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def _require_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load() first")
