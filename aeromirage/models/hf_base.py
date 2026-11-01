"""Shared base for every model run with Hugging Face Transformers.

VideoLLaMA3, VideoChat-Flash, LLaVA-NeXT-Video, LLaVA-OneVision-2, Kimi-VL,
MiniCPM-V 4 and InternVL3.5 each run their checkpoint's official code in
their own Conda environment.
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar, Iterable

from aeromirage.models.base import BaseModelAdapter


# Greedy decoding for every model. Sampling keys are set to None explicitly
# because some checkpoints enable sampling in generation_config.json
# (VideoChat-Flash: do_sample, temperature 0.7, repetition_penalty 1.05, ...).
GREEDY: dict[str, Any] = {
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}


def new_tokens(output_ids: Any, input_ids: Any) -> Any:
    """Drop the prompt when generate() echoes it.

    Plain Transformers models return prompt + answer; models that generate
    from input embeddings (VideoLLaMA3, LLaVA) return only the answer.
    """
    if input_ids is not None:
        length = input_ids.shape[-1]
        if output_ids.shape[-1] >= length and bool(
            (output_ids[:, :length].to(input_ids.device) == input_ids).all()
        ):
            return output_ids[:, length:]
    return output_ids


class HFAdapter(BaseModelAdapter):
    """Holds the loaded model and moves its inputs to the right device."""

    # Extra batch_decode arguments, e.g. from the model's official example.
    decode_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, config, video=None):
        super().__init__(config, video)
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.torch: Any = None
        self.input_device: Any = None  # where input tensors go
        self.dtype: Any = None  # model dtype, for pixel inputs that need casting

    def _to_device(self, batch: Any, cast: Iterable[str] = ()) -> dict[str, Any]:
        """Move every tensor to ``input_device``; keys in ``cast`` also to ``dtype``."""
        cast = set(cast)
        return {
            key: value.to(self.input_device, dtype=self.dtype if key in cast else None)
            if isinstance(value, self.torch.Tensor)
            else value
            for key, value in dict(batch).items()
        }

    def _remote_module(self, name: str) -> Any:
        """A module shipped with the checkpoint, e.g. its conversation.py."""
        package = type(self.model).__module__.rsplit(".", 1)[0]
        return importlib.import_module(f"{package}.{name}")

    def _generate_kwargs(self) -> dict[str, Any]:
        """Model-specific generate() arguments on top of GREEDY."""
        return {}

    def run(self, inputs: dict[str, Any], max_new_tokens: int) -> str:
        """``model.generate`` on ``inputs["model_inputs"]``, decoding only the answer."""
        self._require_loaded("model", "torch")
        model_inputs = inputs["model_inputs"]
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens, **GREEDY, **self._generate_kwargs()
            )
        answer_ids = new_tokens(output_ids, model_inputs.get("input_ids"))
        decoder = self.tokenizer if self.tokenizer is not None else self.processor
        return decoder.batch_decode(answer_ids, skip_special_tokens=True, **self.decode_kwargs)[0].strip()

    def _generate_until_stop(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int,
        stopping_criteria_class: Any,
        **model_kwargs: Any,
    ) -> str:
        """LLaVA-style generate(inputs=..., images=...) that ends at a stop string."""
        self._require_loaded("model", "tokenizer", "torch")
        stop = inputs["stop_str"]
        criteria = stopping_criteria_class([stop], self.tokenizer, inputs["input_ids"])
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"],
                stopping_criteria=[criteria],
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **GREEDY,
                **model_kwargs,
            )
        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return text[: -len(stop)].strip() if text.endswith(stop) else text

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.torch = None
        self.input_device = None
        self.dtype = None
        super().close()
