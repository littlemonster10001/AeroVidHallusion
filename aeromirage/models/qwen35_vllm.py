"""Qwen3.5 through vLLM (2B / 4B / 9B / 27B, with or without thinking)."""

from __future__ import annotations

from typing import Any

from aeromirage.models.qwen3_vl_vllm import Qwen3VLVLLMAdapter
from aeromirage.models.registry import register_model


@register_model("qwen35_vllm")
class Qwen35VLLMAdapter(Qwen3VLVLLMAdapter):
    model_label = "Qwen3.5"

    def load(self) -> None:
        # Reject bad thinking options before spending minutes on engine start-up.
        options = self.config.options
        thinking = options.get("enable_thinking", False)
        if not isinstance(thinking, bool):
            raise TypeError("model.options.enable_thinking must be true or false")
        budget = options.get("thinking_token_budget", 0)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("model.options.thinking_token_budget must be a non-negative integer")
        if budget and not thinking:
            raise ValueError("model.options.thinking_token_budget requires enable_thinking: true")
        super().load()

    def _chat_template_kwargs(self) -> dict[str, Any]:
        # The template opens <think> for thinking mode and inserts an empty
        # <think></think> otherwise; the prompt text itself is identical.
        return {"enable_thinking": self.config.options.get("enable_thinking", False)}

    def run(self, inputs: dict[str, Any], max_new_tokens: int) -> str:
        """With a thinking budget, think first and answer with the rest.

        Without a cap, thinking can use all of max_new_tokens and leave no
        answer. Stage one stops at </think> or the budget (closing the
        thinking if needed); stage two continues the same prompt after
        ``</think>\\n\\n``, as Qwen3.5's template formats a finished turn.
        """
        budget = int(self.config.options.get("thinking_token_budget", 0))
        if not budget:
            return super().run(inputs, max_new_tokens)
        if budget >= max_new_tokens:
            raise ValueError(
                f"thinking_token_budget ({budget}) must be smaller than max_new_tokens "
                f"({max_new_tokens})"
            )
        thinking = self._sample(
            inputs, budget, stop=["</think>"], include_stop_str_in_output=True
        ).text.rstrip()
        if "</think>" not in thinking:
            thinking = f"{thinking}\n</think>" if thinking else "</think>"
        follow_up = {**inputs, "prompt": f"{inputs['prompt']}{thinking}\n\n"}
        answer = self._sample(follow_up, max_new_tokens - budget).text.strip()
        return f"{thinking}\n\n{answer}"
