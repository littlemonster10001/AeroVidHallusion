"""Registry for model-family adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aeromirage.config import ModelConfig

if TYPE_CHECKING:
    from aeromirage.models.base import BaseModelAdapter


MODEL_REGISTRY: dict[str, type[BaseModelAdapter]] = {}


def register_model(name: str):
    def decorator(adapter_class: type[BaseModelAdapter]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Duplicate model adapter registration: {name}")
        MODEL_REGISTRY[name] = adapter_class
        return adapter_class

    return decorator


def create_model_adapter(
    config: ModelConfig, video: dict[str, Any] | None = None
) -> BaseModelAdapter:
    try:
        adapter_class = MODEL_REGISTRY[config.family]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown model family {config.family!r}. Available adapters: {available}"
        ) from exc
    return adapter_class(config, video)
