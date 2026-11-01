"""Model adapters. Importing this package registers every model family."""

from aeromirage.models import (  # noqa: F401 - imported to register adapters
    internvl35_transformers,
    keye_vl_vllm,
    kimi_vl_transformers,
    llava_next_video,
    llava_onevision2_transformers,
    minicpm_v4_transformers,
    qwen25_vl_vllm,
    qwen3_vl_vllm,
    qwen35_vllm,
    videochat_flash_transformers,
    videollama3_transformers,
)
from aeromirage.models.base import BaseModelAdapter
from aeromirage.models.registry import MODEL_REGISTRY, create_model_adapter, register_model

__all__ = ["BaseModelAdapter", "MODEL_REGISTRY", "create_model_adapter", "register_model"]
