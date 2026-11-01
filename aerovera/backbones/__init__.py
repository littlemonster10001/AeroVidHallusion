"""Backbone-specific code: input building, decoding, gated attribution and masking."""

from __future__ import annotations

from typing import Any

FAMILIES = ("qwen25_vl", "videollama3")


def create_backbone(family: str, path: str, device: str, dtype: str) -> Any:
    """The backbone of ``model.family`` (not loaded yet)."""
    if family == "qwen25_vl":
        from aerovera.backbones.qwen25_vl import Qwen25VL

        return Qwen25VL(path, device, dtype)
    if family == "videollama3":
        from aerovera.backbones.videollama3 import VideoLLaMA3

        return VideoLLaMA3(path, device, dtype)
    raise ValueError(f"Unknown model.family {family!r}; expected one of {FAMILIES}")
