#!/usr/bin/env python3
"""Shared text and model utilities for the speech-driven workflow."""

import re
from typing import Optional

from speech_workflow.llm_runtime import load_llm_module


TAG_PATTERN = r"\[(IM|RM|R|D)\]"


def strip_annotation_tags(text: str) -> str:
    """Remove annotation markers while preserving the original ASR text."""
    return re.sub(TAG_PATTERN, "", text).strip()


def clean_model_output(text: str) -> str:
    """Remove common Markdown fences from a plain-text model response."""
    text = text.strip()
    fenced = re.fullmatch(
        r"```(?:text|plaintext|json)?[ \t]*\r?\n(?P<body>.*?)\r?\n?```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group("body").strip() if fenced else text


def safe_model_name(model: str) -> str:
    """Convert a model identifier into a filesystem-safe name."""
    name = re.sub(r"[^0-9A-Za-z]+", "_", model).strip("_").lower()
    return name or "model"


def get_model_registry() -> dict:
    """Load provider configuration only when model execution is requested."""
    return getattr(load_llm_module(), "MODEL_REGISTRY", {})


def parse_model_names(
    model: Optional[str],
    models: Optional[list[str]],
    all_models: bool,
    model_registry: Optional[dict] = None,
) -> list[str]:
    """Normalize single, multiple, or registry-wide model selections."""
    if all_models:
        registry = model_registry if model_registry is not None else get_model_registry()
        return list(registry.keys())

    names = []
    raw_items = []
    if model:
        raw_items.append(model)
    if models:
        raw_items.extend(models)

    for item in raw_items:
        for name in str(item).split(","):
            name = name.strip()
            if name and name not in names:
                names.append(name)
    return names or ["gemini-2.5-pro"]
