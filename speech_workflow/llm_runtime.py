#!/usr/bin/env python3
"""Load the provider-neutral model runner for speech-workflow scripts."""

from __future__ import annotations

import sys
from pathlib import Path


SPEECH_WORKFLOW_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SPEECH_WORKFLOW_DIR.parent


def ensure_import_paths() -> None:
    """Make local scripts importable when files are executed directly."""
    for path in (str(SPEECH_WORKFLOW_DIR), str(BENCHMARK_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_llm_module():
    """Load the shared public model API adapters."""
    ensure_import_paths()
    from visual_workflow import model_api

    return model_api
