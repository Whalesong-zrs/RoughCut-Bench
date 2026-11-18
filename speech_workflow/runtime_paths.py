"""Shared filesystem defaults for the speech-driven workflow."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ROUGHCUT_DATA_ROOT") or PROJECT_ROOT / "data")
OUTPUT_ROOT = Path(os.environ.get("ROUGHCUT_OUTPUT_ROOT") or PROJECT_ROOT / "outputs")
SPEECH_DATA_ROOT = Path(os.environ.get("ROUGHCUT_SPEECH_DATA_ROOT") or DATA_ROOT / "speech")
SPEECH_OUTPUT_ROOT = Path(os.environ.get("ROUGHCUT_SPEECH_OUTPUT_ROOT") or OUTPUT_ROOT / "speech")
