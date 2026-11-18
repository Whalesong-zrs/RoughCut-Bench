#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || \
    fail "Python executable not found: ${PYTHON_BIN}"

if ! "$PYTHON_BIN" - <<'PY'
import sys

minimum = (3, 10)
maximum = (3, 13)
if not minimum <= sys.version_info[:2] < maximum:
    raise SystemExit(
        "RoughCut-Bench requires Python 3.10, 3.11, or 3.12; "
        f"found {sys.version.split()[0]}."
    )
PY
then
    fail "Set PYTHON_BIN to a supported Python interpreter."
fi

command -v ffmpeg >/dev/null 2>&1 || fail \
    "FFmpeg is required. Install it with 'sudo apt-get install ffmpeg' on Ubuntu or 'brew install ffmpeg' on macOS."

FFMPEG_ENCODERS="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
grep -q '[[:space:]]libx264[[:space:]]' <<<"$FFMPEG_ENCODERS" || fail \
    "The installed FFmpeg build does not provide the libx264 encoder."
grep -q '[[:space:]]aac[[:space:]]' <<<"$FFMPEG_ENCODERS" || fail \
    "The installed FFmpeg build does not provide the AAC encoder."

if [[ ! -d "$VENV_DIR" ]]; then
    printf 'Creating virtual environment: %s\n' "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR" || fail \
        "Failed to create the virtual environment. On Ubuntu, install python3-venv first."
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
[[ -x "$VENV_PYTHON" ]] || fail \
    "The existing virtual environment is invalid: ${VENV_DIR}"

printf 'Installing public dependencies from %s\n' "$REQUIREMENTS_FILE"
"$VENV_PYTHON" -m pip install --upgrade 'pip>=24,<27'
"$VENV_PYTHON" -m pip install --requirement "$REQUIREMENTS_FILE"
"$VENV_PYTHON" -m pip check

PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/speech_workflow${PYTHONPATH:+:${PYTHONPATH}}" \
    "$VENV_PYTHON" - <<'PY'
import decord
import pillow_heif
from moviepy.editor import VideoFileClip
from scipy.stats import kendalltau

from benchmark_evaluation import human_aligned_evaluator
from visual_workflow import model_api, model_clients, workflow_utils

pillow_heif.register_heif_opener()
assert kendalltau([1, 2, 3], [1, 2, 3]).statistic == 1.0
assert decord.VideoReader is not None
assert VideoFileClip is not None
assert model_api.MODEL_REGISTRY and model_clients.MODEL_REGISTRY
assert workflow_utils.read_json is not None
assert human_aligned_evaluator.EVALUATOR_PROMPT_VERSION
print("Dependency import check passed.")
PY

printf '\nEnvironment ready. Activate it with:\n  source %s/bin/activate\n' "$VENV_DIR"
