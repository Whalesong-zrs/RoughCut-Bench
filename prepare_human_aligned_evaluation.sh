#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/common.sh

python scripts/tools/prepare_human_aligned_evaluation.py \
    --manifest "${ROUGHCUT_PROJECT_ROOT}/human_aligned_evaluation_runs.tsv" \
    --result-root "${ROUGHCUT_OUTPUT_ROOT}/human_aligned_evaluation" \
    --eval-repeat-count 3
