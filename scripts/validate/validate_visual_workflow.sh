#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

OUTPUT_ROOT="${ROUGHCUT_OUTPUT_ROOT}/vlm_comparison"
MODEL="gemini-2.5-pro"
CUT_PLANNING_PROMPT_VERSION="v33"
TEST_IDS_STR="test1 test2 test3"
SET_NAMES_STR="set1 set2"
IOU_THRESHOLD="0.5"

GT_PATHS=(
    "${ROUGHCUT_SET1_LABELS}"
    "${ROUGHCUT_SET2_LABELS}"
)

read -r -a TEST_IDS <<< "$TEST_IDS_STR"
read -r -a SET_NAMES <<< "$SET_NAMES_STR"

UNDERSTANDING_ROOT="${OUTPUT_ROOT}/${MODEL}/visual_workflow/video_understanding"
CUT_PLANNING_ROOT="${OUTPUT_ROOT}/${MODEL}/visual_workflow/cut_planning/${CUT_PLANNING_PROMPT_VERSION}"
REPORT_DIR="${OUTPUT_ROOT}/${MODEL}/validate/visual_workflow/${CUT_PLANNING_PROMPT_VERSION}"

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

run_cmd python benchmark_evaluation/validate_visual_workflow.py \
    --understanding-root "$UNDERSTANDING_ROOT" \
    --cut-planning-root "$CUT_PLANNING_ROOT" \
    --gt-paths "${GT_PATHS[@]}" \
    --set-names "${SET_NAMES[@]}" \
    --test-ids "${TEST_IDS[@]}" \
    --output-dir "$REPORT_DIR" \
    --iou-threshold "$IOU_THRESHOLD"

echo "Saved visual_workflow validation reports to ${REPORT_DIR}"
