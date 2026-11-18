#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

OUTPUT_ROOT="${ROUGHCUT_OUTPUT_ROOT}/vlm_comparison_ablation/setting"
MODEL="gemini-2.5-pro"
JOINT_VIDEO_UNDERSTANDING_PROMPT_VERSION="v2"
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

VERSION_TAG="u-${JOINT_VIDEO_UNDERSTANDING_PROMPT_VERSION}_a-${CUT_PLANNING_PROMPT_VERSION}"
UNDERSTANDING_ROOT="${OUTPUT_ROOT}/${MODEL}/joint_video_understanding_workflow/video_understanding/${JOINT_VIDEO_UNDERSTANDING_PROMPT_VERSION}"
CUT_PLANNING_ROOT="${OUTPUT_ROOT}/${MODEL}/joint_video_understanding_workflow/cut_planning/${VERSION_TAG}"
REPORT_DIR="${OUTPUT_ROOT}/${MODEL}/validate/joint_video_understanding_workflow/${VERSION_TAG}"

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

run_cmd python benchmark_evaluation/validate_joint_video_understanding_workflow.py \
    --understanding-root "$UNDERSTANDING_ROOT" \
    --cut-planning-root "$CUT_PLANNING_ROOT" \
    --gt-paths "${GT_PATHS[@]}" \
    --set-names "${SET_NAMES[@]}" \
    --test-ids "${TEST_IDS[@]}" \
    --output-dir "$REPORT_DIR" \
    --iou-threshold "$IOU_THRESHOLD"

echo "Saved joint_video_understanding_workflow validation reports to ${REPORT_DIR}"
