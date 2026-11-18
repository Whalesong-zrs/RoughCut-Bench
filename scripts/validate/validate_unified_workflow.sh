#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/common.sh

OUTPUT_ROOT="${ROUGHCUT_OUTPUT_ROOT}/vlm_comparison_ablation/setting"
MODEL="gemini-2.5-pro"
UNIFIED_WORKFLOW_PROMPT_VERSIONS_STR="v18"
TEST_IDS_STR="test1 test2 test3"
SET_NAMES_STR="set1 set2"
IOU_THRESHOLD="0.5"

GT_PATHS=(
    "${ROUGHCUT_SET1_LABELS}"
    "${ROUGHCUT_SET2_LABELS}"
)

read -r -a UNIFIED_WORKFLOW_PROMPT_VERSIONS <<< "$UNIFIED_WORKFLOW_PROMPT_VERSIONS_STR"
read -r -a TEST_IDS <<< "$TEST_IDS_STR"
read -r -a SET_NAMES <<< "$SET_NAMES_STR"

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

for version in "${UNIFIED_WORKFLOW_PROMPT_VERSIONS[@]}"; do
    UNIFIED_ROOT="${OUTPUT_ROOT}/${MODEL}/unified_workflow/${version}"
    REPORT_DIR="${OUTPUT_ROOT}/${MODEL}/validate/unified_workflow/${version}"
    run_cmd python benchmark_evaluation/validate_unified_workflow.py \
        --unified-root "$UNIFIED_ROOT" \
        --gt-paths "${GT_PATHS[@]}" \
        --set-names "${SET_NAMES[@]}" \
        --test-ids "${TEST_IDS[@]}" \
        --output-dir "$REPORT_DIR" \
        --iou-threshold "$IOU_THRESHOLD"
    echo "Saved unified_workflow ${version} validation reports to ${REPORT_DIR}"
done
