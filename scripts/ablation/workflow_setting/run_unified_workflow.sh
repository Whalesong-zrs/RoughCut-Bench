#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/common.sh

OUTPUT_ROOT="${ROUGHCUT_OUTPUT_ROOT}/vlm_comparison_ablation/setting"

TEST_IDS=("test1" "test2" "test3")
SET_NAMES=("set1" "set2")
UNIFIED_WORKFLOW_PROMPT_VERSIONS=("v18")
WORKERS=3

DATASETS=(
    "${ROUGHCUT_SET1_MEDIA}"
    "${ROUGHCUT_SET1_LABELS}"

    "${ROUGHCUT_SET2_MEDIA}"
    "${ROUGHCUT_SET2_LABELS}"
)

MODELS=(
    "gemini-3-pro"
)

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

VIDEO_DIRS=()
GT_PATHS=()
if (( ${#DATASETS[@]} % 2 != 0 )); then
    echo "[ERROR] DATASETS must contain video/gt pairs." >&2
    exit 1
fi
for ((i=0; i<${#DATASETS[@]}; i+=2)); do
    VIDEO_DIRS+=("${DATASETS[i]}")
    GT_PATHS+=("${DATASETS[i+1]}")
done
if (( ${#SET_NAMES[@]} != ${#VIDEO_DIRS[@]} )); then
    echo "[ERROR] SET_NAMES and DATASETS pair count must match." >&2
    exit 1
fi

for model in "${MODELS[@]}"; do
    for version in "${UNIFIED_WORKFLOW_PROMPT_VERSIONS[@]}"; do
        version_root="${OUTPUT_ROOT}/${model}/unified_workflow/${version}"

        for test_id in "${TEST_IDS[@]}"; do
            out_dirs=()
            for ((di=0; di<${#VIDEO_DIRS[@]}; di++)); do
                set_name="${SET_NAMES[di]}"
                out_dirs+=("${version_root}/${test_id}/${set_name}/cut_plans")
            done

            echo "================================================================"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] unified_workflow model=${model} version=${version} test=${test_id}"
            echo "================================================================"

            run_cmd python visual_workflow/unified_workflow.py \
                --model "$model" \
                --unified-workflow-prompt-version "$version" \
                --video-dirs "${VIDEO_DIRS[@]}" \
                --out-dirs "${out_dirs[@]}" \
                --gt-paths "${GT_PATHS[@]}" \
                --workers "$WORKERS" \
                --repair-invalid \
                --max-repair-rounds 5 \
                --require-explicit-understanding

            run_cmd python benchmark_evaluation/evaluate_unified_workflow.py \
                --model "$model" \
                --unified-dirs "${out_dirs[@]}" \
                --gt-paths "${GT_PATHS[@]}" \
                --set-names "${SET_NAMES[@]}"
        done

        run_cmd python benchmark_evaluation/summarize_repeat_metrics.py \
            --model "$model" \
            --chain unified_workflow \
            --version "$version" \
            --version-root "$version_root" \
            --test-ids "${TEST_IDS[@]}" \
            --set-names "${SET_NAMES[@]}"
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Unified workflow repeat eval done."
