#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/common.sh

MODEL="gemini-3-pro"

# Cut-planning input ablation writes to a separate root so it never overwrites main comparison runs.
OUTPUT_ROOT="${ROUGHCUT_OUTPUT_ROOT}/vlm_comparison_ablation/cut_planning_input"

CUT_PLANNING_INPUT_MODES=("time_desc_frame" "time_desc" "time_frame")
TEST_IDS=("test1" "test2" "test3")
SET_NAMES=("set1" "set2")
CUT_PLANNING_PROMPT_VERSIONS=("v33")
VIDEO_UNDERSTANDING_WORKERS=3
CUT_PLANNING_WORKERS=3
REPAIR_INVALID=1
MAX_REPAIR_ROUNDS=5

DATASETS=(
    "${ROUGHCUT_SET1_MEDIA}"
    "${ROUGHCUT_SET1_LABELS}"

    "${ROUGHCUT_SET2_MEDIA}"
    "${ROUGHCUT_SET2_LABELS}"
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

REPAIR_ARGS=()
if [[ "${REPAIR_INVALID}" == "1" ]]; then
    REPAIR_ARGS=(--repair-invalid --max-repair-rounds "$MAX_REPAIR_ROUNDS")
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    for version in "${CUT_PLANNING_PROMPT_VERSIONS[@]}"; do
        if ! grep -q "^cut_planning_prompt_${version}[[:space:]]*=" visual_workflow/prompt/cut_planning_prompt.py; then
            echo "[ERROR] cut_planning_prompt_${version} is not defined. Add it before running cut-planning input ablation." >&2
            exit 1
        fi
    done
fi

for test_id in "${TEST_IDS[@]}"; do
    video_understanding_dirs=()
    for ((di=0; di<${#VIDEO_DIRS[@]}; di++)); do
        set_name="${SET_NAMES[di]}"
        video_understanding_dirs+=("${OUTPUT_ROOT}/${MODEL}/video_understanding/${test_id}/${set_name}")
    done

    echo "================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] cut_planning_input_ablation understanding model=${MODEL} test=${test_id}"
    echo "================================================================"

    run_cmd python visual_workflow/video_understanding_agent.py \
        --model "$MODEL" \
        --video-dirs "${VIDEO_DIRS[@]}" \
        --out-dirs "${video_understanding_dirs[@]}" \
        --gt-paths "${GT_PATHS[@]}" \
        --workers "$VIDEO_UNDERSTANDING_WORKERS" \
        "${REPAIR_ARGS[@]}"

    for input_mode in "${CUT_PLANNING_INPUT_MODES[@]}"; do
        for version in "${CUT_PLANNING_PROMPT_VERSIONS[@]}"; do
            version_root="${OUTPUT_ROOT}/${MODEL}/${input_mode}/visual_workflow/cut_planning/${version}"
            cut_plan_dirs=()
            for ((di=0; di<${#VIDEO_DIRS[@]}; di++)); do
                set_name="${SET_NAMES[di]}"
                cut_plan_dirs+=("${version_root}/${test_id}/${set_name}/cut_plans")
            done

            echo "================================================================"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cut-Planning Agent model=${MODEL} input_mode=${input_mode} version=${version} test=${test_id}"
            echo "================================================================"

            run_cmd python visual_workflow/cut_planning_agent.py \
                --model "$MODEL" \
                --cut-planning-prompt-version "$version" \
                --cut-planning-input-mode "$input_mode" \
                --video-understanding-dirs "${video_understanding_dirs[@]}" \
                --out-dirs "${cut_plan_dirs[@]}" \
                --gt-paths "${GT_PATHS[@]}" \
                --workers "$CUT_PLANNING_WORKERS" \
                "${REPAIR_ARGS[@]}"

            run_cmd python benchmark_evaluation/evaluate_cut_plans.py \
                --model "$MODEL" \
                --cut-plan-dirs "${cut_plan_dirs[@]}" \
                --gt-paths "${GT_PATHS[@]}" \
                --set-names "${SET_NAMES[@]}"
        done
    done
done

for input_mode in "${CUT_PLANNING_INPUT_MODES[@]}"; do
    for version in "${CUT_PLANNING_PROMPT_VERSIONS[@]}"; do
        version_root="${OUTPUT_ROOT}/${MODEL}/${input_mode}/visual_workflow/cut_planning/${version}"
        run_cmd python benchmark_evaluation/summarize_repeat_metrics.py \
            --model "$MODEL" \
            --chain visual_workflow \
            --version "$version" \
            --version-root "$version_root" \
            --test-ids "${TEST_IDS[@]}" \
            --set-names "${SET_NAMES[@]}"
    done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cut-planning input ablation done for model=${MODEL}."
