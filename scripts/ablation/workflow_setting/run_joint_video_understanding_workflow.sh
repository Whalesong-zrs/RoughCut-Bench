#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/common.sh

OUTPUT_ROOT="${ROUGHCUT_OUTPUT_ROOT}/vlm_comparison_ablation/setting"

TEST_IDS=("test1" "test2" "test3")
SET_NAMES=("set1" "set2")
JOINT_VIDEO_UNDERSTANDING_PROMPT_VERSION="v2"
CUT_PLANNING_PROMPT_VERSIONS=("v33")
JOINT_VIDEO_UNDERSTANDING_WORKERS=3
CUT_PLANNING_WORKERS=3
REPAIR_INVALID=1
MAX_REPAIR_ROUNDS=5

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

REPAIR_ARGS=()
if [[ "${REPAIR_INVALID}" == "1" ]]; then
    REPAIR_ARGS=(--repair-invalid --max-repair-rounds "$MAX_REPAIR_ROUNDS")
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    prompt_var="joint_video_understanding_prompt_${JOINT_VIDEO_UNDERSTANDING_PROMPT_VERSION}"
    if ! grep -q "^${prompt_var}[[:space:]]*=" visual_workflow/prompt/joint_video_understanding_prompt.py; then
        echo "[ERROR] ${prompt_var} is not defined." >&2
        exit 1
    fi
    for version in "${CUT_PLANNING_PROMPT_VERSIONS[@]}"; do
        if ! grep -q "^cut_planning_prompt_${version}[[:space:]]*=" visual_workflow/prompt/cut_planning_prompt.py; then
            echo "[ERROR] cut_planning_prompt_${version} is not defined." >&2
            exit 1
        fi
    done
fi

for model in "${MODELS[@]}"; do
    for test_id in "${TEST_IDS[@]}"; do
        video_understanding_dirs=()
        for ((di=0; di<${#VIDEO_DIRS[@]}; di++)); do
            set_name="${SET_NAMES[di]}"
            video_understanding_dirs+=("${OUTPUT_ROOT}/${model}/joint_video_understanding_workflow/video_understanding/${JOINT_VIDEO_UNDERSTANDING_PROMPT_VERSION}/${test_id}/${set_name}")
        done

        echo "================================================================"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Joint Video-Understanding Agent model=${model} version=${JOINT_VIDEO_UNDERSTANDING_PROMPT_VERSION} test=${test_id}"
        echo "================================================================"

        run_cmd python visual_workflow/joint_video_understanding_agent.py \
            --model "$model" \
            --joint-video-understanding-prompt-version "$JOINT_VIDEO_UNDERSTANDING_PROMPT_VERSION" \
            --video-dirs "${VIDEO_DIRS[@]}" \
            --out-dirs "${video_understanding_dirs[@]}" \
            --gt-paths "${GT_PATHS[@]}" \
            --workers "$JOINT_VIDEO_UNDERSTANDING_WORKERS" \
            "${REPAIR_ARGS[@]}"

        for cut_planning_version in "${CUT_PLANNING_PROMPT_VERSIONS[@]}"; do
            version_tag="u-${JOINT_VIDEO_UNDERSTANDING_PROMPT_VERSION}_a-${cut_planning_version}"
            version_root="${OUTPUT_ROOT}/${model}/joint_video_understanding_workflow/cut_planning/${version_tag}"
            cut_plan_dirs=()
            for ((di=0; di<${#VIDEO_DIRS[@]}; di++)); do
                set_name="${SET_NAMES[di]}"
                cut_plan_dirs+=("${version_root}/${test_id}/${set_name}/cut_plans")
            done

            echo "================================================================"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Joint-understanding Cut-Planning Agent model=${model} version=${version_tag} test=${test_id}"
            echo "================================================================"

            run_cmd python visual_workflow/cut_planning_agent.py \
                --model "$model" \
                --cut-planning-prompt-version "$cut_planning_version" \
                --video-understanding-dirs "${video_understanding_dirs[@]}" \
                --out-dirs "${cut_plan_dirs[@]}" \
                --gt-paths "${GT_PATHS[@]}" \
                --workers "$CUT_PLANNING_WORKERS" \
                "${REPAIR_ARGS[@]}"

            run_cmd python benchmark_evaluation/evaluate_cut_plans.py \
                --model "$model" \
                --cut-plan-dirs "${cut_plan_dirs[@]}" \
                --gt-paths "${GT_PATHS[@]}" \
                --set-names "${SET_NAMES[@]}"
        done
    done

done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Joint video-understanding workflow repeat eval done."
