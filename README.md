# RoughCut-Bench: A Benchmark for Automatic Video Rough-Cut with Large Vision-Language Models｜KDD 2027

RoughCut-Bench is a benchmark for automatic video rough-cut. It covers two
practical settings:

- **Visually driven rough-cut:** detect and select highlight segments from
  source videos, determine source-video order, and follow optional user
  requirements.
- **Speech-driven rough-cut:** refine complete ASR transcripts through filler
  removal, expression deduplication, and low-value content trimming.

This repository provides the agentic workflows, annotation-based metrics,
rendering utilities, human-aligned visual evaluator, and experiment launchers
used by RoughCut-Bench. Benchmark data and generated outputs are distributed
separately.

Project page: <https://whalesong-zrs.github.io/RoughCut-Bench-Project-Page/>

## Contents

- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Visual cut-plan format](#visual-cut-plan-format)
- [Runtime configuration](#runtime-configuration)
- [Visual rough-cut](#visual-rough-cut)
- [Speech-driven rough-cut](#speech-driven-rough-cut)
- [Rendered-cut evaluation](#rendered-cut-evaluation)
- [Ablations and validation](#ablations-and-validation)
- [Tests](#tests)
- [Reproducibility notes](#reproducibility-notes)

## Repository Structure

```text
RoughCut-Bench/
|-- visual_workflow/            # Visual agents, model clients, and utilities
|   |-- model_api.py            # Central adapters for official public APIs
|   |-- video_understanding_agent.py
|   |-- cut_planning_agent.py
|   `-- prompt/                 # Prompts for the visual agents and ablations
|-- benchmark_evaluation/       # Annotation-based and human-aligned evaluation
|   `-- human_aligned_evaluator.py
|-- schemas/                    # Machine-readable submission schemas
|-- docs/                       # Format and evaluation documentation
|-- speech_workflow/            # Transcript-Editing Agent and speech metrics
|-- examples/                   # Visual and speech example cases
|-- scripts/
|   |-- visual_workflow/        # Main two-agent visual workflow launchers
|   |-- ablation/               # Visual ablation launchers
|   |-- tools/                  # Rendering and evaluation preparation
|   `-- validate/               # Repeated-run validation
|-- human_aligned_evaluation_runs.tsv
|-- prepare_human_aligned_evaluation.sh
|-- requirements.txt
|-- env_setting.sh
`-- .env.example
```

## Installation

### Requirements

- Linux x86_64
- Python 3.10, 3.11, or 3.12
- FFmpeg with `libx264` and AAC encoders
- Python `venv`

On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv
```

Create the Python environment:

```bash
./env_setting.sh
source .venv/bin/activate
```

The setup script installs the pinned dependencies, runs `pip check`, and
verifies the core video, image, model-client, and metric imports. To use a
different interpreter or environment path:

```bash
PYTHON_BIN=python3.11 VENV_DIR=/path/to/venv ./env_setting.sh
```

## Data Preparation

The default directory layout is:

```text
data/
|-- benchmark_set1/
|   `-- <case_id>/
|       `-- <source_video>
|-- benchmark_set2/
|   `-- <case_id>/
|       `-- <source_video>
|-- set1_label.json
|-- set2_label.json
`-- speech/
    |-- <case_id>/
    |   |-- gt.txt
    |   `-- input.txt           # Optional unannotated ASR text
    `-- ...
```

Each visual case contains the source videos from one coherent creation
scenario. The label JSON files contain professional highlight annotations,
source-video order, and user requirements when available.

For speech cases, `gt.txt` contains the transcript and its IM, RM->R, and D
annotations. The current workflow removes annotation markers from `gt.txt`
before sending the transcript to a model. `input.txt`, when distributed, is the
same ASR transcript without annotation markers; it is not a cleaned reference
and is not required by the current runner.

Data and generated outputs use separate roots. The defaults are `data/` and
`outputs/` inside this repository.

## Visual Cut-Plan Format

RoughCut-Bench uses a workflow-independent JSON format for visual submissions.
Each case file contains the case identifier, source videos in final-cut order,
and the selected timestamp intervals:

```json
{
  "schema_version": "roughcut-bench-cut-plan-v1",
  "case_id": "case_001",
  "ordered_videos": [
    {
      "source_id": "source_video_b",
      "segments": [
        {"start_time": 1.2, "end_time": 4.6}
      ]
    }
  ]
}
```

`source_id` is normally the source filename without its extension and must
match the corresponding annotation key. The order of `ordered_videos` defines
the source-video order in the rendered cut. Model reasoning and workflow
intermediate fields are not required.

Validate a submission directory with:

```bash
python benchmark_evaluation/validate_cut_plan_submission.py \
  --cut-plan-path outputs/my_submission/set1
```

The released agentic workflows retain additional intermediate fields for
auditing. Convert them to the public format with:

```bash
python benchmark_evaluation/convert_workflow_outputs.py \
  --input outputs/my_workflow/set1/cut_plans \
  --output outputs/my_submission/set1
```

The annotation-based evaluator and rendering utility accept both formats.
See `docs/cut_plan_format.md` and `schemas/visual_cut_plan.schema.json` for the
complete specification.

Full benchmark evaluation requires every annotated case and source video to be
present exactly once. Missing cases and unknown source identifiers are rejected
instead of being silently excluded. Use `--allow-partial` only for local
debugging on an intentional subset.

## Runtime Configuration

### Paths

Set custom roots when the benchmark data or outputs live elsewhere:

```bash
export ROUGHCUT_DATA_ROOT=/path/to/roughcut-data
export ROUGHCUT_OUTPUT_ROOT=/path/to/roughcut-outputs
```

The following optional variables override individual locations:

```text
ROUGHCUT_SET1_MEDIA
ROUGHCUT_SET1_LABELS
ROUGHCUT_SET2_MEDIA
ROUGHCUT_SET2_LABELS
ROUGHCUT_SPEECH_DATA_ROOT
ROUGHCUT_SPEECH_OUTPUT_ROOT
```

Load the resolved defaults into the current shell before using commands that
reference these variables directly:

```bash
source scripts/common.sh
```

### Model APIs

Provider credentials are read from environment variables. Never place
credentials in source files or commit them to Git.

Create a private environment file:

```bash
mkdir -p ~/.config/roughcut
cp .env.example ~/.config/roughcut/env.sh
chmod 600 ~/.config/roughcut/env.sh
```

Fill the variables for the models you intend to run, then load the file:

```bash
source ~/.config/roughcut/env.sh
```

The code uses official public endpoints by default. Shared credentials are
`GEMINI_API_KEY`, `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, `ZAI_API_KEY`, and
`ARK_API_KEY`. Model-specific keys, model IDs, and provider base URLs can be
overridden without editing the agents. All request construction and response
parsing is centralized in `visual_workflow/model_api.py`; see
`docs/model_apis.md` for the provider mapping and official references.
That document also provides an explicit credentialed smoke-test command for
checking account access and multimodal requests before a full run.

The released workflows and smoke test do not explicitly set output-token
limits. Each provider's default and model limits still apply; this does not
mean unlimited output. Reasoning tokens may contribute to latency and cost
even when the requested answer is short.

## Visual Rough-Cut

### Workflow

The main visual workflow contains two stages:

1. The **Video-Understanding Agent** processes each source video independently
   using frames sampled at 1 FPS. It returns candidate highlight timestamps,
   reasons, descriptions, and representative key frames.
2. The **Cut-Planning Agent** combines these video-level results, selects the
   final highlight segments, and determines source-video order while following
   user requirements when provided.

The resulting cut plan is stored as one JSON file per visual case.

### Run the Main Workflow

Configure the API key for the selected model first. Inspect the
commands without sending model requests:

```bash
DRY_RUN=1 bash scripts/visual_workflow/run_gemini_2_5_pro.sh
```

Run the full workflow:

```bash
bash scripts/visual_workflow/run_gemini_2_5_pro.sh
```

Launchers are provided for the evaluated models:

```text
gemini-2.5-pro   gemini-3-pro
gpt-5.4          gpt-5.5
qwen-3.5-plus    qwen-3.7-plus
glm-5v-turbo     seed-2.0-lite
```

Each launcher runs both visual stages, computes annotation-based metrics, and
aggregates repeated runs. Model endpoints can be costly; inspect the launcher
settings and use `DRY_RUN=1` before starting a complete experiment.

### Visual Outputs

With the default configuration:

```text
outputs/vlm_comparison/<model>/visual_workflow/
|-- video_understanding/<test_id>/<set_name>/
`-- cut_planning/v33/<test_id>/<set_name>/cut_plans/
```

The `video_understanding` directories contain video-level candidate records.
The `cut_plans` directories contain final structured cut plans and metric
reports.

### Annotation-Based Metrics

The visual evaluation reports:

- precision, recall, and F1 for highlight detection and selection;
- average IoU for temporal overlap;
- Kendall's tau for source-video ordering.

Predicted and professional highlight segments are matched at temporal IoU 0.5
for precision, recall, and F1. The experiment launchers call the metric scripts
automatically.

## Speech-Driven Rough-Cut

### Workflow

The Transcript-Editing Agent processes each complete ASR transcript under two
settings:

- `ZERO_SHOT_PROMPT`: directly returns the cleaned transcript;
- `COT_ZERO_SHOT_PROMPT`: returns structured reasons and a `cleaned_text`
  field.

One-shot prompting is not part of the released workflow.

### Check the Run

Always pass the intended models explicitly. Verify case discovery and task
counts without sending requests:

```bash
python speech_workflow/run_prompt_all_cases.py \
  --dir "${ROUGHCUT_SPEECH_DATA_ROOT:-data/speech}" \
  --out-dir "${ROUGHCUT_SPEECH_OUTPUT_ROOT:-outputs/speech}/cases" \
  --models gemini-2.5-pro \
  --dry-run
```

For one model and 84 cases, the two prompt variants produce 168 tasks.

### Generate Predictions

```bash
python speech_workflow/run_prompt_all_cases.py \
  --dir "${ROUGHCUT_SPEECH_DATA_ROOT:-data/speech}" \
  --out-dir "${ROUGHCUT_SPEECH_OUTPUT_ROOT:-outputs/speech}/cases" \
  --models gemini-2.5-pro \
  --workers 4 \
  --skip-existing
```

Use `--max-cases N` for a small test run. Generated files are isolated from the
benchmark data:

```text
outputs/speech/cases/<case_id>/
|-- ZERO_SHOT_PROMPT/
|   |-- <model>.txt
|   |-- <model>_prompt.txt
|   `-- <model>_raw_response.txt
`-- COT_ZERO_SHOT_PROMPT/
    |-- <model>.json
    |-- <model>_cleaned_text.txt
    |-- <model>_prompt.txt
    `-- <model>_raw_response.txt
```

### Evaluate Predictions

Use the same `--models` and, when applicable, `--max-cases` values used during
generation:

```bash
python speech_workflow/evaluate_prompt_scores.py \
  --dir "${ROUGHCUT_SPEECH_DATA_ROOT:-data/speech}" \
  --pred-dir "${ROUGHCUT_SPEECH_OUTPUT_ROOT:-outputs/speech}/cases" \
  --out-dir "${ROUGHCUT_SPEECH_OUTPUT_ROOT:-outputs/speech}/metrics" \
  --models gemini-2.5-pro
```

Reports are written as CSV and Markdown. IM and D use precision, recall, and F1;
RM->R evaluates whether the removed expression is deleted while its paired
expression is retained. Categories with no applicable annotations are reported
as `N/A`. The command exits with an error when no generated prediction can be
evaluated.

See `speech_workflow/README.md` for the speech-specific file overview.

## Rendered-Cut Evaluation

### Render Cut Plans

Render a cut-plan directory into videos:

```bash
source scripts/common.sh

python scripts/tools/render_cut_plan.py \
  --cut-plan-path "${ROUGHCUT_OUTPUT_ROOT}/vlm_comparison/gemini-2.5-pro/visual_workflow/cut_planning/v33/test1/set1/cut_plans" \
  --media-root "${ROUGHCUT_SET1_MEDIA}" \
  --out-dir "${ROUGHCUT_OUTPUT_ROOT}/rendered/gemini-2.5-pro/set1" \
  --num-workers 4 \
  --skip-existing
```

Rendered videos are muted by default. Add `--keep-audio` to preserve source
audio.

### Human-Aligned Agentic Evaluation

The visual evaluator assesses a rendered cut using the original videos for
context and checks available user requirements. It reports highlight coverage,
redundancy, ordering, requirement following when applicable, and overall visual
quality. Highlight coverage is retained in the complete evaluator output but
is not reported separately in the main benchmark table.

The fixed evaluator prompt is versioned as `v1` and defined by
`build_llm_prompt()` in `benchmark_evaluation/human_aligned_evaluator.py`. Its system prompt contains the
evaluation criteria and scoring rules. Each request includes the rendered cut,
the original videos as contextual evidence, and available user requirements.
The evaluator returns integer scores and corresponding reasons in a structured
JSON object.
Before the prompt was fixed for benchmark evaluation, professional annotators
scored rendered LVLM cuts and provided corresponding reasons. We iteratively
refined the evaluator prompt and scoring criteria by comparing its scores and
reasons with this human feedback.

To prepare an evaluation batch:

1. Update `human_aligned_evaluation_runs.tsv` so each `cut_plan_dir` points to an
   existing visual workflow output.
2. Configure the evaluator model's provider credentials from `.env.example`.
3. Generate the rendering and evaluator commands:

```bash
./prepare_human_aligned_evaluation.sh
```

4. Run the generated scripts:

```bash
bash outputs/human_aligned_evaluation/scripts/run_render.sh
bash outputs/human_aligned_evaluation/scripts/run_human_aligned_evaluator.sh
```

The wrapper requests three independent evaluator runs per rendered cut.
Evaluator outputs are written under `outputs/human_aligned_evaluation/evaluator/`.

### Visual Example Cases

Two candidate example cases are included under `examples/visual_cases/`:

```text
examples/visual_cases/
|-- set1/11_resize/
`-- set2/batch2_1/
```

Each case contains the source videos, the professional reference cut and label,
rendered outputs from the evaluated LVLMs, their structured cut plans, and case
metadata. These cases can be used to inspect evaluator inputs and rendering
outputs without running the complete benchmark. They remain release candidates
and must pass the final privacy and usage-permission review before publication.

### Speech Example Cases

The six complete speech examples shown on the project page are included in
`examples/speech_cases/speech_cases.json`. They cover filler removal, expression
deduplication, and low-value content trimming, with two examples per category.
Each entry includes the annotated transcript, professional reference, model
cleanup records, cleaned transcript, and category-specific metrics. See
`examples/speech_cases/README.md` for the display schema and its distinction
from the executable speech benchmark layout.

## Ablations and Validation

Released visual ablations include:

```text
scripts/ablation/video_understanding_input/ # Frame sampling and video input
scripts/ablation/cut_planning_input/ # Cut-Planning Agent input components
scripts/ablation/workflow_setting/   # Joint-understanding and unified variants
```

All released visual ablation launchers use `gemini-3-pro`, matching the
experimental setup reported in the paper.

All ablation launchers support `DRY_RUN=1`. Structural and repeated-run checks
are available in `scripts/validate/`. Review model names, test IDs, and output
roots in each launcher before running it.

## Tests

Run the no-key test suite with:

```bash
python -m unittest discover -s tests -v
```

These tests validate the public cut-plan schema, strict evaluation coverage,
and request/response formats for all configured provider protocols using a
mock HTTP transport. They do not contact provider endpoints. A credentialed
small-case smoke test is still required to confirm account access, regional
base URLs, current model availability, and provider-side limits.

## Reproducibility Notes

- The main visual workflow uses `cut_planning_prompt_v33`.
- The retained workflow ablations use `unified_workflow_prompt_v18` and
  `joint_video_understanding_prompt_v2`.
- The default Video-Understanding Agent frame-sampling rate is 1 FPS.
- Main visual launchers use three repeated runs unless edited locally.
- Dataset paths, outputs, virtual environments, and private environment files
  are excluded by `.gitignore`.
- Prompt files and raw model responses are retained with generated outputs for
  auditing.

## Common Problems

**No visual cases are found.** Verify the media directories and label paths
resolved by `scripts/common.sh`.

**A model request is rejected.** Check that the provider key is loaded, the
configured model ID is available to your account and region, and any base-URL
override points to the matching provider API. See `docs/model_apis.md`.

**Speech evaluation reports missing outputs.** Pass the same model identifiers,
case limit, and prediction root to generation and evaluation.

**FFmpeg rendering fails.** Confirm that `ffmpeg -encoders` lists both
`libx264` and AAC, then rerun `./env_setting.sh`.
