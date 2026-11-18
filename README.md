## RoughCut-Bench: A Benchmark for Automatic Video Rough-Cut with Large Vision-Language Models｜KDD 2027

<p align="center">
  <a href="https://whalesong-zrs.github.io/RoughCut-Bench-Project-Page/"><img src="https://img.shields.io/badge/Project-Website-orange?style=flat-square" alt="Project Website"></a>
</p>

### 🌟 Overview

RoughCut-Bench evaluates automatic video rough-cut in two settings:

- **Visually driven:** highlight detection and selection, source-video ordering,
  and adherence to user requirements.
- **Speech-driven:** filler removal, expression deduplication, and low-value
  content trimming in ASR transcripts.

This repository provides the agentic workflows, annotation-based metrics,
rendering tools, and human-aligned visual evaluator. Browse qualitative
examples and results on the [project website](https://whalesong-zrs.github.io/RoughCut-Bench-Project-Page/).

### 📊 Dataset

| Component | Size |
| --- | --- |
| Visual cases | 101: 62 general cases + 39 with user requirements |
| Source videos | 831 |
| Professional highlight annotations | 1,476 |
| Speech-driven cases | 84 |

The full benchmark data are distributed separately from this code repository.
Place the data under `data/`, or set `ROUGHCUT_DATA_ROOT` to another location:

```text
data/
|-- benchmark_set1/<case_id>/<source_video>
|-- benchmark_set2/<case_id>/<source_video>
|-- set1_label.json
|-- set2_label.json
`-- speech/<case_id>/
    |-- gt.txt
    `-- input.txt  # Optional unannotated transcript
```

Speech `gt.txt` contains IM, RM→R, and D annotations; the runner removes their
markers before inference. The optional `input.txt` is not a cleaned reference.
See the [data and example details](docs/usage.md#data-and-examples).

### 💻 Installation

Requirements: Linux x86_64, Python 3.10–3.12, and FFmpeg with `libx264` and AAC.

```bash
git clone https://github.com/Whalesong-zrs/RoughCut-Bench.git
cd RoughCut-Bench
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv
./env_setting.sh
source .venv/bin/activate
```

Run all commands below from the repository root.

### 🔑 Configuration

Create a private configuration file:

```bash
mkdir -p ~/.config/roughcut
cp .env.example ~/.config/roughcut/env.sh
chmod 600 ~/.config/roughcut/env.sh
```

Edit that file to set the API keys for the models you use:
`GEMINI_API_KEY`, `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, `ZAI_API_KEY`, or
`ARK_API_KEY`. Set `ROUGHCUT_DATA_ROOT` and `ROUGHCUT_OUTPUT_ROOT` there if
needed; their defaults are repository-local `data/` and `outputs/`.

```bash
source ~/.config/roughcut/env.sh
source scripts/common.sh
```

See [model APIs](docs/model_apis.md) for provider model IDs and overrides.
Model availability depends on your account; the `gemini-3-pro` alias currently
maps to `gemini-3.1-pro-preview`. Workflows do not set explicit output-token
limits; provider defaults still apply. Generation and agentic evaluation
make billable API requests.

### ▶️ Running and Evaluation

#### Visual rough-cut

The two-stage workflow understands source videos, then selects and orders
segments. Choose a launcher under [scripts/visual_workflow](scripts/visual_workflow/):

```bash
# Preview commands without API requests.
DRY_RUN=1 bash scripts/visual_workflow/run_gemini_2_5_pro.sh
# Generate cut plans and run annotation-based evaluation.
bash scripts/visual_workflow/run_gemini_2_5_pro.sh
```

The launcher processes both visual sets with three repeats by default.
To evaluate existing cut plans independently, for example Set-1 / test1:

```bash
python benchmark_evaluation/evaluate_cut_plans.py \
  --model gemini-2.5-pro \
  --cut-plan-dirs "${ROUGHCUT_OUTPUT_ROOT}/vlm_comparison/gemini-2.5-pro/visual_workflow/cut_planning/v33/test1/set1/cut_plans" \
  --gt-paths "${ROUGHCUT_SET1_LABELS}" \
  --set-names set1
```

Metrics are precision, recall, F1 (temporal IoU threshold 0.5), average IoU,
and Kendall's tau. All annotated cases and source videos are required by
default. See [cut-plan format and validation](docs/cut_plan_format.md) and
[rendering and human-aligned evaluation](docs/usage.md#rendered-cut-evaluation).

#### Speech-driven rough-cut

Generate predictions with zero-shot and CoT zero-shot prompts:

```bash
python speech_workflow/run_prompt_all_cases.py \
  --dir "${ROUGHCUT_SPEECH_DATA_ROOT}" \
  --out-dir "${ROUGHCUT_SPEECH_OUTPUT_ROOT}/cases" \
  --models gemini-2.5-pro \
  --workers 4 \
  --skip-existing
```

Evaluate predictions using the same model identifiers:

```bash
python speech_workflow/evaluate_prompt_scores.py \
  --dir "${ROUGHCUT_SPEECH_DATA_ROOT}" \
  --pred-dir "${ROUGHCUT_SPEECH_OUTPUT_ROOT}/cases" \
  --out-dir "${ROUGHCUT_SPEECH_OUTPUT_ROOT}/metrics" \
  --models gemini-2.5-pro
```

IM and D report precision, recall, and F1; RM→R checks removal and retention
together. Use `--dry-run` on generation to inspect tasks, or `--max-cases N`
on both commands for a matching subset. Details are in the
[speech workflow guide](docs/usage.md#speech-driven-rough-cut).

### 📚 Documentation

- [Detailed usage guide](docs/usage.md): configuration, output layouts, rendering,
  human-aligned evaluation, examples, ablations, and troubleshooting.
- [Model API configuration](docs/model_apis.md): credentials and provider mappings.
- [Cut-plan submission format](docs/cut_plan_format.md): schema, conversion, and validation.
- [Speech code overview](speech_workflow/README.md): transcript-editing and metric scripts.

### 📄 License

The code is distributed under the [Apache License 2.0](LICENSE).
