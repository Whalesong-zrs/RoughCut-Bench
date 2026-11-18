# Speech-Driven Rough-Cut

This directory contains the Transcript-Editing Agent workflow and its offline
annotation-based evaluation used by RoughCut-Bench.

## Workflow

- `asr_prompts.py` defines zero-shot and CoT zero-shot prompt variants.
- `run_prompt_case.py` runs all prompt variants for one transcript case.
- `run_prompt_all_cases.py` runs the same workflow over a directory of cases
  and multiple models.
- `transcript_editing_agent.py` and `llm_runtime.py` provide shared text and model
  runtime utilities.

## Evaluation

- `evaluate_f1.py` implements the annotation-based IM, RM→R, and D metrics.
- `batch_evaluate.py` evaluates generated transcripts in batch.
- `evaluate_prompt_scores.py` aggregates precision, recall, and F1 by model
  and prompt variant.
- `gen_gt.py` derives a cleaned reference transcript from an annotated one.

## Case Layout

The generation scripts currently expect one directory per case:

```text
case_name/
  gt.txt
  input.txt  # optional unannotated transcript
```

`gt.txt` contains the annotated transcript. The runner removes its annotation
markers to construct the model input. When present, `input.txt` contains the
same transcript without annotation tags; it is not a cleaned reference and is
not required by the current workflow. Benchmark cases and generated model
outputs are distributed separately and are not included in this code folder.
Generated files are written to
`$ROUGHCUT_SPEECH_OUTPUT_ROOT/cases/<case>/<prompt-variant>/` by default; the
source case directories remain read-only.
