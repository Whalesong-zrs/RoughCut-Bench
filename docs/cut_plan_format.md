# Visual Cut-Plan Submission Format

RoughCut-Bench evaluates visual systems through a workflow-independent cut-plan
format. A submission contains one JSON file per benchmark case. The filename
must be `<case_id>.json`.

```json
{
  "schema_version": "roughcut-bench-cut-plan-v1",
  "case_id": "case_001",
  "ordered_videos": [
    {
      "source_id": "source_video_b",
      "segments": [
        {"start_time": 1.2, "end_time": 4.6},
        {"start_time": 7.0, "end_time": 9.5}
      ]
    },
    {
      "source_id": "source_video_a",
      "segments": [
        {"start_time": 0.0, "end_time": 3.1}
      ]
    }
  ],
  "metadata": {
    "system_name": "example-system"
  }
}
```

## Field Definitions

- `schema_version` must be `roughcut-bench-cut-plan-v1`.
- `case_id` must match both the benchmark case identifier and the JSON filename.
- `ordered_videos` records the source videos in their intended final-cut order.
- `source_id` must match a source-video key in the corresponding annotation
  entry. It is normally the source filename without its extension.
- `segments` contains the retained intervals from that source video. Times are
  expressed in seconds relative to the beginning of the source video.
- `metadata` is optional and is ignored by annotation-based metrics.

For full benchmark evaluation, every source video listed by the corresponding
case annotation must appear exactly once; this is required for source-video
ordering. The JSON Schema enforces uniqueness within a file, while the evaluator
checks coverage against the annotations. Segments within one source video must
be non-empty, chronological, non-overlapping, and satisfy
`0 <= start_time < end_time`. Intermediate descriptions, model reasoning,
prompts, and workflow-specific identifiers are not part of the submission.

The machine-readable JSON Schema is available at
`schemas/visual_cut_plan.schema.json`.

## Validation

Validate one file or a directory before evaluation:

```bash
python benchmark_evaluation/validate_cut_plan_submission.py \
  --cut-plan-path outputs/my_system/set1
```

## Converting the Released Workflow Outputs

The released agentic workflows retain additional intermediate information for
auditing. Convert those files to the public format with:

```bash
python benchmark_evaluation/convert_workflow_outputs.py \
  --input outputs/my_workflow/set1/cut_plans \
  --output outputs/my_submission/set1
```

The annotation-based evaluator and rendering utility accept both the public
format and the legacy workflow output. Third-party systems should use the
public format directly.
