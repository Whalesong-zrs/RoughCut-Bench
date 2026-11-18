# Speech Example Cases

`speech_cases.json` contains the six complete speech-driven examples displayed
on the RoughCut-Bench project page. The examples cover:

- filler removal: `beauty-im` and `makeup-primer-im`;
- expression deduplication: `qa-rm` and `london-rm`;
- low-value content trimming: `lifestyle-d` and `workweek-d`.

Each case contains the complete annotated transcript as ordered
`originalSegments`, the professional `referenceTranscript`, the model's
structured `modelEdits`, the resulting `modelTranscript`, and category-specific
metrics. Segment types use `im` for invalid markers, `rm` and `r` for paired
removed and retained expressions, and `d` for low-value deletion spans.

This JSON follows the project-page display schema. It is intended for inspecting
complete examples and is not the per-case input layout consumed directly by the
scripts in `speech_workflow/`. See `speech_workflow/README.md` for the executable benchmark
layout.
