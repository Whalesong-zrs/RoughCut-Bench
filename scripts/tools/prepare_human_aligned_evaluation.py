#!/usr/bin/env python3
"""Prepare rendering and evaluator commands for human-aligned evaluation.

Manifest columns:
  model,set_name,cut_plan_dir[,media_root,label_path]

The script computes common case ids per set across selected runs, writes
common_cases/*.txt, and generates shell scripts for rendering cut-plan outputs
and running the human-aligned evaluator.
"""

import argparse
import csv
import json
import os
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ROUGHCUT_DATA_ROOT") or PROJECT_ROOT / "data")
OUTPUT_ROOT = Path(os.environ.get("ROUGHCUT_OUTPUT_ROOT") or PROJECT_ROOT / "outputs")
if not os.environ.get("ROUGHCUT_DATA_ROOT"):
    os.environ["ROUGHCUT_DATA_ROOT"] = str(DATA_ROOT)
if not os.environ.get("ROUGHCUT_OUTPUT_ROOT"):
    os.environ["ROUGHCUT_OUTPUT_ROOT"] = str(OUTPUT_ROOT)

DEFAULT_SET_CONFIG = {
    "set1": {
        "media_root": os.environ.get("ROUGHCUT_SET1_MEDIA") or str(DATA_ROOT / "benchmark_set1"),
        "label_path": os.environ.get("ROUGHCUT_SET1_LABELS") or str(DATA_ROOT / "set1_label.json"),
    },
    "set2": {
        "media_root": os.environ.get("ROUGHCUT_SET2_MEDIA") or str(DATA_ROOT / "benchmark_set2"),
        "label_path": os.environ.get("ROUGHCUT_SET2_LABELS") or str(DATA_ROOT / "set2_label.json"),
    },
}


def q(value) -> str:
    return shlex.quote(str(value))


def natural_key(text: str):
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def read_manifest(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for row in reader:
            if not row or not any((v or "").strip() for v in row.values()):
                continue
            rows.append(
                {
                    k: os.path.expanduser(os.path.expandvars((v or "").strip()))
                    for k, v in row.items()
                }
            )
    required = {"model", "set_name", "cut_plan_dir"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise KeyError(f"manifest missing required columns: {sorted(missing)}")
    return rows


def fill_defaults(row: dict, args: argparse.Namespace) -> dict:
    set_name = row["set_name"]
    if set_name not in DEFAULT_SET_CONFIG:
        raise ValueError(f"unknown set_name: {set_name}")
    defaults = DEFAULT_SET_CONFIG[set_name]
    row = dict(row)
    row["media_root"] = row.get("media_root") or getattr(args, f"{set_name}_media") or defaults["media_root"]
    row["label_path"] = row.get("label_path") or getattr(args, f"{set_name}_label") or defaults["label_path"]
    row["model_slug"] = row.get("model_slug") or row["model"].replace("/", "_").replace(" ", "_")
    return row


def cut_plan_cases(cut_plan_dir: Path) -> List[str]:
    if not cut_plan_dir.is_dir():
        raise FileNotFoundError(f"cut_plan_dir not found: {cut_plan_dir}")
    return sorted((p.stem for p in cut_plan_dir.iterdir() if p.suffix.lower() == ".json"), key=natural_key)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip("\n") + "\n")


def shell_header() -> List[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]


def render_command(project_root: Path, row: dict, out_dir: Path, case_ids: List[str], fps: int, workers: int, max_retries: int, retry_sleep_sec: float) -> str:
    parts = [
        "python", "scripts/tools/render_cut_plan.py",
        "--cut-plan-path", row["cut_plan_dir"],
        "--out-dir", str(out_dir),
        "--media-root", row["media_root"],
        "--fps", str(fps),
        "--num-workers", str(workers),
        "--max-retries", str(max_retries),
        "--retry-sleep-sec", str(retry_sleep_sec),
        "--continue-on-error",
        "--skip-existing",
        "--skip-invalid-segments",
    ]
    if case_ids:
        parts.append("--case-ids")
        parts.extend(case_ids)
    return "cd " + q(project_root) + " && " + " ".join(q(part) for part in parts)


def evaluator_command(project_root: Path, row: dict, rendered_dir: Path, out_dir: Path, args: argparse.Namespace, case_ids: List[str]) -> str:
    parts = [
        "python", "benchmark_evaluation/human_aligned_evaluator.py",
        "--result-dir", str(rendered_dir),
        "--source-root", row["media_root"],
        "--labels-path", row["label_path"],
        "--output-dir", str(out_dir),
        "--llm-model", args.llm_model,
        "--evidence-sample-fps", str(args.evidence_sample_fps),
        "--llm-max-edited-frames", str(args.llm_max_edited_frames),
        "--llm-max-source-reference-frames", str(args.llm_max_source_reference_frames),
        "--num-workers", str(args.num_workers),
    ]
    if case_ids:
        parts.extend(["--case-ids", ",".join(case_ids)])
    return "cd " + q(project_root) + " && " + " ".join(q(part) for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV/TSV with model,set_name,cut_plan_dir[,media_root,label_path]")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--result-root", default=str(OUTPUT_ROOT / "human_aligned_evaluation"))
    parser.add_argument("--set1-media", default="")
    parser.add_argument("--set2-media", default="")
    parser.add_argument("--set1-label", default="")
    parser.add_argument("--set2-label", default="")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-workers", type=int, default=16, help="parallel case render workers passed to render_cut_plan.py")
    parser.add_argument("--render-max-retries", type=int, default=2, help="retry count per failed render case")
    parser.add_argument("--render-retry-sleep-sec", type=float, default=1.0, help="sleep seconds between render retries")
    parser.add_argument("--llm-model", default="gemini-3-pro")
    parser.add_argument("--evidence-sample-fps", type=float, default=3.0)
    parser.add_argument("--llm-max-edited-frames", type=int, default=30)
    parser.add_argument("--llm-max-source-reference-frames", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--eval-repeat-count", type=int, default=1,
                        help="number of independent evaluator runs per model and set")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root)
    result_root = Path(args.result_root)
    rows = [fill_defaults(row, args) for row in read_manifest(Path(args.manifest))]

    cases_by_run: Dict[int, List[str]] = {}
    rows_by_set: Dict[str, List[dict]] = defaultdict(list)
    for idx, row in enumerate(rows):
        cases = cut_plan_cases(Path(row["cut_plan_dir"]))
        cases_by_run[idx] = cases
        row["case_count"] = len(cases)
        rows_by_set[row["set_name"]].append(row | {"_idx": idx})

    common_by_set: Dict[str, List[str]] = {}
    for set_name, set_rows in rows_by_set.items():
        common: Optional[set] = None
        for row in set_rows:
            case_set = set(cases_by_run[row["_idx"]])
            common = case_set if common is None else common & case_set
        common_by_set[set_name] = sorted(common or [], key=natural_key)

    common_dir = result_root / "common_cases"
    for set_name, case_ids in common_by_set.items():
        write_lines(common_dir / f"{set_name}.txt", case_ids)

    metadata = {
        "manifest": str(Path(args.manifest).resolve()),
        "project_root": str(project_root),
        "result_root": str(result_root),
        "rows": rows,
        "common_cases": {k: {"count": len(v), "case_ids": v} for k, v in common_by_set.items()},
        "eval_repeat_count": max(1, int(args.eval_repeat_count)),
        "render_workers": max(1, int(args.render_workers)),
        "render_max_retries": max(0, int(args.render_max_retries)),
        "render_retry_sleep_sec": float(args.render_retry_sleep_sec),
    }
    result_root.mkdir(parents=True, exist_ok=True)
    with open(result_root / "selected_runs.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    render_lines = shell_header()
    eval_lines = shell_header()

    for row in rows:
        set_name = row["set_name"]
        case_ids = common_by_set[set_name]
        rendered_dir = result_root / "rendered" / set_name / row["model_slug"]
        evaluator_dir = result_root / "evaluator" / set_name / row["model_slug"]
        render_lines.append(f"echo '[render] {set_name} {row['model']}'")
        render_lines.append(render_command(
            project_root,
            row,
            rendered_dir,
            case_ids,
            args.fps,
            max(1, int(args.render_workers)),
            max(0, int(args.render_max_retries)),
            float(args.render_retry_sleep_sec),
        ))
        render_lines.append("")
        repeat_count = max(1, int(args.eval_repeat_count))
        for repeat_idx in range(1, repeat_count + 1):
            repeat_evaluator_dir = evaluator_dir if repeat_count == 1 else evaluator_dir / f"repeat{repeat_idx}"
            repeat_label = "" if repeat_count == 1 else f" repeat{repeat_idx}/{repeat_count}"
            eval_lines.append(f"echo '[eval] {set_name} {row['model']}{repeat_label}'")
            eval_lines.append(evaluator_command(
                project_root,
                row,
                rendered_dir,
                repeat_evaluator_dir,
                args,
                case_ids,
            ))
            eval_lines.append("")

    script_dir = result_root / "scripts"
    write_lines(script_dir / "run_render.sh", render_lines)
    write_lines(script_dir / "run_human_aligned_evaluator.sh", eval_lines)

    for script in script_dir.glob("*.sh"):
        script.chmod(0o755)

    print(json.dumps({
        "result_root": str(result_root),
        "common_cases": {k: len(v) for k, v in common_by_set.items()},
        "scripts": sorted(str(p) for p in script_dir.glob("*.sh")),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
