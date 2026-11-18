#!/usr/bin/env python3
"""Evaluate public or legacy visual cut plans against benchmark annotations."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_evaluation.cut_plan_schema import load_cut_plan
from benchmark_evaluation.evaluation_inputs import align_evaluation_inputs
from benchmark_evaluation.visual_metrics import calculate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RoughCut-Bench visual cut plans using annotation-based metrics."
    )
    parser.add_argument(
        "--model",
        default="system",
        help="system name used only in the output CSV filename",
    )
    parser.add_argument(
        "--cut-plan-dirs",
        nargs="+",
        required=True,
        help="cut-plan JSON directories to evaluate, one per set",
    )
    parser.add_argument(
        "--gt-paths",
        nargs="+",
        required=True,
        help="annotation JSON paths, one per set",
    )
    parser.add_argument(
        "--video-understanding-model",
        default=None,
        help="optional Video-Understanding Agent name used only for ablation CSV naming",
    )
    parser.add_argument(
        "--set-names",
        nargs="+",
        default=None,
        help="optional set names, one per cut-plan directory",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "evaluate only overlapping cases and sources for local debugging; "
            "benchmark submissions are strict by default"
        ),
    )
    args = parser.parse_args()
    if len(args.cut_plan_dirs) != len(args.gt_paths):
        parser.error("--cut-plan-dirs and --gt-paths must contain the same number of values")
    if args.set_names is not None and len(args.set_names) != len(args.cut_plan_dirs):
        parser.error("--set-names and --cut-plan-dirs must contain the same number of values")
    return args


def build_prediction(cut_plan_dir: str | Path) -> dict:
    cut_plan_dir = Path(cut_plan_dir)
    if not cut_plan_dir.is_dir():
        raise FileNotFoundError(f"cut-plan directory not found: {cut_plan_dir}")

    prediction = {}
    json_files = sorted(
        path for path in cut_plan_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )
    if not json_files:
        raise ValueError(f"no cut-plan JSON files found under {cut_plan_dir}")

    for path in json_files:
        cut_plan = load_cut_plan(path)
        case_id = cut_plan["case_id"]
        if case_id in prediction:
            raise ValueError(f"duplicate cut plan for case_id {case_id!r}: {path}")
        case_prediction = {}
        for video in cut_plan["ordered_videos"]:
            case_prediction[video["source_id"]] = {
                "highlights": [
                    [segment["start_time"], segment["end_time"]]
                    for segment in video["segments"]
                ]
            }
        prediction[case_id] = case_prediction
    return prediction


def write_metrics_csv(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **metrics}
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def evaluate(
    gt_path: str,
    cut_plan_dir: str,
    model_name: str,
    set_name: str | None = None,
    video_understanding_model: str | None = None,
    allow_partial: bool = False,
) -> dict:
    print(f"\n=== Evaluating: {gt_path} ===")
    with open(gt_path, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    prediction = build_prediction(cut_plan_dir)
    filtered_gt, filtered_prediction = align_evaluation_inputs(
        ground_truth,
        prediction,
        allow_partial=allow_partial,
    )
    if not filtered_gt:
        raise ValueError("no common cases between annotations and predictions")

    metrics = calculate_metrics(filtered_gt, filtered_prediction, iou_threshold=0.5)
    metrics.pop("kendalltau_legal", None)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if set_name is None:
        set_name = "set2" if "batch2" in str(gt_path) else "set1"
    if video_understanding_model:
        name_tag = f"vu-{video_understanding_model}_cp-{model_name}"
    else:
        name_tag = model_name
    output_path = Path(cut_plan_dir).parent / f"eval_cut_plans_{set_name}_{name_tag}.csv"
    write_metrics_csv(output_path, metrics)
    print(f"Saved to {output_path}")
    return metrics


def main() -> int:
    args = parse_args()
    print(f"=== Evaluating cut plans for system: {args.model} ===")
    for index, (cut_plan_dir, gt_path) in enumerate(zip(args.cut_plan_dirs, args.gt_paths)):
        evaluate(
            gt_path=gt_path,
            cut_plan_dir=cut_plan_dir,
            model_name=args.model,
            set_name=args.set_names[index] if args.set_names is not None else None,
            video_understanding_model=args.video_understanding_model,
            allow_partial=args.allow_partial,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
