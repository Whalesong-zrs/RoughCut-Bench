#!/usr/bin/env python3
"""Validate public visual cut-plan submission files."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_evaluation.cut_plan_schema import CutPlanValidationError, is_public_cut_plan, load_cut_plan


def cut_plan_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".json")
    raise FileNotFoundError(f"cut-plan path not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RoughCut-Bench visual cut plans.")
    parser.add_argument("--cut-plan-path", required=True, help="public cut-plan JSON file or directory")
    parser.add_argument(
        "--allow-workflow-output",
        action="store_true",
        help="also accept legacy agent-workflow output and validate its normalized cut plan",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.cut_plan_path)
    files = cut_plan_files(path)
    if not files:
        raise SystemExit(f"no JSON files found under {path}")

    failures = []
    for file_path in files:
        try:
            plan = load_cut_plan(file_path)
            if not args.allow_workflow_output:
                with open(file_path, "r", encoding="utf-8") as file:
                    raw = json.load(file)
                if not is_public_cut_plan(raw):
                    raise CutPlanValidationError(
                        "legacy workflow output is not a public submission; convert it first"
                    )
            segment_count = sum(len(video["segments"]) for video in plan["ordered_videos"])
            print(
                f"[OK] {file_path}: case={plan['case_id']} "
                f"videos={len(plan['ordered_videos'])} segments={segment_count}"
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            failures.append((file_path, exc))
            print(f"[FAIL] {file_path}: {type(exc).__name__}: {exc}")

    print(f"Validated {len(files)} file(s): passed={len(files) - len(failures)} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
