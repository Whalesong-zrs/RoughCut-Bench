#!/usr/bin/env python3
"""Convert agent-workflow JSON outputs to the public visual cut-plan schema."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_evaluation.cut_plan_schema import load_cut_plan, write_cut_plan


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"input file must use the .json extension: {path}")
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".json")
    raise FileNotFoundError(f"input path not found: {path}")


def output_path(case_id: str, output: Path, multiple: bool) -> Path:
    if multiple or output.is_dir() or output.suffix.lower() != ".json":
        return output / f"{case_id}.json"
    if output.stem != case_id:
        raise ValueError(
            f"public cut-plan filename must match case_id {case_id!r}: {output.name!r}"
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RoughCut-Bench agent outputs to the public visual cut-plan format."
    )
    parser.add_argument("--input", required=True, help="workflow-output JSON file or directory")
    parser.add_argument("--output", required=True, help="public cut-plan JSON file or directory")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    files = input_files(source)
    if not files:
        raise SystemExit(f"no JSON files found under {source}")

    multiple = source.is_dir() or len(files) > 1
    converted = 0
    seen_targets = set()
    for file_path in files:
        cut_plan = load_cut_plan(file_path)
        target = output_path(cut_plan["case_id"], destination, multiple)
        if target in seen_targets:
            raise ValueError(f"multiple inputs resolve to the same case_id: {cut_plan['case_id']}")
        seen_targets.add(target)
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {target}; pass --overwrite to replace it")
        write_cut_plan(target, cut_plan)
        converted += 1
        print(f"[OK] {file_path} -> {target}")

    print(f"Converted {converted} cut plan(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
