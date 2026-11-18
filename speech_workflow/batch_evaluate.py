#!/usr/bin/env python3
"""Batch offline evaluator based on `evaluate_f1.py`."""

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_workflow.evaluate_f1 import evaluate_v2, print_evaluation_report
from speech_workflow.runtime_paths import SPEECH_DATA_ROOT


def print_average_metrics(results: List[Dict], title: str):
    print(f"\n{title}:")
    print(f"  Files: {len(results)}")
    if not results:
        return

    avg_im_recall = sum(r["im"]["recall"] for r in results) / len(results)
    avg_im_precision = sum(r["im"]["precision"] for r in results) / len(results)
    avg_im_f1 = sum(r["im"]["f1"] for r in results) / len(results)
    print(f"  Average IM: Recall={avg_im_recall * 100:.2f}% Precision={avg_im_precision * 100:.2f}% F1={avg_im_f1 * 100:.2f}%")

    avg_rm_r_recall = sum(r["rm_r"]["recall"] for r in results) / len(results)
    avg_rm_r_precision = sum(r["rm_r"]["precision"] for r in results) / len(results)
    avg_rm_r_f1 = sum(r["rm_r"]["f1"] for r in results) / len(results)
    print(f"  Average RM→R: Recall={avg_rm_r_recall * 100:.2f}% Precision={avg_rm_r_precision * 100:.2f}% F1={avg_rm_r_f1 * 100:.2f}%")

    if all("d" in r for r in results):
        avg_d_recall = sum(r["d"]["recall"] for r in results) / len(results)
        avg_d_precision = sum(r["d"]["precision"] for r in results) / len(results)
        avg_d_f1 = sum(r["d"]["f1"] for r in results) / len(results)
        print(f"  Average D: Recall={avg_d_recall * 100:.2f}% Precision={avg_d_precision * 100:.2f}% F1={avg_d_f1 * 100:.2f}%")

def get_test_type(test_file: Path) -> str:
    if test_file.name.endswith("_test1.txt"):
        return "test1"
    if test_file.name.endswith("_test2.txt"):
        return "test2"
    return "test"


def get_case_stem(test_file: Path) -> str:
    stem = test_file.stem
    for suffix in ("_test1", "_test2", "_test"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def has_annotation_tags(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return any(tag in text for tag in ("[IM]", "[RM]", "[R]", "[D]"))


def find_annotated_file(dir_path: Path, case_stem: str) -> Path:
    # Support current and legacy annotation filenames.
    candidates = [dir_path / f"{case_stem}_gt.txt", dir_path / f"{case_stem}.txt"]
    for candidate in candidates:
        if has_annotation_tags(candidate):
            return candidate
    return candidates[0]


def batch_evaluate(
    dir_path: str = str(SPEECH_DATA_ROOT),
    output_summary_path: str = None,
    test_pattern: str = "*_test*.txt"
):
    dir_path = Path(dir_path)
    all_results: List[Dict] = []

    test_files = sorted(
        p for p in dir_path.glob(test_pattern)
        if p.name.endswith(("_test.txt", "_test1.txt", "_test2.txt"))
    )
    print(f"Found {len(test_files)} test files")

    for test_file in test_files:
        case_stem = get_case_stem(test_file)
        annotated_file = find_annotated_file(dir_path, case_stem)

        if not annotated_file.exists():
            print(f"Warning: Annotated file {annotated_file.name} not found for {test_file.name}, skip")
            continue

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {annotated_file.name} vs {test_file.name}")
        print(f"{'=' * 80}")

        try:
            result = evaluate_v2(str(annotated_file), str(test_file))
            result["file_name"] = case_stem
            result["annotated_file"] = annotated_file.name
            result["test_file"] = test_file.name
            result["test_type"] = get_test_type(test_file)
            all_results.append(result)
            print_evaluation_report(result)
        except Exception as e:
            print(f"Failed to evaluate {annotated_file.name}: {e}")
            continue

    print(f"\n{'=' * 80}")
    print("汇总结果")
    print(f"{'=' * 80}")
    print(f"Total files evaluated: {len(all_results)}")

    if all_results:
        total_exact_match = sum(r["exact_match"] for r in all_results)
        avg_exact_match = total_exact_match / len(all_results)
        print(f"Average Exact Match: {avg_exact_match * 100:.2f}%")

        avg_im_recall = sum(r["im"]["recall"] for r in all_results) / len(all_results)
        avg_im_precision = sum(r["im"]["precision"] for r in all_results) / len(all_results)
        avg_im_f1 = sum(r["im"]["f1"] for r in all_results) / len(all_results)
        print("\nAverage IM:")
        print(f"  Recall: {avg_im_recall * 100:.2f}%")
        print(f"  Precision: {avg_im_precision * 100:.2f}%")
        print(f"  F1: {avg_im_f1 * 100:.2f}%")

        avg_rm_r_recall = sum(r["rm_r"]["recall"] for r in all_results) / len(all_results)
        avg_rm_r_precision = sum(r["rm_r"]["precision"] for r in all_results) / len(all_results)
        avg_rm_r_f1 = sum(r["rm_r"]["f1"] for r in all_results) / len(all_results)
        print("\nAverage RM→R:")
        print(f"  Recall: {avg_rm_r_recall * 100:.2f}%")
        print(f"  Precision: {avg_rm_r_precision * 100:.2f}%")
        print(f"  F1: {avg_rm_r_f1 * 100:.2f}%")

        if all("d" in r for r in all_results):
            avg_d_recall = sum(r["d"]["recall"] for r in all_results) / len(all_results)
            avg_d_precision = sum(r["d"]["precision"] for r in all_results) / len(all_results)
            avg_d_f1 = sum(r["d"]["f1"] for r in all_results) / len(all_results)
            print("\nAverage D:")
            print(f"  Recall: {avg_d_recall * 100:.2f}%")
            print(f"  Precision: {avg_d_precision * 100:.2f}%")
            print(f"  F1: {avg_d_f1 * 100:.2f}%")

        total_fp = sum(r["false_positives"]["count"] for r in all_results)
        print(f"\nTotal FP (误删): {total_fp}")

        for test_type in sorted({r.get("test_type", "test") for r in all_results}):
            grouped = [r for r in all_results if r.get("test_type", "test") == test_type]
            print_average_metrics(grouped, f"{test_type} 分组平均")

    if output_summary_path:
        import json
        with open(output_summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to {output_summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch evaluate all test files in asr_case")
    parser.add_argument("--dir", default=str(SPEECH_DATA_ROOT), help="Directory containing test and annotated files")
    parser.add_argument("--output-json", default=None, help="Path to save summary JSON")
    parser.add_argument("--test-pattern", default="*_test*.txt", help="Glob pattern for test files")
    args = parser.parse_args()

    batch_evaluate(args.dir, args.output_json, args.test_pattern)
