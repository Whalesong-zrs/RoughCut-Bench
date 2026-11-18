#!/usr/bin/env python3
"""Aggregate rule-based F1 metrics across prompts and models."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_workflow.evaluate_f1 import calculate_f1, calculate_precision, evaluate_v2
from speech_workflow.transcript_editing_agent import safe_model_name
from speech_workflow.run_prompt_all_cases import DEFAULT_MODELS, discover_cases
from speech_workflow.run_prompt_case import PROMPT_SPECS
from speech_workflow.runtime_paths import SPEECH_DATA_ROOT, SPEECH_OUTPUT_ROOT


CASE_DIR = SPEECH_DATA_ROOT
OUT_DIR = SPEECH_OUTPUT_ROOT / "metrics"


def output_path(prediction_root: Path, case_name: str, prompt_name: str, suffix: str, model_safe: str) -> Path:
    output_dir = prediction_root / case_name / prompt_name
    if suffix == "json":
        return output_dir / f"{model_safe}_cleaned_text.txt"
    return output_dir / f"{model_safe}.txt"


def empty_score_bucket() -> dict:
    return {
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "total": 0,
    }


def finalize_bucket(bucket: dict) -> dict:
    if bucket["total"] == 0:
        return {
            **bucket,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    precision = calculate_precision(bucket["tp"], bucket["fp"])
    recall = bucket["tp"] / bucket["total"]
    return {
        **bucket,
        "precision": precision,
        "recall": recall,
        "f1": calculate_f1(precision, recall),
    }


def evaluate_group(
    case_dirs: list[Path],
    prediction_root: Path,
    model: str,
    prompt_name: str,
    suffix: str,
) -> dict:
    model_safe = safe_model_name(model)
    im = empty_score_bucket()
    rm_r = empty_score_bucket()
    d = empty_score_bucket()
    evaluated_cases = 0
    missing_cases = []
    failed_cases = []

    for case_dir in case_dirs:
        gt_path = case_dir / "gt.txt"
        pred_path = output_path(prediction_root, case_dir.name, prompt_name, suffix, model_safe)
        if not pred_path.exists() or pred_path.stat().st_size == 0:
            missing_cases.append(case_dir.name)
            continue

        try:
            result = evaluate_v2(str(gt_path), str(pred_path))
        except Exception as exc:
            failed_cases.append(f"{case_dir.name}:{type(exc).__name__}:{exc}")
            continue

        evaluated_cases += 1
        fp_count = result["false_positives"]["count"]

        im["tp"] += result["im"]["tp"]
        im["fn"] += result["im"]["fn"]
        im["fp"] += fp_count
        im["total"] += result["im"]["total"]

        rm_r["tp"] += result["rm_r"]["tp"]
        rm_r["fn"] += result["rm_r"]["fn"]
        rm_r["fp"] += fp_count
        rm_r["total"] += result["rm_r"]["total"]

        d["tp"] += result["d"]["tp"]
        d["fn"] += result["d"]["fn"]
        d["fp"] += result["d"]["fp"]
        d["total"] += result["d"]["total"]

    im_final = finalize_bucket(im)
    rm_r_final = finalize_bucket(rm_r)
    d_final = finalize_bucket(d)

    return {
        "model": model,
        "model_safe": model_safe,
        "prompt": prompt_name,
        "evaluated_cases": evaluated_cases,
        "missing_cases": len(missing_cases),
        "failed_eval_cases": len(failed_cases),
        "missing_case_names": ";".join(missing_cases),
        "failed_eval_case_names": ";".join(failed_cases),
        "im": im_final,
        "rm_r": rm_r_final,
        "d": d_final,
    }


def flatten_row(row: dict) -> dict:
    flat = {
        "model": row["model"],
        "model_safe": row["model_safe"],
        "prompt": row["prompt"],
        "evaluated_cases": row["evaluated_cases"],
        "missing_cases": row["missing_cases"],
        "failed_eval_cases": row["failed_eval_cases"],
        "missing_case_names": row["missing_case_names"],
        "failed_eval_case_names": row["failed_eval_case_names"],
    }
    for metric_name in ("im", "rm_r", "d"):
        metric = row[metric_name]
        for key in ("tp", "fn", "fp", "total", "precision", "recall", "f1"):
            flat[f"{metric_name}_{key}"] = metric[key]
    return flat


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def write_csv(rows: list[dict], path: Path):
    flat_rows = [flatten_row(row) for row in rows]
    fieldnames = list(flat_rows[0].keys()) if flat_rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_markdown(rows: list[dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        f.write("# ASR Prompt Scores By Model\n\n")
        f.write("Scores are micro-aggregated across evaluated cases using `evaluate_f1.evaluate_v2`.\n\n")
        f.write(
            "| model | prompt | cases | missing | "
            "IM P | IM R | IM F1 | "
            "RM→R P | RM→R R | RM→R F1 | "
            "D P | D R | D F1 |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['model']} | {row['prompt']} | {row['evaluated_cases']} | {row['missing_cases']} | "
                f"{format_pct(row['im']['precision'])} | {format_pct(row['im']['recall'])} | {format_pct(row['im']['f1'])} | "
                f"{format_pct(row['rm_r']['precision'])} | {format_pct(row['rm_r']['recall'])} | {format_pct(row['rm_r']['f1'])} | "
                f"{format_pct(row['d']['precision'])} | {format_pct(row['d']['recall'])} | {format_pct(row['d']['f1'])} |\n"
            )

        f.write("\n## Missing Outputs\n\n")
        for row in rows:
            if row["missing_cases"] or row["failed_eval_cases"]:
                f.write(
                    f"- {row['model']} / {row['prompt']}: "
                    f"missing={row['missing_cases']}, failed_eval={row['failed_eval_cases']}\n"
                )
                if row["missing_case_names"]:
                    f.write(f"  missing cases: {row['missing_case_names']}\n")
                if row["failed_eval_case_names"]:
                    f.write(f"  failed eval cases: {row['failed_eval_case_names']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(CASE_DIR), help="Directory containing speech cases")
    parser.add_argument(
        "--pred-dir",
        default=str(SPEECH_OUTPUT_ROOT / "cases"),
        help="Directory containing generated per-case outputs",
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Directory for aggregate reports")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models whose generated outputs should be evaluated")
    parser.add_argument("--max-cases", type=int, default=None, help="Evaluate at most this many discovered cases")
    args = parser.parse_args()

    case_dir = Path(args.dir)
    prediction_root = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = discover_cases(case_dir, args.max_cases)
    rows = [
        evaluate_group(case_dirs, prediction_root, model, prompt_name, suffix)
        for model in args.models
        for prompt_name, _, suffix in PROMPT_SPECS
    ]

    csv_path = out_dir / "prompt_scores_by_model.csv"
    md_path = out_dir / "prompt_scores_by_model.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"cases={len(case_dirs)}")
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    for row in rows:
        print(
            f"{row['model']} {row['prompt']} cases={row['evaluated_cases']} "
            f"missing={row['missing_cases']} "
            f"IM(P/R/F1)={format_pct(row['im']['precision'])}/{format_pct(row['im']['recall'])}/{format_pct(row['im']['f1'])} "
            f"RM_R(P/R/F1)={format_pct(row['rm_r']['precision'])}/{format_pct(row['rm_r']['recall'])}/{format_pct(row['rm_r']['f1'])} "
            f"D(P/R/F1)={format_pct(row['d']['precision'])}/{format_pct(row['d']['recall'])}/{format_pct(row['d']['f1'])}"
        )

    if not any(row["evaluated_cases"] > 0 for row in rows):
        raise SystemExit(
            "No generated predictions were evaluated. Check --dir, --pred-dir, and --models."
        )


if __name__ == "__main__":
    main()
