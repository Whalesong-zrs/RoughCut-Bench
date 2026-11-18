#!/usr/bin/env python3
"""Batch runner for the per-case ASR prompt benchmark."""

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_workflow.transcript_editing_agent import (
    clean_model_output,
    load_llm_module,
    safe_model_name,
    strip_annotation_tags,
)
from speech_workflow.runtime_paths import SPEECH_DATA_ROOT, SPEECH_OUTPUT_ROOT
from speech_workflow.run_prompt_case import (
    PROMPT_SPECS,
    build_prompt,
    extract_json,
    normalize_cot_json,
    prompt_path,
    raw_response_path,
)


DEFAULT_MODELS = [
    "glm-5v-turbo",
    "qwen-3.5-plus",
    "qwen-3.7-plus",
    "seed-2.0-lite",
    "gemini-2.5-pro",
    "gemini-3-pro",
    "gpt-5.4",
    "gpt-5.5",
]


def discover_cases(dir_path: Path, max_cases: int | None = None) -> list[Path]:
    cases = sorted(path for path in dir_path.glob("case*") if path.is_dir() and (path / "gt.txt").exists())
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def result_path(case_output_dir: Path, prompt_name: str, suffix: str, model_safe: str) -> Path:
    output_dir = case_output_dir / prompt_name
    if suffix == "json":
        return output_dir / f"{model_safe}.json"
    return output_dir / f"{model_safe}.txt"


def has_existing_result(case_output_dir: Path, prompt_name: str, suffix: str, model_safe: str) -> bool:
    path = result_path(case_output_dir, prompt_name, suffix, model_safe)
    if not path.exists() or path.stat().st_size == 0:
        return False
    if suffix == "json":
        cleaned_path = path.with_name(f"{model_safe}_cleaned_text.txt")
        return cleaned_path.exists() and cleaned_path.stat().st_size > 0
    return True


def write_error(case_output_dir: Path, prompt_name: str, model_safe: str, message: str):
    output_dir = case_output_dir / prompt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model_safe}.error.txt").write_text(message, encoding="utf-8")


def run_task(task):
    case_dir, case_output_dir, model, prompt_name, template, suffix, skip_existing, temperature, top_p = task
    model_safe = safe_model_name(model)
    output_dir = case_output_dir / prompt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and has_existing_result(case_output_dir, prompt_name, suffix, model_safe):
        return "skip", case_dir.name, model, prompt_name

    res = None
    try:
        input_text = strip_annotation_tags((case_dir / "gt.txt").read_text(encoding="utf-8"))
        prompt = build_prompt(template, input_text)
        prompt_file = prompt_path(output_dir, model_safe)
        prompt_file.write_text(prompt, encoding="utf-8")
        runner = load_llm_module().ModelRunner()
        res, _, _ = runner.run(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        raw_path = raw_response_path(output_dir, model_safe)
        raw_path.write_text(res, encoding="utf-8")

        if suffix == "json":
            data = normalize_cot_json(extract_json(res))
            output_text = json.dumps(data, ensure_ascii=False, indent=4)
            path = output_dir / f"{model_safe}.json"
            path.write_text(output_text + "\n", encoding="utf-8")
            (output_dir / f"{model_safe}_cleaned_text.txt").write_text(
                data.get("cleaned_text", ""),
                encoding="utf-8",
            )
            error_path = output_dir / f"{model_safe}.error.txt"
            if error_path.exists():
                error_path.unlink()
            return "ok", case_dir.name, model, prompt_name

        path = output_dir / f"{model_safe}.txt"
        path.write_text(clean_model_output(res), encoding="utf-8")
        error_path = output_dir / f"{model_safe}.error.txt"
        if error_path.exists():
            error_path.unlink()
        return "ok", case_dir.name, model, prompt_name
    except Exception:
        error_message = traceback.format_exc()
        if res is not None:
            raw_path = raw_response_path(output_dir, model_safe)
            raw_path.write_text(res, encoding="utf-8")
        error_message += f"\nPrompt saved to: {prompt_path(output_dir, model_safe)}\n"
        if (output_dir / f"{model_safe}_raw_response.txt").exists():
            error_message += f"Raw response saved to: {raw_response_path(output_dir, model_safe)}\n"
        write_error(case_output_dir, prompt_name, model_safe, error_message)
        return "fail", case_dir.name, model, prompt_name


def main():
    parser = argparse.ArgumentParser(description="Run all ASR cases for all prompt templates and models.")
    parser.add_argument(
        "--dir",
        default=str(SPEECH_DATA_ROOT),
        help="Directory containing per-case folders",
    )
    parser.add_argument(
        "--out-dir",
        default=str(SPEECH_OUTPUT_ROOT / "cases"),
        help="Directory for generated per-case outputs",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for LLM generation")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling value")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.dir)
    if not data_root.is_dir():
        parser.error(f"speech case directory not found: {data_root}")
    case_dirs = discover_cases(data_root, args.max_cases)
    if not case_dirs:
        parser.error(f"no case*/gt.txt inputs found under {data_root}")

    registry = load_llm_module().MODEL_REGISTRY
    unknown_models = sorted(set(args.models) - set(registry))
    if unknown_models:
        parser.error(f"unknown model aliases: {unknown_models}; available: {sorted(registry)}")

    output_root = Path(args.out_dir)
    tasks = [
        (case_dir, output_root / case_dir.name, model, prompt_name, template, suffix, args.skip_existing)
        for case_dir in case_dirs
        for model in args.models
        for prompt_name, template, suffix in PROMPT_SPECS
    ]
    tasks = [
        (*task, args.temperature, args.top_p)
        for task in tasks
    ]

    print(f"Cases: {len(case_dirs)}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"Models: {len(args.models)} ({', '.join(args.models)})", flush=True)
    print(f"Prompts: {len(PROMPT_SPECS)}", flush=True)
    print(f"Tasks: {len(tasks)}", flush=True)
    print(f"Workers: {max(1, args.workers)}", flush=True)
    print(f"Temperature: {args.temperature}", flush=True)
    print(f"Top-p: {args.top_p}", flush=True)
    if args.dry_run:
        return

    counts = {"ok": 0, "fail": 0, "skip": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), 1):
            status, case_name, model, prompt_name = future.result()
            counts[status] += 1
            print(
                f"[{idx}/{len(tasks)}] {status.upper()} {case_name} {model} {prompt_name} "
                f"(ok={counts['ok']} fail={counts['fail']} skip={counts['skip']})",
                flush=True,
            )

    print(f"Done. ok={counts['ok']} fail={counts['fail']} skip={counts['skip']}", flush=True)


if __name__ == "__main__":
    main()
