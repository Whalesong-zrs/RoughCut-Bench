#!/usr/bin/env python3
"""Run one ASR case through all prompt variants and persist raw/final outputs."""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_workflow.asr_prompts import (
    COT_ZERO_SHOT_PROMPT,
    ZERO_SHOT_PROMPT,
)
from speech_workflow.transcript_editing_agent import (
    clean_model_output,
    load_llm_module,
    safe_model_name,
    strip_annotation_tags,
)
from speech_workflow.runtime_paths import SPEECH_OUTPUT_ROOT


PROMPT_SPECS = [
    ("ZERO_SHOT_PROMPT", ZERO_SHOT_PROMPT, "txt"),
    ("COT_ZERO_SHOT_PROMPT", COT_ZERO_SHOT_PROMPT, "json"),
]


def extract_json(text: str) -> str:
    text = clean_model_output(text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        candidate = match.group(0)
        json.loads(candidate)
        return candidate


def normalize_cot_json(text: str) -> dict:
    data = json.loads(text)

    for edit in data.get("edits", []):
        if "text" not in edit and "" in edit:
            edit["text"] = edit.pop("")
        for key in list(edit):
            if key not in {"text", "kept", "type", "reason"}:
                edit.pop(key)
        edit.setdefault("text", "")
        edit.setdefault("kept", "")
        edit.setdefault("type", "")
        edit.setdefault("reason", "")
    if "cleaned_text" not in data:
        for fallback_key in ("cleaned_", "final_text", "output_text", "result"):
            if fallback_key in data:
                data["cleaned_text"] = data.pop(fallback_key)
                break
    data.setdefault("cleaned_text", "")

    return {
        "edits": data.get("edits", []),
        "cleaned_text": data.get("cleaned_text", ""),
    }


def raw_response_path(output_dir: Path, model_safe: str) -> Path:
    return output_dir / f"{model_safe}_raw_response.txt"


def prompt_path(output_dir: Path, model_safe: str) -> Path:
    return output_dir / f"{model_safe}_prompt.txt"


def build_prompt(template: str, input_text: str) -> str:
    return template.format(input_text=input_text)


def main():
    parser = argparse.ArgumentParser(description="Run one ASR case and save outputs by prompt folder.")
    parser.add_argument("--case-dir", required=True, help="Case folder containing gt.txt")
    parser.add_argument("--out-dir", default=None, help="Output directory for this case")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for LLM generation")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling value")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    case_output_dir = Path(args.out_dir) if args.out_dir else SPEECH_OUTPUT_ROOT / "cases" / case_dir.name
    input_text = strip_annotation_tags((case_dir / "gt.txt").read_text(encoding="utf-8"))
    model_safe = safe_model_name(args.model)
    runner = load_llm_module().ModelRunner()

    for prompt_name, template, suffix in PROMPT_SPECS:
        output_dir = case_output_dir / prompt_name
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt = build_prompt(template, input_text)
        prompt_path(output_dir, model_safe).write_text(prompt, encoding="utf-8")

        res, _, _ = runner.run(
            [{"role": "user", "content": prompt}],
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        raw_response_path(output_dir, model_safe).write_text(res, encoding="utf-8")

        if suffix == "json":
            data = normalize_cot_json(extract_json(res))
            output_text = json.dumps(data, ensure_ascii=False, indent=4)
            json_path = output_dir / f"{model_safe}.json"
            json_path.write_text(output_text + "\n", encoding="utf-8")
            cleaned_text = data.get("cleaned_text", "")
            (output_dir / f"{model_safe}_cleaned_text.txt").write_text(cleaned_text, encoding="utf-8")
            print(f"{prompt_name}: saved {json_path.name}, edits={len(data.get('edits', []))}", flush=True)
        else:
            output_text = clean_model_output(res)
            txt_path = output_dir / f"{model_safe}.txt"
            txt_path.write_text(output_text, encoding="utf-8")
            print(f"{prompt_name}: saved {txt_path.name}", flush=True)


if __name__ == "__main__":
    main()
