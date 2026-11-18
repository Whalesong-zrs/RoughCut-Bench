#!/usr/bin/env python3
"""Run a minimal credentialed request through the centralized model adapters."""

import argparse
import base64
import mimetypes
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_workflow.model_api import MODEL_REGISTRY, ModelRunner


def image_part(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}", "detail": "low"},
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send one small, billable API request per explicitly selected model."
    )
    parser.add_argument("--models", nargs="+", required=True, help="benchmark model aliases")
    parser.add_argument("--image", default="", help="optional image for a multimodal smoke test")
    parser.add_argument("--prompt", default="Reply with the single word OK.")
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = sorted(set(args.models) - set(MODEL_REGISTRY))
    if unknown:
        raise SystemExit(f"unknown model aliases: {unknown}; available: {sorted(MODEL_REGISTRY)}")

    content = [{"type": "text", "text": args.prompt}]
    if args.image:
        content.append(image_part(Path(args.image)))
    messages = [{"role": "user", "content": content}]

    runner = ModelRunner()
    failed = []
    for model in args.models:
        try:
            text, input_tokens, output_tokens = runner.run(messages, model)
            preview = " ".join(text.split())[:160]
            print(
                f"[OK] {model}: input_tokens={input_tokens} "
                f"output_tokens={output_tokens} response={preview!r}"
            )
        except Exception as ex:
            failed.append(model)
            print(f"[FAIL] {model}: {type(ex).__name__}: {ex}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
