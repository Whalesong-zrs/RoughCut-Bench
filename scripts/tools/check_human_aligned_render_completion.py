#!/usr/bin/env python3
"""Check whether videos for human-aligned evaluation have all been rendered.

The script expects the layout produced by prepare_human_aligned_evaluation.py:
  result_root/selected_runs.json
  result_root/rendered/{set_name}/{model_slug}/{case_id}.mp4
  result_root/rendered/{set_name}/{model_slug}/{case_id}.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def natural_key(text: str):
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                items.append({"line_no": line_no, "raw": line, "json_error": True})
    return items


def case_files(rendered_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    if not rendered_dir.exists():
        return {}, {}
    mp4s = {p.stem: p for p in rendered_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"}
    manifests = {p.stem: p for p in rendered_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"}
    return mp4s, manifests


def inspect_manifest(path: Path) -> Tuple[int, int, str]:
    try:
        data = load_json(path)
    except Exception as exc:
        return 0, 0, f"{type(exc).__name__}: {exc}"
    clips = data.get("clips", [])
    skipped = data.get("skipped_segments", [])
    clip_count = len(clips) if isinstance(clips, list) else 0
    skipped_count = len(skipped) if isinstance(skipped, list) else 0
    return clip_count, skipped_count, ""


def collect_one(result_root: Path, row: dict, common_cases: List[str], min_video_bytes: int) -> dict:
    set_name = row["set_name"]
    model = row["model"]
    model_slug = row.get("model_slug") or model.replace("/", "_").replace(" ", "_")
    rendered_dir = result_root / "rendered" / set_name / model_slug
    expected = list(common_cases)
    expected_set = set(expected)

    mp4s, manifests = case_files(rendered_dir)
    failed_log = rendered_dir / "_failed" / "render_failed.jsonl"
    failed_items = read_jsonl(failed_log)
    failed_cases = sorted({str(item.get("case_id") or Path(str(item.get("json_path", ""))).stem) for item in failed_items if item}, key=natural_key)

    missing_video = []
    small_video = []
    missing_manifest = []
    invalid_manifest = []
    zero_clip_cases = []
    skipped_segment_cases = []
    skipped_segments = 0

    valid_video_cases = []
    valid_manifest_cases = []
    for case_id in expected:
        mp4 = mp4s.get(case_id)
        manifest = manifests.get(case_id)
        if mp4 is None:
            missing_video.append(case_id)
        else:
            size = mp4.stat().st_size
            if size < min_video_bytes:
                small_video.append(case_id)
            else:
                valid_video_cases.append(case_id)

        if manifest is None:
            missing_manifest.append(case_id)
        else:
            clip_count, skipped_count, error = inspect_manifest(manifest)
            if error:
                invalid_manifest.append({"case_id": case_id, "error": error})
            else:
                valid_manifest_cases.append(case_id)
                if clip_count <= 0:
                    zero_clip_cases.append(case_id)
                if skipped_count > 0:
                    skipped_segment_cases.append(case_id)
                    skipped_segments += skipped_count

    extra_mp4 = sorted(set(mp4s) - expected_set, key=natural_key)
    extra_manifest = sorted(set(manifests) - expected_set, key=natural_key)
    incomplete_cases = sorted(
        set(missing_video)
        | set(small_video)
        | set(missing_manifest)
        | {item["case_id"] for item in invalid_manifest}
        | set(zero_clip_cases),
        key=natural_key,
    )
    unresolved_failed_cases = sorted(set(failed_cases) & set(incomplete_cases), key=natural_key)

    return {
        "set_name": set_name,
        "model": model,
        "model_slug": model_slug,
        "rendered_dir": str(rendered_dir),
        "expected": len(expected),
        "video_ok": len(valid_video_cases),
        "manifest_ok": len(valid_manifest_cases),
        "missing_video": missing_video,
        "small_video": small_video,
        "missing_manifest": missing_manifest,
        "invalid_manifest": invalid_manifest,
        "zero_clip_cases": zero_clip_cases,
        "extra_mp4": extra_mp4,
        "extra_manifest": extra_manifest,
        "failed_log": str(failed_log),
        "failed_log_cases": failed_cases,
        "unresolved_failed_cases": unresolved_failed_cases,
        "skipped_segment_cases": skipped_segment_cases,
        "skipped_segments": skipped_segments,
        "complete": len(incomplete_cases) == 0,
        "incomplete_cases": incomplete_cases,
    }


def limited(items: Iterable, limit: int) -> str:
    items = list(items)
    if not items:
        return "-"
    shown = items[:limit]
    suffix = "" if len(items) <= limit else f", ... total={len(items)}"
    return ", ".join(map(str, shown)) + suffix


def print_table(rows: List[dict], detail_limit: int) -> None:
    print("=== Human-aligned evaluation render completion ===")
    headers = [
        "set", "model", "expected", "video_ok", "manifest_ok", "missing_video",
        "small_video", "missing_manifest", "failed_log", "skipped_seg_cases", "skipped_segments", "complete",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            row["set_name"],
            row["model"],
            row["expected"],
            row["video_ok"],
            row["manifest_ok"],
            len(row["missing_video"]),
            len(row["small_video"]),
            len(row["missing_manifest"]),
            len(row["failed_log_cases"]),
            len(row["skipped_segment_cases"]),
            row["skipped_segments"],
            "yes" if row["complete"] else "no",
        ]
        print("| " + " | ".join(map(str, values)) + " |")

    incomplete = [row for row in rows if not row["complete"]]
    skipped = [row for row in rows if row["skipped_segments"] > 0]
    failed_logs = [row for row in rows if row["failed_log_cases"]]

    if incomplete:
        print("\n=== Incomplete details ===")
        for row in incomplete:
            print(f"[{row['set_name']}] {row['model']} -> {row['rendered_dir']}")
            print(f"  missing_video: {limited(row['missing_video'], detail_limit)}")
            print(f"  small_video: {limited(row['small_video'], detail_limit)}")
            print(f"  missing_manifest: {limited(row['missing_manifest'], detail_limit)}")
            if row["invalid_manifest"]:
                invalid = [f"{item['case_id']} ({item['error']})" for item in row["invalid_manifest"]]
                print(f"  invalid_manifest: {limited(invalid, detail_limit)}")
            print(f"  zero_clip_cases: {limited(row['zero_clip_cases'], detail_limit)}")
            print(f"  unresolved_failed_cases: {limited(row['unresolved_failed_cases'], detail_limit)}")

    if failed_logs:
        print("\n=== Failed-log cases ===")
        for row in failed_logs:
            print(f"[{row['set_name']}] {row['model']}: {limited(row['failed_log_cases'], detail_limit)}")

    if skipped:
        print("\n=== Rendered with skipped invalid segments ===")
        for row in skipped:
            print(f"[{row['set_name']}] {row['model']}: cases={limited(row['skipped_segment_cases'], detail_limit)} segments={row['skipped_segments']}")


def write_outputs(rows: List[dict], json_path: str, md_path: str) -> None:
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    if md_path:
        path = Path(md_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "set", "model", "expected", "video_ok", "manifest_ok", "missing_video",
            "small_video", "missing_manifest", "failed_log", "skipped_seg_cases", "skipped_segments", "complete",
        ]
        lines = ["# Human-Aligned Evaluation Render Completion", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            values = [
                row["set_name"], row["model"], row["expected"], row["video_ok"], row["manifest_ok"],
                len(row["missing_video"]), len(row["small_video"]), len(row["missing_manifest"]),
                len(row["failed_log_cases"]), len(row["skipped_segment_cases"]), row["skipped_segments"],
                "yes" if row["complete"] else "no",
            ]
            lines.append("| " + " | ".join(map(str, values)) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[2]
    output_root = Path(os.environ.get("ROUGHCUT_OUTPUT_ROOT") or project_root / "outputs")
    parser.add_argument("--result-root", default=str(output_root / "human_aligned_evaluation"))
    parser.add_argument("--selected-runs", default="", help="default: {result-root}/selected_runs.json")
    parser.add_argument("--min-video-bytes", type=int, default=1024, help="mp4 smaller than this is treated as invalid")
    parser.add_argument("--detail-limit", type=int, default=30)
    parser.add_argument("--write-json", default="")
    parser.add_argument("--write-md", default="")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = Path(args.result_root)
    selected_runs = Path(args.selected_runs) if args.selected_runs else result_root / "selected_runs.json"
    metadata = load_json(selected_runs)
    common_cases = metadata.get("common_cases", {})
    rows = []
    for row in metadata.get("rows", []):
        set_name = row["set_name"]
        case_info = common_cases.get(set_name, {})
        case_ids = case_info.get("case_ids", []) if isinstance(case_info, dict) else []
        rows.append(collect_one(result_root, row, case_ids, args.min_video_bytes))
    rows.sort(key=lambda item: (item["set_name"], natural_key(item["model"])))

    print_table(rows, args.detail_limit)
    write_outputs(rows, args.write_json, args.write_md)

    all_complete = all(row["complete"] for row in rows)
    print(f"\nALL_COMPLETE={str(all_complete).lower()}")
    if args.fail_on_incomplete and not all_complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
