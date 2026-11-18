#!/usr/bin/env python3
"""Render rough-cut preview videos from public or legacy cut-plan JSON."""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_evaluation.cut_plan_schema import load_cut_plan


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_cut_plan_jsons(cut_plan_path):
    cut_plan_path = Path(cut_plan_path)
    if cut_plan_path.is_file():
        if cut_plan_path.suffix.lower() != ".json":
            raise ValueError(f"cut-plan-path must be a .json file or directory: {cut_plan_path}")
        return [cut_plan_path]
    if cut_plan_path.is_dir():
        return sorted(p for p in cut_plan_path.iterdir() if p.suffix.lower() == ".json")
    raise FileNotFoundError(f"cut-plan-path not found: {cut_plan_path}")


def case_id_from_json_path(path):
    return Path(path).stem


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTS


def resolve_media_path(raw_path, case_id, media_root=None):
    raw_path = Path(str(raw_path))
    if raw_path.exists():
        return raw_path

    if media_root:
        media_root = Path(media_root)
        candidates = [
            media_root / case_id / "source_videos" / raw_path.name,
            media_root / case_id / raw_path.name,
            media_root / "source_videos" / raw_path.name,
            media_root / raw_path.name,
        ]
        # If raw_path contains a case-relative suffix, preserve it when possible.
        parts = raw_path.parts
        if case_id in parts:
            idx = parts.index(case_id)
            candidates.insert(0, media_root / Path(*parts[idx:]))
        for candidate in candidates:
            if candidate.exists():
                return candidate

        search_dirs = [
            media_root / case_id / "source_videos",
            media_root / case_id,
            media_root / "source_videos",
            media_root,
        ]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for ext in sorted(VIDEO_EXTS | IMAGE_EXTS):
                candidate = search_dir / f"{raw_path.stem}{ext}"
                if candidate.exists():
                    return candidate

    raise FileNotFoundError(f"media file not found: {raw_path}")


def normalize_segment(segment, media_path):
    start = float(segment.get("start_time", segment.get("highlight_start", 0.0)))
    end = float(segment.get("end_time", segment.get("highlight_end", 0.0)))
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"invalid segment time: start={start}, end={end}, media={media_path}")
    return start, end


def make_clip(media_path, start, end, mute, fps):
    duration = end - start
    if is_image_file(media_path):
        clip = ImageClip(str(media_path)).set_duration(duration).set_fps(fps)
        return clip.without_audio()

    if not is_video_file(media_path):
        raise ValueError(f"unsupported media type: {media_path}")

    video = VideoFileClip(str(media_path))
    safe_start = max(0.0, min(start, max(video.duration - 0.01, 0.0)))
    safe_end = max(safe_start + 0.01, min(end, video.duration))
    clip = video.subclip(safe_start, safe_end)
    if mute:
        clip = clip.without_audio()
    return clip


def write_video(clips, output_path, fps, mute):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_clip = concatenate_videoclips(clips, method="compose")
    try:
        final_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio=(not mute),
            audio_codec="aac",
            logger=None,
        )
    finally:
        final_clip.close()
        for clip in clips:
            clip.close()


def render_one(json_path, out_dir, media_root, mute, fps, skip_invalid_segments=True):
    cut_plan = load_cut_plan(json_path)
    case_id = cut_plan["case_id"]

    clips = []
    manifest = {
        "case_id": case_id,
        "source_json": str(json_path),
        "schema_version": cut_plan["schema_version"],
        "mute": mute,
        "fps": fps,
        "clips": [],
        "skipped_segments": [],
    }

    for video in cut_plan["ordered_videos"]:
        source_id = video["source_id"]
        media_path = resolve_media_path(source_id, case_id, media_root)
        for segment_idx, segment in enumerate(video["segments"]):
            try:
                start, end = normalize_segment(segment, media_path)
            except (TypeError, ValueError, KeyError) as exc:
                if not skip_invalid_segments:
                    raise
                manifest["skipped_segments"].append(
                    {
                        "source_id": source_id,
                        "segment_index": segment_idx,
                        "media_path": str(media_path),
                        "segment": segment,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                continue
            clip = make_clip(media_path, start, end, mute, fps)
            clips.append(clip)
            manifest["clips"].append(
                {
                    "source_id": source_id,
                    "media_path": str(media_path),
                    "start_time": start,
                    "end_time": end,
                    "duration": round(end - start, 3),
                }
            )

    if not clips:
        raise ValueError(f"no clips to render: {json_path}")

    out_dir = Path(out_dir)
    output_video = out_dir / f"{case_id}.mp4"
    output_manifest = out_dir / f"{case_id}.json"
    write_video(clips, output_video, fps=fps, mute=mute)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return output_video


def append_jsonl(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def render_one_status(json_path, out_dir, media_root, mute, fps, max_retries, retry_sleep_sec, skip_existing, skip_invalid_segments):
    case_id = case_id_from_json_path(json_path)
    output_video = Path(out_dir) / f"{case_id}.mp4"
    output_manifest = Path(out_dir) / f"{case_id}.json"
    if skip_existing and output_video.exists() and output_manifest.exists():
        return {
            "ok": True,
            "skipped_existing": True,
            "json_path": str(json_path),
            "case_id": case_id,
            "output": str(output_video),
            "attempt": 0,
            "max_attempts": max(1, int(max_retries) + 1),
        }

    attempts = max(1, int(max_retries) + 1)
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            output = render_one(
                json_path=json_path,
                out_dir=out_dir,
                media_root=media_root,
                mute=mute,
                fps=fps,
                skip_invalid_segments=skip_invalid_segments,
            )
            return {
                "ok": True,
                "json_path": str(json_path),
                "case_id": case_id_from_json_path(json_path),
                "output": str(output),
                "attempt": attempt,
                "max_attempts": attempts,
            }
        except Exception as exc:
            last_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if attempt < attempts and retry_sleep_sec > 0:
                time.sleep(float(retry_sleep_sec))

    return {
        "ok": False,
        "json_path": str(json_path),
        "case_id": case_id_from_json_path(json_path),
        "attempt": attempts,
        "max_attempts": attempts,
        "error_type": last_error["type"] if last_error else "UnknownError",
        "error_message": last_error["message"] if last_error else "unknown error",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut-plan-path", required=True, help="cut-plan JSON file or cut-plan directory")
    parser.add_argument("--out-dir", required=True, help="directory to write rendered rough cuts")
    parser.add_argument("--media-root", default=None, help="optional media root used when JSON video_name paths are missing")
    parser.add_argument("--case-ids", nargs="*", default=None, help="optional case ids to render")
    parser.add_argument("--keep-audio", action="store_true", help="keep original video audio; default is muted")
    parser.add_argument("--fps", type=int, default=30, help="output fps, default 30")
    parser.add_argument("--num-workers", type=int, default=1, help="number of parallel case render workers")
    parser.add_argument("--max-retries", type=int, default=2, help="retry count per failed case; total attempts=max_retries+1")
    parser.add_argument("--retry-sleep-sec", type=float, default=1.0, help="sleep seconds between render retries")
    parser.add_argument("--continue-on-error", action="store_true", help="finish remaining cases and exit 0 even if some cases fail")
    parser.add_argument("--skip-existing", action="store_true", help="skip cases whose output mp4 and manifest json already exist")
    parser.add_argument("--skip-invalid-segments", dest="skip_invalid_segments", action="store_true", help="skip segments with invalid/non-positive time ranges; enabled by default")
    parser.add_argument("--fail-invalid-segments", dest="skip_invalid_segments", action="store_false", help="strict mode: fail the whole case when a segment has invalid/non-positive time range")
    parser.set_defaults(skip_invalid_segments=True)
    parser.add_argument("--failed-log", default="", help="jsonl path for unresolved render failures; default: out_dir/_failed/render_failed.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    wanted = set(args.case_ids) if args.case_ids else None
    json_paths = iter_cut_plan_jsons(args.cut_plan_path)
    if wanted:
        json_paths = [p for p in json_paths if p.stem in wanted]

    success = 0
    failed = []
    failed_log = Path(args.failed_log) if args.failed_log else Path(args.out_dir) / "_failed" / "render_failed.jsonl"
    if failed_log.exists():
        failed_log.unlink()

    def render_kwargs(json_path):
        return {
            "json_path": json_path,
            "out_dir": args.out_dir,
            "media_root": args.media_root,
            "mute": (not args.keep_audio),
            "fps": args.fps,
            "max_retries": args.max_retries,
            "retry_sleep_sec": args.retry_sleep_sec,
            "skip_existing": args.skip_existing,
            "skip_invalid_segments": args.skip_invalid_segments,
        }

    def handle_status(status):
        nonlocal success
        if status.get("ok"):
            success += 1
            if status.get("skipped_existing"):
                print(f"[SKIP_EXISTING] {status['json_path']} -> {status['output']}")
                return
            retry_note = "" if int(status.get("attempt", 1)) <= 1 else f" after attempt {status.get('attempt')}/{status.get('max_attempts')}"
            print(f"[OK] {status['json_path']} -> {status['output']}{retry_note}")
        else:
            failed.append(status)
            append_jsonl(failed_log, status)
            print(
                f"[FAIL] {status['json_path']}: "
                f"{status.get('error_type')}: {status.get('error_message')} "
                f"attempts={status.get('attempt')}/{status.get('max_attempts')}"
            )

    if args.num_workers <= 1 or len(json_paths) <= 1:
        for json_path in json_paths:
            handle_status(render_one_status(**render_kwargs(json_path)))
    else:
        worker_count = min(int(args.num_workers), len(json_paths))
        print(f"=== Rendering {len(json_paths)} cases with workers={worker_count}, max_retries={args.max_retries} ===")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_path = {
                executor.submit(render_one_status, **render_kwargs(json_path)): json_path
                for json_path in json_paths
            }
            for future in as_completed(future_to_path):
                json_path = future_to_path[future]
                try:
                    handle_status(future.result())
                except Exception as exc:
                    status = {
                        "ok": False,
                        "json_path": str(json_path),
                        "case_id": case_id_from_json_path(json_path),
                        "attempt": 1,
                        "max_attempts": max(1, int(args.max_retries) + 1),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    handle_status(status)

    print(f"=== Render cut-plan summary: success={success} failed={len(failed)} ===")
    if failed:
        print(f"Unresolved render failures written to: {failed_log}")
        if not args.continue_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
