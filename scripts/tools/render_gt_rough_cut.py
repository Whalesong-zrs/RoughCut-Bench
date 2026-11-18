#!/usr/bin/env python3
"""Render rough-cut preview videos from GT label JSON."""

import argparse
import json
from pathlib import Path

from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTS


def resolve_media_path(video_info, case_id, video_name, media_root):
    media_root = Path(media_root)
    raw_path = video_info.get("path")
    candidates = []
    stems = [Path(str(video_name)).stem]
    if raw_path:
        raw_path = Path(str(raw_path))
        stems.insert(0, raw_path.stem)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        candidates.append(media_root / raw_path)
        candidates.append(media_root / case_id / "source_videos" / raw_path.name)
        candidates.append(media_root / case_id / raw_path.name)
        candidates.append(media_root / "source_videos" / raw_path.name)
        candidates.append(media_root / raw_path.name)
    candidates.append(media_root / case_id / "source_videos" / video_name)
    candidates.append(media_root / case_id / video_name)
    candidates.append(media_root / "source_videos" / video_name)
    candidates.append(media_root / video_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_dirs = [media_root / case_id, media_root]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for stem in dict.fromkeys(stems):
            for ext in sorted(VIDEO_EXTS | IMAGE_EXTS):
                candidate = search_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate

    raise FileNotFoundError(f"media file not found for case={case_id}, video={video_name}")


def normalize_highlight(highlight, media_path):
    if not isinstance(highlight, (list, tuple)) or len(highlight) < 2:
        raise ValueError(f"invalid highlight: {highlight}")
    start = float(highlight[0])
    end = float(highlight[1])
    if end <= start:
        raise ValueError(f"invalid highlight time: start={start}, end={end}, media={media_path}")
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


def render_case(case_id, case_data, out_dir, media_root, mute, fps):
    clips = []
    manifest = {
        "case_id": case_id,
        "media_root": str(media_root),
        "mute": mute,
        "fps": fps,
        "clips": [],
    }

    for video_name, video_info in case_data.items():
        media_path = resolve_media_path(video_info, case_id, video_name, media_root)
        for highlight in video_info.get("highlights", []) or []:
            start, end = normalize_highlight(highlight, media_path)
            clip = make_clip(media_path, start, end, mute, fps)
            clips.append(clip)
            manifest["clips"].append(
                {
                    "video_name": video_name,
                    "media_path": str(media_path),
                    "start_time": start,
                    "end_time": end,
                    "duration": round(end - start, 3),
                    "type": video_info.get("type"),
                }
            )

    if not clips:
        raise ValueError(f"no clips to render for case: {case_id}")

    out_dir = Path(out_dir)
    output_video = out_dir / f"{case_id}.mp4"
    output_manifest = out_dir / f"{case_id}.json"
    write_video(clips, output_video, fps=fps, mute=mute)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return output_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-path", required=True, help="GT label JSON path")
    parser.add_argument("--media-root", required=True, help="raw media root")
    parser.add_argument("--out-dir", required=True, help="directory to write rendered GT rough cuts")
    parser.add_argument("--case-ids", nargs="*", default=None, help="optional case ids to render")
    parser.add_argument("--keep-audio", action="store_true", help="keep original video audio; default is muted")
    parser.add_argument("--fps", type=int, default=30, help="output fps, default 30")
    return parser.parse_args()


def main():
    args = parse_args()
    gt = load_json(args.gt_path)
    wanted = set(args.case_ids) if args.case_ids else None

    success = 0
    failed = []
    for case_id, case_data in gt.items():
        if wanted and case_id not in wanted:
            continue
        try:
            output = render_case(
                case_id=case_id,
                case_data=case_data,
                out_dir=args.out_dir,
                media_root=args.media_root,
                mute=(not args.keep_audio),
                fps=args.fps,
            )
            success += 1
            print(f"[OK] {case_id} -> {output}")
        except Exception as exc:
            failed.append((case_id, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {case_id}: {type(exc).__name__}: {exc}")

    print(f"=== Render GT rough cut summary: success={success} failed={len(failed)} ===")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
