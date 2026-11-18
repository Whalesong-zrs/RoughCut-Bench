"""Public visual cut-plan schema and legacy workflow-output conversion."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "roughcut-bench-cut-plan-v1"


class CutPlanValidationError(ValueError):
    """Raised when a visual cut plan does not satisfy the public schema."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CutPlanValidationError(message)


def _source_id_from_path(value: Any) -> str:
    normalized = str(value).replace("\\", "/")
    return PurePosixPath(normalized).stem


def _validate_segment(segment: Any, location: str) -> dict:
    _require(isinstance(segment, dict), f"{location} must be an object")
    allowed = {"start_time", "end_time"}
    extra = set(segment) - allowed
    _require(not extra, f"{location} has unsupported fields: {sorted(extra)}")
    _require("start_time" in segment, f"{location}.start_time is required")
    _require("end_time" in segment, f"{location}.end_time is required")

    start = segment["start_time"]
    end = segment["end_time"]
    _require(
        isinstance(start, (int, float)) and not isinstance(start, bool),
        f"{location}.start_time must be a number",
    )
    _require(
        isinstance(end, (int, float)) and not isinstance(end, bool),
        f"{location}.end_time must be a number",
    )
    start = float(start)
    end = float(end)
    _require(math.isfinite(start) and math.isfinite(end), f"{location} times must be finite")
    _require(start >= 0, f"{location}.start_time must be non-negative")
    _require(end > start, f"{location}.end_time must be greater than start_time")
    return {"start_time": start, "end_time": end}


def validate_public_cut_plan(data: Any, expected_case_id: str | None = None) -> dict:
    """Validate and normalize one public visual cut plan."""
    _require(isinstance(data, dict), "cut plan must be a JSON object")
    allowed = {"schema_version", "case_id", "ordered_videos", "metadata"}
    extra = set(data) - allowed
    _require(not extra, f"cut plan has unsupported fields: {sorted(extra)}")
    _require(data.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION!r}")

    case_id = data.get("case_id")
    _require(isinstance(case_id, str) and bool(case_id.strip()), "case_id must be a non-empty string")
    case_id = case_id.strip()
    if expected_case_id is not None:
        _require(case_id == expected_case_id, f"case_id {case_id!r} does not match filename {expected_case_id!r}")

    ordered_videos = data.get("ordered_videos")
    _require(isinstance(ordered_videos, list) and ordered_videos, "ordered_videos must be a non-empty array")

    normalized_videos = []
    seen_source_ids = set()
    for video_index, video in enumerate(ordered_videos):
        location = f"ordered_videos[{video_index}]"
        _require(isinstance(video, dict), f"{location} must be an object")
        extra = set(video) - {"source_id", "segments"}
        _require(not extra, f"{location} has unsupported fields: {sorted(extra)}")

        source_id = video.get("source_id")
        _require(isinstance(source_id, str) and bool(source_id.strip()), f"{location}.source_id must be a non-empty string")
        source_id = source_id.strip()
        _require("/" not in source_id and "\\" not in source_id, f"{location}.source_id must not contain path separators")
        _require(source_id not in seen_source_ids, f"duplicate source_id: {source_id}")
        seen_source_ids.add(source_id)

        segments = video.get("segments")
        _require(isinstance(segments, list) and segments, f"{location}.segments must be a non-empty array")
        normalized_segments = [
            _validate_segment(segment, f"{location}.segments[{segment_index}]")
            for segment_index, segment in enumerate(segments)
        ]
        for previous, current in zip(normalized_segments, normalized_segments[1:]):
            _require(
                current["start_time"] >= previous["end_time"],
                f"{location}.segments must be chronological and non-overlapping",
            )

        normalized_videos.append({"source_id": source_id, "segments": normalized_segments})

    normalized = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "ordered_videos": normalized_videos,
    }
    if "metadata" in data:
        _require(isinstance(data["metadata"], dict), "metadata must be an object")
        normalized["metadata"] = data["metadata"]
    return normalized


def is_public_cut_plan(data: Any) -> bool:
    return isinstance(data, dict) and "schema_version" in data and "ordered_videos" in data


def _legacy_planned_videos(data: dict) -> list:
    try:
        cut_plan = data["get_arrange_result"]["arrange_result"]
    except (KeyError, TypeError) as exc:
        raise CutPlanValidationError(
            "input is neither a public cut plan nor a supported workflow output"
        ) from exc

    duration_adjustment = cut_plan.get("duration_adjustment")
    if duration_adjustment:
        planned_videos = duration_adjustment.get("adjusted_arrange_results")
    else:
        planned_videos = cut_plan.get("arrange_results")
    _require(isinstance(planned_videos, list) and planned_videos, "workflow output contains no planned videos")
    return planned_videos


def _legacy_understanding(data: dict) -> dict:
    for key in ("raw_understanding_result", "shuffled_understand_result", "understanding_results"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    raise CutPlanValidationError("workflow output contains no source-video mapping")


def convert_workflow_output(data: Any, case_id: str) -> dict:
    """Convert one legacy agent-workflow output into the public cut-plan format."""
    _require(isinstance(data, dict), "workflow output must be a JSON object")
    _require(isinstance(case_id, str) and bool(case_id.strip()), "case_id must be a non-empty string")
    planned_videos = _legacy_planned_videos(data)
    understanding = _legacy_understanding(data)

    ordered_videos = []
    for video_index, video in enumerate(planned_videos):
        location = f"workflow planned videos[{video_index}]"
        _require(isinstance(video, dict), f"{location} must be an object")
        video_id = video.get("video_id")
        _require(video_id in understanding, f"{location} references unknown video_id: {video_id!r}")
        video_info = understanding[video_id]
        _require(isinstance(video_info, dict), f"source mapping for {video_id!r} must be an object")
        source_path = video_info.get("video_name") or video_info.get("path")
        _require(source_path, f"source mapping for {video_id!r} has no video_name or path")

        segments = video.get("adjusted_segments") or video.get("segments")
        _require(isinstance(segments, list) and segments, f"{location} contains no segments")
        ordered_videos.append(
            {
                "source_id": _source_id_from_path(source_path),
                "segments": [
                    {
                        "start_time": segment.get("start_time", segment.get("highlight_start")),
                        "end_time": segment.get("end_time", segment.get("highlight_end")),
                    }
                    for segment in segments
                ],
            }
        )

    return validate_public_cut_plan(
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id.strip(),
            "ordered_videos": ordered_videos,
        },
        expected_case_id=case_id.strip(),
    )


def normalize_cut_plan(data: Any, expected_case_id: str) -> dict:
    """Normalize either the public schema or a legacy workflow output."""
    if is_public_cut_plan(data):
        return validate_public_cut_plan(data, expected_case_id=expected_case_id)
    legacy_case_id = expected_case_id
    if isinstance(data, dict) and data.get("entry"):
        legacy_case_id = _source_id_from_path(data["entry"])
    return convert_workflow_output(data, case_id=legacy_case_id)


def load_cut_plan(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return normalize_cut_plan(data, expected_case_id=path.stem)


def write_cut_plan(path: str | Path, cut_plan: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(cut_plan, file, ensure_ascii=False, indent=2)
        file.write("\n")
