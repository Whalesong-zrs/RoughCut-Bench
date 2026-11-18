import argparse
import base64
import hashlib
import copy
import json
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from visual_workflow.model_api import ModelRunner, VLMDataInspectionError
from visual_workflow.model_clients import CutPlanningLLM
from visual_workflow.prompt import cut_planning_prompt_v33
from visual_workflow.workflow_utils import (
    build_keyframes_by_video,
    build_processed_understanding,
    filter_keyframes_by_selected_segments,
    read_json,
    select_cut_planning_keyframe_segments,
    write_json,
)

cut_planning_prompt = cut_planning_prompt_v33


CUT_PLANNING_INPUT_MODES = ("time_desc_frame", "time_desc", "time_frame")


def _count_visual_segments(understand_result: dict) -> int:
    return sum(len(video_info.get("visual_segments", [])) for video_info in understand_result.values())


def _empty_cut_planning_keyframe_limit_info(model_name: str, total_segments: int):
    return {
        "enabled": False,
        "model_name": model_name,
        "max_keyframes": None,
        "total_segments": total_segments,
        "selected_segments": total_segments,
        "dropped_segments": 0,
        "selected_by_video": {},
        "dropped_by_video": {},
    }


def build_empty_keyframes_by_video(understand_result: dict):
    return {
        video_id: [None] * len(video_info.get("visual_segments", []))
        for video_id, video_info in understand_result.items()
    }


def build_cut_planning_input(understand_result: dict, input_mode: str):
    if input_mode not in CUT_PLANNING_INPUT_MODES:
        raise ValueError(f"unknown cut-planning input mode: {input_mode}")

    cut_planning_input = copy.deepcopy(understand_result)
    if input_mode == "time_frame":
        for video_info in cut_planning_input.values():
            for seg in video_info.get("visual_segments", []):
                if isinstance(seg, dict):
                    seg.pop("description", None)
    return cut_planning_input



def _is_qwen_model(model_name: str) -> bool:
    return str(model_name).startswith("qwen-")


def _is_qwen_data_inspection_error(model_name: str, ex: Exception) -> bool:
    return (
        _is_qwen_model(model_name)
        and isinstance(ex, VLMDataInspectionError)
    )


def _json_hash(data) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_filename(value) -> str:
    chars = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


def _write_debug_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _append_debug_jsonl(path: str, item):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _save_b64_jpeg(path: str, image_b64: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(image_b64))


def _qwen_safety_case_dir(out_dir: str, sample_key: str) -> str:
    return os.path.join(out_dir, "_qwen_safety", sample_key)


def _iter_cut_planning_keyframe_records(raw_understanding_result, cut_planning_input, keyframes_by_video):
    global_index = 0
    for video_id, video_info in cut_planning_input.items():
        raw_video_info = raw_understanding_result.get(video_id, {})
        visual_segments = video_info.get("visual_segments", [])
        frames = keyframes_by_video.get(video_id, [])
        for segment_index, seg in enumerate(visual_segments):
            img_b64 = frames[segment_index] if segment_index < len(frames) else None
            if img_b64 is None:
                continue
            start_time = float(seg.get("start_time", 0) or 0)
            end_time = float(seg.get("end_time", start_time) or start_time)
            yield {
                "global_index": global_index,
                "video_id": video_id,
                "segment_index": segment_index,
                "start_time": start_time,
                "end_time": end_time,
                "mid_time": round((start_time + end_time) / 2.0, 3),
                "duration": seg.get("duration"),
                "description": seg.get("description", ""),
                "video_name": raw_video_info.get("video_name", ""),
                "image_b64": img_b64,
            }
            global_index += 1


def _filter_keyframes_by_failed_records(keyframes_by_video, failed_records):
    filtered = copy.deepcopy(keyframes_by_video)
    failed_keys = {
        (item.get("video_id"), int(item.get("segment_index")))
        for item in failed_records
        if item.get("video_id") is not None and item.get("segment_index") is not None
    }
    for video_id, segment_index in failed_keys:
        frames = filtered.get(video_id)
        if frames is not None and 0 <= segment_index < len(frames):
            frames[segment_index] = None
    return filtered


def _probe_qwen_cut_planning_keyframe(model_name: str, record: dict):
    runner = ModelRunner()
    question = (
        "请判断这张视频关键帧是否可以正常用于视频剪辑理解。"
        "如果画面正常，请只回答 normal。"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{question}\n"
                        f"video_id={record['video_id']} segment_index={record['segment_index']} "
                        f"time={record['mid_time']}s"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{record['image_b64']}"},
                },
            ],
        }
    ]
    return runner.run(messages, model_name)


def _load_qwen_safety_cache(cache_path: str, expected_meta: dict):
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    metadata = data.get("metadata", {})
    for key, value in expected_meta.items():
        if metadata.get(key) != value:
            return None
    return data


def apply_qwen_frame_safety_filter(
    cfg,
    sample_key,
    model_name,
    understanding_file,
    raw_understanding_result,
    cut_planning_input,
    keyframes_by_video,
    input_mode,
):
    records = list(
        _iter_cut_planning_keyframe_records(
            raw_understanding_result,
            cut_planning_input,
            keyframes_by_video,
        )
    )
    safety_dir = _qwen_safety_case_dir(cfg["out_dir"], sample_key)
    failed_frames_dir = os.path.join(safety_dir, "frames", "failed")
    cache_path = os.path.join(safety_dir, "probe_results.json")
    failed_jsonl = os.path.join(safety_dir, "failed_frames.jsonl")

    expected_meta = {
        "model_name": model_name,
        "sample_key": sample_key,
        "cut_planning_input_mode": input_mode,
        "understanding_hash": _json_hash(understanding_file),
        "keyframe_count": len(records),
        "probe_policy": "data_inspection_as_frame_v2",
    }

    cached = _load_qwen_safety_cache(cache_path, expected_meta)
    if cached is not None:
        failed_records = cached.get("failed_frames", [])
        info = {
            "triggered": True,
            "cache_hit": True,
            "cache_path": cache_path,
            "probe_frame_count": len(records),
            "filtered_frame_count": len(failed_records),
            "failed_frames": failed_records,
            "probe_errors": cached.get("probe_errors", []),
        }
        return _filter_keyframes_by_failed_records(keyframes_by_video, failed_records), info

    if os.path.exists(failed_jsonl):
        os.remove(failed_jsonl)
    if os.path.isdir(failed_frames_dir):
        for name in os.listdir(failed_frames_dir):
            path = os.path.join(failed_frames_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    failed_records = []
    probe_errors = []
    probe_records = []
    os.makedirs(failed_frames_dir, exist_ok=True)

    print(f"[QWEN_SAFETY] probing cut-planning keyframes sample={sample_key} count={len(records)}")
    for record in records:
        record_for_log = {k: v for k, v in record.items() if k != "image_b64"}
        try:
            _probe_qwen_cut_planning_keyframe(model_name, record)
            row = {**record_for_log, "ok": True}
            probe_records.append(row)
        except VLMDataInspectionError as ex:
            row = {
                **record_for_log,
                "ok": False,
                "error_type": type(ex).__name__,
                "inspection_type": getattr(ex, "inspection_type", "unknown"),
                "error_message": str(ex),
                "debug_path": getattr(ex, "debug_path", None),
            }
            probe_records.append(row)
            stem = (
                f"{record['global_index']:04d}_"
                f"{_safe_filename(record['video_id'])}_seg{int(record['segment_index']):03d}_"
                f"{record['mid_time']}s.jpg"
            )
            saved_frame = os.path.join(failed_frames_dir, stem)
            _save_b64_jpeg(saved_frame, record["image_b64"])
            failed_record = {**row, "saved_frame": saved_frame}
            failed_records.append(failed_record)
            _append_debug_jsonl(failed_jsonl, failed_record)
            print(
                f"[QWEN_SAFETY] flagged frame sample={sample_key} "
                f"idx={record['global_index']} {record['video_id']} seg={record['segment_index']} "
                f"t={record['mid_time']}s inspection={getattr(ex, 'inspection_type', 'unknown')} "
                f"saved={saved_frame}"
            )
        except Exception as ex:
            row = {
                **record_for_log,
                "ok": False,
                "error_type": type(ex).__name__,
                "inspection_type": "probe_error",
                "error_message": str(ex),
            }
            probe_records.append(row)
            probe_errors.append(row)
            print(
                f"[QWEN_SAFETY] probe error sample={sample_key} "
                f"idx={record['global_index']} err={type(ex).__name__}: {ex}"
            )

    cache_data = {
        "metadata": expected_meta,
        "failed_frames": failed_records,
        "probe_errors": probe_errors,
        "probe_records": probe_records,
    }
    _write_debug_json(cache_path, cache_data)

    info = {
        "triggered": True,
        "cache_hit": False,
        "cache_path": cache_path,
        "probe_frame_count": len(records),
        "filtered_frame_count": len(failed_records),
        "failed_frames": failed_records,
        "probe_errors": probe_errors,
    }
    return _filter_keyframes_by_failed_records(keyframes_by_video, failed_records), info


def run_one_entry(entry_filename: str, cfg: dict):
    entry = entry_filename
    out_path = os.path.join(cfg["out_dir"], entry)
    tools = cfg["tools"]
    model_name_by_tool = cfg["model_name_by_tool"]
    cut_planning_input_mode = cfg.get("cut_planning_input_mode", "time_desc_frame")
    if cut_planning_input_mode not in CUT_PLANNING_INPUT_MODES:
        raise ValueError(f"unknown cut-planning input mode: {cut_planning_input_mode}")

    understanding_path = os.path.join(cfg["understanding_json_dir"], entry)

    sample_key = entry.replace('.json', '')
    gt_for_sample = cfg.get("_gt", {}).get(sample_key, {})
    if gt_for_sample:
        first_inner = next(iter(gt_for_sample.values()))
        user_prompt = first_inner.get("prompt", "")
    else:
        user_prompt = ""

    understanding_file = read_json(understanding_path)

    outputs = {
        "entry": entry,
        "tools": cfg["tools"],
        "model_name_by_tool": cfg["model_name_by_tool"],
        "cut_planning_input_mode": cut_planning_input_mode,
    }


    try:
        raw_understanding_result = build_processed_understanding(understanding_file, None, delete_asr_prob=0.85)
        cut_planning_model_name = model_name_by_tool.get("get_arrange_result", "")
        cut_planning_uses_keyframes = cut_planning_input_mode != "time_desc"

        if cut_planning_uses_keyframes:
            selected_segments_by_video, keyframe_limit_info = select_cut_planning_keyframe_segments(
                raw_understanding_result,
                cut_planning_model_name,
                max_keyframes=120,
            )
        else:
            selected_segments_by_video = None
            keyframe_limit_info = _empty_cut_planning_keyframe_limit_info(
                cut_planning_model_name,
                _count_visual_segments(raw_understanding_result),
            )

        if not cut_planning_uses_keyframes:
            keyframes_by_video_raw = build_empty_keyframes_by_video(raw_understanding_result)
        elif keyframe_limit_info["enabled"]:
            keyframes_by_video_raw = build_keyframes_by_video(raw_understanding_result, selected_segments_by_video)
        else:
            keyframes_by_video_raw = build_keyframes_by_video(raw_understanding_result)

    except Exception as ex:
        print(
            f"[FAIL] {sample_key} could not prepare Cut-Planning Agent inputs: "
            f"{type(ex).__name__}: {ex}"
        )
        return 0


    outputs["user_prompt"] = user_prompt

    keyframes_by_video  = keyframes_by_video_raw
    outputs.update(
        {
            "raw_understanding_result": raw_understanding_result,
            "cut_planning_keyframe_limit_info": keyframe_limit_info,
        }
    )

    understand_result_del_video_name = copy.deepcopy(raw_understanding_result)
    for _vid, _info in understand_result_del_video_name.items():
        if isinstance(_info, dict) and "video_name" in _info:
            del _info["video_name"]

    cut_plan = None
 
    
    for tool in tools:
        model_name = model_name_by_tool[tool]

        if tool == "get_arrange_result":
            cut_planning_llm = CutPlanningLLM(
                model_name,
                cut_planning_prompt,
                input_mode=cut_planning_input_mode,
            )
            ok = False
            qwen_frame_safety_filter_info = None
            for attempt in range(5):
                cut_planning_input = build_cut_planning_input(
                    understand_result_del_video_name,
                    cut_planning_input_mode,
                )
                if cut_planning_input_mode == "time_desc":
                    cut_planning_keyframes_by_video = build_empty_keyframes_by_video(cut_planning_input)
                else:
                    cut_planning_keyframes_by_video = filter_keyframes_by_selected_segments(
                        keyframes_by_video,
                        selected_segments_by_video,
                    )
                try:
                    cut_plan = cut_planning_llm.run(
                        cut_planning_input,
                        user_prompt,
                        cut_planning_keyframes_by_video,
                    )
                    if not cut_plan:
                        raise ValueError(f"empty cut plan for {entry}")
                    ok = True
                    break
                except VLMDataInspectionError as ex:
                    if _is_qwen_data_inspection_error(model_name, ex) and cut_planning_input_mode != "time_desc":
                        print(
                            f"[QWEN_SAFETY] cut-planning data inspection failed "
                            f"sample={sample_key} reported_type={getattr(ex, 'inspection_type', 'unknown')}; "
                            f"probing keyframes"
                        )
                        filtered_keyframes_by_video, qwen_frame_safety_filter_info = apply_qwen_frame_safety_filter(
                            cfg,
                            sample_key,
                            model_name,
                            understanding_file,
                            raw_understanding_result,
                            cut_planning_input,
                            cut_planning_keyframes_by_video,
                            cut_planning_input_mode,
                        )
                        qwen_frame_safety_filter_info["trigger_error_type"] = type(ex).__name__
                        qwen_frame_safety_filter_info["trigger_error_message"] = str(ex)
                        qwen_frame_safety_filter_info["trigger_debug_path"] = getattr(ex, "debug_path", None)

                        if qwen_frame_safety_filter_info.get("filtered_frame_count", 0) <= 0:
                            print(f"[QWEN_SAFETY] no single-frame data inspection failures found for sample={sample_key}; cannot auto-repair")
                            return 0

                        try:
                            cut_plan = cut_planning_llm.run(
                                cut_planning_input,
                                user_prompt,
                                filtered_keyframes_by_video,
                            )
                            if not cut_plan:
                                raise ValueError(f"empty cut plan after Qwen frame filter for {entry}")
                            qwen_frame_safety_filter_info["repair_ok"] = True
                            ok = True
                            break
                        except Exception as repair_ex:
                            qwen_frame_safety_filter_info["repair_ok"] = False
                            qwen_frame_safety_filter_info["repair_error_type"] = type(repair_ex).__name__
                            qwen_frame_safety_filter_info["repair_error_message"] = str(repair_ex)
                            print(f"[QWEN_SAFETY] filtered cut-planning retry failed sample={sample_key} err={type(repair_ex).__name__}: {repair_ex}")
                            return 0

                    print(
                        f"[SKIP] {entry} tool={tool} data_inspection "
                        f"type={getattr(ex, 'inspection_type', 'unknown')} err={ex}"
                    )
                    return 0
                except Exception as ex:
                    if attempt >= 4:
                        print(f"[SKIP] {entry} tool={tool} retries=5 err={ex}")
                        return 0
            if not ok:
                return 0

            gemini_expected_duration = None
            final_duration = 0.0
            predict_sort_answer = []
            try:
                if cut_plan['duration_adjustment']:
                    planned_videos = cut_plan['duration_adjustment']['adjusted_arrange_results']
                    gemini_expected_duration = cut_plan['duration_adjustment'].get("expected_duration")
                else:
                    planned_videos = cut_plan["arrange_results"]

                for res in planned_videos:
                    predict_sort_answer.append(res["video_id"])
                    segments = res["adjusted_segments"] if "adjusted_segments" in res else res["segments"]
                    for seg in segments:
                        final_duration += seg["end_time"] - seg["start_time"]
                final_duration = round(final_duration, 1)
            except Exception:
                pass

            outputs[tool] = {
                "model_name": model_name,
                "cut_planning_input_mode": cut_planning_input_mode,
                "gemini_expected_duration": gemini_expected_duration,
                "final_duration": final_duration,
                "predict_sort_answer": predict_sort_answer,
                "qwen_frame_safety_filter_info": qwen_frame_safety_filter_info,
                "arrange_result": copy.deepcopy(cut_plan),
            }
            if qwen_frame_safety_filter_info is not None:
                outputs["qwen_frame_safety_filter_info"] = qwen_frame_safety_filter_info
            continue
        raise KeyError(f"unknown tool: {tool}")

    write_json(out_path, outputs)
    return 1



def make_cut_planning_status(sample_key, ok, error_type=None, error_message=None, skipped_existing=False):
    return {
        "sample_key": sample_key,
        "ok": ok,
        "error_type": error_type,
        "error_message": error_message,
        "skipped_existing": skipped_existing,
    }


def invalid_case(sample_key, error_type, error_message, rerun=True):
    status = make_cut_planning_status(sample_key, False, error_type, error_message)
    status["rerun"] = rerun
    return status


def video_name_from_path(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def _mapping_video_name(mapping, video_id):
    video_info = mapping[video_id]
    if isinstance(video_info, str):
        return video_name_from_path(video_info)
    if isinstance(video_info, dict):
        video_name = video_info.get("video_name") or video_info.get("path")
        if video_name:
            return video_name_from_path(video_name)
    raise KeyError(video_id)


def _cut_plan_result_dict(file_content):
    cut_plan = file_content["get_arrange_result"]["arrange_result"]
    if not isinstance(cut_plan, dict):
        raise TypeError("get_arrange_result.arrange_result must be a dict")
    return cut_plan


def _planned_videos_for_metrics(cut_plan):
    duration_adjustment = cut_plan.get("duration_adjustment")
    if duration_adjustment:
        adjusted = duration_adjustment.get("adjusted_arrange_results")
        if not isinstance(adjusted, list):
            raise KeyError("invalid duration_adjustment.adjusted_arrange_results")
        return adjusted
    planned_videos = cut_plan.get("arrange_results")
    if not isinstance(planned_videos, list):
        raise KeyError("missing arrange_results")
    return planned_videos


def _segments_for_metrics(video_result):
    segments = video_result.get("adjusted_segments") or video_result.get("segments", [])
    if not isinstance(segments, list):
        raise TypeError("segments must be a list")
    return segments


def _validate_segment(segment, video_id):
    if not isinstance(segment, dict):
        raise TypeError(f"invalid segment for {video_id}")
    start = segment.get("start_time", segment.get("highlight_start"))
    end = segment.get("end_time", segment.get("highlight_end"))
    if start is None or end is None:
        raise KeyError(f"missing start/end time for {video_id}")
    start = float(start)
    end = float(end)
    if end < start:
        raise ValueError(f"invalid segment time range for {video_id}: {start}-{end}")


def resolve_understanding_mapping(file_content):
    if "raw_understanding_result" in file_content:
        return file_content["raw_understanding_result"]
    if "understanding_results" in file_content:
        return file_content["understanding_results"]
    if "shuffled_understand_result" in file_content:
        return file_content["shuffled_understand_result"]
    raise KeyError("raw_understanding_result/understanding_results/shuffled_understand_result")


def validate_cut_planning_case_output(out_dir, sample_key, gt_for_sample):
    json_path = os.path.join(out_dir, sample_key + ".json")
    if not os.path.exists(json_path):
        return invalid_case(sample_key, "missing_case", "case output json is missing")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            file_content = json.load(f)
    except Exception as ex:
        return invalid_case(sample_key, "malformed_json", f"{type(ex).__name__}: {ex}")

    try:
        cut_plan = _cut_plan_result_dict(file_content)
        planned_videos = _planned_videos_for_metrics(cut_plan)
        video_mapping = resolve_understanding_mapping(file_content)
        if not isinstance(video_mapping, dict) or not video_mapping:
            raise TypeError("understanding mapping must be a non-empty dict")

        gt_video_names = set((gt_for_sample or {}).keys())
        seen_gt_video_names = set()
        seen_video_ids = []
        for video_result in planned_videos:
            if not isinstance(video_result, dict):
                raise TypeError("cut-plan video item must be a dict")
            video_id = video_result.get("video_id")
            if video_id is None:
                raise KeyError("cut-plan video item missing video_id")
            seen_video_ids.append(video_id)
            video_name = _mapping_video_name(video_mapping, video_id)
            if gt_video_names and video_name not in gt_video_names:
                raise KeyError(f"extra video predicted: {video_id} -> {video_name}")
            segments = _segments_for_metrics(video_result)
            if not segments:
                raise ValueError(f"empty segments for {video_id}")
            for segment in segments:
                _validate_segment(segment, video_id)
            if gt_video_names:
                seen_gt_video_names.add(video_name)

        duplicate_video_ids = sorted({vid for vid in seen_video_ids if seen_video_ids.count(vid) > 1})
        if duplicate_video_ids:
            raise KeyError(f"duplicate cut-plan video_id: {duplicate_video_ids}")

        if gt_video_names:
            missing_gt_videos = sorted(gt_video_names - seen_gt_video_names)
            if missing_gt_videos:
                raise KeyError(f"missing gt videos: {missing_gt_videos}")

        duration_adjustment = cut_plan.get("duration_adjustment")
        if isinstance(duration_adjustment, dict) and duration_adjustment:
            adjusted = duration_adjustment.get("adjusted_arrange_results")
            if not isinstance(adjusted, list):
                raise KeyError("invalid duration_adjustment.adjusted_arrange_results")
    except Exception as ex:
        return invalid_case(sample_key, "invalid_case_output", f"{type(ex).__name__}: {ex}")

    return make_cut_planning_status(sample_key, True)


def validate_cut_planning_output_dir(cfg, sample_keys):
    expected = set(sample_keys)
    invalid_items = []
    json_files = [name for name in os.listdir(cfg["out_dir"]) if name.endswith(".json")]
    for file_name in sorted(json_files):
        sample_key = file_name[:-5]
        if sample_key not in expected:
            invalid_items.append(
                invalid_case(sample_key, "extra_case", "case output json is not in gt", rerun=False)
            )

    for sample_key in sample_keys:
        invalid = validate_cut_planning_case_output(
            cfg["out_dir"],
            sample_key,
            cfg.get("_gt", {}).get(sample_key, {}),
        )
        if not invalid.get("ok"):
            invalid_items.append(invalid)
    return invalid_items


def remove_case_artifacts(out_dir, sample_key):
    json_path = os.path.join(out_dir, sample_key + ".json")
    if os.path.exists(json_path):
        os.remove(json_path)


def clear_repair_logs(out_dir):
    failed_dir = os.path.join(out_dir, "_failed")
    for name in ["cut_planning_failed.jsonl", "cut_planning_unresolved_after_repair.jsonl"]:
        path = os.path.join(failed_dir, name)
        if os.path.exists(path):
            os.remove(path)


def write_cut_planning_failed_log(out_dir, failed_items):
    if not failed_items:
        return None
    failed_dir = os.path.join(out_dir, "_failed")
    os.makedirs(failed_dir, exist_ok=True)
    failed_path = os.path.join(failed_dir, "cut_planning_failed.jsonl")
    with open(failed_path, "w", encoding="utf-8") as f:
        for item in failed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return failed_path


def write_unresolved_repair_log(out_dir, invalid_items):
    failed_dir = os.path.join(out_dir, "_failed")
    os.makedirs(failed_dir, exist_ok=True)
    failed_path = os.path.join(failed_dir, "cut_planning_unresolved_after_repair.jsonl")
    with open(failed_path, "w", encoding="utf-8") as f:
        for item in invalid_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return failed_path


def run_one_entry_status(entry_filename, cfg):
    sample_key = entry_filename[:-5] if entry_filename.endswith(".json") else entry_filename
    try:
        out_path = os.path.join(cfg["out_dir"], sample_key + ".json")
        if cfg.get("skip_existing", True) and os.path.exists(out_path):
            return make_cut_planning_status(sample_key, True, skipped_existing=True)
        ok = run_one_entry(sample_key + ".json", cfg)
        if ok:
            return make_cut_planning_status(sample_key, True)
        return make_cut_planning_status(
            sample_key,
            False,
            "cut_planning_run_failed",
            "run_one_entry returned 0",
        )
    except Exception as ex:
        traceback.print_exc()
        return make_cut_planning_status(
            sample_key,
            False,
            "unexpected_worker_exception",
            f"{type(ex).__name__}: {ex}",
        )


def run_entry_batch(entries, cfg, workers):
    results = []
    if workers <= 1:
        for entry in tqdm(entries):
            results.append(run_one_entry_status(entry, cfg))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for r in tqdm(executor.map(run_one_entry_status, entries, [cfg] * len(entries)), total=len(entries)):
                if r is None:
                    r = make_cut_planning_status(
                        "<unknown>",
                        False,
                        "unknown_worker_failure",
                        "worker returned None",
                    )
                results.append(r)

    failed = [item for item in results if not item.get("ok")]
    ok_total = len(results) - len(failed)
    failed_path = write_cut_planning_failed_log(cfg["out_dir"], failed)
    if failed_path:
        print(f"[FAIL] wrote cut-planning failed log: {failed_path}")
    print(f"done ok={ok_total} failed={len(failed)} total={len(entries)} workers={workers}")
    return results, failed


def run_repair_loop(cfg, sample_keys, workers, max_repair_rounds):
    invalid_items = validate_cut_planning_output_dir(cfg, sample_keys)
    if not invalid_items:
        clear_repair_logs(cfg["out_dir"])
        print("[REPAIR] all cut-planning cases already valid")
        return []

    for round_idx in range(1, max_repair_rounds + 1):
        preview = ", ".join(item["sample_key"] for item in invalid_items[:10])
        suffix = "" if len(invalid_items) <= 10 else f", ... total={len(invalid_items)}"
        print(f"[REPAIR] cut-planning round={round_idx}/{max_repair_rounds} invalid={len(invalid_items)} ({preview}{suffix})")

        rerun_entries = []
        for item in invalid_items:
            sample_key = item["sample_key"]
            remove_case_artifacts(cfg["out_dir"], sample_key)
            if item.get("rerun", True) and sample_key in sample_keys:
                rerun_entries.append(sample_key + ".json")

        if rerun_entries:
            run_entry_batch(sorted(set(rerun_entries)), cfg, workers)

        invalid_items = validate_cut_planning_output_dir(cfg, sample_keys)
        if not invalid_items:
            clear_repair_logs(cfg["out_dir"])
            print(f"[REPAIR] cut planning passed after round {round_idx}")
            return []

    failed_path = write_unresolved_repair_log(cfg["out_dir"], invalid_items)
    print(f"[REPAIR] unresolved cut-planning cases after {max_repair_rounds} rounds: {failed_path}")
    return invalid_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemini-2.5-pro',
                        help='one of the aliases in visual_workflow/model_api.py MODEL_REGISTRY')
    parser.add_argument('--video-understanding-dirs', nargs='+', required=True,
                        help='Video-Understanding Agent output directories, one per set')
    parser.add_argument('--out-dirs', nargs='+', required=True,
                        help='where to write cut-plan JSON, one per set')
    parser.add_argument('--gt-paths', nargs='+', required=True,
                        help='ground-truth JSON paths, one per set')
    parser.add_argument('--cut-planning-prompt-version', default='v33',
                        help='prompt version defined in visual_workflow/prompt/cut_planning_prompt.py (default: v33)')
    parser.add_argument('--cut-planning-input-mode', default='time_desc_frame', choices=CUT_PLANNING_INPUT_MODES,
                        help='Cut-Planning Agent input ablation: time_desc_frame, time_desc, or time_frame')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of worker processes for cut-planning cases in each set')
    parser.add_argument('--repair-invalid', action='store_true',
                        help='repair invalid or missing cut plans by deleting invalid case JSON and rerunning it')
    parser.add_argument('--max-repair-rounds', type=int, default=5,
                        help='maximum repair rounds per output directory when --repair-invalid is enabled')
    args = parser.parse_args()
    assert len(args.video_understanding_dirs) == len(args.out_dirs) == len(args.gt_paths), \
        f'--video-understanding-dirs ({len(args.video_understanding_dirs)}), --out-dirs ({len(args.out_dirs)}), and --gt-paths ({len(args.gt_paths)}) must all have the same length'
    if args.max_repair_rounds < 1:
        raise SystemExit('--max-repair-rounds must be >= 1')
    MODEL_NAME = args.model
    print(f'=== Using cut-planning model: {MODEL_NAME} ===')
    print(f'=== Using cut-planning input mode: {args.cut_planning_input_mode} ===')

    global cut_planning_prompt
    _ap_name = f'cut_planning_prompt_{args.cut_planning_prompt_version}'
    try:
        cut_planning_prompt = globals()[_ap_name]
    except KeyError as ex:
        raise SystemExit(f'Unknown Cut-Planning Agent prompt: {_ap_name}.') from ex
    print(f'=== Using cut-planning prompt: {_ap_name} ===')

    EVAL_TARGETS = [
        {
            "understanding_json_dir": ud.rstrip('/'),
            "out_dir":                od.rstrip('/'),
            "label_path":             gt,
        }
        for ud, od, gt in zip(args.video_understanding_dirs, args.out_dirs, args.gt_paths)
    ]

    BASE_CFG = {
        "workers": int(args.workers),
        "skip_existing": True,
        "tools": ["get_arrange_result"],
        "model_name_by_tool": {
            "get_arrange_result": MODEL_NAME,
        },
        "cut_planning_input_mode": args.cut_planning_input_mode,
    }

    all_failed = []
    for target in EVAL_TARGETS:
        print(f'\n=== Processing: {target["label_path"]} ===')
        cfg = {**BASE_CFG, **target}

        if cfg.get("label_path"):
            cfg["_gt"] = read_json(cfg["label_path"])
        else:
            cfg["_gt"] = {}

        os.makedirs(cfg["out_dir"], exist_ok=True)
        workers = int(cfg["workers"])

        if args.repair_invalid:
            sample_keys = sorted(cfg["_gt"].keys())
            print(f"sample count: {len(sample_keys)}")
            failed = run_repair_loop(cfg, sample_keys, workers, args.max_repair_rounds)
            all_failed.extend(failed)
            continue

        common = set(os.listdir(cfg["understanding_json_dir"]))
        entries = sorted([x for x in common if x.endswith(".json")])

        if cfg["skip_existing"]:
            entries = [e for e in entries if not os.path.exists(os.path.join(cfg["out_dir"], e))]
        print(len(entries))
        run_entry_batch(entries, cfg, workers)

    if all_failed:
        raise SystemExit(f'cut-planning unresolved invalid cases: {len(all_failed)}')


if __name__ == "__main__":
    main()
