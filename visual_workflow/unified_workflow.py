"""Unified visual workflow used in the workflow-design ablation.

The workflow processes the sampled frames from all source videos in one model
request and jointly performs video understanding and cut planning. Its output
uses the same cut-plan schema as the main visual workflow.

输出目录约定: <video_dir>_output/output_<MODEL>_unified_workflow/cut_plans/<sample>.json
"""

import argparse
import json
import os
import sys
import traceback
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

from visual_workflow.workflow_utils import (
    read_json,
    write_json,
    parallel_extract_video_frames,
    my_encode_img,
)
from visual_workflow.model_clients import UnifiedWorkflowLLM, LLMJsonParseError
from visual_workflow.prompt import unified_workflow_prompt_v18


unified_workflow_prompt = unified_workflow_prompt_v18


_VALID_EXT = {"mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png", "heic"}


def _is_video_or_image(filename: str) -> bool:
    if filename.startswith("."):
        return False
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return ext in _VALID_EXT


def make_status(sample_key, ok, error_type=None, error_message=None, attempt=None, frames_by_video=None, video_paths=None, skipped_existing=False, attempt_errors=None):
    if frames_by_video is not None:
        video_names = [v.get("video_name") for v in frames_by_video.values()]
    elif video_paths is not None:
        video_names = list(video_paths)
    else:
        video_names = []
    return {
        "sample_key": sample_key,
        "ok": ok,
        "skipped_existing": skipped_existing,
        "error_type": error_type,
        "error_message": error_message,
        "attempt": attempt,
        "video_count": len(video_names),
        "video_names": video_names,
        "attempt_errors": attempt_errors or [],
    }


def classify_unified_exception(ex):
    if isinstance(ex, (json.JSONDecodeError, LLMJsonParseError)):
        return "llm_json_parse_failed"
    msg = str(ex)
    low = msg.lower()
    if "unified output" in msg or "missing videos" in msg or "unknown videos" in msg or "duplicate videos" in msg:
        return "video_id_validation_failed"
    if "timeout" in low or "timed out" in low or "rate" in low or "transport" in low:
        return "api_or_timeout_failed"
    return "unified_llm_failed"


def write_failed_log(out_dir, failed_items):
    if not failed_items:
        return None
    failed_dir = os.path.join(out_dir, "_failed")
    os.makedirs(failed_dir, exist_ok=True)
    failed_path = os.path.join(failed_dir, "unified_failed.jsonl")
    with open(failed_path, "w", encoding="utf-8") as f:
        for item in failed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return failed_path


def write_raw_response_debug(out_dir, sample_key, attempt, ex):
    raw_response = getattr(ex, "raw_response", None)
    normalized_response = getattr(ex, "normalized_response", None)
    if raw_response is None and normalized_response is None:
        return {}
    raw_dir = os.path.join(out_dir, "_failed_raw")
    os.makedirs(raw_dir, exist_ok=True)
    attempt_label = "final" if attempt is None else f"attempt{attempt}"
    prefix = os.path.join(raw_dir, f"{sample_key}_{attempt_label}")
    paths = {}
    if raw_response is not None:
        raw_path = prefix + "_raw.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_response)
        paths["raw_response_path"] = raw_path
        paths["raw_response_chars"] = len(raw_response)
    if normalized_response is not None:
        normalized_path = prefix + "_normalized.txt"
        with open(normalized_path, "w", encoding="utf-8") as f:
            f.write(normalized_response)
        paths["normalized_response_path"] = normalized_path
        paths["normalized_response_chars"] = len(normalized_response)
    return paths


def list_video_paths(video_dir: str, sample_key: str):
    base = os.path.join(video_dir, sample_key)
    if not os.path.isdir(base):
        return None
    video_files = [n for n in sorted(os.listdir(base)) if _is_video_or_image(n)]
    return [os.path.join(base, n) for n in video_files]


def build_frames_from_video_paths(video_paths, sample_fps: int = 1, target_short_edge: int = 224) -> "OrderedDict":
    pil_collects, ts_collects = parallel_extract_video_frames(video_paths)

    frames_by_video: "OrderedDict[str, dict]" = OrderedDict()
    for idx, (pil_list, timestamps, vpath) in enumerate(zip(pil_collects, ts_collects, video_paths)):
        resized = []
        for img in pil_list:
            w, h = img.size
            if w < h:
                nw = target_short_edge
                nh = int(round(h / w * target_short_edge))
            else:
                nh = target_short_edge
                nw = int(round(w / h * target_short_edge))
            resized.append(img.resize((nw, nh), Image.Resampling.LANCZOS))
        encoded = [(round(float(t), 1), my_encode_img(im)) for t, im in zip(timestamps, resized)]
        duration = float(timestamps[-1]) if len(timestamps) > 0 else 0.0
        frames_by_video[f"video_{idx}"] = {
            "video_name": vpath,
            "duration": duration,
            "frames": encoded,
        }
    return frames_by_video


def build_frames_by_video(video_dir: str, sample_key: str, sample_fps: int = 1, target_short_edge: int = 224) -> "OrderedDict | None":
    """扫 case 文件夹, 对每个视频抽 sample_fps 帧, resize 后 base64 编码, 返回 OrderedDict.

    返回结构跟 UnifiedWorkflowLLM.run 的 frames_by_video 形参一致.
    video_id 编号顺序由 sorted(os.listdir(...)) 决定, 跨机一致.
    """
    video_paths = list_video_paths(video_dir, sample_key)
    if not video_paths:
        return None
    return build_frames_from_video_paths(video_paths, sample_fps=sample_fps, target_short_edge=target_short_edge)



def run_one_entry(sample_key: str, cfg: dict) -> dict:
    out_path = os.path.join(cfg["out_dir"], sample_key + ".json")
    if cfg.get("skip_existing", True) and os.path.exists(out_path):
        return make_status(sample_key, True, skipped_existing=True)

    gt_for_sample = cfg.get("_gt", {}).get(sample_key, {})
    if gt_for_sample:
        first_inner = next(iter(gt_for_sample.values()))
        user_prompt = first_inner.get("prompt", "")
    else:
        user_prompt = ""

    video_paths = list_video_paths(cfg["video_dir"], sample_key)
    if video_paths is None:
        print(f"[FAIL] {sample_key} missing sample directory")
        return make_status(sample_key, False, "missing_sample_dir", "sample directory does not exist")
    if not video_paths:
        print(f"[FAIL] {sample_key} no valid videos under sample")
        return make_status(sample_key, False, "no_valid_videos", "no valid videos under sample", video_paths=[])

    try:
        frames_by_video = build_frames_from_video_paths(video_paths)
    except Exception as ex:
        print(f"[FAIL] {sample_key} frame-extract err={type(ex).__name__}: {ex}")
        return make_status(sample_key, False, "frame_extract_failed", f"{type(ex).__name__}: {ex}", video_paths=video_paths)
    if not frames_by_video:
        print(f"[FAIL] {sample_key} empty frames_by_video")
        return make_status(sample_key, False, "empty_frames", "frames_by_video is empty", video_paths=video_paths)

    unified_llm = UnifiedWorkflowLLM(cfg["model_name"], unified_workflow_prompt)
    try:
        cut_plan = unified_llm.run(frames_by_video, user_prompt)
        if not cut_plan:
            raise ValueError(f"empty unified output for {sample_key}")
    except Exception as ex:
        error_type = classify_unified_exception(ex)
        raw_debug = write_raw_response_debug(cfg["out_dir"], sample_key, None, ex)
        attempt_error = {
            "attempt": None,
            "error_type": error_type,
            "error_message": f"{type(ex).__name__}: {ex}",
        }
        attempt_error.update(raw_debug)
        print(f"[FAIL] {sample_key} unified failed after internal retry err={type(ex).__name__}: {ex}")
        if raw_debug:
            print(f"[FAIL] {sample_key} raw response debug: {raw_debug}")
        return make_status(
            sample_key,
            False,
            error_type,
            f"{type(ex).__name__}: {ex}",
            attempt=None,
            frames_by_video=frames_by_video,
            attempt_errors=[attempt_error],
        )

    understanding_results = None
    if isinstance(cut_plan, dict):
        understanding_results = cut_plan.pop("understanding_results", None)

    outputs = {
        "entry": sample_key + ".json",
        "tools": ["get_arrange_result"],
        "model_name_by_tool": {"get_arrange_result": cfg["model_name"]},
        "user_prompt": user_prompt,
        "chain": "unified_workflow",
    }
    if understanding_results is not None:
        outputs["understanding_results"] = understanding_results
    outputs["get_arrange_result"] = {
        "model_name": cfg["model_name"],
        "arrange_result": cut_plan,
        "gemini_expected_duration": None,
        "final_duration": 0.0,
        "predict_sort_answer": [],
    }
    outputs["shuffled_understand_result"] = {
        vid: {"video_name": v["video_name"], "duration": v["duration"]}
        for vid, v in frames_by_video.items()
    }
    try:
        write_json(out_path, outputs)
    except Exception as ex:
        print(f"[FAIL] {sample_key} write-output err={type(ex).__name__}: {ex}")
        return make_status(sample_key, False, "write_output_failed", f"{type(ex).__name__}: {ex}", frames_by_video=frames_by_video)
    return make_status(sample_key, True, frames_by_video=frames_by_video)




def invalid_case(sample_key, error_type, error_message, rerun=True):
    status = make_status(sample_key, False, error_type, error_message)
    status["rerun"] = rerun
    return status


def video_name_from_path(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def _validate_video_id_list(stage_name, video_ids, expected_video_ids):
    if not isinstance(video_ids, list):
        raise TypeError(f"{stage_name} must be a list")
    missing = [vid for vid in expected_video_ids if vid not in video_ids]
    extra = [vid for vid in video_ids if vid not in expected_video_ids]
    duplicate = sorted({vid for vid in video_ids if video_ids.count(vid) > 1})
    if missing:
        raise KeyError(f"{stage_name} missing videos: {missing}")
    if extra:
        raise KeyError(f"{stage_name} has unknown videos: {extra}")
    if duplicate:
        raise KeyError(f"{stage_name} has duplicate videos: {duplicate}")


def _mapping_video_name(mapping, video_id):
    video_info = mapping[video_id]
    if isinstance(video_info, str):
        return video_name_from_path(video_info)
    if isinstance(video_info, dict):
        video_name = video_info.get("video_name") or video_info.get("path")
        if video_name:
            return video_name_from_path(video_name)
    raise KeyError(video_id)


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


def _understanding_result_items(file_content, cut_plan):
    if "understanding_results" in file_content:
        understanding_results = file_content["understanding_results"]
    elif "understanding_results" in cut_plan:
        understanding_results = cut_plan["understanding_results"]
    else:
        raise KeyError("missing understanding_results")

    if isinstance(understanding_results, dict):
        return [
            {"video_id": video_id, **video_result}
            for video_id, video_result in understanding_results.items()
            if isinstance(video_result, dict)
        ]
    if isinstance(understanding_results, list):
        return understanding_results
    raise TypeError(f"unsupported understanding_results type: {type(understanding_results)}")


def validate_unified_case_output(out_dir, sample_key, gt_for_sample, require_explicit_understanding=False):
    json_path = os.path.join(out_dir, sample_key + ".json")
    if not os.path.exists(json_path):
        return invalid_case(sample_key, "missing_case", "case output json is missing")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            file_content = json.load(f)
    except Exception as ex:
        return invalid_case(sample_key, "malformed_json", f"{type(ex).__name__}: {ex}")

    try:
        cut_plan = file_content["get_arrange_result"]["arrange_result"]
        if not isinstance(cut_plan, dict):
            raise TypeError("get_arrange_result.arrange_result must be a dict")

        video_mapping = file_content["shuffled_understand_result"]
        if not isinstance(video_mapping, dict) or not video_mapping:
            raise TypeError("shuffled_understand_result must be a non-empty dict")

        expected_video_ids = list(video_mapping.keys())
        gt_video_names = set(gt_for_sample.keys())

        sort_result = cut_plan.get("sort", {}).get("result")
        _validate_video_id_list("sort.result", sort_result, expected_video_ids)

        planned_videos = _planned_videos_for_metrics(cut_plan)
        planned_video_ids = []
        seen_gt_video_names = set()
        for video_result in planned_videos:
            if not isinstance(video_result, dict):
                raise TypeError("cut-plan video item must be a dict")
            video_id = video_result.get("video_id")
            if video_id is None:
                raise KeyError("cut-plan video item missing video_id")
            planned_video_ids.append(video_id)
            video_name = _mapping_video_name(video_mapping, video_id)
            if video_name not in gt_video_names:
                raise KeyError(f"extra video predicted: {video_id} -> {video_name}")
            segments = _segments_for_metrics(video_result)
            if not segments:
                raise ValueError(f"empty segments for {video_id}")
            for segment in segments:
                if not isinstance(segment, dict):
                    raise TypeError(f"invalid segment for {video_id}")
                float(segment["start_time"])
                float(segment["end_time"])
            seen_gt_video_names.add(video_name)
        _validate_video_id_list("arrange_results", planned_video_ids, expected_video_ids)

        missing_gt_videos = sorted(gt_video_names - seen_gt_video_names)
        if missing_gt_videos:
            raise KeyError(f"missing gt videos: {missing_gt_videos}")

        duration_adjustment = cut_plan.get("duration_adjustment")
        if isinstance(duration_adjustment, dict) and duration_adjustment:
            adjusted_results = duration_adjustment.get("adjusted_arrange_results")
            adjusted_video_ids = [
                item.get("video_id") if isinstance(item, dict) else None
                for item in adjusted_results
            ] if isinstance(adjusted_results, list) else adjusted_results
            _validate_video_id_list(
                "duration_adjustment.adjusted_arrange_results",
                adjusted_video_ids,
                expected_video_ids,
            )

        if require_explicit_understanding:
            understanding_items = _understanding_result_items(file_content, cut_plan)
            understanding_video_ids = []
            for item in understanding_items:
                if not isinstance(item, dict):
                    raise TypeError("understanding result item must be a dict")
                video_id = item.get("video_id")
                if not video_id:
                    raise KeyError("understanding result item missing video_id")
                understanding_video_ids.append(video_id)
                _mapping_video_name(video_mapping, video_id)
                highlights = item.get("highlights")
                if not isinstance(highlights, list) or not highlights:
                    raise ValueError(f"empty highlights for {video_id}")
                for highlight in highlights:
                    if not isinstance(highlight, dict):
                        raise TypeError(f"invalid highlight for {video_id}")
                    float(highlight["highlight_start"])
                    float(highlight["highlight_end"])
            _validate_video_id_list("understanding_results", understanding_video_ids, expected_video_ids)
    except Exception as ex:
        return invalid_case(sample_key, "invalid_case_output", f"{type(ex).__name__}: {ex}")

    return make_status(sample_key, True)


def validate_unified_output_dir(cfg, sample_keys, require_explicit_understanding=False):
    expected = set(sample_keys)
    invalid_items = []
    json_files = [
        name for name in os.listdir(cfg["out_dir"])
        if name.endswith(".json")
    ]
    for file_name in sorted(json_files):
        sample_key = file_name[:-5]
        if sample_key not in expected:
            invalid_items.append(
                invalid_case(sample_key, "extra_case", "case output json is not in gt", rerun=False)
            )

    for sample_key in sample_keys:
        invalid = validate_unified_case_output(
            cfg["out_dir"],
            sample_key,
            cfg["_gt"].get(sample_key, {}),
            require_explicit_understanding=require_explicit_understanding,
        )
        if not invalid.get("ok"):
            invalid_items.append(invalid)
    return invalid_items


def remove_case_artifacts(out_dir, sample_key):
    json_path = os.path.join(out_dir, sample_key + ".json")
    if os.path.exists(json_path):
        os.remove(json_path)

    raw_dir = os.path.join(out_dir, "_failed_raw")
    if os.path.isdir(raw_dir):
        prefix = sample_key + "_"
        for name in os.listdir(raw_dir):
            if name.startswith(prefix):
                path = os.path.join(raw_dir, name)
                if os.path.isfile(path):
                    os.remove(path)


def clear_repair_logs(out_dir):
    failed_dir = os.path.join(out_dir, "_failed")
    for name in ["unified_failed.jsonl", "unified_unresolved_after_repair.jsonl"]:
        path = os.path.join(failed_dir, name)
        if os.path.exists(path):
            os.remove(path)


def write_unresolved_repair_log(out_dir, invalid_items):
    failed_dir = os.path.join(out_dir, "_failed")
    os.makedirs(failed_dir, exist_ok=True)
    failed_path = os.path.join(failed_dir, "unified_unresolved_after_repair.jsonl")
    with open(failed_path, "w", encoding="utf-8") as f:
        for item in invalid_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return failed_path


def run_sample_batch(sample_keys, cfg, workers):
    results = []
    if workers <= 1:
        for sk in tqdm(sample_keys):
            try:
                results.append(run_one_entry(sk, cfg))
            except Exception as ex:
                print(f"[FAIL] {sk} {type(ex).__name__}: {ex}")
                traceback.print_exc()
                results.append(make_status(sk, False, "unexpected_worker_exception", f"{type(ex).__name__}: {ex}"))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for r in tqdm(
                executor.map(run_one_entry, sample_keys, [cfg] * len(sample_keys)),
                total=len(sample_keys),
            ):
                if r is None:
                    r = make_status("<unknown>", False, "unknown_worker_failure", "worker returned None")
                results.append(r)

    failed = [item for item in results if not item.get("ok")]
    ok_total = len(results) - len(failed)
    failed_path = write_failed_log(cfg["out_dir"], failed)
    if failed_path:
        print(f"[FAIL] wrote unified failed log: {failed_path}")
    print(f"done ok={ok_total} failed={len(failed)} total={len(sample_keys)} workers={workers}")
    return results, failed


def run_repair_loop(cfg, sample_keys, workers, max_repair_rounds, require_explicit_understanding):
    invalid_items = validate_unified_output_dir(
        cfg,
        sample_keys,
        require_explicit_understanding=require_explicit_understanding,
    )
    if not invalid_items:
        clear_repair_logs(cfg["out_dir"])
        print("[REPAIR] all cases already valid")
        return []

    for round_idx in range(1, max_repair_rounds + 1):
        preview = ", ".join(item["sample_key"] for item in invalid_items[:10])
        suffix = "" if len(invalid_items) <= 10 else f", ... total={len(invalid_items)}"
        print(f"[REPAIR] round={round_idx}/{max_repair_rounds} invalid={len(invalid_items)} ({preview}{suffix})")

        rerun_keys = []
        for item in invalid_items:
            sample_key = item["sample_key"]
            remove_case_artifacts(cfg["out_dir"], sample_key)
            if item.get("rerun", True) and sample_key in cfg["_gt"]:
                rerun_keys.append(sample_key)

        if rerun_keys:
            run_sample_batch(sorted(set(rerun_keys)), cfg, workers)

        invalid_items = validate_unified_output_dir(
            cfg,
            sample_keys,
            require_explicit_understanding=require_explicit_understanding,
        )
        if not invalid_items:
            clear_repair_logs(cfg["out_dir"])
            print(f"[REPAIR] passed after round {round_idx}")
            return []

    failed_path = write_unresolved_repair_log(cfg["out_dir"], invalid_items)
    print(f"[REPAIR] unresolved invalid cases after {max_repair_rounds} rounds: {failed_path}")
    return invalid_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="one of the aliases in visual_workflow/model_api.py MODEL_REGISTRY",
    )
    parser.add_argument(
        "--video-dirs",
        nargs="+",
        required=True,
        help="raw video directories, one per set (parallel with --out-dirs and --gt-paths)",
    )
    parser.add_argument(
        "--out-dirs",
        nargs="+",
        required=True,
        help="where to write unified cut-plan JSON, one per set",
    )
    parser.add_argument(
        "--gt-paths",
        nargs="+",
        required=True,
        help="ground-truth JSON paths, one per set",
    )
    parser.add_argument(
        "--unified-workflow-prompt-version",
        default="v18",
        help="prompt version defined in visual_workflow/prompt/unified_workflow_prompt.py (default: v18)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes for samples in each set",
    )
    parser.add_argument(
        "--repair-invalid",
        action="store_true",
        help="repair invalid/missing case outputs by deleting bad case jsons and rerunning them",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=5,
        help="maximum repair rounds per output directory when --repair-invalid is enabled",
    )
    parser.add_argument(
        "--require-explicit-understanding",
        action="store_true",
        help="require explicit understanding_results in unified output validation",
    )
    args = parser.parse_args()
    assert (
        len(args.video_dirs) == len(args.out_dirs) == len(args.gt_paths)
    ), f"video-dirs/out-dirs/gt-paths 必须等长 ({len(args.video_dirs)}, {len(args.out_dirs)}, {len(args.gt_paths)})"
    MODEL_NAME = args.model
    if args.max_repair_rounds < 1:
        raise SystemExit("--max-repair-rounds must be >= 1")
    print(f"=== Using unified chain model: {MODEL_NAME} ===")

    global unified_workflow_prompt
    _up_name = f"unified_workflow_prompt_{args.unified_workflow_prompt_version}"
    try:
        unified_workflow_prompt = globals()[_up_name]
    except KeyError as ex:
        raise SystemExit(
            f"Unknown unified workflow prompt: {_up_name}."
        ) from ex
    print(f"=== Using unified prompt: {_up_name} ===")

    register_heif_opener()

    all_failed = []
    for vd, od, gt in zip(args.video_dirs, args.out_dirs, args.gt_paths):
        print(f"\n=== Processing: {gt} ===")
        cfg = {
            "model_name": MODEL_NAME,
            "video_dir": vd.rstrip("/"),
            "out_dir": od.rstrip("/"),
            "skip_existing": True,
            "_gt": read_json(gt) if gt else {},
        }
        os.makedirs(cfg["out_dir"], exist_ok=True)

        all_sample_keys = sorted(cfg["_gt"].keys())
        workers = int(args.workers)

        if args.repair_invalid:
            print(f"sample count: {len(all_sample_keys)}")
            failed = run_repair_loop(
                cfg,
                all_sample_keys,
                workers,
                args.max_repair_rounds,
                args.require_explicit_understanding,
            )
        else:
            sample_keys = all_sample_keys
            if cfg["skip_existing"]:
                sample_keys = [
                    k for k in sample_keys
                    if not os.path.exists(os.path.join(cfg["out_dir"], k + ".json"))
                ]
            print(f"sample count: {len(sample_keys)}")
            _, failed = run_sample_batch(sample_keys, cfg, workers)

        all_failed.extend(failed)

    if all_failed:
        raise SystemExit(f"unified failed samples: {len(all_failed)}")


if __name__ == "__main__":
    main()
