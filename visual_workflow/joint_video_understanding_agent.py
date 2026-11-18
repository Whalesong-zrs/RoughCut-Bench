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
from tqdm import tqdm

from visual_workflow.workflow_utils import (
    read_json,
    write_json,
    parallel_extract_video_frames,
    my_encode_img,
)
from visual_workflow.model_clients import JointVideoUnderstandingLLM
from visual_workflow.prompt import joint_video_understanding_prompt_v2


joint_video_understanding_prompt = joint_video_understanding_prompt_v2

_VALID_EXT = {"mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png", "heic"}


def make_status(sample_key, ok, error_type=None, error_message=None, video_count=None, skipped_existing=False):
    return {
        "sample_key": sample_key,
        "ok": ok,
        "error_type": error_type,
        "error_message": error_message,
        "video_count": video_count,
        "skipped_existing": skipped_existing,
    }


def invalid_case(sample_key, error_type, error_message, rerun=True):
    status = make_status(sample_key, False, error_type, error_message)
    status["rerun"] = rerun
    return status


def _is_video_or_image(filename: str) -> bool:
    if filename.startswith("."):
        return False
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return ext in _VALID_EXT


def video_name_from_path(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def normalize_time_value(value):
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def build_frames_by_video(video_dir: str, sample_key: str, sample_fps: int = 1, target_short_edge: int = 224):
    base = os.path.join(video_dir, sample_key)
    if not os.path.isdir(base):
        return None

    video_files = [name for name in sorted(os.listdir(base)) if _is_video_or_image(name)]
    gt_for_sample = gt_by_sample.get(sample_key, {}) if "gt_by_sample" in globals() else {}
    if gt_for_sample:
        expected_video_names = set(gt_for_sample.keys())
        video_files = [name for name in video_files if video_name_from_path(name) in expected_video_names]
    if not video_files:
        return None

    video_paths = [os.path.join(base, name) for name in video_files]
    pil_collects, ts_collects = parallel_extract_video_frames(video_paths)

    frames_by_video = OrderedDict()
    for idx, (pil_list, timestamps, video_path) in enumerate(zip(pil_collects, ts_collects, video_paths)):
        resized = []
        for img in pil_list:
            width, height = img.size
            if width < height:
                new_width = target_short_edge
                new_height = int(round(height / width * target_short_edge))
            else:
                new_height = target_short_edge
                new_width = int(round(width / height * target_short_edge))
            resized.append(img.resize((new_width, new_height), Image.Resampling.LANCZOS))

        encoded = [(round(float(t), 1), my_encode_img(img)) for t, img in zip(timestamps, resized)]
        duration = float(timestamps[-1]) if len(timestamps) > 0 else 0.0
        frames_by_video[f"video_{idx}"] = {
            "video_name": video_path,
            "duration": duration,
            "frames": encoded,
        }
    return frames_by_video


class JointUnderstandingProcessor:
    def __init__(self, model):
        self.llm = JointVideoUnderstandingLLM(model, joint_video_understanding_prompt)

    def process(self, sample_key):
        out_path = os.path.join(save_dir, sample_key + ".json")
        if skip_existing and os.path.exists(out_path):
            return make_status(sample_key, True, skipped_existing=True)

        try:
            frames_by_video = build_frames_by_video(video_dir, sample_key)
            if not frames_by_video:
                print(f"[FAIL] {sample_key} no valid videos under sample")
                return make_status(sample_key, False, "no_valid_videos", "no valid videos under sample", video_count=0)

            result, input_token, output_token = self.llm.run(frames_by_video)
            result["input_tokens"] = input_token
            result["output_tokens"] = output_token
            write_json(out_path, result)
            return make_status(sample_key, True, video_count=len(frames_by_video))
        except Exception as ex:
            print(f"[FAIL] {sample_key} joint understanding err={type(ex).__name__}: {ex}")
            traceback.print_exc()
            return make_status(sample_key, False, "joint_video_understanding_failed", f"{type(ex).__name__}: {ex}")


def write_failed_log(save_dir, failed_items):
    if not failed_items:
        return None
    failed_dir = os.path.join(save_dir, "_failed")
    os.makedirs(failed_dir, exist_ok=True)
    failed_path = os.path.join(failed_dir, "joint_video_understanding_failed.jsonl")
    with open(failed_path, "w", encoding="utf-8") as f:
        for item in failed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return failed_path


def validate_joint_video_understanding_case_output(save_dir, sample_key, gt_for_sample=None):
    json_path = os.path.join(save_dir, sample_key + ".json")
    if not os.path.exists(json_path):
        return invalid_case(sample_key, "missing_case", "case output json is missing")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            file_content = json.load(f)
    except Exception as ex:
        return invalid_case(sample_key, "malformed_json", f"{type(ex).__name__}: {ex}")

    try:
        understanding_results = file_content["understanding_results"]
        if not isinstance(understanding_results, dict) or not understanding_results:
            raise TypeError("understanding_results must be a non-empty dict")

        gt_video_names = set((gt_for_sample or {}).keys())
        seen_gt_video_names = set()
        seen_video_names = set()
        for video_id, video_result in understanding_results.items():
            if not isinstance(video_result, dict):
                raise TypeError(f"{video_id} result must be a dict")
            video_name = video_result.get("video_name")
            if not video_name:
                raise KeyError(f"{video_id} missing video_name")
            base_name = video_name_from_path(video_name)
            if base_name in seen_video_names:
                raise KeyError(f"duplicate video_name: {base_name}")
            seen_video_names.add(base_name)
            if gt_video_names:
                if base_name not in gt_video_names:
                    raise KeyError(f"extra video predicted: {video_id} -> {base_name}")
                seen_gt_video_names.add(base_name)

            highlights = video_result.get("highlights")
            if not isinstance(highlights, list) or not highlights:
                raise ValueError(f"empty highlights for {video_id}")
            for highlight in highlights:
                if not isinstance(highlight, dict):
                    raise TypeError(f"invalid highlight for {video_id}")
                start = normalize_time_value(highlight.get("highlight_start"))
                end = normalize_time_value(highlight.get("highlight_end"))
                if start is None or end is None:
                    raise KeyError(f"missing highlight_start/highlight_end for {video_id}")
                if end < start:
                    raise ValueError(f"invalid highlight time range for {video_id}: {start}-{end}")

        if gt_video_names:
            missing = sorted(gt_video_names - seen_gt_video_names)
            if missing:
                raise KeyError(f"missing gt videos: {missing}")
    except Exception as ex:
        return invalid_case(sample_key, "invalid_case_output", f"{type(ex).__name__}: {ex}")

    return make_status(sample_key, True)


def validate_joint_video_understanding_output_dir(save_dir, sample_keys, gt=None):
    expected = set(sample_keys)
    invalid_items = []
    json_files = [name for name in os.listdir(save_dir) if name.endswith(".json")]
    for file_name in sorted(json_files):
        sample_key = file_name[:-5]
        if sample_key not in expected:
            invalid_items.append(
                invalid_case(sample_key, "extra_case", "case output json is not in gt", rerun=False)
            )

    gt = gt or {}
    for sample_key in sample_keys:
        invalid = validate_joint_video_understanding_case_output(save_dir, sample_key, gt.get(sample_key, {}))
        if not invalid.get("ok"):
            invalid_items.append(invalid)
    return invalid_items


def remove_case_artifacts(save_dir, sample_key):
    json_path = os.path.join(save_dir, sample_key + ".json")
    if os.path.exists(json_path):
        os.remove(json_path)


def clear_repair_logs(save_dir):
    failed_dir = os.path.join(save_dir, "_failed")
    for name in ["joint_video_understanding_failed.jsonl", "joint_video_understanding_unresolved_after_repair.jsonl"]:
        path = os.path.join(failed_dir, name)
        if os.path.exists(path):
            os.remove(path)


def write_unresolved_repair_log(save_dir, invalid_items):
    failed_dir = os.path.join(save_dir, "_failed")
    os.makedirs(failed_dir, exist_ok=True)
    failed_path = os.path.join(failed_dir, "joint_video_understanding_unresolved_after_repair.jsonl")
    with open(failed_path, "w", encoding="utf-8") as f:
        for item in invalid_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return failed_path


def parallel_process(sample_keys, workers):
    print("Total Reqs: %d" % len(sample_keys))
    results = []
    if workers <= 1:
        for sample_key in tqdm(sample_keys):
            try:
                data = runner.process(sample_key)
            except Exception as ex:
                traceback.print_exc()
                data = make_status(sample_key, False, "unexpected_worker_exception", f"{type(ex).__name__}: {ex}")
            if data is None:
                data = make_status(sample_key, False, "unknown_worker_failure", "worker returned None")
            results.append(data)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for sample_key, data in zip(sample_keys, tqdm(executor.map(runner.process, sample_keys), total=len(sample_keys))):
                if data is None:
                    data = make_status(sample_key, False, "unknown_worker_failure", "worker returned None")
                results.append(data)

    failed = [item for item in results if not item.get("ok")]
    ok_count = len(results) - len(failed)
    print(f"done ok={ok_count} failed={len(failed)} total={len(results)} workers={workers}")
    return results, failed


def run_repair_loop(sample_keys, workers, gt, max_repair_rounds):
    invalid_items = validate_joint_video_understanding_output_dir(save_dir, sample_keys, gt)
    if not invalid_items:
        clear_repair_logs(save_dir)
        print("[REPAIR] all joint understanding cases already valid")
        return []

    for round_idx in range(1, max_repair_rounds + 1):
        preview = ", ".join(item["sample_key"] for item in invalid_items[:10])
        suffix = "" if len(invalid_items) <= 10 else f", ... total={len(invalid_items)}"
        print(f"[REPAIR] joint understanding round={round_idx}/{max_repair_rounds} invalid={len(invalid_items)} ({preview}{suffix})")

        rerun_keys = []
        for item in invalid_items:
            sample_key = item["sample_key"]
            remove_case_artifacts(save_dir, sample_key)
            if item.get("rerun", True) and sample_key in sample_keys:
                rerun_keys.append(sample_key)

        if rerun_keys:
            _, failed = parallel_process(sorted(set(rerun_keys)), workers)
            failed_path = write_failed_log(save_dir, failed)
            if failed_path:
                print(f"[FAIL] wrote joint understanding failed log: {failed_path}")

        invalid_items = validate_joint_video_understanding_output_dir(save_dir, sample_keys, gt)
        if not invalid_items:
            clear_repair_logs(save_dir)
            print(f"[REPAIR] joint understanding passed after round {round_idx}")
            return []

    failed_path = write_unresolved_repair_log(save_dir, invalid_items)
    print(f"[REPAIR] unresolved joint understanding invalid cases after {max_repair_rounds} rounds: {failed_path}")
    return invalid_items


def list_sample_keys(video_root, gt_path=None):
    if gt_path:
        gt = read_json(gt_path)
        return sorted(gt.keys())
    return sorted([
        name for name in os.listdir(video_root)
        if os.path.isdir(os.path.join(video_root, name))
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--video-dirs", nargs="+", required=True)
    parser.add_argument("--out-dirs", nargs="+", required=True)
    parser.add_argument("--gt-paths", nargs="+", default=None)
    parser.add_argument("--joint-video-understanding-prompt-version", default="v2")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repair-invalid", action="store_true")
    parser.add_argument("--max-repair-rounds", type=int, default=5)
    args = parser.parse_args()

    if args.gt_paths is not None:
        assert len(args.gt_paths) == len(args.video_dirs), \
            f"--gt-paths ({len(args.gt_paths)}) and --video-dirs ({len(args.video_dirs)}) must have same length"
    assert len(args.video_dirs) == len(args.out_dirs), \
        f"--video-dirs ({len(args.video_dirs)}) and --out-dirs ({len(args.out_dirs)}) must have same length"
    if args.max_repair_rounds < 1:
        raise SystemExit("--max-repair-rounds must be >= 1")

    prompt_name = f"joint_video_understanding_prompt_{args.joint_video_understanding_prompt_version}"
    try:
        selected_prompt = globals()[prompt_name]
    except KeyError as ex:
        raise SystemExit(f"unknown joint understanding prompt version: {prompt_name}") from ex

    global joint_video_understanding_prompt
    joint_video_understanding_prompt = selected_prompt
    print(f"=== Using joint understanding model: {args.model} ===")
    print(f"=== Using joint understanding prompt: {prompt_name} ===")

    global runner
    runner = JointUnderstandingProcessor(args.model)

    all_failed = []
    for idx, (video_root, output_root) in enumerate(zip(args.video_dirs, args.out_dirs)):
        gt_path = args.gt_paths[idx] if args.gt_paths is not None else None
        gt = read_json(gt_path) if gt_path else {}
        print(f"\n=== Processing: {video_root} ===")
        os.makedirs(output_root, exist_ok=True)

        sample_keys = list_sample_keys(video_root.rstrip("/"), gt_path)

        global video_dir, save_dir, skip_existing, gt_by_sample
        video_dir = video_root.rstrip("/")
        save_dir = output_root.rstrip("/")
        gt_by_sample = gt
        skip_existing = True

        if args.repair_invalid:
            print(f"sample count: {len(sample_keys)}")
            failed = run_repair_loop(sample_keys, args.workers, gt, args.max_repair_rounds)
        else:
            existing = set(name[:-5] for name in os.listdir(output_root) if name.endswith(".json"))
            unprocessed = [key for key in sample_keys if key not in existing]
            print(len(unprocessed))
            _, failed = parallel_process(unprocessed, args.workers)
            failed_path = write_failed_log(save_dir, failed)
            if failed_path:
                print(f"[FAIL] wrote joint understanding failed log: {failed_path}")
        all_failed.extend(failed)

    if all_failed:
        raise SystemExit(f"joint understanding failed samples: {len(all_failed)}")


if __name__ == "__main__":
    main()
