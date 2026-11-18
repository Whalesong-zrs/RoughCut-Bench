
import base64
import io
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import math
import decord
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from moviepy.editor import VideoFileClip



def shuffle_understanding(understand_result: dict):
    shuffled_understand_result = {}
    video_ids = [i for i in understand_result]
    shuffled_ids = video_ids.copy()
    random.shuffle(shuffled_ids)

    mapper = {}
    for idx, old_id in enumerate(shuffled_ids):
        new_id = "video_" + str(idx)
        shuffled_understand_result[new_id] = understand_result[old_id]
        mapper[old_id] = new_id

    sort_answer = [mapper[i] for i in video_ids]

    return shuffled_understand_result, sort_answer, mapper


def build_processed_understanding(understanding_file: dict, asr_file: dict, delete_asr_prob=0.5):
    asr_contents = None 
    asr_result = None 
    if asr_file:
        asr_contents = asr_file['extracted_asr']
        asr_result = asr_file['processed_asr']
    

    if "understanding_results" in understanding_file:
        sample_content = understanding_file["understanding_results"]
        if isinstance(sample_content, dict) and "understanding_results" in sample_content:
            sample_content = sample_content["understanding_results"]
    elif "filter_results" in understanding_file:
        sample_content = understanding_file["filter_results"]
        if isinstance(sample_content, dict) and "filter_results" in sample_content:
            sample_content = sample_content["filter_results"]
    else:
        raise KeyError(
            "video-understanding output must contain 'understanding_results' "
            "or legacy 'filter_results'"
        )
    new_content = {}

    delete_asr_for_sample = random.random() < delete_asr_prob

    for idx, k in enumerate(sample_content):
        video_id = "video_" + str(idx)
        video_content = sample_content[k]
        video_content_new = {}
        video_content_new["time"] = video_content["time"]
        video_content_new["scene/location"] = video_content["scene/location"]
        new_visual_segs = []
        for idxx, seg in enumerate(video_content["highlights"]):
            seg_duration = round((seg["highlight_end"] - seg["highlight_start"]), 1)
            new_visual_segs.append(
                {
                    "start_time": round(seg["highlight_start"], 1),
                    "end_time": round(seg["highlight_end"], 1),
                    "duration": str(seg_duration),
                    "description": seg["description"],
                }
            )
            
        new_asr_segs = []


        if asr_contents and asr_result:
  
            asr_content = asr_contents[video_id]
            if not delete_asr_for_sample:
                asr_select_segments = asr_result[video_id]["asr_select_segments"]
                asr_select_segments = [int(i) for i in asr_select_segments]
                asr_select_segments = sorted(set(asr_select_segments))
           
                utterances = asr_content["utterances"]
  
                selected_raw = []
                for idxx in asr_select_segments:
                    if 0 <= idxx < len(utterances):
                        seg = utterances[idxx]
                        selected_raw.append((seg["start_time"] * 1e-3, seg["end_time"] * 1e-3, seg["text"]))

                merged_raw = []
                for st_s_raw, et_s_raw, text in selected_raw:
                    if merged_raw and (st_s_raw - merged_raw[-1][1]) < 0.2:
                        last_st, _, last_text = merged_raw[-1]
                        merged_raw[-1] = (last_st, et_s_raw, (last_text.rstrip() + " " + text.lstrip()).strip())
                    else:
                        merged_raw.append((st_s_raw, et_s_raw, text))

                for st_s_raw, et_s_raw, text in merged_raw:
                    st_s = round(st_s_raw, 1)
                    et_s = round(et_s_raw, 1)
                    new_asr_segs.append(
                        {
                            "start_time": st_s,
                            "end_time": et_s,
                            "duration": round(et_s - st_s, 1),
                            "transcription": text,
                        }
                    )

            
                filtered_visual_segs = []
                for v_seg in new_visual_segs:
                    is_overlap = False
                    for u_seg in utterances:
                        u_st = u_seg.get("start_time")
                        u_et = u_seg.get("end_time")
                    

                        a_st = u_st * 1e-3
                        a_et = u_et * 1e-3

                        if v_seg["start_time"] < a_et and a_st < v_seg["end_time"]:
                            is_overlap = True
                            break
                    if not is_overlap:
                        filtered_visual_segs.append(v_seg)
                new_visual_segs = filtered_visual_segs
    
        video_name = video_content["video_name"]
        if video_name.split(".")[-1].lower() in ["mp4", "mov", "avi", "mkv"]:
            video_duration = VideoFileClip(video_name).duration
        else:
            video_duration = video_content["video_abs_frame_stamps"][-1]
            
        video_content_new["video_duration"] = video_duration
        video_content_new["visual_segments"] = new_visual_segs
        video_content_new["asr_segments"] = new_asr_segs
        video_content_new["video_name"] = video_content["video_name"]
        new_content[video_id] = video_content_new

    return new_content

def build_processed_asr(asr_file: dict):
    asr_contents = asr_file["asr_results"]
    new_content = {}
    for idx, k in enumerate(asr_contents):
        asr_content = asr_contents[k]
        new_asr_segs = {}
        for idxx, seg in enumerate(asr_content["utterances"]):
            new_asr_segs[idxx] = seg["text"]
        new_content[k] = new_asr_segs

    return new_content




def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _segment_duration(seg: dict):
    duration = _safe_float(seg.get("duration"))
    if duration is not None:
        return duration

    start_time = _safe_float(seg.get("start_time"))
    end_time = _safe_float(seg.get("end_time"))
    if start_time is not None and end_time is not None:
        return max(0.0, end_time - start_time)
    return 0.0


def _metadata_time_value(value):
    parsed = _safe_float(value)
    if parsed is None:
        return "" if value is None else value
    return round(parsed, 3)


def _segment_limit_record(video_id, segment_index, seg):
    return {
        "video_id": video_id,
        "segment_index": segment_index,
        "start_time": _metadata_time_value(seg.get("start_time")),
        "end_time": _metadata_time_value(seg.get("end_time")),
        "duration": round(_segment_duration(seg), 3),
    }


def select_cut_planning_keyframe_segments(understand_result: dict, model_name: str, max_keyframes: int = 120):
    segment_entries = []
    original_order = 0
    for video_id, video_info in understand_result.items():
        visual_segments = video_info.get("visual_segments", [])
        for segment_index, seg in enumerate(visual_segments):
            segment_entries.append(
                {
                    "video_id": video_id,
                    "segment_index": segment_index,
                    "duration": _segment_duration(seg),
                    "order": original_order,
                    "record": _segment_limit_record(video_id, segment_index, seg),
                }
            )
            original_order += 1

    total_segments = len(segment_entries)
    limit_keyframe_models = ("glm-5v-turbo",)
    enabled = bool(
        model_name
        and (model_name.startswith("gpt-") or model_name in limit_keyframe_models)
        and total_segments > max_keyframes
    )

    if enabled:
        selected_entries = sorted(segment_entries, key=lambda item: (-item["duration"], item["order"]))[:max_keyframes]
        selected_keys = {(item["video_id"], item["segment_index"]) for item in selected_entries}
    else:
        selected_keys = {(item["video_id"], item["segment_index"]) for item in segment_entries}

    selected_by_video = {}
    dropped_by_video = {}
    for item in segment_entries:
        key = (item["video_id"], item["segment_index"])
        target = selected_by_video if key in selected_keys else dropped_by_video
        target.setdefault(item["video_id"], []).append(item["record"])

    selected_segments_by_video = None
    if enabled:
        selected_segments_by_video = {
            video_id: {record["segment_index"] for record in records}
            for video_id, records in selected_by_video.items()
        }

    limit_info = {
        "enabled": enabled,
        "model_name": model_name,
        "max_keyframes": max_keyframes,
        "total_segments": total_segments,
        "selected_segments": len(selected_keys),
        "dropped_segments": total_segments - len(selected_keys),
        "selected_by_video": selected_by_video,
        "dropped_by_video": dropped_by_video,
    }
    return selected_segments_by_video, limit_info


def filter_keyframes_by_selected_segments(keyframes_by_video: dict, selected_segments_by_video):
    if selected_segments_by_video is None:
        return keyframes_by_video

    filtered_keyframes_by_video = {}
    for video_id, frames_list in keyframes_by_video.items():
        selected_indices = selected_segments_by_video.get(video_id, set())
        filtered_keyframes_by_video[video_id] = [
            img_b64 if segment_index in selected_indices else None
            for segment_index, img_b64 in enumerate(frames_list)
        ]
    return filtered_keyframes_by_video


def build_keyframes_by_video(understand_result: dict, selected_segments_by_video=None):
    keyframes_by_video = {}
    for video_id in understand_result:
        video_info = understand_result[video_id]
        video_path = video_info["video_name"]
        if not video_path:
            raise KeyError(f"missing video_name for {video_id}")

        visual_segments = video_info["visual_segments"]
        keyframes_b64_list = [None] * len(visual_segments)

        if selected_segments_by_video is None:
            selected_indices = list(range(len(visual_segments)))
        else:
            selected_indices = sorted(
                idx
                for idx in selected_segments_by_video.get(video_id, set())
                if 0 <= idx < len(visual_segments)
            )

        if not selected_indices:
            keyframes_by_video[video_id] = keyframes_b64_list
            continue

        seg_mid_timestamps = [
            (visual_segments[idx]["start_time"] + visual_segments[idx]["end_time"]) / 2.0
            for idx in selected_indices
        ]

        pil_list = decode_with_timestamps(video_path, seg_mid_timestamps)
        if len(pil_list) != len(seg_mid_timestamps):
            raise ValueError(f"keyframes count mismatch for {video_id}: highlights={len(seg_mid_timestamps)} frames={len(pil_list)}")

        for segment_index, pil_img in zip(selected_indices, pil_list):
            pil_img = pil_img.convert("RGB")
            pil_img = resize_short_side(pil_img, 224)
            keyframes_b64_list[segment_index] = my_encode_img(pil_img)
        keyframes_by_video[video_id] = keyframes_b64_list

    return keyframes_by_video


def apply_meta_time_injection(understand_result: dict):
    meta_injection_mode = "none"
    meta_missing_video_ids = []

    meta_inject_prob = 0.2
    if random.random() >= meta_inject_prob:
        return meta_injection_mode, meta_missing_video_ids

    video_ids_for_meta = list(understand_result.keys())
    base_times = generate_constrained_time_list(len(video_ids_for_meta))

    r = random.random()
    if r < 0.25:
        meta_injection_mode = "correct"
        assigned_times = base_times
    elif r < 0.50:
        meta_injection_mode = "wrong"
        assigned_times = base_times.copy()
        random.shuffle(assigned_times)
        if assigned_times == base_times and len(assigned_times) > 1:
            assigned_times = assigned_times[1:] + assigned_times[:1]
    elif r < 0.75:
        meta_injection_mode = "correct_partial_missing"
        assigned_times = base_times
    else:
        meta_injection_mode = "wrong_partial_missing"
        assigned_times = base_times.copy()
        random.shuffle(assigned_times)
        if assigned_times == base_times and len(assigned_times) > 1:
            assigned_times = assigned_times[1:] + assigned_times[:1]

    for idx, k in enumerate(video_ids_for_meta):
        understand_result[k]["time"] = assigned_times[idx]

    if meta_injection_mode in {"correct_partial_missing", "wrong_partial_missing"}:
        missing_ratio = random.uniform(0.2, 0.5)
        missing_k = max(1, int(round(len(video_ids_for_meta) * missing_ratio)))
        meta_missing_video_ids = random.sample(video_ids_for_meta, k=min(missing_k, len(video_ids_for_meta)))
        for k in meta_missing_video_ids:
            understand_result[k]["time"] = ""

    return meta_injection_mode, meta_missing_video_ids



def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def generate_constrained_time_list(n: int):
    base_year = random.choice([2023, 2024, 2025])
    same_month = random.random() < 0.95
    base_month = random.randint(1, 12) if same_month else None

    rand_day = random.random()
    fixed_day = None
    start_day = None

    if same_month:
        month = base_month
        if month in [4, 6, 9, 11]:
            max_day = 30
        elif month == 2:
            max_day = 29 if base_year == 2024 else 28
        else:
            max_day = 31

        if rand_day < 0.7:
            fixed_day = random.randint(1, max_day)
        elif rand_day < 0.85:
            start_day = random.randint(1, max_day - 3) if max_day >= 4 else 1

    times = []
    for _ in range(n):
        year = base_year
        if same_month:
            month = base_month
        else:
            month = random.randint(1, 12)

        if month in [4, 6, 9, 11]:
            current_max_day = 30
        elif month == 2:
            current_max_day = 29 if year == 2024 else 28
        else:
            current_max_day = 31

        if same_month:
            if rand_day < 0.7:
                current_day = fixed_day
            elif rand_day < 0.85:
                current_day = start_day + random.randint(0, 3)
                current_day = min(current_day, current_max_day)
            else:
                current_day = random.randint(1, current_max_day)
        else:
            current_day = random.randint(1, current_max_day)

        base_hour = random.randint(0, 22)
        if random.random() < 0.5:
            hour = base_hour + random.randint(0, 2)
            hour = min(hour, 23)
        else:
            hour = random.randint(0, 23)

        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        current_time = datetime(year, month, current_day, hour, minute, second)
        times.append(current_time)

    times.sort()
    return [t.strftime("%Y/%m/%d %H:%M:%S") for t in times]


def resize_short_side(img: Image.Image, short_side: int = 224) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    if min(w, h) == short_side:
        return img
    scale = float(short_side) / float(min(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BICUBIC)


def decode_with_timestamps(video_path: str, timestamps):

    decord.bridge.set_bridge("torch")
    if video_path.split(".")[-1].lower() in ["mp4", "mov", "avi", "mkv"]:
        video_reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)
        if total_frames <= 0:
            raise ValueError(f"empty video: {video_path}")

        times_in_vid = video_reader.get_frame_timestamp(range(total_frames)).mean(-1)
        indices = np.searchsorted(times_in_vid, timestamps)
        safe_indices = [int(min(max(i, 0), total_frames - 1)) for i in indices]
        video_data = video_reader.get_batch(safe_indices)
        return [Image.fromarray(video_data[i].numpy()) for i in range(video_data.shape[0])]
    elif video_path.split(".")[-1].lower() in ["jpg", "jpeg", "png", "webp", "heic", "heif", "bmp"]:
        register_heif_opener()
        img = Image.open(video_path).convert("RGB")
        return [img for _ in range(len(timestamps))]
    else:
        import os
        if os.path.isdir(video_path):
            files = [f for f in os.listdir(video_path) if f.endswith(".jpg") and "cut_time_" in f]
            if not files:
                raise ValueError(f"No valid frames found in folder: {video_path}")
            
            times_and_files = []
            for f in files:
                try:
                    time_str = f.split("cut_time_")[1].replace(".jpg", "")
                    times_and_files.append((float(time_str), os.path.join(video_path, f)))
                except Exception:
                    pass
            
            if not times_and_files:
                raise ValueError(f"Failed to parse time from frames in folder: {video_path}")
                
            times_and_files.sort(key=lambda x: x[0])
            times_in_folder = np.array([x[0] for x in times_and_files])
            file_paths = [x[1] for x in times_and_files]
            
            indices = np.searchsorted(times_in_folder, timestamps)
            safe_indices = [int(min(max(i, 0), len(file_paths) - 1)) for i in indices]
            
            return [Image.open(file_paths[i]).convert("RGB") for i in safe_indices]
            
        raise ValueError(f"Unsupported file format: {video_path}")


def normalize_llm_json_response(res: str):
    if not isinstance(res, str):
        return res
    res = res.strip()
    fenced = re.fullmatch(
        r"```(?:json|python)?[ \t]*\r?\n(?P<body>.*?)\r?\n?```",
        res,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group("body").strip() if fenced else res





def my_encode_img(img):
    img_byte = io.BytesIO()
    img.save(img_byte, format="JPEG")
    binary = img_byte.getvalue()
    base64_encoded_data = base64.b64encode(binary).decode("utf-8")
    return base64_encoded_data


def decode_with_fps(video_path, sample_fps=1):
    decord.bridge.set_bridge("torch")

    is_video = video_path.split(".")[-1].lower() in ["mp4", "mov", "avi", "mkv"]

    video_duration = 2
    if is_video:
        video_duration = VideoFileClip(video_path).duration

    end_t = max(0.02, video_duration - 0.1)
    n_frames = max(math.ceil(video_duration * sample_fps + 1), 1)
    timestamps = np.linspace(0.02, end_t, n_frames, endpoint=True).tolist()

    if is_video:
        video_reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
        times_in_vid = video_reader.get_frame_timestamp(range(len(video_reader))).mean(-1)
        indices = np.searchsorted(times_in_vid, timestamps)
        n = len(video_reader)
        indices = [min(max(int(i), 0), n - 1) for i in indices]

        video_data = video_reader.get_batch([_ for _ in indices if _ <= len(video_reader)])
        pil_list = list()
        for index in range(video_data.shape[0]):
            pil_list.append(Image.fromarray(video_data[index].numpy()))
    else:
        img = Image.open(video_path).convert("RGB")
        pil_list = [img for _ in range(len(timestamps))]

    assert len(pil_list) == len(timestamps)
    return pil_list, timestamps


def parallel_extract_video_frames(video_names, sample_fps=1):
    pil_collects = list()
    frame_stamps = list()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(decode_with_fps, video_name, sample_fps=sample_fps) for video_name in video_names]
        for future in futures:
            pil_list, timestamps = future.result()
            pil_collects.append(pil_list)
            frame_stamps.append(timestamps)
    return pil_collects, frame_stamps
