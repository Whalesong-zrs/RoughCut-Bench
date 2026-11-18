import base64
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from retrying import retry
from moviepy.editor import VideoFileClip

from visual_workflow.prompt.cut_planning_prompt import build_cut_planning_prompt_for_input_mode
from visual_workflow.workflow_utils import normalize_llm_json_response
from visual_workflow.model_api import (
    ModelRunner,
    LLMJsonParseError,
    _should_retry_llm_exception,
    get_model_cfg,
)
from visual_workflow.video_input import gemini_supports_inline_video, gemini_video_mime_type


GEMINI_VIDEO_PROXY_THRESHOLD_MB = 14.0
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_GEMINI_VIDEO_PROXY_ROOT = (
    Path(os.environ.get("ROUGHCUT_OUTPUT_ROOT") or _PROJECT_ROOT / "outputs")
    / "_cache"
    / "gemini_video_proxy"
)
GEMINI_VIDEO_PROXY_ROOT = os.environ.get("GEMINI_VIDEO_PROXY_ROOT") or str(
    _DEFAULT_GEMINI_VIDEO_PROXY_ROOT
)
GEMINI_VIDEO_PROXY_PRESETS = [
    {"width": 720, "fps": 12, "crf": 32},
    {"width": 540, "fps": 10, "crf": 35},
    {"width": 480, "fps": 8, "crf": 38},
    {"width": 360, "fps": 6, "crf": 40},
    {"width": 320, "fps": 5, "crf": 42},
]


class VideoUnderstandingLLM:
    def __init__(self, model: str, prompt):
        self.model = model
        self.runner = ModelRunner()
        self.prompt = prompt

    @retry(
        stop_max_attempt_number=5,
        wait_random_min=3000,
        wait_random_max=5000,
        retry_on_exception=_should_retry_llm_exception,
    )
    def run(self, binary_list, time_stamps):
        user_input_collects = []
        for img_idx, binary_img in enumerate(binary_list):
            user_input_collects.append({"type": "text", "text": "%ss:" % time_stamps[img_idx]})
            user_input_collects.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{binary_img}", "detail": "low"},
                }
            )

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.prompt}] + user_input_collects,
            }
        ]

        t0 = time.time()
        res, input_token, output_token = self.runner.run(messages, model=self.model)
        print("Understanding Time Consumption: %.2f" % (time.time() - t0))
        print("Input Tokens: %s | Output Tokens: %s" % (input_token, output_token))

        res = json.loads(normalize_llm_json_response(res))
        print(res)

        for highlight in res['highlights']:
            if highlight['highlight_start'] not in time_stamps:
                closest = time_stamps[np.argmin(abs(np.array(time_stamps) - highlight['highlight_start']))]
                print('Refine Highlight Start: %f -> %f' % (highlight['highlight_start'], closest))
                highlight['highlight_start'] = closest
            if highlight['highlight_end'] not in time_stamps:
                closest = time_stamps[np.argmin(abs(np.array(time_stamps) - highlight['highlight_end']))]
                print('Refine Highlight End: %f -> %f' % (highlight['highlight_end'], closest))
                highlight['highlight_end'] = closest
        res['video_abs_frame_stamps'] = time_stamps

        return res, input_token, output_token

    def _encode_video(self, video_path):
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _video_mime_type(self, video_path):
        return gemini_video_mime_type(video_path)

    def _video_size_mb(self, video_path):
        return os.path.getsize(video_path) / 1024.0 / 1024.0

    def _probe_video_duration(self, video_path):
        clip = None
        try:
            clip = VideoFileClip(video_path)
            return float(clip.duration)
        except Exception as ex:
            print(f"[VIDEO_INPUT] failed to probe duration for {video_path}: {ex}")
            return None
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass

    def _build_video_proxy(self, source_video, preset):
        source_stat = os.stat(source_video)
        source_identity = (
            f"{os.path.abspath(source_video)}|{source_stat.st_size}|{source_stat.st_mtime_ns}"
        )
        source_hash = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()[:16]
        output_dir = os.path.join(GEMINI_VIDEO_PROXY_ROOT, source_hash)
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(source_video))[0]
        proxy_path = os.path.join(
            output_dir,
            f"{stem}.w{preset['width']}.fps{preset['fps']}.crf{preset['crf']}.mp4",
        )
        if os.path.exists(proxy_path) and os.path.getsize(proxy_path) > 0:
            print(f"[VIDEO_INPUT] reuse proxy: {proxy_path} ({self._video_size_mb(proxy_path):.2f} MB)")
            return proxy_path

        vf = (
            f"scale={preset['width']}:{preset['width']}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"fps={preset['fps']}"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_video,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(preset["crf"]),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-movflags",
            "+faststart",
            proxy_path,
        ]
        print("[VIDEO_INPUT] build proxy:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg proxy build failed for {source_video}, "
                f"returncode={proc.returncode}, stderr={proc.stderr[-2000:]}"
            )
        print(f"[VIDEO_INPUT] proxy ready: {proxy_path} ({self._video_size_mb(proxy_path):.2f} MB)")
        return proxy_path

    def _pick_initial_video_input(self, source_video):
        size_mb = self._video_size_mb(source_video)
        extension = Path(source_video).suffix.lower()
        if not gemini_supports_inline_video(source_video):
            print(
                f"[VIDEO_INPUT] {extension or '<no extension>'} is not listed by Gemini; "
                "convert to an MP4 proxy"
            )
            return self._build_video_proxy(source_video, GEMINI_VIDEO_PROXY_PRESETS[0]), 0
        if size_mb < GEMINI_VIDEO_PROXY_THRESHOLD_MB:
            return source_video, None
        print(
            f"[VIDEO_INPUT] source video is {size_mb:.2f} MB >= "
            f"{GEMINI_VIDEO_PROXY_THRESHOLD_MB:.2f} MB; build proxy"
        )
        return self._build_video_proxy(source_video, GEMINI_VIDEO_PROXY_PRESETS[0]), 0

    def _is_payload_too_large(self, error):
        text = str(error).lower()
        patterns = [
            "payload too large",
            "request too large",
            "request entity too large",
            "entity too large",
            "body too large",
            "content length",
            "exceed",
            "exceeds",
            "too many bytes",
            "413",
        ]
        return any(pattern in text for pattern in patterns)

    def _build_video_prompt(self, video_duration):
        direct_video_instruction = """

---
## 输入方式更新
本次输入不是按 FPS 抽取的视频帧，而是完整视频文件。请直接理解完整视频的连续视觉内容；如果视频包含音轨，也请结合原始音频辅助判断内容和高光。
原 prompt 中“从输入帧时间戳中选择 highlight_start/highlight_end”的要求不再适用；请基于完整视频自行判断高光边界，时间戳单位为秒，保留 1 位小数。
高光时间必须在视频真实时长范围内，且 highlight_start < highlight_end。
"""
        if video_duration is not None:
            direct_video_instruction += f"视频总时长约为 {video_duration:.3f} 秒。\n"
        return self.prompt + direct_video_instruction

    @retry(
        stop_max_attempt_number=5,
        wait_random_min=3000,
        wait_random_max=5000,
        retry_on_exception=_should_retry_llm_exception,
    )
    def run_video(self, video_path):
        cfg = get_model_cfg(self.model)
        if cfg.get("client_type") != "gemini_native":
            raise ValueError(f"--input-mode video only supports gemini_native models, got {self.model}")

        video_duration = self._probe_video_duration(video_path)
        prompt = self._build_video_prompt(video_duration)
        analysis_video, preset_idx = self._pick_initial_video_input(video_path)
        attempted = {os.path.abspath(analysis_video)}

        def call_once(path):
            mime_type = self._video_mime_type(path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:{mime_type};base64,{self._encode_video(path)}",
                                "mime_type": mime_type,
                            },
                        },
                    ],
                }
            ]
            t0 = time.time()
            response_text, input_token, output_token = self.runner.run(messages, model=self.model)
            print("Understanding Video Time Consumption: %.2f" % (time.time() - t0))
            print("Input Tokens: %s | Output Tokens: %s" % (input_token, output_token))
            return response_text, input_token, output_token

        try:
            res, input_token, output_token = call_once(analysis_video)
        except Exception as ex:
            if not self._is_payload_too_large(ex):
                raise
            last_error = ex
            print(f"[VIDEO_INPUT] request too large, retry with smaller proxy: {ex}")
            start_idx = 0 if preset_idx is None else preset_idx + 1
            for idx in range(start_idx, len(GEMINI_VIDEO_PROXY_PRESETS)):
                retry_video = self._build_video_proxy(video_path, GEMINI_VIDEO_PROXY_PRESETS[idx])
                retry_abs = os.path.abspath(retry_video)
                if retry_abs in attempted:
                    continue
                attempted.add(retry_abs)
                try:
                    analysis_video = retry_video
                    res, input_token, output_token = call_once(analysis_video)
                    break
                except Exception as retry_ex:
                    if not self._is_payload_too_large(retry_ex):
                        raise
                    last_error = retry_ex
                    print(f"[VIDEO_INPUT] proxy still too large: {retry_video}, error: {retry_ex}")
            else:
                raise last_error

        normalized = normalize_llm_json_response(res)
        parsed = json.loads(normalized)
        print(parsed)

        if video_duration is not None:
            for highlight in parsed.get("highlights", []):
                if "highlight_start" in highlight:
                    highlight["highlight_start"] = round(max(0.0, min(float(highlight["highlight_start"]), video_duration)), 1)
                if "highlight_end" in highlight:
                    highlight["highlight_end"] = round(max(0.0, min(float(highlight["highlight_end"]), video_duration)), 1)

        parsed["video_duration"] = video_duration
        parsed["understanding_input_mode"] = "video"
        if os.path.abspath(analysis_video) != os.path.abspath(video_path):
            parsed["analysis_proxy_video"] = analysis_video
        return parsed, input_token, output_token



class JointVideoUnderstandingLLM:
    """Joint-understanding LLM: one call per sample, understands all videos together.

    Input frames_by_video shape follows UnifiedWorkflowLLM. Output is compatible with
    Video-Understanding Agent JSON: {"understanding_results": {video_id: ...}}.
    """

    def __init__(self, model: str, prompt):
        self.model = model
        self.runner = ModelRunner()
        self.prompt = prompt

    @retry(
        stop_max_attempt_number=5,
        wait_random_min=3000,
        wait_random_max=5000,
        retry_on_exception=_should_retry_llm_exception,
    )
    def run(self, frames_by_video):
        user_input_collects = []
        user_input_collects.append({"type": "text", "text": self.prompt})
        user_input_collects.append({
            "type": "text",
            "text": f"\n\n## 输入视频帧序列 (共 {len(frames_by_video)} 个视频)\n",
        })

        for video_id, vinfo in frames_by_video.items():
            user_input_collects.append({
                "type": "text",
                "text": f"\n--- {video_id} (duration: {float(vinfo['duration']):.1f}s, {len(vinfo['frames'])} frames) ---\n",
            })
            for local_time, b64 in vinfo["frames"]:
                label = json.dumps(
                    {"video_id": video_id, "local_time": round(float(local_time), 1)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                user_input_collects.append({"type": "text", "text": label})
                user_input_collects.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    },
                })

        messages = [{"role": "user", "content": user_input_collects}]

        t0 = time.time()
        res, input_token, output_token = self.runner.run(messages, model=self.model)
        print("Joint Understanding Time Consumption: %.2f" % (time.time() - t0))
        print("Input Tokens: %s | Output Tokens: %s" % (input_token, output_token))

        res = normalize_llm_json_response(res)
        parsed = json.loads(res)
        if "understanding_results" not in parsed:
            raise KeyError("joint understanding output missing understanding_results")

        understanding_results = parsed["understanding_results"]
        expected_video_ids = list(frames_by_video.keys())
        missing = [vid for vid in expected_video_ids if vid not in understanding_results]
        extra = [vid for vid in understanding_results.keys() if vid not in frames_by_video]
        if missing:
            raise KeyError(f"joint understanding output missing videos: {missing}")
        if extra:
            raise KeyError(f"joint understanding output has unknown videos: {extra}")

        normalized_results = {}
        for video_id in expected_video_ids:
            vinfo = frames_by_video[video_id]
            video_result = understanding_results[video_id]
            time_stamps = [round(float(t), 1) for t, _ in vinfo["frames"]]
            if not time_stamps:
                raise ValueError(f"empty frame timestamps for {video_id}")
            highlights = video_result.get("highlights")
            if not isinstance(highlights, list) or not highlights:
                raise ValueError(f"empty highlights for {video_id}")

            for highlight in highlights:
                for field in ["highlight_start", "highlight_end"]:
                    if field not in highlight:
                        raise KeyError(f"missing {field} for {video_id}")
                    value = round(float(highlight[field]), 1)
                    if value not in time_stamps:
                        closest = time_stamps[int(np.argmin(abs(np.array(time_stamps) - value)))]
                        print("Refine %s %s: %f -> %f" % (video_id, field, value, closest))
                        value = closest
                    highlight[field] = value

            video_result["video_abs_frame_stamps"] = time_stamps
            video_result["video_name"] = vinfo["video_name"]
            normalized_results[video_id] = video_result

        parsed["understanding_results"] = normalized_results
        return parsed, input_token, output_token



def _validate_video_id_permutation(stage_name, video_ids, expected_video_ids):
    if not isinstance(video_ids, list):
        raise TypeError(f"{stage_name} video ids must be a list")

    missing = [vid for vid in expected_video_ids if vid not in video_ids]
    extra = [vid for vid in video_ids if vid not in expected_video_ids]
    duplicate = sorted({vid for vid in video_ids if video_ids.count(vid) > 1})

    if missing:
        raise KeyError(f"unified output {stage_name} missing videos: {missing}")
    if extra:
        raise KeyError(f"unified output {stage_name} has unknown videos: {extra}")
    if duplicate:
        raise KeyError(f"unified output {stage_name} has duplicate videos: {duplicate}")


def _extract_understanding_video_ids(understanding_results):
    if isinstance(understanding_results, dict):
        return list(understanding_results.keys())
    if isinstance(understanding_results, list):
        video_ids = []
        for item in understanding_results:
            if not isinstance(item, dict):
                raise TypeError("unified output understanding_results item must be a dict")
            if "video_id" not in item:
                raise KeyError("unified output understanding_results item missing video_id")
            video_ids.append(item["video_id"])
        return video_ids
    raise TypeError(f"unsupported unified understanding_results type: {type(understanding_results)}")


def _extract_result_video_ids(stage_name, results):
    if not isinstance(results, list):
        raise TypeError(f"unified output {stage_name} must be a list")
    video_ids = []
    for item in results:
        if not isinstance(item, dict):
            raise TypeError(f"unified output {stage_name} item must be a dict")
        if "video_id" not in item:
            raise KeyError(f"unified output {stage_name} item missing video_id")
        video_ids.append(item["video_id"])
    return video_ids


def _validate_unified_video_ids(parsed, expected_video_ids):
    if not isinstance(parsed, dict):
        raise TypeError("unified output must be a JSON object")

    understanding_results = parsed.get("understanding_results")
    if understanding_results is not None:
        _validate_video_id_permutation(
            "understanding_results",
            _extract_understanding_video_ids(understanding_results),
            expected_video_ids,
        )

    sort_result = parsed.get("sort", {}).get("result")
    _validate_video_id_permutation("sort.result", sort_result, expected_video_ids)

    arrange_results = parsed.get("arrange_results")
    _validate_video_id_permutation(
        "arrange_results",
        _extract_result_video_ids("arrange_results", arrange_results),
        expected_video_ids,
    )

    duration_adjustment = parsed.get("duration_adjustment")
    if duration_adjustment is not None and not isinstance(duration_adjustment, dict):
        raise TypeError("unified output duration_adjustment must be a dict")
    if isinstance(duration_adjustment, dict) and duration_adjustment:
        adjusted_results = duration_adjustment.get("adjusted_arrange_results")
        _validate_video_id_permutation(
            "duration_adjustment.adjusted_arrange_results",
            _extract_result_video_ids("duration_adjustment.adjusted_arrange_results", adjusted_results),
            expected_video_ids,
        )



class CutPlanningLLM:
    def __init__(self, model: str, prompt, input_mode="time_desc_frame"):
        self.model = model
        self.runner = ModelRunner()
        self.input_mode = input_mode
        self.prompt = build_cut_planning_prompt_for_input_mode(prompt, input_mode)

    @retry(
        stop_max_attempt_number=5,
        wait_random_min=3000,
        wait_random_max=5000,
        retry_on_exception=_should_retry_llm_exception,
    )
    def run(self, shuffled_understand_result, user_prompt, keyframes_by_video):
        prompt1 = self.prompt
        user_input_collects = []
        user_input_collects.append({"type": "text", "text": prompt1})
        user_input_collects.append({"type": "text", "text": "\n提供的素材理解结果为\n"})
        user_input_collects.append({"type": "text", "text": "{"})

        video_ids = list(shuffled_understand_result.keys())
        meta_keys = ["time", "scene/location", "video_duration"]

        for video_id in video_ids:
            video_info = shuffled_understand_result[video_id]
            meta_parts = []
            for mk in meta_keys:
                meta_parts.append(
                    json.dumps(mk, ensure_ascii=False, separators=(",", ":"))
                    + ":"
                    + json.dumps(video_info.get(mk, ""), ensure_ascii=False, separators=(",", ":"))
                )

            video_prefix = (
                json.dumps(video_id, ensure_ascii=False, separators=(",", ":"))
                + ":{"
                + ",".join(meta_parts)
                + ',"visual_segments":['
            )
            user_input_collects.append({"type": "text", "text": video_prefix})

            visual_segments = video_info.get("visual_segments", [])
            frames_list = keyframes_by_video.get(video_id)
            if frames_list is None:
                raise KeyError(f"missing keyframes for {video_id}")
            if len(frames_list) != len(visual_segments):
                raise ValueError(f"keyframes mismatch for {video_id}: visual_segments={len(visual_segments)} keyframes={len(frames_list)}")

            for seg, img_b64 in zip(visual_segments, frames_list):
                seg_str = json.dumps(seg, ensure_ascii=False, separators=(",", ":"))
                if img_b64 is None:
                    user_input_collects.append({"type": "text", "text": seg_str + ","})
                    continue

                seg_prefix = seg_str[:-1] + ',"keyframe":"'
                user_input_collects.append({"type": "text", "text": seg_prefix})
                user_input_collects.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"},
                    }
                )
                user_input_collects.append({"type": "text", "text": '"},'})

            if visual_segments:
                last_seg = user_input_collects.pop()
                user_input_collects.append({"type": "text", "text": last_seg["text"][:-1]})

            user_input_collects.append({"type": "text", "text": '],"asr_segments":'})
            user_input_collects.append({"type": "text", "text": json.dumps(video_info.get("asr_segments", []), ensure_ascii=False, separators=(",", ":"))})
            user_input_collects.append({"type": "text", "text": "},"})

        last_video = user_input_collects.pop()
        user_input_collects.append({"type": "text", "text": last_video["text"][:-1]})
        user_input_collects.append({"type": "text", "text": "}"})
        user_input_collects.append({"type": "text", "text": f"\n用户输入的prompt为{user_prompt}\n"})

        messages = [{"role": "user", "content": user_input_collects}]

        cut_planning_start = time.time()
        res, input_token, output_token = self.runner.run(messages, model=self.model)
        cut_planning_end = time.time()
        print("Cut-Planning Time Consumption: %.2f" % (cut_planning_end - cut_planning_start))
        print("Input Tokens: %s | Output Tokens: %s" % (input_token, output_token))

        res = normalize_llm_json_response(res)
        return json.loads(res)




class UnifiedWorkflowLLM:
    """Unified-workflow client used in the workflow-design ablation.

    Unlike the two-agent visual workflow, this client consumes all sampled
    source-video frames in one request and returns a cut plan directly.

    输入 frames_by_video 形态 (OrderedDict, 保留 video_id 顺序):
        {
            "video_0": {
                "video_name": "/path/to/IMG_001.mp4",
                "duration": 12.5,
                "frames": [
                    (0.0, base64_str),
                    (1.0, base64_str),
                    ...
                ],
            },
            "video_1": {...},
            ...
        }

    The output schema follows the active unified prompt.
    """

    def __init__(self, model: str, prompt):
        self.model = model
        self.runner = ModelRunner()
        self.prompt = prompt

    @retry(
        stop_max_attempt_number=50,
        wait_random_min=3000,
        wait_random_max=5000,
        retry_on_exception=_should_retry_llm_exception,
    )
    def run(self, frames_by_video, user_prompt):
        user_input_collects = []
        user_input_collects.append({"type": "text", "text": self.prompt})

        user_input_collects.append({
            "type": "text",
            "text": f"\n\n## 输入视频帧序列 (共 {len(frames_by_video)} 个视频)\n",
        })

        for video_id, vinfo in frames_by_video.items():
            user_input_collects.append({
                "type": "text",
                "text": f"\n--- {video_id} (duration: {float(vinfo['duration']):.1f}s, {len(vinfo['frames'])} frames) ---\n",
            })
            for local_time, b64 in vinfo["frames"]:
                label = json.dumps(
                    {"video_id": video_id, "local_time": round(float(local_time), 1)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                user_input_collects.append({"type": "text", "text": label})
                user_input_collects.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    },
                })

        user_input_collects.append({
            "type": "text",
            "text": f"\n\n用户输入的 prompt 为: {user_prompt}\n",
        })

        messages = [{"role": "user", "content": user_input_collects}]

        t0 = time.time()
        res, input_token, output_token = self.runner.run(messages, model=self.model)
        print("Unified Time Consumption: %.2f" % (time.time() - t0))
        print("Input Tokens: %s | Output Tokens: %s" % (input_token, output_token))

        raw_res = res
        normalized_res = normalize_llm_json_response(res)
        try:
            parsed = json.loads(normalized_res)
        except json.JSONDecodeError as ex:
            raise LLMJsonParseError(
                f"JSONDecodeError: {ex}",
                raw_response=raw_res,
                normalized_response=normalized_res,
            ) from ex
        _validate_unified_video_ids(parsed, list(frames_by_video.keys()))
        return parsed
