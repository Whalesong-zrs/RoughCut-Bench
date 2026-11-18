#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import decord
import numpy as np
from PIL import Image, ImageDraw

from visual_workflow.model_api import ModelRunner, get_model_cfg


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
EVALUATOR_PROMPT_VERSION = "v1"

DATA_ROOT = Path(os.environ.get("ROUGHCUT_DATA_ROOT") or PROJECT_ROOT / "data")
OUTPUT_ROOT = Path(os.environ.get("ROUGHCUT_OUTPUT_ROOT") or PROJECT_ROOT / "outputs")
SET1_MEDIA_ROOT = Path(os.environ.get("ROUGHCUT_SET1_MEDIA") or DATA_ROOT / "benchmark_set1")
SET1_LABELS_PATH = Path(os.environ.get("ROUGHCUT_SET1_LABELS") or DATA_ROOT / "set1_label.json")

@dataclass
class SegmentRef:
    segment_id: str
    media_path: str
    media_name: str
    start: float
    end: float
    duration: float
    label: str
    source_type: str
    score_weight: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human-aligned evaluation of rendered rough-cut videos.")
    parser.add_argument(
        "--result-dir",
        default=str(OUTPUT_ROOT / "rendered_cuts"),
        help="Directory containing rendered cuts.",
    )
    parser.add_argument(
        "--source-root",
        default=str(SET1_MEDIA_ROOT),
        help="Directory containing source folders like 1_resize, 2_resize, ...",
    )
    parser.add_argument(
        "--labels-path",
        default=str(SET1_LABELS_PATH),
        help="Label JSON used to read per-case user requirements. Requirements may be empty.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT / "human_aligned_evaluation"),
        help="Directory to save per-case JSON and summary CSV/JSON.",
    )
    parser.add_argument(
        "--report-csv-path",
        default="",
        help="Optional report CSV providing prompts/scenes and receiving augmented scores.",
    )
    parser.add_argument(
        "--report-output-csv",
        default="",
        help="Output CSV path for the augmented report. Defaults to output_dir/<original_name>_llm_eval.csv",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated case ids to evaluate, e.g. 1,2,3. Empty means all.",
    )
    parser.add_argument(
        "--test-case-count",
        "--limit",
        dest="test_case_count",
        type=int,
        default=-1,
        help="Only evaluate the first N cases after filtering. -1 means evaluate all cases.",
    )
    parser.add_argument(
        "--evidence-sample-fps",
        type=float,
        default=3.0,
        help="How many frames per second to sample from both edited and source videos.",
    )
    parser.add_argument(
        "--llm-max-edited-frames",
        type=int,
        default=30,
        help="Max number of sampled edited frames sent to the LLM.",
    )
    parser.add_argument(
        "--llm-max-source-reference-frames",
        type=int,
        default=60,
        help="Max number of sampled source frames sent to the LLM.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("VISUAL_EVAL_LLM_MODEL", "gemini-3-pro"),
        help="Model alias from visual_workflow/model_api.py for the evaluator.",
    )
    parser.add_argument(
        "--max-still-images-per-case",
        type=int,
        default=0,
        help="Optional cap for still images per case. 0 means no extra cap.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for case-level parallel evaluation. 1 means single process.",
    )
    return parser.parse_args()


def is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def natural_key(text: str) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def extract_case_id(edited_path: Path) -> Optional[str]:
    stem = edited_path.stem
    if re.match(r"^\d+_resize$", stem) or re.match(r"^batch2_\d+$", stem):
        return stem

    match = re.match(r"^(\d+)(?:_|-|\b)", stem)
    if match:
        return f"{match.group(1)}_resize"

    return stem or None


def case_sort_key(case_id: Any) -> List[object]:
    return natural_key(str(case_id))


def case_report_key(case_id: str) -> Optional[int]:
    match = re.match(r"^(\d+)(?:_|$)", str(case_id))
    if not match:
        return None
    return int(match.group(1))


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    content = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n?```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        content = fenced.group("body").strip()
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


class ModelJudgeLLM:
    """Adapter from evaluator inputs to the shared provider-neutral runner."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.provider = get_model_cfg(model)["provider"]
        self.runner = ModelRunner()

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        with open(image_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def chat(self, system: str, user: str, image_paths: Optional[List[str]] = None) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user}]
        for image_path in image_paths or []:
            content.append({"type": "text", "text": f"Evidence image: {Path(image_path).name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image_path)},
                }
            )
        response, _, _ = self.runner.run(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            self.model,
        )
        return response


def build_llm_judge(args: argparse.Namespace) -> ModelJudgeLLM:
    return ModelJudgeLLM(args.llm_model)


class MediaSampler:
    def __init__(self) -> None:
        self._video_cache: Dict[str, Tuple[decord.VideoReader, np.ndarray]] = {}
        self._image_cache: Dict[str, Image.Image] = {}
        decord.bridge.set_bridge("torch")

    def _load_video(self, path: str) -> Tuple[decord.VideoReader, np.ndarray]:
        cached = self._video_cache.get(path)
        if cached is not None:
            return cached
        reader = decord.VideoReader(path, ctx=decord.cpu(0))
        if len(reader) <= 0:
            raise ValueError(f"empty video: {path}")
        timestamps = reader.get_frame_timestamp(range(len(reader))).mean(-1)
        cached = (reader, np.asarray(timestamps))
        self._video_cache[path] = cached
        return cached

    def duration(self, path: str, fallback: Optional[float] = None) -> float:
        if is_image_path(path):
            return float(fallback if fallback is not None else 2.0)
        if not is_video_path(path):
            if fallback is not None:
                return float(fallback)
            raise ValueError(f"unsupported media: {path}")
        _, timestamps = self._load_video(path)
        if len(timestamps) == 0:
            return float(fallback or 0.0)
        return float(timestamps[-1])

    def get_frames(self, path: str, timestamps: Sequence[float]) -> List[Image.Image]:
        if is_image_path(path):
            if path not in self._image_cache:
                self._image_cache[path] = Image.open(path).convert("RGB")
            return [self._image_cache[path].copy() for _ in timestamps]

        reader, times_in_vid = self._load_video(path)
        indices = np.searchsorted(times_in_vid, timestamps)
        safe_indices = [int(min(max(int(i), 0), len(reader) - 1)) for i in indices]
        batch = reader.get_batch(safe_indices)
        return [Image.fromarray(batch[i].numpy()).convert("RGB") for i in range(batch.shape[0])]

    def clear_cache(self) -> None:
        self._video_cache.clear()
        for image in self._image_cache.values():
            try:
                image.close()
            except Exception:
                pass
        self._image_cache.clear()


def select_evenly_spaced_indices(length: int, max_items: int) -> List[int]:
    if length <= 0 or max_items <= 0:
        return []
    if length <= max_items:
        return list(range(length))
    indices = np.linspace(0, length - 1, max_items, endpoint=True)
    return sorted({int(round(x)) for x in indices})


def build_frame_preview(
    sampler: MediaSampler,
    media_path: str,
    timestamp: float,
    title: str,
    out_path: Path,
) -> str:
    if is_image_path(media_path):
        timestamps = [0.0]
    else:
        timestamps = [max(0.0, float(timestamp))]

    frames = sampler.get_frames(media_path, timestamps)
    frame = frames[0].resize((320, 320)).convert("RGB")
    pad = 12
    header_h = 56
    canvas = Image.new("RGB", (320 + pad * 2, 320 + header_h + pad * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header_h), fill=(240, 243, 247))
    draw.text((12, 10), title[:80], fill=(0, 0, 0))
    draw.text((12, 32), f"{timestamps[0]:.2f}s", fill=(70, 70, 70))
    x = pad
    y = header_h + pad
    canvas.paste(frame, (x, y))
    draw.rectangle((x, y, x + 90, y + 22), fill=(0, 0, 0))
    draw.text((x + 6, y + 3), f"{timestamps[0]:.2f}s", fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=90)
    return str(out_path)


def read_labels(labels_path: str) -> Dict[str, dict]:
    if not labels_path or not os.path.exists(labels_path):
        return {}
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_report_column_map(report_csv_path: str, column_name: str) -> Dict[int, str]:
    if not report_csv_path or not os.path.exists(report_csv_path):
        return {}

    with open(report_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}

    header = rows[1]
    seq_idx = -1
    value_idx = -1
    for idx, name in enumerate(header):
        normalized = str(name).strip().lower()
        if normalized == "序号":
            seq_idx = idx
        elif str(name).strip() == column_name:
            value_idx = idx

    if seq_idx < 0 or value_idx < 0:
        return {}

    value_map: Dict[int, str] = {}
    for row in rows[2:]:
        if seq_idx >= len(row) or value_idx >= len(row):
            continue
        seq_text = str(row[seq_idx]).strip()
        if not seq_text.isdigit():
            continue
        value_text = str(row[value_idx]).strip()
        if value_text:
            value_map[int(seq_text)] = value_text
    return value_map


def read_report_prompts(report_csv_path: str) -> Dict[int, str]:
    return read_report_column_map(report_csv_path, "Prompt")


def read_report_scenes(report_csv_path: str) -> Dict[int, str]:
    return read_report_column_map(report_csv_path, "场景")


def build_case_prompt(case_labels: Dict[str, dict]) -> str:
    for meta in case_labels.values():
        prompt = meta.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    return ""


def build_metric_reason(result_dict: Optional[Dict[str, Any]], reason_key: str) -> str:
    if not result_dict:
        return ""
    value = result_dict.get(reason_key, "")
    return value if isinstance(value, str) else ""


def build_llm_prompt(
    case_id: str,
    prompt: str,
    scene: str,
    edited_duration: float,
    edited_frame_count: int,
    edited_entries: List[dict],
    source_reference_entries: List[dict],
    evidence_sample_fps: float,
) -> Tuple[str, str]:
    system_prompt = """
你是一位专业的视频剪辑专家，要对其他剪辑助理输出的粗剪成片结果进行打分。
你需要基于「用户剪辑指令」「原片参考帧证据」「成片采样帧证据」对成片做视觉层面的评估。

特别注意：
1. 只评估视觉内容，不考虑音频、配乐、旁白、口播、字幕、ASR。
2. 不需要关注转场包装、特效包装、调色等后期因素，重点看粗剪阶段的素材选择、时长和排序。
3. 所有证据图片都是按固定采样频率抽出来的单帧图片，并附带时间戳信息。
4. 原片参考帧和成片采样帧都按统一采样规则得到，采样频率约为每秒 %.3f 帧。
5. 证据按时间顺序给出，你需要结合顺序进行整体判断。
6. 一定要注意图片中的视觉信息是否能够传达出内部逻辑. 比如:
   - 在某些特效中, 当完成某件事情, 计数增加或者某些元素变化, 这会反应片段内部的时间逻辑.
   - 在某些片段中体现了时空或状态变化, 当他们能够被判断是同一类物品或者场景的变化时,内部隐含了关联和逻辑.
   

## 当前输入格式
输入是一个 JSON dict，主要字段如下：
1. `user_prompt`
   - 用户对成片的剪辑要求。

2. `scene`
   - 当前 case 的场景类别，如果提供则可作为辅助背景信息，但不要把它当成硬约束。

3. `edited_summary`
   - `edited_duration`：成片总时长（秒）
   - `edited_frame_count`：送入模型的成片采样帧数量

4. `edited_frames`
   - 一个列表，按成片时间顺序排列。
   - 每个元素包含：
     - `id`
     - `image`
     - `timestamp`

5. `source_reference_frames`
   - 一个列表，表示原片按时间流采样得到的参考帧。
   - 每个元素包含：
     - `id`
     - `image`
      - `timestamp`

## 评分维度
请对下列 5 个维度进行 1-5 分打分，5 分最好，1 分最差，并给出中文简要原因。

1. 用户 prompt 遵循程度（prompt_follow_score）
   - 重点看成片是否满足用户的剪辑指令要求.
   - 只关注剪辑指令是否得到满足, 诸如用户希望的顺序, 主体, 主题, 偏好, 风格, 整体时长等, 不要因为高光, 排序等具体的细节因素而影响扣分.
   - 当用户不提供prompt的时候, 完全忽略这一项的评估, 返回0分, 并且不要让他影响overall分数的判断.
   - 例如：即使剪辑出来的视频中, 某些片段信息不够精炼, 或者没有截全画面, 但只要符合用户需要的指令,就应该给高分.
   - 例如, 用户期望时长一分钟, 而得到的视频也是一分钟, 即使内部顺序或高光有问题,或者某个片段过长, 某些重点片段过短, 也应该给高分.
   - 只根据当前可见的视觉信息判断，不要因为缺少音频信息而扣分。

2. 高光覆盖度（highlight_score）
   - 重点看成片是否覆盖了原片参考流中的关键视觉内容、代表性场景和重要事件.
   - 严格且审慎的判断 片段是否关键. 关键的片段需要提供增量信息，对主线有所帮助或者是必须的.
   - 截取部分关键片段不连贯等, 应该降低分数.
   - 如果成片明显漏掉原片中多个关键场景、主体或事件，应降低分数, 如果成片覆盖了原片里多数重要内容，且覆盖比较充分，则应提高分数。
   - 如果成片选择的片段不完整, 应该降低分数, 比如, 在剪辑竞技类游戏的时候, 高光操作或击杀场景是需要重点保留的,尤其是越猛的击杀表现越需要保留.

3. 素材冗余度（redundancy_score）
   - 重点看成片选择的内容质量是否足够高, 是否存在较多的冗余内容, 是否选择了主题正确但是质量低的内容, 某些素材的时长是否舒适, 节奏是否拖沓或者过碎.
   - 需要综合比较：成片采样帧 vs 原片参考帧。
   - 如果成片选入了较多冗余、低信息量或明显不关键的内容，应降低分数。
   - 如果成片选入了较多主题正确但是质量低的内容，应降低分数.

4. 排序准确度（ordering_score）
   - 重点看成片镜头顺序、场景组织、故事推进是否自然。
   - 注意空间线/场景线是否混乱，是否有明显跳跃、来回切换、视觉组织不稳定的问题, 以及场景间的组织和变换是否突兀.
   - 注意成片中各个片段的组织是否具备关联性:
        - [同类产品的连续介绍]: 同一内容(或产品)的介绍一般要连续出现, 并且在单个产品的介绍片段内的排序呈现出递进或者逻辑关系
        - [vlog等视频的时间顺序]: 而vlog等视频往往要按照合理的时间顺序排列. 
        - [变化前后对应的片段]: 比如展示一个事物的前后变化的时候, 如果按照"变化前的所有片段 + 变化后的所有片段", 前后两段的每个子片段的内部排序应具备一致性和关联性, 能够相互呼应.
        - 当这种具备明显一致性或关联性的片段被分散的时候, 应该降低分数, 如果分散程度较高甚至随机, 需要严格扣分.

5. 整体粗剪质量（overall_visual_quality_score）
   - 这是综合印象分。
   - 需要综合考虑前面所有维度，但不要简单平均，而是给出整体观看感受上的总评。

## 打分要求
1. 分数必须是 1 到 5 的整数。
2. 原因必须使用中文，简洁明确，避免空话。
3. 如果证据不足以支持极端判断，请使用中间分，不要过度自信。
4. 不要输出推理过程，不要输出 markdown，不要输出代码块，只输出一个合法 JSON dict。

## 输出格式
{
  "prompt_follow_score": 1,
  "prompt_follow_score_reason": "",
  "highlight_score": 1,
  "highlight_score_reason": "",
  "redundancy_score": 1,
  "redundancy_score_reason": "",
  "ordering_score": 1,
  "ordering_score_reason": "",
  "overall_visual_quality_score": 1,
  "overall_visual_quality_score_reason": ""
}
""".strip()

    user_payload = {
        "task": "Judge a rough-cut edited video from visual evidence only.",
        "case_id": case_id,
        "user_prompt": prompt,
        "scene": scene,
        "edited_summary": {
            "edited_duration": edited_duration,
            "edited_frame_count": edited_frame_count,
        },
        "evidence_description": {
            "edited_frames": f"These are sampled frames from the edited video in chronological order at about {evidence_sample_fps} fps.",
            "source_reference_frames": f"These are sampled frames from the original source materials in chronological order at about {evidence_sample_fps} fps.",
        },
        "edited_frames": edited_entries,
        "source_reference_frames": source_reference_entries,
        "instruction": "Please score strictly according to the schema in the system prompt. Use the evidence image filenames to inspect the attached images. Output concise Chinese reasons.",
    }
    return system_prompt % evidence_sample_fps, json.dumps(user_payload, ensure_ascii=False, indent=2)


def build_case_evidence(
    sampler: MediaSampler,
    edited_path: Path,
    edited_frames: List[SegmentRef],
    source_reference_frames: List[SegmentRef],
    case_assets_dir: Path,
    args: argparse.Namespace,
) -> Tuple[List[str], List[dict], List[dict]]:
    image_paths: List[str] = []
    edited_entries: List[dict] = []
    source_reference_entries: List[dict] = []

    case_assets_dir.mkdir(parents=True, exist_ok=True)

    chosen_edited_frames = choose_sampled_frames(edited_frames, args.llm_max_edited_frames)
    for export_idx, frame_ref in enumerate(chosen_edited_frames):
        out_path = case_assets_dir / f"edited_{export_idx:03d}.jpg"
        image_paths.append(
            build_frame_preview(
                sampler=sampler,
                media_path=str(edited_path),
                timestamp=frame_ref.start,
                title=f"Edited Frame E{export_idx + 1}",
                out_path=out_path,
            )
        )
        edited_entries.append(
            {
                "id": f"E{export_idx + 1}",
                "image": out_path.name,
                "timestamp": round(frame_ref.start, 3),
            }
        )

    chosen_refs = choose_sampled_frames(source_reference_frames, args.llm_max_source_reference_frames)
    for idx, seg in enumerate(chosen_refs):
        out_path = case_assets_dir / f"source_ref_{idx:03d}.jpg"
        image_paths.append(
            build_frame_preview(
                sampler=sampler,
                media_path=seg.media_path,
                timestamp=seg.start,
                title=f"Source Frame R{idx + 1}",
                out_path=out_path,
            )
        )
        source_reference_entries.append(
            {
                "id": f"R{idx + 1}",
                "image": out_path.name,
                "timestamp": round(seg.start, 3),
            }
        )

    return image_paths, edited_entries, source_reference_entries


def collect_source_media_paths(source_case_dir: Path) -> List[str]:
    if not source_case_dir.exists():
        return []
    nested_source_dir = source_case_dir / "source_videos"
    if nested_source_dir.is_dir():
        source_case_dir = nested_source_dir
    media_paths = []
    for entry in sorted(source_case_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if entry.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
            media_paths.append(str(entry))
    return media_paths


def build_sampled_frame_segments(
    sampler: MediaSampler,
    media_paths: List[str],
    sample_fps: float,
    max_still_images_per_case: int,
) -> List[SegmentRef]:
    segments: List[SegmentRef] = []
    sample_fps = max(0.1, float(sample_fps))
    still_count = 0
    for media_path in media_paths:
        media_name = Path(media_path).stem
        if is_image_path(media_path):
            if max_still_images_per_case > 0 and still_count >= max_still_images_per_case:
                continue
            segments.append(
                SegmentRef(
                    segment_id=f"{media_name}_frame_0",
                    media_path=media_path,
                    media_name=media_name,
                    start=0.0,
                    end=0.0,
                    duration=0.0,
                    label="reference",
                    source_type="image",
                    score_weight=1.0,
                )
            )
            still_count += 1
            continue

        media_duration = sampler.duration(media_path, fallback=0.0)
        if media_duration <= 0.0:
            continue
        frame_count = max(int(math.ceil(media_duration * sample_fps)), 1)
        timestamps = np.linspace(0.0, max(media_duration - 1e-3, 1e-3), frame_count, endpoint=True).tolist()
        for idx, timestamp in enumerate(timestamps):
            segments.append(
                SegmentRef(
                    segment_id=f"{media_name}_frame_{idx}",
                    media_path=media_path,
                    media_name=media_name,
                    start=float(timestamp),
                    end=float(timestamp),
                    duration=0.0,
                    label="reference",
                    source_type="video",
                    score_weight=1.0,
                )
            )
    return segments


def choose_sampled_frames(segments: List[SegmentRef], max_segments: int) -> List[SegmentRef]:
    if max_segments <= 0:
        return list(segments)
    chosen_indices = select_evenly_spaced_indices(len(segments), max_segments)
    return [segments[idx] for idx in chosen_indices]

def evaluate_case(
    llm_judge: ModelJudgeLLM,
    edited_path: Path,
    case_id: str,
    source_root: Path,
    labels_data: Dict[str, dict],
    report_prompts: Dict[int, str],
    report_scenes: Dict[int, str],
    sampler: MediaSampler,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    batch_key = str(case_id)
    source_case_dir = source_root / batch_key
    source_media_paths = collect_source_media_paths(source_case_dir)
    if not source_media_paths:
        raise FileNotFoundError(f"no source media found for case {case_id} under {source_case_dir}")
    case_labels = labels_data.get(batch_key, {})
    report_key = case_report_key(batch_key)
    prompt = (report_prompts.get(report_key, "") if report_key is not None else "").strip() or build_case_prompt(case_labels)
    scene = (report_scenes.get(report_key, "") if report_key is not None else "").strip()
    edited_duration = round(sampler.duration(str(edited_path), fallback=0.0), 3)
    edited_frames = build_sampled_frame_segments(
        sampler=sampler,
        media_paths=[str(edited_path)],
        sample_fps=args.evidence_sample_fps,
        max_still_images_per_case=args.max_still_images_per_case,
    )
    source_reference_frames = build_sampled_frame_segments(
        sampler=sampler,
        media_paths=source_media_paths,
        sample_fps=args.evidence_sample_fps,
        max_still_images_per_case=args.max_still_images_per_case,
    )
    case_assets_dir = output_dir / "llm_assets" / f"case_{case_id}"
    image_paths, edited_entries, source_reference_entries = build_case_evidence(
        sampler=sampler,
        edited_path=edited_path,
        edited_frames=edited_frames,
        source_reference_frames=source_reference_frames,
        case_assets_dir=case_assets_dir,
        args=args,
    )

    system_prompt, user_prompt = build_llm_prompt(
        case_id=case_id,
        prompt=prompt,
        scene=scene,
        edited_duration=edited_duration,
        edited_frame_count=len(edited_entries),
        edited_entries=edited_entries,
        source_reference_entries=source_reference_entries,
        evidence_sample_fps=args.evidence_sample_fps,
    )

    raw_response = llm_judge.chat(system=system_prompt, user=user_prompt, image_paths=image_paths)
    parsed = safe_json_loads(raw_response)
    if parsed is None:
        raise ValueError(
            f"case {case_id}: evaluator returned invalid JSON: {raw_response[:500]!r}"
        )

    metrics: Dict[str, Optional[float]] = {}
    reasons: Dict[str, str] = {}
    metric_keys = [
        "highlight_score",
        "redundancy_score",
        "ordering_score",
        "prompt_follow_score",
        "overall_visual_quality_score",
    ]
    for key in metric_keys:
        value = parsed.get(key)
        try:
            numeric_value = float(value)
            if key == "prompt_follow_score":
                metrics[key] = round(clamp(numeric_value, 0.0, 5.0), 3)
            else:
                metrics[key] = round(clamp(numeric_value, 1.0, 5.0), 3)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"case {case_id}: evaluator response has invalid {key}: {value!r}"
            ) from exc
        reasons[key] = build_metric_reason(parsed, f"{key}_reason")

    return {
        "case_id": case_id,
        "batch_key": batch_key,
        "edited_video": str(edited_path),
        "source_case_dir": str(source_case_dir),
        "source_exists": source_case_dir.exists(),
        "source_media_count": len(source_media_paths),
        "prompt": prompt,
        "metrics": metrics,
        "reasons": reasons,
        "details": {
            "edited_duration": edited_duration,
            "edited_frame_count": len(edited_entries),
            "edited_frames": edited_entries,
            "source_reference_frames": source_reference_entries,
            "evidence_images": image_paths,
        },
        "llm": {
            "provider": llm_judge.provider,
            "model": args.llm_model,
            "prompt_version": EVALUATOR_PROMPT_VERSION,
            "raw_response": raw_response,
            "parsed": parsed,
        },
    }


def evaluate_case_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    args = argparse.Namespace(**job["args"])
    edited_path = Path(job["edited_path"])
    case_id = str(job["case_id"])
    print(f"[eval] case={case_id} file={edited_path.name}")
    sampler = MediaSampler()
    llm_judge = build_llm_judge(args)
    try:
        return evaluate_case(
            llm_judge=llm_judge,
            edited_path=edited_path,
            case_id=case_id,
            source_root=Path(args.source_root),
            labels_data=job["labels_data"],
            report_prompts=job["report_prompts"],
            report_scenes=job["report_scenes"],
            sampler=sampler,
            output_dir=Path(args.output_dir),
            args=args,
        )
    finally:
        sampler.clear_cache()


def iter_edited_videos(result_dir: Path, case_ids: Optional[set], test_case_count: int) -> Iterable[Path]:
    count = 0
    for path in sorted(result_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        case_id = extract_case_id(path)
        if case_id is None:
            continue
        if case_ids and case_id not in case_ids:
            continue
        yield path
        count += 1
        if test_case_count > 0 and count >= test_case_count:
            break


def write_summary_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "case_id",
        "batch_key",
        "edited_video",
        "source_exists",
        "source_media_count",
        "highlight_score",
        "redundancy_score",
        "ordering_score",
        "prompt_follow_score",
        "overall_visual_quality_score",
        "edited_duration",
        "edited_frame_count",
        "prompt",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "case_id": item["case_id"],
                    "batch_key": item["batch_key"],
                    "edited_video": item["edited_video"],
                    "source_exists": item["source_exists"],
                    "source_media_count": item["source_media_count"],
                    "highlight_score": item["metrics"].get("highlight_score"),
                    "redundancy_score": item["metrics"].get("redundancy_score"),
                    "ordering_score": item["metrics"].get("ordering_score"),
                    "prompt_follow_score": item["metrics"].get("prompt_follow_score"),
                    "overall_visual_quality_score": item["metrics"].get("overall_visual_quality_score"),
                    "edited_duration": item["details"].get("edited_duration"),
                    "edited_frame_count": item["details"].get("edited_frame_count"),
                    "prompt": item["prompt"],
                }
            )


def build_report_row_map(results: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    row_map: Dict[int, Dict[str, Any]] = {}
    for item in results:
        report_key = case_report_key(str(item["case_id"]))
        if report_key is None:
            continue
        metrics = item.get("metrics", {})
        reasons = item.get("reasons", {})
        details = item.get("details", {})
        llm_info = item.get("llm", {})

        score_summary = " | ".join(
            [
                f"prompt_follow={metrics.get('prompt_follow_score', '')}",
                f"highlight={metrics.get('highlight_score', '')}",
                f"redundancy={metrics.get('redundancy_score', '')}",
                f"ordering={metrics.get('ordering_score', '')}",
                f"overall={metrics.get('overall_visual_quality_score', '')}",
            ]
        )

        reason_summary = " | ".join(
            [
                f"prompt_follow: {reasons.get('prompt_follow_score', '')}",
                f"highlight: {reasons.get('highlight_score', '')}",
                f"redundancy: {reasons.get('redundancy_score', '')}",
                f"ordering: {reasons.get('ordering_score', '')}",
                f"overall: {reasons.get('overall_visual_quality_score', '')}",
            ]
        )

        row_map[report_key] = {
            "llm_score_summary": score_summary,
            "llm_reason_summary": reason_summary,
            "llm_raw_response": llm_info.get("raw_response", ""),
            "llm_prompt": item.get("prompt", ""),
            "llm_provider": llm_info.get("provider", ""),
            "llm_model": llm_info.get("model", ""),
            "llm_edited_duration": details.get("edited_duration"),
            "llm_edited_frame_count": details.get("edited_frame_count"),
        }
    return row_map


def write_augmented_report_csv(
    source_csv_path: Path,
    output_csv_path: Path,
    results: List[Dict[str, Any]],
) -> None:
    if not source_csv_path.exists():
        return

    with open(source_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return

    extra_columns = [
        "llm_score_summary",
        "llm_reason_summary",
        "llm_raw_response",
        "llm_prompt",
        "llm_provider",
        "llm_model",
        "llm_edited_duration",
        "llm_edited_frame_count",
    ]
    result_map = build_report_row_map(results)

    output_rows: List[List[str]] = []
    if len(rows) >= 1:
        output_rows.append(rows[0] + [""] * len(extra_columns))
    if len(rows) >= 2:
        output_rows.append(rows[1] + extra_columns)

    for row in rows[2:]:
        if not row:
            output_rows.append(row)
            continue

        seq_text = row[0].strip() if len(row) > 0 else ""
        result_payload: Dict[str, Any] = {}
        if seq_text.isdigit():
            result_payload = result_map.get(int(seq_text), {})

        appended = []
        for column in extra_columns:
            value = result_payload.get(column, "")
            appended.append("" if value is None else str(value))

        output_rows.append(row + appended)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)


def build_aggregate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"case_count": 0}

    def avg_metric(key: str) -> Optional[float]:
        values = [item["metrics"].get(key) for item in results if item["metrics"].get(key) is not None]
        if not values:
            return None
        return round(float(np.mean(values)), 3)

    return {
        "case_count": len(results),
        "avg_highlight_score": avg_metric("highlight_score"),
        "avg_redundancy_score": avg_metric("redundancy_score"),
        "avg_ordering_score": avg_metric("ordering_score"),
        "avg_prompt_follow_score": avg_metric("prompt_follow_score"),
        "avg_overall_visual_quality_score": avg_metric("overall_visual_quality_score"),
    }


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir)
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = None
    if args.case_ids.strip():
        case_ids = {part.strip() for part in args.case_ids.split(",") if part.strip()}

    labels_data = read_labels(args.labels_path)
    report_prompts = read_report_prompts(args.report_csv_path)
    report_scenes = read_report_scenes(args.report_csv_path)
    edited_paths = list(
        iter_edited_videos(
            result_dir,
            case_ids=case_ids,
            test_case_count=args.test_case_count,
        )
    )
    if not edited_paths:
        raise FileNotFoundError(f"no rendered cuts found under {result_dir}")

    results: List[Dict[str, Any]] = []
    if args.num_workers <= 1:
        sampler = MediaSampler()
        llm_judge = build_llm_judge(args)
        for edited_path in edited_paths:
            case_id = extract_case_id(edited_path)
            if case_id is None:
                continue
            print(f"[eval] case={case_id} file={edited_path.name}")
            result = evaluate_case(
                llm_judge=llm_judge,
                edited_path=edited_path,
                case_id=case_id,
                source_root=source_root,
                labels_data=labels_data,
                report_prompts=report_prompts,
                report_scenes=report_scenes,
                sampler=sampler,
                output_dir=output_dir,
                args=args,
            )
            results.append(result)
            with open(output_dir / f"case_{case_id}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        jobs: List[Dict[str, Any]] = []
        args_dict = vars(args).copy()
        for edited_path in edited_paths:
            case_id = extract_case_id(edited_path)
            if case_id is None:
                continue
            jobs.append(
                {
                    "args": args_dict,
                    "edited_path": str(edited_path),
                    "case_id": case_id,
                    "labels_data": labels_data,
                    "report_prompts": report_prompts,
                    "report_scenes": report_scenes,
                }
            )

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_case_id = {
                executor.submit(evaluate_case_worker, job): job["case_id"] for job in jobs
            }
            for future in as_completed(future_to_case_id):
                case_id = future_to_case_id[future]
                result = future.result()
                results.append(result)
                with open(output_dir / f"case_{case_id}.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

    results.sort(key=lambda item: case_sort_key(item["case_id"]))

    summary = build_aggregate_summary(results)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_csv(results, output_dir / "summary.csv")

    if args.report_csv_path.strip():
        report_csv_path = Path(args.report_csv_path)
        if args.report_output_csv.strip():
            report_output_csv = Path(args.report_output_csv)
        else:
            report_output_csv = output_dir / f"{report_csv_path.stem}_llm_eval.csv"
        write_augmented_report_csv(report_csv_path, report_output_csv, results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
