"""Validate repeated outputs from the unified-workflow ablation.

Unified-chain outputs keep all usable predictions under each arrange JSON:

- explicit understanding, if enabled:
  understanding_results
  or legacy get_arrange_result.arrange_result.understanding_results
- final arrange prediction:
  get_arrange_result.arrange_result.arrange_results
  or duration_adjustment.adjusted_arrange_results
- video_id -> real file-name mapping:
  shuffled_understand_result

The script always validates cut-plan results. It validates explicit
understanding only when the unified prompt version outputs
understanding_results, so it can also be used for without-understanding and
without-description ablations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_evaluation.visual_metrics import calculate_metrics


UNDERSTANDING_COLUMNS = [
    ("precision", "p"),
    ("recall", "r"),
    ("avg_iou", "iou"),
    ("empty_case", "empty_case"),
    ("extra_case", "extra_case"),
    ("empty_video", "empty_video"),
    ("extra_video", "extra_video"),
]

CUT_PLANNING_COLUMNS = [
    ("precision", "p"),
    ("recall", "r"),
    ("avg_iou", "iou"),
    ("kendalltau", "k"),
    ("empty_case", "empty_case"),
    ("extra_case", "extra_case"),
    ("empty_video", "empty_video"),
    ("extra_video", "extra_video"),
    ("empty_segments", "empty_segments"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value):
    if isinstance(value, float):
        text = f"{value:.3f}"
        return text.rstrip("0").rstrip(".") if "." in text else text
    return str(value)


def video_name_from_path(path):
    return os.path.splitext(str(path).split("/")[-1])[0]


def normalize_highlight(highlight):
    start = highlight.get("highlight_start", highlight.get("start_time"))
    end = highlight.get("highlight_end", highlight.get("end_time"))
    if start is None or end is None:
        return None
    try:
        return [round(float(start), 1), round(float(end), 1)]
    except (TypeError, ValueError):
        return None


def normalize_segment(segment):
    start = segment.get("start_time", segment.get("highlight_start"))
    end = segment.get("end_time", segment.get("highlight_end"))
    if start is None or end is None:
        return None
    try:
        return [round(float(start), 1), round(float(end), 1)]
    except (TypeError, ValueError):
        return None


def discover_test_ids(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        [child.name for child in root.iterdir() if child.is_dir() and child.name.startswith("test")],
        key=lambda x: (len(x), x),
    )


def resolve_cut_plan_dir(root, test_id, set_name):
    root = Path(root)
    direct = root / test_id / set_name / "cut_plans"
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"cannot find unified cut plans under {direct}")


def base_coverage(groundtruth, pred_case_ids):
    gt_cases = set(groundtruth.keys())
    pred_case_ids = set(pred_case_ids)
    missing_cases = sorted(gt_cases - pred_case_ids)
    extra_cases = sorted(pred_case_ids - gt_cases)
    return {
        "gt_cases": len(gt_cases),
        "pred_cases": len(pred_case_ids),
        "common_cases": len(gt_cases & pred_case_ids),
        "empty_case": len(missing_cases),
        "extra_case": len(extra_cases),
        "empty_case_names": missing_cases,
        "extra_case_names": extra_cases,
        "gt_videos_total": sum(len(videos) for videos in groundtruth.values()),
    }


def resolve_unified_arrange_result(file_content):
    return file_content["get_arrange_result"]["arrange_result"]


def resolve_video_mapping(file_content):
    if "shuffled_understand_result" in file_content:
        return file_content["shuffled_understand_result"]
    if "raw_understanding_result" in file_content:
        return file_content["raw_understanding_result"]
    if "understanding_results" in file_content:
        return file_content["understanding_results"]
    raise KeyError("shuffled_understand_result/raw_understanding_result/understanding_results")


def resolve_video_name(mapping, video_id):
    video_info = mapping[video_id]
    if isinstance(video_info, str):
        return video_name_from_path(video_info)
    if isinstance(video_info, dict):
        video_name = video_info.get("video_name") or video_info.get("path")
        if video_name:
            return video_name_from_path(video_name)
    raise KeyError(video_id)


def resolve_arrange_results(arrange_result_dict, json_path):
    duration_adjustment = arrange_result_dict.get("duration_adjustment")
    if duration_adjustment:
        adjusted = duration_adjustment.get("adjusted_arrange_results")
        if not isinstance(adjusted, list):
            raise KeyError(f"{json_path} has invalid adjusted_arrange_results")
        return adjusted

    arrange_results = arrange_result_dict.get("arrange_results")
    if not isinstance(arrange_results, list):
        raise KeyError(f"{json_path} has no valid arrange_results")
    return arrange_results


def final_segments(video):
    segments = video.get("adjusted_segments") or video.get("segments", [])
    if isinstance(segments, list):
        return segments
    return []


def resolve_understanding_results(file_content, arrange_result_dict):
    if "understanding_results" in file_content:
        understanding_results = file_content["understanding_results"]
    else:
        understanding_results = arrange_result_dict["understanding_results"]
    if isinstance(understanding_results, dict):
        return [
            {"video_id": video_id, **video_result}
            for video_id, video_result in understanding_results.items()
            if isinstance(video_result, dict)
        ]
    if isinstance(understanding_results, list):
        return understanding_results
    raise TypeError(f"unsupported understanding_results type: {type(understanding_results)}")


def has_explicit_understanding_result(file_content, arrange_result_dict):
    return "understanding_results" in file_content or "understanding_results" in arrange_result_dict


def filter_and_score(groundtruth, prediction, iou_threshold):
    common_case_ids = sorted(set(groundtruth.keys()) & set(prediction.keys()))
    gt_filtered = {case_id: groundtruth[case_id] for case_id in common_case_ids}
    pred_filtered = {case_id: prediction[case_id] for case_id in common_case_ids}
    metrics = calculate_metrics(gt_filtered, pred_filtered, iou_threshold=iou_threshold)
    metrics.pop("kendalltau_legal", None)
    return metrics


def evaluate_cut_plan_dir(cut_plan_dir, groundtruth, iou_threshold):
    json_files = sorted(Path(cut_plan_dir).glob("*.json"))
    coverage = base_coverage(groundtruth, [path.stem for path in json_files])

    prediction = {}
    empty_video = 0
    extra_video = 0
    empty_segments = 0
    malformed_case = 0
    unresolved_video_id = 0

    for json_path in json_files:
        sample_id = json_path.stem
        if sample_id not in groundtruth:
            continue

        prediction[sample_id] = {}
        gt_video_names = set(groundtruth[sample_id].keys())
        seen_pred_gt_videos = set()

        try:
            file_content = load_json(json_path)
            arrange_result_dict = resolve_unified_arrange_result(file_content)
            arrange_results = resolve_arrange_results(arrange_result_dict, json_path)
            video_mapping = resolve_video_mapping(file_content)
        except Exception:
            malformed_case += 1
            empty_video += len(gt_video_names)
            continue

        for video in arrange_results:
            if not isinstance(video, dict):
                extra_video += 1
                continue

            video_id = video.get("video_id")
            if video_id is None:
                extra_video += 1
                continue

            try:
                video_name = resolve_video_name(video_mapping, video_id)
            except Exception:
                unresolved_video_id += 1
                extra_video += 1
                continue

            if video_name not in gt_video_names:
                extra_video += 1
                continue

            prediction[sample_id][video_name] = {"highlights": []}
            for segment in final_segments(video):
                if not isinstance(segment, dict):
                    continue
                normalized = normalize_segment(segment)
                if normalized is not None:
                    prediction[sample_id][video_name]["highlights"].append(normalized)

            seen_pred_gt_videos.add(video_name)
            if not prediction[sample_id][video_name]["highlights"]:
                empty_segments += 1

        empty_video += len(gt_video_names - seen_pred_gt_videos)

    metrics = filter_and_score(groundtruth, prediction, iou_threshold)
    coverage.update({
        "empty_video": empty_video,
        "extra_video": extra_video,
        "empty_segments": empty_segments,
        "malformed_case": malformed_case,
        "unresolved_video_id": unresolved_video_id,
    })
    return {**metrics, **coverage}


def evaluate_understanding_dir(cut_plan_dir, groundtruth, iou_threshold):
    json_files = sorted(Path(cut_plan_dir).glob("*.json"))
    coverage = base_coverage(groundtruth, [path.stem for path in json_files])

    prediction = {}
    empty_video = 0
    extra_video = 0
    malformed_case = 0
    unresolved_video_id = 0
    missing_understanding_case = 0
    empty_highlights = 0
    has_explicit_understanding = False

    for json_path in json_files:
        sample_id = json_path.stem
        if sample_id not in groundtruth:
            continue

        gt_video_names = set(groundtruth[sample_id].keys())

        try:
            file_content = load_json(json_path)
            arrange_result_dict = resolve_unified_arrange_result(file_content)
            if not has_explicit_understanding_result(file_content, arrange_result_dict):
                missing_understanding_case += 1
                empty_video += len(gt_video_names)
                prediction[sample_id] = {}
                continue
            understanding_results = resolve_understanding_results(file_content, arrange_result_dict)
            video_mapping = resolve_video_mapping(file_content)
        except Exception:
            malformed_case += 1
            empty_video += len(gt_video_names)
            prediction[sample_id] = {}
            continue

        has_explicit_understanding = True
        prediction[sample_id] = {}
        seen_pred_gt_videos = set()

        for video_result in understanding_results:
            if not isinstance(video_result, dict):
                extra_video += 1
                continue

            video_id = video_result.get("video_id")
            if not video_id:
                extra_video += 1
                continue

            try:
                video_name = resolve_video_name(video_mapping, video_id)
            except Exception:
                unresolved_video_id += 1
                extra_video += 1
                continue

            if video_name not in gt_video_names:
                extra_video += 1
                continue

            prediction[sample_id][video_name] = {"highlights": []}
            highlights = video_result.get("highlights", [])
            if not isinstance(highlights, list):
                highlights = []
            for highlight in highlights:
                if not isinstance(highlight, dict):
                    continue
                normalized = normalize_highlight(highlight)
                if normalized is not None:
                    prediction[sample_id][video_name]["highlights"].append(normalized)

            seen_pred_gt_videos.add(video_name)
            if not prediction[sample_id][video_name]["highlights"]:
                empty_highlights += 1

        empty_video += len(gt_video_names - seen_pred_gt_videos)

    metrics = filter_and_score(groundtruth, prediction, iou_threshold)
    coverage.update({
        "empty_video": empty_video,
        "extra_video": extra_video,
        "malformed_case": malformed_case,
        "unresolved_video_id": unresolved_video_id,
        "missing_understanding_case": missing_understanding_case,
        "empty_highlights": empty_highlights,
        "has_explicit_understanding": has_explicit_understanding,
    })
    return {**metrics, **coverage}


def markdown_table(rows, set_names, columns):
    header1 = [""]
    header2 = ["test"]
    for idx, set_name in enumerate(set_names):
        if idx:
            header1.append("")
            header2.append("")
        header1.extend([set_name] + [""] * (len(columns) - 1))
        header2.extend([label for _, label in columns])

    lines = []
    lines.append("| " + " | ".join(header1) + " |")
    lines.append("| " + " | ".join(["---"] * len(header1)) + " |")
    lines.append("| " + " | ".join(header2) + " |")

    for test_id, by_set in rows:
        row = [test_id]
        for idx, set_name in enumerate(set_names):
            if idx:
                row.append("")
            values = by_set.get(set_name, {})
            row.extend(fmt(values.get(key, "")) for key, _ in columns)
        lines.append("| " + " | ".join(row) + " |")

    if rows:
        avg_row = ["average"]
        for idx, set_name in enumerate(set_names):
            if idx:
                avg_row.append("")
            for key, _ in columns:
                numeric_values = []
                for _, by_set in rows:
                    value = by_set.get(set_name, {}).get(key)
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)):
                        numeric_values.append(float(value))
                avg_row.append(fmt(sum(numeric_values) / len(numeric_values)) if numeric_values else "")
        lines.append("| " + " | ".join(avg_row) + " |")

    return "\n".join(lines)


def detail_lines(rows, set_names):
    lines = []
    for test_id, by_set in rows:
        for set_name in set_names:
            result = by_set.get(set_name, {})
            empty_cases = result.get("empty_case_names", [])
            extra_cases = result.get("extra_case_names", [])
            if empty_cases:
                lines.append(
                    f"- {test_id} / {set_name}: empty_case={len(empty_cases)} "
                    f"({', '.join(empty_cases)})"
                )
            if extra_cases:
                preview = ", ".join(extra_cases[:20])
                suffix = "" if len(extra_cases) <= 20 else f", ... total={len(extra_cases)}"
                lines.append(
                    f"- {test_id} / {set_name}: extra_case={len(extra_cases)} "
                    f"({preview}{suffix})"
                )
            for key in [
                "malformed_case",
                "unresolved_video_id",
                "missing_understanding_case",
                "empty_highlights",
            ]:
                value = result.get(key, 0)
                if value:
                    lines.append(f"- {test_id} / {set_name}: {key}={value}")
    if not lines:
        lines.append("- 无 case 级别新增/遗漏；无 malformed case。")
    return "\n".join(lines)


def write_stage_report(output_path, title, root, gt_paths, set_names, rows, columns, iou_threshold):
    notes = [
        "- `empty_case`：GT 中存在但该 test 输出目录缺少整个 case JSON。",
        "- `extra_case`：输出目录中存在但 GT 中不存在的 case JSON。",
        "- `empty_video`：在 GT/prediction common case 内，GT video 没有对应预测。",
        "- `extra_video`：在 GT/prediction common case 内，预测出现 GT 不存在的 video；无法解析的 video_id 也计入该列。",
    ]
    if any(key == "empty_segments" for key, _ in columns):
        notes.append("- `empty_segments`：cut plan 最终参与指标计算的视频输出了空 segments。")

    content = [
        f"# {title}",
        "",
        f"- root: `{root}`",
        f"- iou_threshold: `{iou_threshold}`",
        f"- set_names: `{', '.join(set_names)}`",
        f"- gt_paths: `{', '.join(str(p) for p in gt_paths)}`",
        "",
        "说明：",
        "",
        *notes,
        "",
        markdown_table(rows, set_names, columns),
        "",
        "## 覆盖细节",
        "",
        detail_lines(rows, set_names),
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def write_no_understanding_report(output_path, title, root, gt_paths, set_names, rows, iou_threshold):
    content = [
        f"# {title}",
        "",
        f"- root: `{root}`",
        f"- iou_threshold: `{iou_threshold}`",
        f"- set_names: `{', '.join(set_names)}`",
        f"- gt_paths: `{', '.join(str(p) for p in gt_paths)}`",
        "",
        "未检测到显式 `understanding_results` 或旧格式 `get_arrange_result.arrange_result.understanding_results`。",
        "该 unified 版本属于 without-understanding 输出链路，understanding 阶段没有可单独评测的中间结果；请查看 `unified_arrange.md`。",
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def evaluate_stage(root, test_ids, set_names, gt_paths, iou_threshold, stage):
    rows = []
    for test_id in test_ids:
        by_set = {}
        for set_name, gt_path in zip(set_names, gt_paths):
            gt = load_json(gt_path)
            cut_plan_dir = resolve_cut_plan_dir(root, test_id, set_name)
            if stage == "understanding":
                result = evaluate_understanding_dir(cut_plan_dir, gt, iou_threshold)
            elif stage == "cut_planning":
                result = evaluate_cut_plan_dir(cut_plan_dir, gt, iou_threshold)
            else:
                raise KeyError(stage)
            by_set[set_name] = result
        rows.append((test_id, by_set))
    return rows


def has_any_explicit_understanding(rows):
    return any(
        result.get("has_explicit_understanding")
        for _, by_set in rows
        for result in by_set.values()
    )


def detect_explicit_understanding(root, test_ids, set_names):
    for test_id in test_ids:
        for set_name in set_names:
            cut_plan_dir = resolve_cut_plan_dir(root, test_id, set_name)
            for json_path in sorted(cut_plan_dir.glob("*.json")):
                try:
                    file_content = load_json(json_path)
                    arrange_result_dict = resolve_unified_arrange_result(file_content)
                except Exception:
                    continue
                if has_explicit_understanding_result(file_content, arrange_result_dict):
                    return True
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-root", required=True)
    parser.add_argument("--gt-paths", nargs="+", required=True)
    parser.add_argument("--set-names", nargs="+", required=True)
    parser.add_argument("--test-ids", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--understanding-output-name", default="unified_understanding.md")
    parser.add_argument("--cut-planning-output-name", default="unified_cut_planning.md")
    parser.add_argument("--understanding-title", default="unified explicit understanding validation")
    parser.add_argument("--cut-planning-title", default="unified cut-planning validation")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-understanding",
        action="store_true",
        help="Only validate cut-plan results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.gt_paths) != len(args.set_names):
        raise SystemExit("--gt-paths and --set-names must have the same length.")

    root = Path(args.unified_root)
    test_ids = args.test_ids or discover_test_ids(root)
    if not test_ids:
        raise SystemExit("No test ids found. Pass --test-ids explicitly.")

    gt_paths = [Path(p) for p in args.gt_paths]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cut_plan_rows = evaluate_stage(root, test_ids, args.set_names, gt_paths, args.iou_threshold, "cut_planning")
    cut_plan_output = output_dir / args.cut_planning_output_name
    write_stage_report(
        cut_plan_output,
        args.cut_planning_title,
        args.unified_root,
        args.gt_paths,
        args.set_names,
        cut_plan_rows,
        CUT_PLANNING_COLUMNS,
        args.iou_threshold,
    )
    print(f"Saved {cut_plan_output}")

    if args.skip_understanding:
        return

    understanding_output = output_dir / args.understanding_output_name
    if not detect_explicit_understanding(root, test_ids, args.set_names):
        write_no_understanding_report(
            understanding_output,
            args.understanding_title,
            args.unified_root,
            args.gt_paths,
            args.set_names,
            [],
            args.iou_threshold,
        )
        print(f"Saved {understanding_output}")
        return

    understanding_rows = evaluate_stage(root, test_ids, args.set_names, gt_paths, args.iou_threshold, "understanding")
    if has_any_explicit_understanding(understanding_rows):
        write_stage_report(
            understanding_output,
            args.understanding_title,
            args.unified_root,
            args.gt_paths,
            args.set_names,
            understanding_rows,
            UNDERSTANDING_COLUMNS,
            args.iou_threshold,
        )
    else:
        write_no_understanding_report(
            understanding_output,
            args.understanding_title,
            args.unified_root,
            args.gt_paths,
            args.set_names,
            understanding_rows,
            args.iou_threshold,
        )
    print(f"Saved {understanding_output}")


if __name__ == "__main__":
    main()
