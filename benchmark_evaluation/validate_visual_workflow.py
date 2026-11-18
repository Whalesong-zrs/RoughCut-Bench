"""Validate repeated outputs from the two-agent visual workflow.

The script uses the same metric function as the existing evaluation scripts,
but adds coverage checks:

- empty_case: GT cases with no prediction JSON.
- extra_case: prediction JSON cases not present in GT.
- empty_video: GT videos missing inside common GT/prediction cases.
- extra_video: predicted videos not present in GT inside common cases.
- empty_segments: cut-plan videos used by metrics whose final segments are empty.

Examples:
python benchmark_evaluation/validate_visual_workflow.py \
  --understanding-root /path/to/visual_workflow/_understanding \
  --arrange-root /path/to/visual_workflow/v33 \
  --gt-paths /path/to/labels_final_prompt.json /path/to/labels_final_batch2_fixed.json \
  --set-names set1 set2 \
  --test-ids test1 test2 test3 test4 test5 test6 \
  --output-dir /path/to/report_dir
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


def video_name_from_result(video_result):
    video_name = video_result.get("video_name")
    if not video_name:
        return None
    return video_name_from_path(video_name)


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


def discover_test_ids(*roots):
    test_ids = set()
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("test"):
                test_ids.add(child.name)
    return sorted(test_ids, key=lambda x: (len(x), x))


def resolve_understanding_dir(root, test_id, set_name):
    root = Path(root)
    direct = root / test_id / set_name
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"cannot find Video-Understanding Agent outputs under {direct}")


def resolve_cut_plan_dir(root, test_id, set_name):
    root = Path(root)
    direct = root / test_id / set_name / "cut_plans"
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"cannot find cut plans under {direct}")


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


def evaluate_understanding_dir(understanding_dir, groundtruth, iou_threshold):
    json_files = sorted(Path(understanding_dir).glob("*.json"))
    pred_case_ids = [path.stem for path in json_files]
    coverage = base_coverage(groundtruth, pred_case_ids)

    prediction = {}
    empty_video = 0
    extra_video = 0
    malformed_case = 0
    empty_highlights = 0

    for json_path in json_files:
        sample_id = json_path.stem
        if sample_id not in groundtruth:
            continue

        prediction[sample_id] = {}
        gt_video_names = set(groundtruth[sample_id].keys())
        seen_pred_gt_videos = set()

        try:
            file_content = load_json(json_path)
            understanding_results = file_content["understanding_results"]
        except Exception:
            malformed_case += 1
            empty_video += len(gt_video_names)
            continue

        for video_id, video_result in understanding_results.items():
            if not isinstance(video_result, dict):
                continue
            video_name = video_name_from_result(video_result)
            if not video_name:
                extra_video += 1
                continue
            if video_name not in gt_video_names:
                extra_video += 1
                continue

            seen_pred_gt_videos.add(video_name)
            prediction[sample_id][video_name] = {"highlights": []}
            for highlight in video_result.get("highlights", []) or []:
                if not isinstance(highlight, dict):
                    continue
                segment = normalize_highlight(highlight)
                if segment is not None:
                    prediction[sample_id][video_name]["highlights"].append(segment)
            if not prediction[sample_id][video_name]["highlights"]:
                empty_highlights += 1

        empty_video += len(gt_video_names - seen_pred_gt_videos)

    common_case_ids = sorted(set(groundtruth.keys()) & set(prediction.keys()))
    gt_filtered = {case_id: groundtruth[case_id] for case_id in common_case_ids}
    pred_filtered = {case_id: prediction[case_id] for case_id in common_case_ids}
    metrics = calculate_metrics(gt_filtered, pred_filtered, iou_threshold=iou_threshold)
    metrics.pop("kendalltau_legal", None)

    coverage.update({
        "empty_video": empty_video,
        "extra_video": extra_video,
        "malformed_case": malformed_case,
        "empty_highlights": empty_highlights,
    })
    return {**metrics, **coverage}


def resolve_arrange_results(file_content, json_path):
    arrange_result_dict = file_content["get_arrange_result"]["arrange_result"]
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


def resolve_understand_result(file_content):
    if "raw_understanding_result" in file_content:
        return file_content["raw_understanding_result"]
    if "understanding_results" in file_content:
        return file_content["understanding_results"]
    if "shuffled_understand_result" in file_content:
        return file_content["shuffled_understand_result"]
    raise KeyError("raw_understanding_result/understanding_results/shuffled_understand_result")


def final_segments(video):
    segments = video.get("adjusted_segments") or video.get("segments", [])
    if isinstance(segments, list):
        return segments
    return []


def evaluate_cut_plan_dir(arrange_dir, groundtruth, iou_threshold):
    json_files = sorted(Path(arrange_dir).glob("*.json"))
    pred_case_ids = [path.stem for path in json_files]
    coverage = base_coverage(groundtruth, pred_case_ids)

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
            arrange_results = resolve_arrange_results(file_content, json_path)
            understand_result = resolve_understand_result(file_content)
        except Exception:
            malformed_case += 1
            empty_video += len(gt_video_names)
            continue

        for video in arrange_results:
            video_id = video.get("video_id")
            if video_id is None:
                extra_video += 1
                continue
            try:
                video_name = video_name_from_result(understand_result[video_id])
            except Exception:
                unresolved_video_id += 1
                extra_video += 1
                continue
            if not video_name or video_name not in gt_video_names:
                extra_video += 1
                continue

            segments = final_segments(video)
            prediction[sample_id][video_name] = {"highlights": []}
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                normalized = normalize_segment(segment)
                if normalized is not None:
                    prediction[sample_id][video_name]["highlights"].append(normalized)
            if prediction[sample_id][video_name]["highlights"]:
                seen_pred_gt_videos.add(video_name)
            else:
                empty_segments += 1
                seen_pred_gt_videos.add(video_name)

        empty_video += len(gt_video_names - seen_pred_gt_videos)

    common_case_ids = sorted(set(groundtruth.keys()) & set(prediction.keys()))
    gt_filtered = {case_id: groundtruth[case_id] for case_id in common_case_ids}
    pred_filtered = {case_id: prediction[case_id] for case_id in common_case_ids}
    metrics = calculate_metrics(gt_filtered, pred_filtered, iou_threshold=iou_threshold)
    metrics.pop("kendalltau_legal", None)

    coverage.update({
        "empty_video": empty_video,
        "extra_video": extra_video,
        "empty_segments": empty_segments,
        "malformed_case": malformed_case,
        "unresolved_video_id": unresolved_video_id,
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
            malformed = result.get("malformed_case", 0)
            if malformed:
                lines.append(f"- {test_id} / {set_name}: malformed_case={malformed}")
            unresolved = result.get("unresolved_video_id", 0)
            if unresolved:
                lines.append(f"- {test_id} / {set_name}: unresolved_video_id={unresolved}")
    if not lines:
        lines.append("- 无 case 级别新增/遗漏；无 malformed case。")
    return "\n".join(lines)


def write_stage_report(output_path, title, root, gt_paths, set_names, rows, columns, iou_threshold):
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
        "- `empty_case`：GT 中存在但该 test 输出目录缺少整个 case JSON。",
        "- `extra_case`：输出目录中存在但 GT 中不存在的 case JSON。",
        "- `empty_video`：在 GT/prediction common case 内，GT video 没有对应预测。",
        "- `extra_video`：在 GT/prediction common case 内，预测出现 GT 不存在的 video；cut plans 中无法解析的 video_id 也计入该列。",
        "",
        markdown_table(rows, set_names, columns),
        "",
        "## 覆盖细节",
        "",
        detail_lines(rows, set_names),
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def evaluate_stage(root, test_ids, set_names, gt_paths, iou_threshold, stage):
    rows = []
    for test_id in test_ids:
        by_set = {}
        for set_name, gt_path in zip(set_names, gt_paths):
            gt = load_json(gt_path)
            if stage == "understanding":
                stage_dir = resolve_understanding_dir(root, test_id, set_name)
                result = evaluate_understanding_dir(stage_dir, gt, iou_threshold)
            elif stage == "cut_planning":
                stage_dir = resolve_cut_plan_dir(root, test_id, set_name)
                result = evaluate_cut_plan_dir(stage_dir, gt, iou_threshold)
            else:
                raise KeyError(stage)
            by_set[set_name] = result
        rows.append((test_id, by_set))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--understanding-root", default=None)
    parser.add_argument("--cut-planning-root", default=None)
    parser.add_argument("--gt-paths", nargs="+", required=True)
    parser.add_argument("--set-names", nargs="+", required=True)
    parser.add_argument("--test-ids", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--understanding-output-name", default="video_understanding.md")
    parser.add_argument("--cut-planning-output-name", default="cut_planning.md")
    parser.add_argument("--understanding-title", default="Video-Understanding Agent validation")
    parser.add_argument("--cut-planning-title", default="Cut-Planning Agent validation")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.understanding_root and not args.cut_planning_root:
        raise SystemExit("At least one of --understanding-root or --cut-planning-root is required.")
    if len(args.gt_paths) != len(args.set_names):
        raise SystemExit("--gt-paths and --set-names must have the same length.")

    test_ids = args.test_ids or discover_test_ids(args.understanding_root, args.cut_planning_root)
    if not test_ids:
        raise SystemExit("No test ids found. Pass --test-ids explicitly.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.understanding_root:
        rows = evaluate_stage(
            Path(args.understanding_root),
            test_ids,
            args.set_names,
            [Path(p) for p in args.gt_paths],
            args.iou_threshold,
            "understanding",
        )
        output_path = output_dir / args.understanding_output_name
        write_stage_report(
            output_path,
            args.understanding_title,
            args.understanding_root,
            args.gt_paths,
            args.set_names,
            rows,
            UNDERSTANDING_COLUMNS,
            args.iou_threshold,
        )
        print(f"Saved {output_path}")

    if args.cut_planning_root:
        rows = evaluate_stage(
            Path(args.cut_planning_root),
            test_ids,
            args.set_names,
            [Path(p) for p in args.gt_paths],
            args.iou_threshold,
            "cut_planning",
        )
        output_path = output_dir / args.cut_planning_output_name
        write_stage_report(
            output_path,
            args.cut_planning_title,
            args.cut_planning_root,
            args.gt_paths,
            args.set_names,
            rows,
            CUT_PLANNING_COLUMNS,
            args.iou_threshold,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
