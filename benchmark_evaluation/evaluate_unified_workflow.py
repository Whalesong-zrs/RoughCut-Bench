"""Evaluate cut plans produced by the unified-workflow ablation.

The persisted JSON schema stores selected segments under
``get_arrange_result.arrange_result.arrange_results``. When duration adjustment
is present, the evaluator uses ``adjusted_arrange_results`` instead.

Output: eval_unified_cut_plans_<set>_<MODEL>.csv
"""

import argparse
import csv
import os
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_evaluation.evaluation_inputs import align_evaluation_inputs
from benchmark_evaluation.visual_metrics import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemini-2.5-pro',
                        help='one of the aliases in visual_workflow/model_api.py MODEL_REGISTRY')
    parser.add_argument('--unified-dirs', nargs='+', required=True,
                        help='unified-chain output directories, one per set (parallel with --gt-paths)')
    parser.add_argument('--gt-paths', nargs='+', required=True,
                        help='ground-truth JSON paths, one per set (parallel with --unified-dirs)')
    parser.add_argument('--set-names', nargs='+', default=None,
                        help='optional explicit set names, one per set (parallel with --unified-dirs)')
    parser.add_argument('--allow-partial', action='store_true',
                        help='evaluate an intentional subset instead of requiring full case/source coverage')
    return parser.parse_args()


def _resolve_video_name(understand_result, video_id):
    """video_id -> 文件名 (不含扩展名), 与 GT json 的 key 对齐."""
    return os.path.splitext(understand_result[video_id]['video_name'].split('/')[-1])[0]


def build_prediction(unified_path):
    """Build {sample: {video_name: {"highlights": [[s, e], ...]}}} from unified output.

    读 arrange_results[i].segments 的 start_time/end_time;
    若 duration_adjustment 非空, 优先取 adjusted_arrange_results 的 adjusted_segments.
    """
    prediction = {}
    for file in sorted(os.listdir(unified_path)):
        if not file.endswith('.json'):
            continue
        sample_id = file.split('.json')[0]
        prediction[sample_id] = {}

        with open(os.path.join(unified_path, file), encoding='utf-8') as f:
            file_content = json.load(f)
        arrange_result_dict = file_content['get_arrange_result']['arrange_result']
        understand_result = file_content['shuffled_understand_result']

        if arrange_result_dict.get('duration_adjustment'):
            arrange_results = arrange_result_dict['duration_adjustment']['adjusted_arrange_results']
        else:
            arrange_results = arrange_result_dict['arrange_results']

        for video in arrange_results:
            video_id = video['video_id']
            video_name = _resolve_video_name(understand_result, video_id)
            prediction[sample_id][video_name] = {"highlights": []}

            items = video.get('adjusted_segments') or video.get('segments', [])
            for item in items:
                hi_start = round(float(item['start_time']), 1)
                hi_end = round(float(item['end_time']), 1)
                prediction[sample_id][video_name]['highlights'].append([hi_start, hi_end])
    return prediction


def _write_csv(eval_csv, final_metrics):
    row = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    row.update(final_metrics)
    fieldnames = ['timestamp'] + list(final_metrics.keys())
    with open(eval_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f'Saved to {eval_csv}')


def evaluate(gt_name, unified_path, model_name, set_name=None, allow_partial=False):
    print(f'\n=== Evaluating: {gt_name} ===')

    with open(gt_name, encoding='utf-8') as f:
        gt = json.load(f)

    output_root = os.path.dirname(unified_path)
    if set_name is None:
        set_name = 'set2' if 'batch2' in gt_name else 'set1'

    prediction = build_prediction(unified_path)
    gt_filtered, pred_filtered = align_evaluation_inputs(
        gt,
        prediction,
        allow_partial=allow_partial,
    )
    if not gt_filtered:
        raise ValueError('no common cases between annotations and predictions')

    final_metrics = calculate_metrics(gt_filtered, pred_filtered, iou_threshold=0.5)
    final_metrics.pop('kendalltau_legal', None)

    print(f'[unified-cut-plans] {list(final_metrics.keys())}')
    print(f'[unified-cut-plans] {"    ".join(str(v) for v in final_metrics.values())}')

    eval_csv = os.path.join(output_root, f'eval_unified_cut_plans_{set_name}_{model_name}.csv')
    _write_csv(eval_csv, final_metrics)


def main():
    args = parse_args()
    if len(args.unified_dirs) != len(args.gt_paths):
        raise ValueError(
            f'--unified-dirs ({len(args.unified_dirs)}) and '
            f'--gt-paths ({len(args.gt_paths)}) must have same length'
        )
    if args.set_names is not None and len(args.set_names) != len(args.unified_dirs):
        raise ValueError(
            f'--set-names ({len(args.set_names)}) and '
            f'--unified-dirs ({len(args.unified_dirs)}) must have same length'
        )
    print(f'=== Evaluating unified chain for model: {args.model} ===')
    for index, (unified_dir, gt_path) in enumerate(zip(args.unified_dirs, args.gt_paths)):
        set_name = args.set_names[index] if args.set_names is not None else None
        evaluate(gt_path, unified_dir.rstrip('/'), args.model, set_name, args.allow_partial)


if __name__ == '__main__':
    main()
