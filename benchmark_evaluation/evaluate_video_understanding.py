import argparse
import os
import csv
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
    parser.add_argument('--video-understanding-dirs', nargs='+', required=True,
                        help='Video-Understanding Agent output directories, one per set')
    parser.add_argument('--gt-paths', nargs='+', required=True,
                        help='ground-truth JSON paths, one per set')
    parser.add_argument('--allow-partial', action='store_true',
                        help='evaluate an intentional subset instead of requiring full case/source coverage')
    return parser.parse_args()


def evaluate(gt_name, understanding_path, model_name, allow_partial=False):
    print(f'\n=== Evaluating: {gt_name} ===')

    with open(gt_name, 'r', encoding='utf-8') as f:
        groundtruth = json.load(f)

    prediction = {}
    for file in sorted(os.listdir(understanding_path)):
        if not file.endswith('.json'):
            continue
        case_id = os.path.splitext(file)[0]
        path = os.path.join(understanding_path, file)
        with open(path, encoding='utf-8') as f:
            file_content = json.load(f)
        understanding_result = file_content.get('understanding_results')
        if not isinstance(understanding_result, dict):
            raise ValueError(f'{path}: missing understanding_results object')

        case_prediction = {}
        for video_id, video_result in understanding_result.items():
            try:
                highlights = video_result['highlights']
                video_name = os.path.splitext(os.path.basename(video_result['video_name']))[0]
                case_prediction[video_name] = {
                    'highlights': [
                        [highlight['highlight_start'], highlight['highlight_end']]
                        for highlight in highlights
                    ]
                }
            except (KeyError, TypeError) as exc:
                raise ValueError(f'{path}: invalid result for {video_id!r}: {exc}') from exc
        prediction[case_id] = case_prediction

    groundtruth, prediction = align_evaluation_inputs(
        groundtruth,
        prediction,
        allow_partial=allow_partial,
    )
    if not groundtruth:
        raise ValueError('no common cases between annotations and predictions')

    final_metrics = calculate_metrics(groundtruth, prediction, iou_threshold=0.5)
    print(list(final_metrics.keys()))
    print('    '.join([str(i) for i in list(final_metrics.values())]))

    output_root = os.path.dirname(understanding_path)
    set_name = 'set2' if 'batch2' in gt_name else 'set1'
    task = 'understanding'
    eval_csv = os.path.join(output_root, f'eval_{task}_{set_name}_{model_name}.csv')

    row = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    row.update({k: str(v) for k, v in final_metrics.items()})

    fieldnames = ['timestamp'] + list(final_metrics.keys())
    with open(eval_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f'Saved to {eval_csv}')


def main():
    args = parse_args()
    if len(args.video_understanding_dirs) != len(args.gt_paths):
        raise ValueError(
            f'--video-understanding-dirs ({len(args.video_understanding_dirs)}) and '
            f'--gt-paths ({len(args.gt_paths)}) must have same length'
        )
    print(f'=== Evaluating understanding for model: {args.model} ===')
    for understanding_dir, gt_path in zip(args.video_understanding_dirs, args.gt_paths):
        evaluate(gt_path, understanding_dir.rstrip('/'), args.model, args.allow_partial)


if __name__ == '__main__':
    main()
