import argparse
import json
import logging
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import tqdm
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from visual_workflow.model_clients import VideoUnderstandingLLM
from visual_workflow.prompt import video_understanding_prompt_v7
from visual_workflow.workflow_utils import (
    decode_with_fps,
    my_encode_img,
    parallel_extract_video_frames,
    read_json,
)


def make_status(input_id, ok, error_type=None, error_message=None, video_idx=None, video_name=None, video_count=None):
    return {
        'input_id': input_id,
        'ok': ok,
        'error_type': error_type,
        'error_message': error_message,
        'video_idx': video_idx,
        'video_name': video_name,
        'video_count': video_count,
    }


def write_failed_log(save_dir, failed_items):
    if not failed_items:
        return None
    failed_dir = os.path.join(save_dir, '_failed')
    os.makedirs(failed_dir, exist_ok=True)
    failed_path = os.path.join(failed_dir, 'understanding_failed.jsonl')
    with open(failed_path, 'w', encoding='utf-8') as f:
        for item in failed_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return failed_path



def invalid_case(input_id, error_type, error_message, rerun=True):
    status = make_status(input_id, False, error_type, error_message)
    status['rerun'] = rerun
    return status


def video_name_from_path(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def is_video_asset(path):
    return str(path).lower().rsplit('.', 1)[-1] in {'mp4', 'mkv', 'mov', 'avi', 'webm'}


def list_sample_keys(video_root, gt=None):
    if gt:
        return sorted(gt.keys())
    return sorted([
        name for name in os.listdir(video_root)
        if os.path.isdir(os.path.join(video_root, name))
    ])


def normalize_time_value(value):
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def validate_understanding_case_output(save_dir, sample_key, gt_for_sample=None):
    json_path = os.path.join(save_dir, sample_key + '.json')
    if not os.path.exists(json_path):
        return invalid_case(sample_key, 'missing_case', 'case output json is missing')

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
    except Exception as ex:
        return invalid_case(sample_key, 'malformed_json', f'{type(ex).__name__}: {ex}')

    try:
        understanding_results = file_content['understanding_results']
        if not isinstance(understanding_results, dict) or not understanding_results:
            raise TypeError('understanding_results must be a non-empty dict')

        gt_video_names = set((gt_for_sample or {}).keys())
        seen_gt_video_names = set()
        seen_video_names = set()
        for video_id, video_result in understanding_results.items():
            if not isinstance(video_result, dict):
                raise TypeError(f'{video_id} result must be a dict')
            video_name = video_result.get('video_name')
            if not video_name:
                raise KeyError(f'{video_id} missing video_name')
            base_name = video_name_from_path(video_name)
            if base_name in seen_video_names:
                raise KeyError(f'duplicate video_name: {base_name}')
            seen_video_names.add(base_name)
            if gt_video_names:
                if base_name not in gt_video_names:
                    raise KeyError(f'extra video predicted: {video_id} -> {base_name}')
                seen_gt_video_names.add(base_name)

            highlights = video_result.get('highlights')
            if not isinstance(highlights, list) or not highlights:
                raise ValueError(f'empty highlights for {video_id}')
            for highlight in highlights:
                if not isinstance(highlight, dict):
                    raise TypeError(f'invalid highlight for {video_id}')
                start = normalize_time_value(highlight.get('highlight_start'))
                end = normalize_time_value(highlight.get('highlight_end'))
                if start is None or end is None:
                    raise KeyError(f'missing highlight_start/highlight_end for {video_id}')
                if end < start:
                    raise ValueError(f'invalid highlight time range for {video_id}: {start}-{end}')

        if gt_video_names:
            missing = sorted(gt_video_names - seen_gt_video_names)
            if missing:
                raise KeyError(f'missing gt videos: {missing}')
    except Exception as ex:
        return invalid_case(sample_key, 'invalid_case_output', f'{type(ex).__name__}: {ex}')

    return make_status(sample_key, True)


def validate_understanding_output_dir(save_dir, sample_keys, gt=None):
    expected = set(sample_keys)
    invalid_items = []
    json_files = [name for name in os.listdir(save_dir) if name.endswith('.json')]
    for file_name in sorted(json_files):
        sample_key = file_name[:-5]
        if sample_key not in expected:
            invalid_items.append(
                invalid_case(sample_key, 'extra_case', 'case output json is not in gt', rerun=False)
            )

    gt = gt or {}
    for sample_key in sample_keys:
        invalid = validate_understanding_case_output(save_dir, sample_key, gt.get(sample_key, {}))
        if not invalid.get('ok'):
            invalid_items.append(invalid)
    return invalid_items


def remove_case_artifacts(save_dir, sample_key):
    json_path = os.path.join(save_dir, sample_key + '.json')
    if os.path.exists(json_path):
        os.remove(json_path)


def clear_repair_logs(save_dir):
    failed_dir = os.path.join(save_dir, '_failed')
    for name in ['understanding_failed.jsonl', 'understanding_unresolved_after_repair.jsonl']:
        path = os.path.join(failed_dir, name)
        if os.path.exists(path):
            os.remove(path)


def write_unresolved_repair_log(save_dir, invalid_items):
    failed_dir = os.path.join(save_dir, '_failed')
    os.makedirs(failed_dir, exist_ok=True)
    failed_path = os.path.join(failed_dir, 'understanding_unresolved_after_repair.jsonl')
    with open(failed_path, 'w', encoding='utf-8') as f:
        for item in invalid_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return failed_path


def run_repair_loop(sample_keys, workers, gt, max_repair_rounds):
    invalid_items = validate_understanding_output_dir(english_data_save_dir, sample_keys, gt)
    if not invalid_items:
        clear_repair_logs(english_data_save_dir)
        print('[REPAIR] all understanding cases already valid')
        return []

    for round_idx in range(1, max_repair_rounds + 1):
        preview = ', '.join(item['input_id'] for item in invalid_items[:10])
        suffix = '' if len(invalid_items) <= 10 else f', ... total={len(invalid_items)}'
        print(f'[REPAIR] understanding round={round_idx}/{max_repair_rounds} invalid={len(invalid_items)} ({preview}{suffix})')

        rerun_keys = []
        for item in invalid_items:
            sample_key = item['input_id']
            remove_case_artifacts(english_data_save_dir, sample_key)
            if item.get('rerun', True) and sample_key in sample_keys:
                rerun_keys.append(sample_key)

        if rerun_keys:
            _, failed = parallel_process(sorted(set(rerun_keys)), workers)
            failed_path = write_failed_log(english_data_save_dir, failed)
            if failed_path:
                print(f'[FAIL] wrote understanding failed log: {failed_path}')

        invalid_items = validate_understanding_output_dir(english_data_save_dir, sample_keys, gt)
        if not invalid_items:
            clear_repair_logs(english_data_save_dir)
            print(f'[REPAIR] understanding passed after round {round_idx}')
            return []

    failed_path = write_unresolved_repair_log(english_data_save_dir, invalid_items)
    print(f'[REPAIR] unresolved understanding invalid cases after {max_repair_rounds} rounds: {failed_path}')
    return invalid_items


class VideoUnderstandingRunner:
    def __init__(self, model='gemini-2.5-pro'):
        self.llm = VideoUnderstandingLLM(model, video_understanding_prompt_v7)
        self.model = model
        self.safety_probe_prompt = (
            '请判断这张图片能否正常用于视频内容理解。'
            '如果画面存在明显违规风险，请简要说明；否则只回答 OK。'
        )
        self.max_img_limits = 120

    def _is_data_inspection_error(self, error):
        text = str(error)
        markers = [
            'data_inspection_failed',
            'DataInspectionFailed',
            'Input image data may contain inappropriate content',
            'inappropriate content',
        ]
        return any(marker in text for marker in markers)

    def _resize_understanding_frames(self, pil_lists):
        resized_pil_list = []
        for img in pil_lists:
            width, height = img.size
            if width < height:
                new_width = 224
                new_height = int((height / width) * new_width)
            else:
                new_height = 224
                new_width = int((width / height) * new_height)
            resized_pil_list.append(img.resize((new_width, new_height), Image.Resampling.LANCZOS))
        return resized_pil_list

    def _probe_frame_passes_data_inspection(self, binary_img, timestamp):
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': self.safety_probe_prompt + f'\n时间戳 {timestamp}s'},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{binary_img}', 'detail': 'low'},
                    },
                ],
            }
        ]
        try:
            self.llm.runner.run(messages, model=self.model)
            return True, None
        except Exception as ex:
            if self._is_data_inspection_error(ex):
                return False, str(ex)
            raise

    def _drop_data_inspection_frames(self, binary_list, time_stamps):
        kept_binary = []
        kept_time_stamps = []
        dropped = []
        for binary_img, timestamp in zip(binary_list, time_stamps):
            ok, error_message = self._probe_frame_passes_data_inspection(binary_img, timestamp)
            if ok:
                kept_binary.append(binary_img)
                kept_time_stamps.append(timestamp)
            else:
                dropped.append({
                    'timestamp': timestamp,
                    'error_message': error_message,
                })
                print(f'[SAFETY_SKIP] drop understanding frame timestamp={timestamp}s reason=data_inspection_failed')
        return kept_binary, kept_time_stamps, dropped

    def run_single_video_understanding(self, pil_lists, time_stamps):
        time_stamps = [round(t, 1) for t in time_stamps]
        resized_pil_list = self._resize_understanding_frames(pil_lists)
        binary_list = [my_encode_img(pil_img) for pil_img in resized_pil_list]

        try:
            return self.llm.run(binary_list, time_stamps)
        except Exception as ex:
            if not self._is_data_inspection_error(ex):
                raise
            print(
                f'[SAFETY_SKIP] understanding request hit data inspection; '
                f'probing {len(binary_list)} frames and retrying without blocked frames'
            )
            kept_binary, kept_time_stamps, dropped = self._drop_data_inspection_frames(binary_list, time_stamps)
            if not dropped:
                raise RuntimeError(
                    'understanding request was blocked by data inspection, '
                    'but no individual blocked frame was found during probing'
                ) from ex
            if not kept_binary:
                raise RuntimeError(
                    f'all understanding frames were blocked by data inspection; '
                    f'dropped_timestamps={[item["timestamp"] for item in dropped]}'
                ) from ex
            result, input_token, output_token = self.llm.run(kept_binary, kept_time_stamps)
            result['skipped_data_inspection_frames'] = dropped
            return result, input_token, output_token

    # Split requests that exceed the provider image limit.
    def batch_run_single_video_understanding(self, pil_lists, time_stamps):
        total_img_num = len(pil_lists)
        assert len(pil_lists) == len(time_stamps)
        final_result = dict()
        if total_img_num > self.max_img_limits:
            logging.info('Exceed the Max Image Limits of Agent: %d Images' % total_img_num)
            tmp_len = 0
            while tmp_len < len(pil_lists):
                tmp_pil_list = pil_lists[tmp_len: tmp_len + self.max_img_limits]
                tmp_time_stamps = time_stamps[tmp_len: tmp_len + self.max_img_limits]
                tmp_len += self.max_img_limits
                assert len(tmp_pil_list) == len(tmp_time_stamps)

                tmp_result, input_token, output_token = self.run_single_video_understanding(tmp_pil_list, tmp_time_stamps)
                if 'highlights' in final_result.keys():
                    final_result['highlights'].extend(tmp_result['highlights'])
                    final_result['video_abs_frame_stamps'].extend(tmp_result['video_abs_frame_stamps'])
                    final_result['summary'] += (';' + tmp_result['summary'])
                    final_result['event/action'] += (';' + tmp_result['event/action'])
                    final_result['characters'] += (';' + tmp_result['characters'])
                    final_result['time'] += (';' + tmp_result['time'])
                    final_result['scene/location'] += (';' + tmp_result['scene/location'])
                else:
                    final_result = tmp_result
        else:
            tmp_result, input_token, output_token = self.run_single_video_understanding(pil_lists, time_stamps)

            final_result = tmp_result

        return final_result, input_token, output_token

    def run_single_video_understanding_from_video(self, video_path):
        return self.llm.run_video(video_path)




class VideoUnderstandingPreprocessor:
    def __init__(self, model='gemini-2.5-pro', sample_fps=1.0, input_mode='frames'):
        self.understanding_runner = VideoUnderstandingRunner(model)
        self.sample_fps = sample_fps
        self.input_mode = input_mode

    def process(self, input_id):
        base_video_path = os.path.join(all_cut_videos, input_id)
        try:
            video_names = list()

            for video_name in sorted(os.listdir(base_video_path)):
                if 'split_videos_new' in base_video_path and not video_name.lower().split('.')[-1] in ['mp4', 'mkv', 'mov', 'avi']:
                    continue
                abs_video_name = '%s/%s' % (base_video_path, video_name)
                video_names.append(abs_video_name)
            gt_for_sample = gt_by_sample.get(input_id, {}) if 'gt_by_sample' in globals() else {}
            if gt_for_sample:
                expected_video_names = set(gt_for_sample.keys())
                video_names = [p for p in video_names if video_name_from_path(p) in expected_video_names]

            if not video_names:
                print(f'[FAIL] empty video list for {input_id}, not saving')
                return make_status(input_id, False, 'empty_video_list', 'no valid videos found', video_count=0)

            video_idx = 0
            highlights_collects = {'understanding_results': dict()}
            if self.input_mode == 'video':
                iterable = [(video_name, None, None) for video_name in video_names]
            else:
                pil_collects, frame_stamps = parallel_extract_video_frames(video_names, sample_fps=self.sample_fps)
                iterable = list(zip(video_names, pil_collects, frame_stamps))

            for video_name, pil_list, frame_stamp in iterable:
                if self.input_mode == 'video':
                    print(f'[VIDEO_INPUT] {video_name}')
                else:
                    print(len(pil_list), len(frame_stamp), video_name)
                try:
                    if self.input_mode == 'video' and is_video_asset(video_name):
                        result, input_token, output_token = self.understanding_runner.run_single_video_understanding_from_video(video_name)
                    else:
                        if self.input_mode == 'video':
                            print(f'[VIDEO_INPUT] non-video asset fallback to frame input: {video_name}')
                            pil_list, frame_stamp = decode_with_fps(video_name, sample_fps=self.sample_fps)
                        result, input_token, output_token = self.understanding_runner.batch_run_single_video_understanding(pil_list, frame_stamp)
                except Exception as e:
                    print(
                        f'[FAIL] understanding video failed input_id={input_id} '
                        f'video_idx={video_idx} video_name={video_name} '
                        f'err={type(e).__name__}: {e}'
                    )
                    return make_status(
                        input_id,
                        False,
                        'video_understanding_failed',
                        f'{type(e).__name__}: {e}',
                        video_idx=video_idx,
                        video_name=video_name,
                        video_count=len(video_names),
                    )
                print(result)
                highlights_collects['understanding_results'][f'video_{video_idx}'] = result
                highlights_collects['understanding_results'][f'video_{video_idx}']['video_name'] = video_name
                video_idx += 1
            if not highlights_collects['understanding_results']:
                print(f'[FAIL] empty understanding for {input_id}, not saving')
                return make_status(input_id, False, 'empty_understanding', 'no understanding results produced', video_count=len(video_names))

            save_json_name = os.path.join(english_data_save_dir, input_id+'.json')
            with open(save_json_name, 'w') as f:
                json.dump(highlights_collects, f, ensure_ascii=False, indent=4)
            return make_status(input_id, True, video_count=len(video_names))
        except Exception as e:
            print('Error in Video: %s' % base_video_path, e)
            return make_status(input_id, False, 'case_processing_failed', f'{type(e).__name__}: {e}')



def parallel_process(english_unprocessed_ids, workers):
    print('Total Reqs: %d' % len(english_unprocessed_ids))
    results = []
    if workers <= 1:
        for input_id in tqdm(english_unprocessed_ids):
            try:
                data = runner.process(input_id)
            except Exception as ex:
                traceback.print_exc()
                data = make_status(input_id, False, 'unexpected_worker_exception', f'{type(ex).__name__}: {ex}')
            if data is None:
                data = make_status(input_id, False, 'unknown_worker_failure', 'worker returned None')
            results.append(data)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for input_id, data in zip(english_unprocessed_ids, executor.map(runner.process, english_unprocessed_ids)):
                if data is None:
                    data = make_status(input_id, False, 'unknown_worker_failure', 'worker returned None')
                results.append(data)
    failed = [item for item in results if not item.get('ok')]
    ok_count = len(results) - len(failed)
    print(f'done ok={ok_count} failed={len(failed)} total={len(results)} workers={workers}')
    return results, failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemini-2.5-pro',
                        help='one of the aliases in visual_workflow/model_api.py MODEL_REGISTRY')
    parser.add_argument('--video-dirs', nargs='+', required=True,
                        help='input video directories, one per set (parallel with --out-dirs)')
    parser.add_argument('--out-dirs', nargs='+', required=True,
                        help='where to write Video-Understanding Agent outputs, one per set')
    parser.add_argument('--gt-paths', nargs='+', default=None,
                        help='optional ground-truth JSON paths, one per set; repair mode uses these as authoritative sample/video lists')
    parser.add_argument('--workers', type=int, default=9,
                        help='number of worker processes for source videos')
    parser.add_argument('--sample-fps', type=float, default=1.0,
                        help='frame-sampling rate; default: 1.0 FPS')
    parser.add_argument('--input-mode', choices=['frames', 'video'], default='frames',
                        help='input representation: sampled frames or encoded video')
    parser.add_argument('--repair-invalid', action='store_true',
                        help='repair invalid/missing understanding outputs by deleting bad case jsons and rerunning them')
    parser.add_argument('--max-repair-rounds', type=int, default=5,
                        help='maximum repair rounds per output directory when --repair-invalid is enabled')
    args = parser.parse_args()
    assert len(args.video_dirs) == len(args.out_dirs), \
        f'--video-dirs ({len(args.video_dirs)}) and --out-dirs ({len(args.out_dirs)}) must have same length'
    if args.gt_paths is not None:
        assert len(args.gt_paths) == len(args.video_dirs), \
            f'--gt-paths ({len(args.gt_paths)}) and --video-dirs ({len(args.video_dirs)}) must have same length'
    if args.max_repair_rounds < 1:
        raise SystemExit('--max-repair-rounds must be >= 1')
    if args.sample_fps <= 0:
        raise SystemExit('--sample-fps must be > 0')

    MODEL_NAME = args.model
    print(f'=== Using understanding model: {MODEL_NAME} ===')
    print(f'=== Using understanding input_mode: {args.input_mode} ===')
    print(f'=== Using understanding sample_fps: {args.sample_fps} ===')
    runner = VideoUnderstandingPreprocessor(MODEL_NAME, sample_fps=args.sample_fps, input_mode=args.input_mode)

    PROCESS_TARGETS = []
    for idx, (vd, od) in enumerate(zip(args.video_dirs, args.out_dirs)):
        gt_path = args.gt_paths[idx] if args.gt_paths is not None else None
        PROCESS_TARGETS.append({
            'all_cut_videos': vd.rstrip('/'),
            'save_dir': od.rstrip('/'),
            'gt_path': gt_path,
        })

    all_failed = []
    for target in PROCESS_TARGETS:
        # Worker processes inherit these values when the executor forks.
        all_cut_videos = target['all_cut_videos']
        english_data_save_dir = target['save_dir']
        gt_by_sample = read_json(target['gt_path']) if target.get('gt_path') else {}
        print(f'\n=== Processing: {all_cut_videos} ===')
        os.makedirs(english_data_save_dir, exist_ok=True)

        sample_keys = list_sample_keys(all_cut_videos, gt_by_sample)
        if args.repair_invalid:
            print(f'sample count: {len(sample_keys)}')
            failed = run_repair_loop(sample_keys, args.workers, gt_by_sample, args.max_repair_rounds)
        else:
            already_have = set(os.listdir(english_data_save_dir))
            english_unprocessed_ids = [
                key for key in sample_keys
                if key + '.json' not in already_have
            ]
            print(len(english_unprocessed_ids))
            _, failed = parallel_process(english_unprocessed_ids, args.workers)
            failed_path = write_failed_log(english_data_save_dir, failed)
            if failed_path:
                print(f'[FAIL] wrote understanding failed log: {failed_path}')

        all_failed.extend(failed)

    if all_failed:
        raise SystemExit(f'understanding failed samples: {len(all_failed)}')
