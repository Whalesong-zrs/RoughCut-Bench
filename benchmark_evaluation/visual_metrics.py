
import logging
import numpy as np
from scipy.stats import kendalltau as calculate_kt
import math 

DEFAULT_IOU_THRESHOLD = 0.5

def calculate_segment_iou(seg1, seg2):
    """
    计算单个片段的IOU（交并比）
    seg: [start, end] 片段起止时间（注：若输入是长度>2的列表，取前两个元素）
    """
    start1, end1 = seg1[0], seg1[1]
    start2, end2 = seg2[0], seg2[1]

    inter_start = max(start1, start2)
    inter_end = min(end1, end2)

    if inter_end <= inter_start:
        return 0.0

    inter_duration = inter_end - inter_start
    union_duration = (end1 - start1) + (end2 - start2) - inter_duration

    if union_duration == 0:
        return 0.0
    
    return inter_duration / union_duration


def calculate_video_match(
    gt_highlights,
    gt_confidences,
    pred_segments,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
):
    """
    单个视频的片段匹配（按GT遍历匹配Pred，支持置信度优先级）
    :param gt_highlights: 该视频的GT片段列表 [[s1,e1], [s2,e2], ...]
    :param gt_confidences: 该视频的GT片段置信度列表 [0,1,None,...]
    :param pred_segments: 该视频的预测片段列表 [[s1,e1], [s2,e2], ...]
    :param iou_threshold: 匹配阈值（IoU >= 0.5 视为匹配）
    :return: 
        matched_pred_count: 匹配成功的预测片段数
        matched_high_conf_gt_count: 匹配成功的高置信度GT片段数
        total_high_conf_gt: 高置信度GT片段总数
        matched_ious: 每个Pred对应的最大IOU列表（即使不满足阈值也记录）
    """
    high_conf_gt_idxs = []
    low_conf_gt_idxs = []
    for idx, conf in enumerate(gt_confidences):
        if conf == 0:
            low_conf_gt_idxs.append(idx)
        else:
            high_conf_gt_idxs.append(idx)
    total_high_conf_gt = len(high_conf_gt_idxs)
    
    gt_matched = [False] * len(gt_highlights)
    pred_matched = [False] * len(pred_segments)
    pred_max_iou = [0.0] * len(pred_segments)

    # High-confidence references have matching priority.
    for gt_idx in high_conf_gt_idxs:
        if gt_matched[gt_idx]:
            continue
        gt_seg = gt_highlights[gt_idx]
        
        max_iou = 0.0
        best_pred_idx = -1
        for pred_idx, pred_seg in enumerate(pred_segments):
            if pred_matched[pred_idx]:
                continue
            iou = calculate_segment_iou(gt_seg, pred_seg)
            if iou > max_iou:
                max_iou = iou
                best_pred_idx = pred_idx
            if iou > pred_max_iou[pred_idx]:
                pred_max_iou[pred_idx] = iou

        if max_iou >= iou_threshold and best_pred_idx != -1:
            gt_matched[gt_idx] = True
            pred_matched[best_pred_idx] = True
    
    for gt_idx in low_conf_gt_idxs:
        if gt_matched[gt_idx]:
            continue
        gt_seg = gt_highlights[gt_idx]
        
        max_iou = 0.0
        best_pred_idx = -1
        for pred_idx, pred_seg in enumerate(pred_segments):
            if pred_matched[pred_idx]:
                continue
            iou = calculate_segment_iou(gt_seg, pred_seg)
            if iou > max_iou:
                max_iou = iou
                best_pred_idx = pred_idx
            if iou > pred_max_iou[pred_idx]:
                pred_max_iou[pred_idx] = iou

        if max_iou >= iou_threshold and best_pred_idx != -1:
            gt_matched[gt_idx] = True
            pred_matched[best_pred_idx] = True
    
    matched_ious = pred_max_iou
    matched_pred_count = sum(pred_matched)
    matched_high_conf_gt_count = sum([gt_matched[idx] for idx in high_conf_gt_idxs])
    
    return matched_pred_count, matched_high_conf_gt_count, total_high_conf_gt, matched_ious
def calculate_metrics(groundtruth, prediction, iou_threshold=DEFAULT_IOU_THRESHOLD):
    """
    计算所有样本的平均指标（precision/recall/IOU）
    核心规则：
    1. Recall仅计算高置信度（非0/非None）GT片段的匹配率
    2. Precision计算所有预测片段的匹配率（匹配低置信度GT也视为准确）
    3. 仅同名视频间做片段匹配，样本内合并所有视频的统计结果
    :param groundtruth: 字典，格式 {样本名: {视频名: {'highlights': [...], 'confidence': [...]}, ...}, ...}
    :param prediction: 字典，格式 {样本名: {视频名: {'highlights': [...]}, ...}, ...}
    :param iou_threshold: 匹配阈值，默认 0.5
    :return: 字典，包含样本平均的precision、recall、avg_iou
    """
    sample_precisions = []
    sample_recalls = []
    sample_avg_ious = []
    sample_kendatall = []
    kendalltau_legal = 0
    kendalltau_illegal = 0
    for sample_name, gt_videos in groundtruth.items():
        if sample_name not in prediction or prediction[sample_name]=={}:
            print('missing prediction sample ', sample_name )
            continue

        gt_order = list(groundtruth[sample_name].keys())
        prediction_order = list(prediction[sample_name].keys())

        if len(gt_order) == 1:
            pass
        elif len(gt_order) == len(prediction_order) and set(gt_order) == set(prediction_order):
            kendalltau_legal+=1
            map_dict = {key: value for value, key in enumerate(gt_order)}
            map_pred = [map_dict[key] for key in prediction_order]    
            kendalltau = calculate_kt(map_pred, list(range(len(gt_order))))[0]
            sample_kendatall.append(kendalltau)
        else:
            logging.warning(f"[{sample_name}] gt_order={gt_order} vs prediction_order={prediction_order}: video set mismatch, computing kendalltau on filtered intersection.")
            kendalltau_illegal+=1
            common_videos = set(gt_order) & set(prediction_order)
            gt_order_filtered = [video for video in gt_order if video in common_videos]
            prediction_order_filtered = [video for video in prediction_order if video in common_videos]
            if len(common_videos) >= 2:
                map_dict = {key: value for value, key in enumerate(gt_order_filtered)}
                map_pred = [map_dict[key] for key in prediction_order_filtered]
                kendalltau = calculate_kt(map_pred, list(range(len(gt_order_filtered))))[0]
                if not math.isnan(kendalltau):
                    sample_kendatall.append(kendalltau)
  

        pred_videos = prediction.get(sample_name, {})
        
        common_video_names = set(gt_videos.keys()) & set(pred_videos.keys())

        video_precisions = []
        video_recalls = []
        video_avg_ious = []
        for video_name in common_video_names:
            gt_data = gt_videos[video_name]
            gt_highlights = [seg for seg in gt_data['highlights'] if seg[1] > seg[0]]
            gt_confidences = gt_data['confidence']

            
            pred_data = pred_videos[video_name]
            pred_segments = [seg for seg in pred_data['highlights']]

            total_pred_segments = len(pred_segments)

            vid_matched_pred, vid_matched_high_gt, vid_total_high_gt, vid_ious = calculate_video_match(
                gt_highlights, gt_confidences, pred_segments, iou_threshold
            )

            if total_pred_segments > 0:
                video_precisions.append(vid_matched_pred / total_pred_segments)
                video_avg_ious.append(np.mean(vid_ious))
            else:
                print('video pred none segments ', video_name)
            if vid_total_high_gt > 0:
                recall = vid_matched_high_gt / vid_total_high_gt
                video_recalls.append(recall)

        gt_only_video_names = set(gt_videos.keys()) - set(pred_videos.keys())
        for video_name in gt_only_video_names:
            gt_confidences = gt_videos[video_name].get('confidence', [])
            vid_total_high_gt = sum(1 for c in gt_confidences if c not in (0, '0'))
            if vid_total_high_gt > 0:
                video_recalls.append(0.0)
                print('video missing from pred ', video_name)

        if video_precisions:
            sample_precisions.append(np.mean(video_precisions))
        if video_recalls:
            sample_recalls.append(np.mean(video_recalls))
        if video_avg_ious:
            sample_avg_ious.append(np.mean(video_avg_ious))

    precision = float(np.mean(sample_precisions)) if sample_precisions else 0.0
    recall = float(np.mean(sample_recalls)) if sample_recalls else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ordering_cases = kendalltau_legal + kendalltau_illegal
    final_metrics = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "avg_iou": round(np.mean(sample_avg_ious),3) if sample_avg_ious else 0.0,
        'kendalltau': round(np.mean(sample_kendatall),3) if sample_kendatall else 0.0,
        'kendalltau_legal': round(kendalltau_legal / ordering_cases, 3) if ordering_cases else 0.0,
    }

    return final_metrics

if __name__ == "__main__":
    groundtruth = {
        "sample1": {
            "video_0": {
                "highlights": [[1.0, 3.0], [5.0, 8.0], [9.0, 12.0]],
                "confidence": [1, 0, None]
            },
            "video_1": {
                "highlights": [[10.0, 15.0]],
                "confidence": [1]
            }
        },
        "sample2": {
            "video_0": {
                "highlights": [[2.0, 4.0]],
                "confidence": [0]
            }
        }
    }
    
    prediction = {
        "sample1": {
            "video_0": {
                "highlights": [[1.5, 2.5], [6.0, 9.0], [9.5, 11.5]]
            },
        },
        "sample2": {
            "video_0": {
                "highlights": [[1.0, 5.0]]
            },
            "video_1": {
                "highlights": [[0.0, 1.0]]
            }
        }
    }
    
    metrics = calculate_metrics(groundtruth, prediction)
    print("最终指标（所有样本平均）：")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Average IOU: {metrics['avg_iou']:.4f}")
