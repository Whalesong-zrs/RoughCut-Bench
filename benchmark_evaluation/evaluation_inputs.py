"""Coverage checks shared by visual annotation-based evaluators."""

from __future__ import annotations


def align_evaluation_inputs(
    ground_truth: dict,
    prediction: dict,
    *,
    allow_partial: bool = False,
) -> tuple[dict, dict]:
    """Validate case/source coverage and return aligned evaluation mappings."""
    missing_cases = sorted(set(ground_truth) - set(prediction))
    extra_cases = sorted(set(prediction) - set(ground_truth))
    if not allow_partial and (missing_cases or extra_cases):
        raise ValueError(
            "prediction case coverage does not match annotations: "
            f"missing={missing_cases}, extra={extra_cases}"
        )
    if allow_partial and missing_cases:
        print(f"[WARN] missing predicted cases: {missing_cases}")
    if allow_partial and extra_cases:
        print(f"[WARN] ignoring unknown predicted cases: {extra_cases}")

    common_case_ids = sorted(set(ground_truth) & set(prediction))
    print(f"Common cases: {len(common_case_ids)}")
    filtered_gt = {case_id: ground_truth[case_id] for case_id in common_case_ids}
    filtered_prediction = {case_id: prediction[case_id] for case_id in common_case_ids}

    for case_id, case_prediction in filtered_prediction.items():
        expected_sources = set(filtered_gt[case_id])
        predicted_sources = set(case_prediction)
        missing_sources = sorted(expected_sources - predicted_sources)
        unknown_sources = sorted(predicted_sources - expected_sources)
        if not allow_partial and (missing_sources or unknown_sources):
            raise ValueError(
                f"{case_id}: source coverage does not match annotations: "
                f"missing={missing_sources}, extra={unknown_sources}"
            )
        if allow_partial and missing_sources:
            print(f"[WARN] {case_id}: missing source IDs: {missing_sources}")
        if allow_partial and unknown_sources:
            print(f"[WARN] {case_id}: ignoring unknown source IDs: {unknown_sources}")
            filtered_prediction[case_id] = {
                source_id: value
                for source_id, value in case_prediction.items()
                if source_id in expected_sources
            }
    return filtered_gt, filtered_prediction
