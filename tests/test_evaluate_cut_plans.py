import unittest

from benchmark_evaluation.evaluation_inputs import align_evaluation_inputs


def _source():
    return {"highlights": [[0.0, 1.0]], "confidence": [1]}


class EvaluateCutPlansTest(unittest.TestCase):
    def test_strict_mode_rejects_missing_cases(self):
        ground_truth = {"case_a": {"video_a": _source()}, "case_b": {"video_b": _source()}}
        prediction = {"case_a": {"video_a": {"highlights": [[0.0, 1.0]]}}}
        with self.assertRaisesRegex(ValueError, "missing=.*case_b"):
            align_evaluation_inputs(ground_truth, prediction)

    def test_strict_mode_rejects_source_mismatch(self):
        ground_truth = {"case_a": {"video_a": _source(), "video_b": _source()}}
        prediction = {
            "case_a": {
                "video_a": {"highlights": [[0.0, 1.0]]},
                "video_c": {"highlights": [[0.0, 1.0]]},
            }
        }
        with self.assertRaisesRegex(ValueError, "source coverage"):
            align_evaluation_inputs(ground_truth, prediction)

    def test_partial_mode_keeps_common_inputs(self):
        ground_truth = {"case_a": {"video_a": _source(), "video_b": _source()}}
        prediction = {
            "case_a": {
                "video_a": {"highlights": [[0.0, 1.0]]},
                "video_c": {"highlights": [[0.0, 1.0]]},
            },
            "case_extra": {"video_x": {"highlights": [[0.0, 1.0]]}},
        }
        filtered_gt, filtered_prediction = align_evaluation_inputs(
            ground_truth,
            prediction,
            allow_partial=True,
        )
        self.assertEqual(set(filtered_gt), {"case_a"})
        self.assertEqual(set(filtered_prediction["case_a"]), {"video_a"})


if __name__ == "__main__":
    unittest.main()
