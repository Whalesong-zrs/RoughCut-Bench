import copy
import unittest
from pathlib import Path

from benchmark_evaluation.convert_workflow_outputs import output_path
from benchmark_evaluation.cut_plan_schema import (
    CutPlanValidationError,
    SCHEMA_VERSION,
    convert_workflow_output,
    validate_public_cut_plan,
)


class CutPlanSchemaTest(unittest.TestCase):
    def setUp(self):
        self.public_plan = {
            "schema_version": SCHEMA_VERSION,
            "case_id": "case_001",
            "ordered_videos": [
                {
                    "source_id": "video_b",
                    "segments": [
                        {"start_time": 1.0, "end_time": 2.5},
                        {"start_time": 3.0, "end_time": 4.0},
                    ],
                },
                {
                    "source_id": "video_a",
                    "segments": [{"start_time": 0.0, "end_time": 1.5}],
                },
            ],
        }

    def test_valid_public_plan(self):
        normalized = validate_public_cut_plan(self.public_plan, expected_case_id="case_001")
        self.assertEqual([item["source_id"] for item in normalized["ordered_videos"]], ["video_b", "video_a"])

    def test_rejects_duplicate_source(self):
        plan = copy.deepcopy(self.public_plan)
        plan["ordered_videos"][1]["source_id"] = "video_b"
        with self.assertRaisesRegex(CutPlanValidationError, "duplicate source_id"):
            validate_public_cut_plan(plan)

    def test_rejects_overlapping_segments(self):
        plan = copy.deepcopy(self.public_plan)
        plan["ordered_videos"][0]["segments"][1]["start_time"] = 2.0
        with self.assertRaisesRegex(CutPlanValidationError, "non-overlapping"):
            validate_public_cut_plan(plan)

    def test_converts_legacy_workflow_output(self):
        legacy = {
            "entry": "case_001.json",
            "raw_understanding_result": {
                "video_0": {"video_name": "source_videos/video_a.mp4"},
                "video_1": {"video_name": "source_videos/video_b.mp4"},
            },
            "get_arrange_result": {
                "arrange_result": {
                    "arrange_results": [
                        {
                            "video_id": "video_1",
                            "segments": [{"start_time": 1.0, "end_time": 2.0}],
                        },
                        {
                            "video_id": "video_0",
                            "segments": [{"start_time": 0.0, "end_time": 1.0}],
                        },
                    ],
                    "duration_adjustment": {},
                }
            },
        }
        converted = convert_workflow_output(legacy, "case_001")
        self.assertEqual(
            [item["source_id"] for item in converted["ordered_videos"]],
            ["video_b", "video_a"],
        )

    def test_converter_names_directory_outputs_by_case_id(self):
        self.assertEqual(
            output_path("case_001", Path("converted"), multiple=True),
            Path("converted/case_001.json"),
        )

    def test_converter_rejects_explicit_filename_that_differs_from_case_id(self):
        with self.assertRaisesRegex(ValueError, "filename must match case_id"):
            output_path("case_001", Path("other_name.json"), multiple=False)


if __name__ == "__main__":
    unittest.main()
