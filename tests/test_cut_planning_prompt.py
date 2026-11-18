import unittest

from visual_workflow.prompt.cut_planning_prompt import build_cut_planning_prompt_for_input_mode


class CutPlanningPromptTest(unittest.TestCase):
    def setUp(self):
        self.prompt = (
            "        - description：视觉高光片段的内容描述\n"
            "        - keyframe：该视觉片段的中间帧图像，用于辅助细粒度判断\n"
            "summary、segments描述与keyframe视觉信息"
        )

    def test_full_input_preserves_prompt(self):
        self.assertEqual(
            build_cut_planning_prompt_for_input_mode(self.prompt, "time_desc_frame"),
            self.prompt,
        )

    def test_description_only_removes_keyframe_instruction(self):
        result = build_cut_planning_prompt_for_input_mode(self.prompt, "time_desc")
        self.assertIn("不包含 keyframe 图像", result)
        self.assertNotIn("- keyframe：", result)

    def test_frame_only_removes_description_instruction(self):
        result = build_cut_planning_prompt_for_input_mode(self.prompt, "time_frame")
        self.assertIn("不包含 description 字段", result)
        self.assertNotIn("- description：", result)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown cut-planning input mode"):
            build_cut_planning_prompt_for_input_mode(self.prompt, "unknown")


if __name__ == "__main__":
    unittest.main()
