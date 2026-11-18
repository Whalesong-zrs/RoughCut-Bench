import unittest

from visual_workflow.video_input import gemini_supports_inline_video, gemini_video_mime_type


class VideoInputTest(unittest.TestCase):
    def test_official_gemini_video_mime_types(self):
        self.assertEqual(gemini_video_mime_type("sample.mov"), "video/quicktime")
        self.assertEqual(gemini_video_mime_type("sample.avi"), "video/avi")
        self.assertEqual(gemini_video_mime_type("sample.webm"), "video/webm")
        with self.assertRaisesRegex(ValueError, "does not list .mkv"):
            gemini_video_mime_type("sample.mkv")

    def test_unsupported_video_format_requires_proxy(self):
        self.assertTrue(gemini_supports_inline_video("sample.mp4"))
        self.assertFalse(gemini_supports_inline_video("sample.mkv"))


if __name__ == "__main__":
    unittest.main()
