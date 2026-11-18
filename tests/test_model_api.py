import os
import unittest
from unittest.mock import patch

from visual_workflow.model_api import (
    MODEL_REGISTRY,
    ModelRunner,
    ModelConfigurationError,
    ModelHTTPError,
    _should_retry_llm_exception,
    get_model_cfg,
)


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = "fake response"

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, response_data):
        self.response_data = response_data
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return FakeResponse(self.response_data)


class ModelApiTest(unittest.TestCase):
    def test_all_model_aliases_resolve_to_public_provider_configs(self):
        expected = {
            "gemini-2.5-pro": ("gemini_generate_content", "gemini-2.5-pro"),
            "gemini-3-pro": ("gemini_generate_content", "gemini-3.1-pro-preview"),
            "gpt-5.4": ("openai_responses", "gpt-5.4"),
            "gpt-5.5": ("openai_responses", "gpt-5.5"),
            "qwen-3.5-plus": ("openai_chat", "qwen3.5-plus"),
            "qwen-3.7-plus": ("openai_chat", "qwen3.7-plus"),
            "glm-5v-turbo": ("openai_chat", "glm-5v-turbo"),
            "seed-2.0-lite": ("ark_responses", "doubao-seed-2-0-lite-260215"),
        }
        with patch.dict(os.environ, {}, clear=True):
            for alias, (provider, model_id) in expected.items():
                cfg = get_model_cfg(alias)
                self.assertEqual(cfg["provider"], provider)
                self.assertEqual(cfg["model_id"], model_id)
                self.assertTrue(cfg["base_url"].startswith("https://"))

    def test_empty_optional_timeout_uses_default(self):
        with patch.dict(os.environ, {"ROUGHCUT_API_TIMEOUT_SECONDS": ""}, clear=True):
            self.assertEqual(get_model_cfg("gpt-5.4")["timeout_s"], 360)

    def test_missing_key_is_a_non_retryable_configuration_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ModelConfigurationError) as context:
                ModelRunner(FakeSession({})).run(
                    [{"role": "user", "content": "hello"}], "gpt-5.4"
                )
        self.assertFalse(_should_retry_llm_exception(context.exception))

    def test_http_retry_policy_distinguishes_auth_and_transient_errors(self):
        self.assertFalse(_should_retry_llm_exception(ModelHTTPError("unauthorized", 401)))
        self.assertTrue(_should_retry_llm_exception(ModelHTTPError("rate limited", 429)))
        self.assertTrue(_should_retry_llm_exception(ModelHTTPError("provider failure", 503)))

    def test_qwen_uses_bearer_chat_completions_and_normalizes_images(self):
        session = FakeSession(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpg;base64,AA==", "detail": "low"},
                    },
                ],
            }
        ]
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            result = ModelRunner(session).run(messages, "qwen-3.7-plus")
        call = session.calls[0]
        self.assertEqual(result, ("ok", 3, 2))
        self.assertEqual(call["url"], "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")
        image_url = call["json"]["messages"][0]["content"][1]["image_url"]
        self.assertEqual(image_url, {"url": "data:image/jpeg;base64,AA=="})
        self.assertNotIn("max_completion_tokens", call["json"])
        self.assertNotIn("max_tokens", call["json"])

    def test_glm_uses_official_chat_completions_endpoint(self):
        session = FakeSession({"choices": [{"message": {"content": "ok"}}], "usage": {}})
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            ModelRunner(session).run([{"role": "user", "content": "hello"}], "glm-5v-turbo")
        self.assertEqual(
            session.calls[0]["url"],
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        )
        self.assertNotIn("max_tokens", session.calls[0]["json"])
        self.assertNotIn("max_completion_tokens", session.calls[0]["json"])

    def test_all_models_omit_output_limits_by_default(self):
        responses = {
            "openai_chat": {"choices": [{"message": {"content": "ok"}}]},
            "openai_responses": {"output_text": "ok"},
            "ark_responses": {"output_text": "ok"},
            "gemini_generate_content": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
        }
        for alias, registered in MODEL_REGISTRY.items():
            for overrides in ({}, {"max_output_tokens": None}):
                with self.subTest(model=alias, overrides=overrides):
                    session = FakeSession(responses[registered["provider"]])
                    with patch.dict(os.environ, {registered["key_envs"][0]: "test-key"}, clear=True):
                        self.assertNotIn("max_output_tokens", get_model_cfg(alias))
                        ModelRunner(session).run(
                            [{"role": "user", "content": "hello"}], alias, **overrides
                        )
                    payload = session.calls[0]["json"]
                    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                        self.assertNotIn(key, payload)
                    self.assertNotIn("maxOutputTokens", payload.get("generationConfig", {}))

    def test_explicit_output_limit_override_uses_provider_field(self):
        cases = [
            ("qwen-3.5-plus", "DASHSCOPE_API_KEY", "max_completion_tokens",
             {"choices": [{"message": {"content": "ok"}}]}),
            ("glm-5v-turbo", "ZAI_API_KEY", "max_tokens",
             {"choices": [{"message": {"content": "ok"}}]}),
            ("gpt-5.4", "OPENAI_API_KEY", "max_output_tokens", {"output_text": "ok"}),
            ("seed-2.0-lite", "ARK_API_KEY", "max_output_tokens", {"output_text": "ok"}),
            ("gemini-2.5-pro", "GEMINI_API_KEY", "maxOutputTokens",
             {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}),
        ]
        for alias, key_env, field, response in cases:
            for override_name in ("max_output_tokens", "max_tokens"):
                with self.subTest(model=alias, override=override_name):
                    session = FakeSession(response)
                    with patch.dict(os.environ, {key_env: "test-key"}, clear=True):
                        ModelRunner(session).run(
                            [{"role": "user", "content": "hello"}],
                            alias,
                            **{override_name: 1024},
                        )
                    payload = session.calls[0]["json"]
                    token_payload = payload.get("generationConfig", payload)
                    self.assertEqual(token_payload[field], 1024)

    def test_openai_uses_responses_multimodal_format(self):
        session = FakeSession(
            {
                "output": [{"content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }
        )
        messages = [
            {"role": "system", "content": "judge"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpg;base64,AA=="}},
                ],
            },
        ]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            result = ModelRunner(session).run(messages, "gpt-5.4")
        call = session.calls[0]
        self.assertEqual(result, ("ok", 4, 2))
        self.assertEqual(call["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(call["json"]["input"][1]["content"][0]["type"], "input_text")
        self.assertEqual(call["json"]["input"][1]["content"][1]["type"], "input_image")
        self.assertEqual(
            call["json"]["input"][1]["content"][1]["image_url"],
            "data:image/jpeg;base64,AA==",
        )

    def test_openai_omits_unsupported_sampling_parameters(self):
        session = FakeSession({"output_text": "ok", "usage": {}})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            ModelRunner(session).run(
                [{"role": "user", "content": "hello"}],
                "gpt-5.5",
                temperature=0.7,
                top_p=0.95,
            )
        payload = session.calls[0]["json"]
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)

    def test_ark_uses_responses_endpoint(self):
        session = FakeSession({"output_text": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}})
        with patch.dict(os.environ, {"ARK_API_KEY": "test-key"}, clear=True):
            ModelRunner(session).run([{"role": "user", "content": "hello"}], "seed-2.0-lite")
        call = session.calls[0]
        self.assertEqual(call["url"], "https://ark.cn-beijing.volces.com/api/v3/responses")
        self.assertEqual(call["json"]["model"], "doubao-seed-2-0-lite-260215")

    def test_gemini_preserves_system_instruction_and_inline_mime(self):
        session = FakeSession(
            {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
            }
        )
        messages = [
            {"role": "system", "content": "judge"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            },
        ]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            result = ModelRunner(session).run(messages, "gemini-2.5-pro")
        call = session.calls[0]
        self.assertEqual(result, ("ok", 5, 2))
        self.assertEqual(
            call["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
        )
        self.assertEqual(call["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(call["json"]["systemInstruction"]["parts"], [{"text": "judge"}])
        inline_data = call["json"]["contents"][0]["parts"][1]["inlineData"]
        self.assertEqual(inline_data["mimeType"], "image/png")

    def test_chat_adapter_forwards_sampling_parameters(self):
        session = FakeSession({"choices": [{"message": {"content": "ok"}}], "usage": {}})
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            ModelRunner(session).run(
                [{"role": "user", "content": "hello"}],
                model="qwen-3.5-plus",
                temperature=0.7,
                top_p=0.95,
            )
        payload = session.calls[0]["json"]
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.95)

    def test_sampling_parameters_use_gemini_field_names(self):
        session = FakeSession(
            {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {},
            }
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            ModelRunner(session).run(
                [{"role": "user", "content": "hello"}],
                "gemini-2.5-pro",
                temperature=0.4,
                top_p=0.8,
            )
        generation_config = session.calls[0]["json"]["generationConfig"]
        self.assertEqual(generation_config["temperature"], 0.4)
        self.assertEqual(generation_config["topP"], 0.8)


if __name__ == "__main__":
    unittest.main()
