"""Official public API adapters used by all RoughCut-Bench workflows.

The benchmark keeps its paper-facing model aliases stable while allowing users
to override provider model IDs, base URLs, and credentials through environment
variables. All provider-specific request construction and response parsing lives
in this module so endpoint changes do not require edits to the agents.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import re
import time
from typing import Any

import requests


DEFAULT_TIMEOUT_SECONDS = 360


class LLMJsonParseError(ValueError):
    def __init__(self, message, raw_response=None, normalized_response=None):
        super().__init__(message)
        self.raw_response = raw_response
        self.normalized_response = normalized_response


class VLMDataInspectionError(RuntimeError):
    def __init__(self, message, inspection_type="unknown", response_text=None, debug_path=None):
        super().__init__(message)
        self.inspection_type = inspection_type
        self.response_text = response_text
        self.debug_path = debug_path


class ModelConfigurationError(RuntimeError):
    """Raised when a required local model setting is missing or invalid."""


class ModelHTTPError(RuntimeError):
    """Raised for a non-successful response from a provider API."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = int(status_code)


def _should_retry_llm_exception(ex):
    if isinstance(ex, (ModelConfigurationError, VLMDataInspectionError)):
        return False
    if isinstance(ex, ModelHTTPError):
        return ex.status_code in {408, 409, 425, 429} or ex.status_code >= 500
    return True


def _first_env(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _load_api_keys(names: tuple[str, ...]) -> list[str]:
    value = _first_env(names)
    return [key.strip() for key in value.split(",") if key.strip()]


# Defaults follow the providers' public APIs. Model-specific variables take
# precedence over shared provider variables. Legacy *_DEPLOYMENT and *_ENDPOINT
# names remain accepted to avoid breaking existing experiment launchers.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "glm-5v-turbo": {
        "provider": "openai_chat",
        "client_type": "openai_chat",
        "default_model_id": "glm-5v-turbo",
        "model_envs": ("ROUGHCUT_GLM_5V_TURBO_MODEL_ID", "ROUGHCUT_GLM_5V_TURBO_DEPLOYMENT"),
        "key_envs": ("ROUGHCUT_GLM_5V_TURBO_API_KEYS", "ZAI_API_KEY", "ZHIPUAI_API_KEY"),
        "base_url_envs": ("ROUGHCUT_GLM_5V_TURBO_BASE_URL", "ZAI_BASE_URL"),
        "endpoint_envs": ("ROUGHCUT_GLM_5V_TURBO_ENDPOINT",),
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "strip_image_detail": True,
    },
    "qwen-3.5-plus": {
        "provider": "openai_chat",
        "client_type": "openai_chat",
        "default_model_id": "qwen3.5-plus",
        "model_envs": ("ROUGHCUT_QWEN_3_5_PLUS_MODEL_ID", "ROUGHCUT_QWEN_3_5_PLUS_DEPLOYMENT"),
        "key_envs": ("ROUGHCUT_QWEN_3_5_PLUS_API_KEYS", "DASHSCOPE_API_KEY"),
        "base_url_envs": ("ROUGHCUT_QWEN_3_5_PLUS_BASE_URL", "DASHSCOPE_BASE_URL"),
        "endpoint_envs": ("ROUGHCUT_QWEN_3_5_PLUS_ENDPOINT",),
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_tokens_field": "max_completion_tokens",
        "strip_image_detail": True,
    },
    "qwen-3.7-plus": {
        "provider": "openai_chat",
        "client_type": "openai_chat",
        "default_model_id": "qwen3.7-plus",
        "model_envs": ("ROUGHCUT_QWEN_3_7_PLUS_MODEL_ID", "ROUGHCUT_QWEN_3_7_PLUS_DEPLOYMENT"),
        "key_envs": ("ROUGHCUT_QWEN_3_7_PLUS_API_KEYS", "DASHSCOPE_API_KEY"),
        "base_url_envs": ("ROUGHCUT_QWEN_3_7_PLUS_BASE_URL", "DASHSCOPE_BASE_URL"),
        "endpoint_envs": ("ROUGHCUT_QWEN_3_7_PLUS_ENDPOINT",),
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_tokens_field": "max_completion_tokens",
        "strip_image_detail": True,
    },
    "seed-2.0-lite": {
        "provider": "ark_responses",
        "client_type": "ark_responses",
        "default_model_id": "doubao-seed-2-0-lite-260215",
        "model_envs": ("ROUGHCUT_SEED_2_0_LITE_MODEL_ID", "ROUGHCUT_SEED_2_0_LITE_DEPLOYMENT"),
        "key_envs": ("ROUGHCUT_SEED_2_0_LITE_API_KEYS", "ARK_API_KEY"),
        "base_url_envs": ("ROUGHCUT_SEED_2_0_LITE_BASE_URL", "ARK_BASE_URL"),
        "endpoint_envs": ("ROUGHCUT_SEED_2_0_LITE_ENDPOINT",),
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    "gemini-2.5-pro": {
        "provider": "gemini_generate_content",
        "client_type": "gemini_native",
        "default_model_id": "gemini-2.5-pro",
        "model_envs": ("ROUGHCUT_GEMINI_2_5_PRO_MODEL_ID", "ROUGHCUT_GEMINI_2_5_PRO_DEPLOYMENT"),
        "key_envs": ("ROUGHCUT_GEMINI_2_5_PRO_API_KEYS", "GEMINI_API_KEY"),
        "base_url_envs": ("ROUGHCUT_GEMINI_2_5_PRO_BASE_URL", "GEMINI_BASE_URL"),
        "endpoint_envs": ("ROUGHCUT_GEMINI_2_5_PRO_ENDPOINT",),
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    "gemini-3-pro": {
        "provider": "gemini_generate_content",
        "client_type": "gemini_native",
        # The paper-facing Gemini 3 Pro alias is retained, but Google's exact
        # preview model was shut down on 2026-03-09. Use its public successor
        # by default and allow exact provider IDs to be supplied via the env.
        "default_model_id": "gemini-3.1-pro-preview",
        "model_envs": ("ROUGHCUT_GEMINI_3_PRO_MODEL_ID", "ROUGHCUT_GEMINI_3_PRO_DEPLOYMENT"),
        "key_envs": ("ROUGHCUT_GEMINI_3_PRO_API_KEYS", "GEMINI_API_KEY"),
        "base_url_envs": ("ROUGHCUT_GEMINI_3_PRO_BASE_URL", "GEMINI_BASE_URL"),
        "endpoint_envs": ("ROUGHCUT_GEMINI_3_PRO_ENDPOINT",),
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "generation_config": {"mediaResolution": "MEDIA_RESOLUTION_MEDIUM"},
    },
    "gpt-5.4": {
        "provider": "openai_responses",
        "client_type": "openai_responses",
        "default_model_id": "gpt-5.4",
        "model_envs": ("ROUGHCUT_GPT_5_4_MODEL_ID", "ROUGHCUT_GPT_5_4_DEPLOYMENT"),
        "key_envs": ("ROUGHCUT_GPT_5_4_API_KEYS", "OPENAI_API_KEY"),
        "base_url_envs": ("ROUGHCUT_GPT_5_4_BASE_URL", "OPENAI_BASE_URL"),
        "endpoint_envs": ("ROUGHCUT_GPT_5_4_ENDPOINT",),
        "default_base_url": "https://api.openai.com/v1",
        "supports_sampling_params": False,
    },
    "gpt-5.5": {
        "provider": "openai_responses",
        "client_type": "openai_responses",
        "default_model_id": "gpt-5.5",
        "model_envs": ("ROUGHCUT_GPT_5_5_MODEL_ID", "ROUGHCUT_GPT_5_5_DEPLOYMENT"),
        "key_envs": ("ROUGHCUT_GPT_5_5_API_KEYS", "OPENAI_API_KEY"),
        "base_url_envs": ("ROUGHCUT_GPT_5_5_BASE_URL", "OPENAI_BASE_URL"),
        "endpoint_envs": ("ROUGHCUT_GPT_5_5_ENDPOINT",),
        "default_base_url": "https://api.openai.com/v1",
        "supports_sampling_params": False,
    },
}


def get_model_cfg(model: str) -> dict[str, Any]:
    if model not in MODEL_REGISTRY:
        raise KeyError(f"unknown model: {model}. available: {list(MODEL_REGISTRY)}")

    cfg = copy.deepcopy(MODEL_REGISTRY[model])
    cfg["alias"] = model
    cfg["model_id"] = _first_env(cfg["model_envs"], cfg["default_model_id"])
    cfg["deployment"] = cfg["model_id"]
    cfg["api_keys"] = _load_api_keys(cfg["key_envs"])
    cfg["api_key_env"] = " or ".join(cfg["key_envs"])
    cfg["base_url"] = _first_env(cfg["base_url_envs"], cfg["default_base_url"]).rstrip("/")
    cfg["endpoint"] = _first_env(cfg["endpoint_envs"])
    cfg["timeout_s"] = int(os.environ.get("ROUGHCUT_API_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS)
    return cfg


def _join_endpoint(base_url: str, suffix: str) -> str:
    suffix = "/" + suffix.lstrip("/")
    return base_url.rstrip("/") + suffix


def _data_url_parts(url: str) -> tuple[str, str]:
    match = re.fullmatch(r"data:([^;,]+);base64,(.+)", url, flags=re.DOTALL)
    if not match:
        raise ValueError("Gemini inline media must be a base64 data URL")
    return match.group(1), match.group(2)


def _inspection_type(text: str) -> str:
    lowered = text.lower()
    if "image" in lowered:
        return "image"
    if "text" in lowered:
        return "text"
    return "unknown"


class ModelRunner:
    """Provider-neutral entry point for all configured public model APIs."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def _choose_api_key(self, cfg: dict[str, Any]) -> str:
        keys = cfg.get("api_keys") or []
        if not keys:
            raise ModelConfigurationError(
                f"No API key configured for {cfg['alias']}. Set {cfg['api_key_env']}."
            )
        fixed_index = cfg.get("fixed_api_key_index")
        return keys[int(fixed_index)] if fixed_index is not None else random.choice(keys)

    def _endpoint(self, cfg: dict[str, Any], suffix: str) -> str:
        return cfg.get("endpoint") or _join_endpoint(cfg["base_url"], suffix)

    @staticmethod
    def _sampling_params(cfg: dict[str, Any], *, gemini: bool = False) -> dict[str, float]:
        params: dict[str, float] = {}
        if not cfg.get("supports_sampling_params", True):
            return params
        if cfg.get("temperature") is not None:
            temperature = float(cfg["temperature"])
            if not 0.0 <= temperature < 2.0:
                raise ValueError("temperature must be in [0, 2)")
            params["temperature"] = temperature
        if cfg.get("top_p") is not None:
            top_p = float(cfg["top_p"])
            if not 0.0 < top_p <= 1.0:
                raise ValueError("top_p must be in (0, 1]")
            params["topP" if gemini else "top_p"] = top_p
        return params

    def _post(self, cfg: dict[str, Any], endpoint: str, payload: dict[str, Any], headers: dict[str, str]) -> dict:
        response = self.session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=int(cfg.get("timeout_s", DEFAULT_TIMEOUT_SECONDS)),
        )
        if response.status_code < 200 or response.status_code >= 300:
            message = f"{cfg['provider']} request failed with HTTP {response.status_code}: {response.text}"
            if "data_inspection_failed" in response.text:
                raise VLMDataInspectionError(
                    message,
                    inspection_type=_inspection_type(response.text),
                    response_text=response.text,
                )
            raise ModelHTTPError(message, response.status_code)
        try:
            data = response.json()
        except ValueError as ex:
            raise RuntimeError(f"{cfg['provider']} returned invalid JSON: {response.text[:1000]}") from ex
        if data.get("error"):
            raise RuntimeError(f"{cfg['provider']} API error: {data['error']}")
        return data

    @staticmethod
    def _normalize_chat_messages(messages: list[dict], cfg: dict[str, Any]) -> list[dict]:
        normalized = copy.deepcopy(messages)
        for message in normalized:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image_url":
                    continue
                image_url = item.get("image_url")
                if not isinstance(image_url, dict):
                    continue
                if cfg.get("strip_image_detail"):
                    image_url.pop("detail", None)
                if isinstance(image_url.get("url"), str):
                    image_url["url"] = image_url["url"].replace(
                        "data:image/jpg;base64,", "data:image/jpeg;base64,"
                    )
        return normalized

    @staticmethod
    def _responses_input(messages: list[dict]) -> list[dict]:
        result = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                result.append({"role": message.get("role", "user"), "content": content})
                continue
            parts = []
            for item in content:
                item_type = item.get("type")
                if item_type in {"text", "input_text"}:
                    parts.append({"type": "input_text", "text": item.get("text", "")})
                elif item_type in {"image_url", "input_image"}:
                    image_value = item.get("image_url", "")
                    if isinstance(image_value, dict):
                        detail = image_value.get("detail")
                        image_value = image_value.get("url", "")
                    else:
                        detail = item.get("detail")
                    if isinstance(image_value, str):
                        image_value = image_value.replace(
                            "data:image/jpg;base64,", "data:image/jpeg;base64,"
                        )
                    part = {"type": "input_image", "image_url": image_value}
                    if detail in {"low", "high", "auto"}:
                        part["detail"] = detail
                    parts.append(part)
                else:
                    raise ValueError(f"Unsupported Responses API content type: {item_type}")
            result.append({"role": message.get("role", "user"), "content": parts})
        return result

    @staticmethod
    def _gemini_request(messages: list[dict], cfg: dict[str, Any]) -> dict[str, Any]:
        system_parts = []
        contents = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            items = [{"type": "text", "text": content}] if isinstance(content, str) else content
            parts = []
            for item in items:
                item_type = item.get("type")
                if item_type == "text":
                    parts.append({"text": item.get("text", "")})
                elif item_type == "image_url":
                    image_value = item.get("image_url", "")
                    url = image_value.get("url", "") if isinstance(image_value, dict) else str(image_value)
                    mime_type, data = _data_url_parts(url)
                    if mime_type == "image/jpg":
                        mime_type = "image/jpeg"
                    parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
                elif item_type == "video_url":
                    video_value = item.get("video_url", "")
                    url = video_value.get("url", "") if isinstance(video_value, dict) else str(video_value)
                    mime_type, data = _data_url_parts(url)
                    parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
                else:
                    raise ValueError(f"Unsupported Gemini content type: {item_type}")
            if role in {"system", "developer"}:
                system_parts.extend(part for part in parts if "text" in part)
            else:
                contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})

        generation_config = copy.deepcopy(cfg.get("generation_config", {}))
        if cfg.get("max_output_tokens") is not None:
            generation_config["maxOutputTokens"] = int(cfg["max_output_tokens"])
        generation_config.update(ModelRunner._sampling_params(cfg, gemini=True))
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        return payload

    @staticmethod
    def _chat_text(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Chat Completions response missing choices: {data}")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        raise RuntimeError(f"Chat Completions response missing text: {data}")

    @staticmethod
    def _responses_text(data: dict) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        texts = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                texts.append(item["text"])
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        if not texts:
            raise RuntimeError(f"Responses API response missing text: {data}")
        return "".join(texts)

    @staticmethod
    def _usage(data: dict) -> tuple[int, int]:
        usage = data.get("usage") or data.get("usageMetadata") or {}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", usage.get("promptTokenCount", 0)))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", usage.get("candidatesTokenCount", 0)))
        return int(input_tokens or 0), int(output_tokens or 0)

    def _run_openai_chat(self, messages: list[dict], cfg: dict[str, Any]) -> tuple[str, int, int]:
        payload = {
            "model": cfg["model_id"],
            "messages": self._normalize_chat_messages(messages, cfg),
            "stream": False,
        }
        if cfg.get("max_output_tokens") is not None:
            max_tokens_field = cfg.get("max_tokens_field", "max_tokens")
            payload[max_tokens_field] = int(cfg["max_output_tokens"])
        payload.update(self._sampling_params(cfg))
        data = self._post(
            cfg,
            self._endpoint(cfg, "chat/completions"),
            payload,
            {"Authorization": f"Bearer {self._choose_api_key(cfg)}", "Content-Type": "application/json"},
        )
        text = self._chat_text(data).strip()
        return text, *self._usage(data)

    # Retained for the Qwen frame-inspection fallback in cut_planning_agent.py.
    def _run_vlm_native(self, messages: list[dict], cfg: dict[str, Any]) -> tuple[str, int, int]:
        return self._run_openai_chat(messages, cfg)

    def _run_responses(self, messages: list[dict], cfg: dict[str, Any]) -> tuple[str, int, int]:
        payload = {
            "model": cfg["model_id"],
            "input": self._responses_input(messages),
        }
        if cfg.get("max_output_tokens") is not None:
            payload["max_output_tokens"] = int(cfg["max_output_tokens"])
        payload.update(self._sampling_params(cfg))
        data = self._post(
            cfg,
            self._endpoint(cfg, "responses"),
            payload,
            {"Authorization": f"Bearer {self._choose_api_key(cfg)}", "Content-Type": "application/json"},
        )
        text = self._responses_text(data).strip()
        return text, *self._usage(data)

    def _run_gemini(self, messages: list[dict], cfg: dict[str, Any]) -> tuple[str, int, int]:
        endpoint = cfg.get("endpoint") or _join_endpoint(
            cfg["base_url"], f"models/{cfg['model_id']}:generateContent"
        )
        data = self._post(
            cfg,
            endpoint,
            self._gemini_request(messages, cfg),
            {"x-goog-api-key": self._choose_api_key(cfg), "Content-Type": "application/json"},
        )
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini response missing candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if not text:
            raise RuntimeError(f"Gemini response missing text: {data}")
        return text, *self._usage(data)

    def run(self, messages: list[dict], model: str, **overrides: Any) -> tuple[str, int, int]:
        cfg = get_model_cfg(model)
        if "max_tokens" in overrides:
            overrides["max_output_tokens"] = overrides.pop("max_tokens")
        cfg.update(overrides)
        started = time.time()
        provider = cfg["provider"]
        if provider == "openai_chat":
            result = self._run_openai_chat(messages, cfg)
        elif provider in {"openai_responses", "ark_responses"}:
            result = self._run_responses(messages, cfg)
        elif provider == "gemini_generate_content":
            result = self._run_gemini(messages, cfg)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        logging.info(
            "Model request: alias=%s provider=%s model_id=%s time=%.2fs",
            model,
            provider,
            cfg["model_id"],
            time.time() - started,
        )
        return result


# Backward-compatible import for code written before the public adapter was
# renamed. Released workflows use ModelRunner directly.
GPTRunner = ModelRunner
