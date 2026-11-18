# Public Model APIs

All visual, speech, and evaluator requests use
`visual_workflow/model_api.py`. The paper-facing aliases remain stable, while
provider model IDs, credentials, and base URLs are configurable through
environment variables.

## Provider Mapping

| Benchmark alias | Provider protocol | Default public base URL | Default provider model ID |
|---|---|---|---|
| `gemini-2.5-pro` | Gemini `generateContent` | `https://generativelanguage.googleapis.com/v1beta` | `gemini-2.5-pro` |
| `gemini-3-pro` | Gemini `generateContent` | `https://generativelanguage.googleapis.com/v1beta` | `gemini-3.1-pro-preview` |
| `gpt-5.4` | OpenAI Responses API | `https://api.openai.com/v1` | `gpt-5.4` |
| `gpt-5.5` | OpenAI Responses API | `https://api.openai.com/v1` | `gpt-5.5` |
| `qwen-3.5-plus` | Model Studio OpenAI-compatible Chat Completions | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.5-plus` |
| `qwen-3.7-plus` | Model Studio OpenAI-compatible Chat Completions | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.7-plus` |
| `glm-5v-turbo` | Z.ai OpenAI-compatible Chat Completions | `https://open.bigmodel.cn/api/paas/v4` | `glm-5v-turbo` |
| `seed-2.0-lite` | Volcengine Ark Responses API | `https://ark.cn-beijing.volces.com/api/v3` | `doubao-seed-2-0-lite-260215` |

The defaults describe public provider protocols, not guaranteed account
entitlements. Preview, dated, or paper-specific model IDs may be unavailable in
a particular account or region. Set the corresponding
`ROUGHCUT_<MODEL>_MODEL_ID` variable to an accessible model version without
changing workflow code.

Released workflows, keyframe checks, and the smoke-test command do not set an
explicit output-token limit. Requests omit `max_tokens`,
`max_completion_tokens`, `max_output_tokens`, and Gemini's `maxOutputTokens`
unless a caller explicitly supplies an override. Provider defaults and model
limits still apply; omitting a limit does not mean unlimited output.

For programmatic callers that explicitly pass `max_output_tokens` (or the
legacy `max_tokens` alias) to `ModelRunner.run()`, the adapter maps it to the
provider-specific field, including `max_completion_tokens` for Qwen3.5-Plus
and Qwen3.7-Plus. Optional `temperature` and
`top_p` values are forwarded only for model interfaces that document support;
they are omitted from GPT-5.4 and GPT-5.5 Responses requests.

The paper-facing `gemini-3-pro` alias requires special handling. Google shut
down `gemini-3-pro-preview` on March 9, 2026 and recommends
`gemini-3.1-pro-preview`; the runnable public default therefore uses that
successor. This preserves the workflow interface but does not reproduce the
retired model exactly. `gemini-2.5-pro` is currently scheduled for shutdown on
October 16, 2026, so its provider ID may also need an override after that date.

## Credentials

Set the shared key for each provider:

```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export DASHSCOPE_API_KEY=...
export ZAI_API_KEY=...
export ARK_API_KEY=...
```

A model-specific `ROUGHCUT_<MODEL>_API_KEYS` value takes precedence and may
contain comma-separated keys. See `.env.example` for model-ID and base-URL
overrides. Alibaba Cloud recommends workspace-specific Model Studio domains;
set `DASHSCOPE_BASE_URL` when one is assigned to your workspace.

After loading credentials, verify only the providers you intend to use. This
command sends a real, billable request for every alias passed to `--models`:

```bash
python scripts/tools/smoke_test_model_api.py \
  --models gemini-2.5-pro gpt-5.4
```

Add `--image path/to/small.jpg` to verify the multimodal payload as well as the
text endpoint. No model is selected implicitly, which avoids accidental calls
to all providers.

The smoke test also uses provider-default output limits. A short prompt does
not guarantee a small reasoning-token budget or a fixed cost.

## Official References

- Gemini GenerateContent and multimodal input:
  <https://ai.google.dev/api/generate-content>
- Gemini video input limits and Files API guidance:
  <https://ai.google.dev/gemini-api/docs/video-understanding>
- Gemini model lifecycle and shutdown dates:
  <https://ai.google.dev/gemini-api/docs/deprecations>
- OpenAI Responses API and image input:
  <https://platform.openai.com/docs/api-reference/responses>
- GPT-5.4 and GPT-5.5 model pages:
  <https://developers.openai.com/api/docs/models/gpt-5.4> and
  <https://developers.openai.com/api/docs/models/gpt-5.5>
- Alibaba Cloud Model Studio OpenAI-compatible Chat Completions:
  <https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions>
- Qwen visual model IDs and input limits:
  <https://help.aliyun.com/en/model-studio/vision-model/>
- Z.ai GLM-5V-Turbo API:
  <https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5v-turbo>
- Volcengine Ark Chat Completions:
  <https://api.volcengine.com/api-docs/view?action=ChatCompletions&serviceCode=ark&version=2024-01-01>
- Volcengine Ark Responses quickstart:
  <https://www.volcengine.com/docs/82379/1795150>

## Changing an API Adapter

Provider-specific logic is confined to `visual_workflow/model_api.py`:

- `MODEL_REGISTRY` defines aliases and environment-variable precedence;
- `_normalize_chat_messages()` builds Qwen and GLM messages;
- `_responses_input()` builds OpenAI and Ark inputs;
- `_gemini_request()` builds Gemini inputs;
- `_chat_text()`, `_responses_text()`, and `_usage()` parse responses.

The agent implementations consume only `ModelRunner.run()` and do not contain
provider URLs, authentication headers, or response parsing.

## Direct Gemini Video Input

The optional direct-video workflow uses Gemini `generateContent` with inline
base64 data. To leave room for base64 expansion and the text prompt, source
videos at or above 14 MB are converted to a smaller MP4 proxy. Formats that are
not listed in Gemini's supported video MIME types, including MKV, are also
converted before the request. The proxy cache is stored below
`outputs/_cache/gemini_video_proxy` by default and is invalidated when the
source file size or modification time changes.

This path is intended for the video-input ablation. For larger or repeatedly
used videos, Google's Files API is the preferred public interface; it is not
used here because the released experiment sends each proxy only once.

## Validation Scope

`tests/test_model_api.py` checks the request URL, authentication header,
provider-specific multimodal payload, generation parameters, usage parsing, and
text extraction with a mock HTTP transport. It does not make billable network
requests. Before a public release, run at least one small credentialed case for
every provider because account entitlements, regional endpoints, and preview
model availability cannot be verified without provider credentials.
