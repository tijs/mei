# OpenAI compatibility contract — Chat Completions P0

This document **freezes** the version-pinned P0 OpenAI-compatible Chat
Completions contract for Mei: which official reference Mei tracks, which
request/response surface is shipped and covered by tests, which fields are
deliberately deferred, the status-code/error-envelope shape, and the
`max_completion_tokens` versus `max_tokens` policy. It is a living reference:
every "shipped" claim below is derived from the current source and tests and is
labeled with the exact source location; anything not yet pinned by tests is
explicitly marked **planned acceptance** and must not be presented as runtime
behavior.

- Mei side pin: `ServerConfig.version = "0.5.0"` (`Sources/MeiCore/ServerConfig.swift:7`),
  HEAD `950e8c2fdfa13ca6ed1dbfbd7794abc96705553a` (2026-09-14).
- Base URL: `http://127.0.0.1:8024/v1` (default; `--host`/`--port` reconfigurable).

## 1. Official reference pin

| Reference | URL | Retrieved |
|---|---|---|
| Create chat completion (canonical, current) | `https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create` | 2026-09-23 (UTC) |
| Create chat completion (legacy canonical; JS shell, redirects to the above) | `https://platform.openai.com/docs/api-reference/chat/create` | 2026-09-23 (UTC) |
| Error codes guide (status codes, envelope semantics) | `https://developers.openai.com/api/docs/guides/error-codes` | 2026-09-23 (UTC) |

Version pin: OpenAI publishes **no version number** on this reference, so the
pin is the exact URL plus the retrieval date above. Re-retrieval is
reproducible: OpenAI serves markdown renderings of documentation pages by
appending `.md` to the page URL, and the full doc index is advertised as
`llms.txt` on the same host. Any future behavior change must be dated and
diffed against this pin first.

Facts pinned by this retrieval (direct quotes):

- The endpoint is `POST /chat/completions`; "Returns a chat completion object,
  or a streamed sequence of chat completion chunk objects if the request is
  streamed."
- `max_completion_tokens`: "An upper bound for the number of tokens that can be
  generated for a completion, including visible output tokens and reasoning
  tokens."
- `max_tokens`: "The maximum number of tokens that can be generated in the chat
  completion." — "This value is now **deprecated in favor of
  max_completion_tokens**, and is not compatible with o-series models."
- The reference documents `stream_options` ("Options for streaming response.
  Only set this when you set `stream: true`"), `tool_choice`
  (`none`/`auto`/`required`/forced `{"type":"function","function":{"name":...}}`),
  `tools`, `stop`, `temperature` (0–2), `top_p` (0–1), `reasoning_effort`,
  `seed` (deprecated/removed upstream; Mei still honors it — see §7), and the
  `chat.completion` response object with `id`, `object`, `created`, `model`,
  `choices` (index, message, logprobs, finish_reason), `usage` (prompt/completion/total
  tokens + details), and `service_tier`.
- The error-codes guide confirms standard status semantics (400 invalid
  request, 401 auth, 403, 404, 422, 429 rate limit, 500, 503 overloaded) and
  that error payloads carry `error.message`, `error.type`, `error.param`, and
  `error.code`.

**Open ambiguity (upstream, unresolved):** the reference does **not** document
behavior when both `max_tokens` and `max_completion_tokens` are supplied in one
request on a model that accepts both (it only says `max_tokens` is incompatible
with o-series models). Mei therefore states its own deterministic policy in §6
rather than pretending upstream defines one.

## 2. P0 subset (relevant to Mei/Hermes)

Shipped and covered by tests at this pin:

1. `POST /v1/chat/completions`, non-streaming — single-choice completion with
   usage.
2. `POST /v1/chat/completions`, streaming — SSE `chat.completion.chunk` deltas,
   finish chunk, optional usage chunk via `stream_options.include_usage`,
   `[DONE]` terminator.
3. Native function tool calls — `tools`/`tool_choice` in requests, `tool_calls`
   in assistant messages and responses (streaming and non-streaming), `role:
   "tool"` + `tool_call_id` request messages.
4. Reasoning — `reasoning_content` request message field, response
   message field, and stream deltas (`--emit-reasoning true` default); request
   `reasoning_effort` passed to the engine.
5. Usage — `prompt_tokens`, `completion_tokens`, `total_tokens`,
   `prompt_tokens_details.cached_tokens`, plus Mei engine extensions
   (`tokens_per_second`, `prompt_tokens_per_second`, `prefill_ms`,
   `generate_ms`, `mei_memory_active_bytes/cache/peak`) when non-zero.
6. Identity/health — `GET /v1/models` (exact served model id), `GET /healthz`
   and `GET /health`.
7. Error envelope and status codes as specified in §5.

Everything else from the official reference is **deferred** (§7).

## 3. Request surface (shipped)

Decoder: `ChatRequest(json:)` — `Sources/MeiCore/OpenAITypes.swift:218-314`;
router dispatch — `Sources/MeiCore/Router.swift:59-116`.

| Field | Type | Shipped behavior / notes |
|---|---|---|
| `model` | string, **required** | Decoded, but **not validated** against the served id (no comparison anywhere in `Router`/`Engine`). Any string or `null` value is accepted. |
| `messages` | array, **required** | Roles are passed through to the chat template **unvalidated** (`MessageMapping.templateDictionary`, `Sources/MeiCore/MessageMapping.swift:9-49`); `user`/`system`/`assistant`/`tool` are the exercised ones. |
| `messages[].content` | string or array | Array of `{"type":"text","text":...}` parts joined with `"\n"`; parts without a text field are dropped (`FlexibleString`, `OpenAITypes.swift:159-171`). Image/audio parts are not decoded. |
| `messages[].tool_call_id` | string | Passed as `tool_call_id` for `role:"tool"` (`MessageMapping.swift:28-30`). |
| `messages[].tool_calls` | array | `{id?, type?, function:{name, arguments}}`; `name` required, `arguments` defaults to `"{}"` (`OpenAITypes.swift:266-286`). |
| `messages[].reasoning_content` | string | Passed through to the template (`MessageMapping.swift:20-23`). |
| `temperature` | number | Not range-validated (no 0–2 clamp); request wins over server default 0.6 (`Engine.swift:900-947`). |
| `top_p` | number | No range validation; default 0.95. |
| `top_k` | integer | Mei extension (not in upstream reference); default 20; `FlexibleInt` also accepts numeric strings/floats (truncated). |
| `min_p` | number | Mei extension; default 0.0. |
| `max_tokens` | integer | See §6. `FlexibleInt` (`OpenAITypes.swift:184-196`). |
| `stream` | boolean | Default `false`. `true` switches to SSE streaming. |
| `stop` | string or array | Up to N sequences passed as `extraStopStrings`; single string normalized to one-element array (`FlexibleStop`, `OpenAITypes.swift:198-210`). Cardinality not validated. |
| `tools` | array | Function tools are the supported P0 case; entries are copied verbatim into the template context (`MessageMapping.templateTools`, `MessageMapping.swift:51-63`). Non-function tool types are also passed through unvalidated — no acceptance coverage exists for them. |
| `tool_choice` | string or object | `"none"`/`"auto"`/`"required"` map to the template's `tool_choice` verbatim; any other string is a forced tool name (`tool_choice:"required"` + `tool_choice_name`). The nested OpenAI forced form `{"type":"function","function":{"name":X}}` extracts the name from `function.name` — a top-level `name` still wins when both are present — and pins it as `tool_choice_name` (`MessageMapping.additionalContext`, `MessageMapping.swift:65-112`). CoCore's forced-tool canary shape (`report_status`, nested `tool_choice`, strict schema, `max_tokens` 96, `temperature` 0; graze-social/cocore PR #237) is covered by `OpenAITypesTests.testCoCoreForcedToolCanaryPinsReportStatus`. |
| `repetition_penalty` / `presence_penalty` / `frequency_penalty` | number | Passed to the generator (defaults from config). |
| `seed` | unsigned int | Honored (`parameters.randomSeed`). Deprecated upstream; Mei keeps it for reproducible benchmark rows (§7). |
| `reasoning_effort` | string | Passed into the engine's thinking decision (`Engine.swift:261,285-286,342-349`). Values are not whitelisted. |
| `stream_options.include_usage` | boolean | Probed from the raw top level of the payload (`OpenAITypes.swift:311-313`); when `true` the stream's terminal sequence includes a usage chunk. Not gated on `stream:true` (upstream says only set when streaming). |
| Any other field | — | **Silently ignored** — `JSONDecoder` is non-strict; unknown keys produce no error. This is shipped behavior and the reason §7 fields are "inert" rather than rejected. |

Missing `model` or `messages`, or unparseable JSON, throws during decode → 400
(§5).

## 4. Response surface (shipped)

Serializer: `JSONEncoder` with `.sortedKeys` — deterministic key order
(`Router.swift:8-13`).

### Non-streaming `POST /v1/chat/completions` — `Router.completionResponse` (`Router.swift:158-173`)

- `object: "chat.completion"`, `id: "chatcmpl-" + 24 lowercase hex`,
  `created`: unix seconds, `model`: exact served model id.
- `choices: [{ index: 0, message, logprobs: null, finish_reason }]`.
- `message`: `role: "assistant"`; `content` present unless the run produced only
  tool calls (then `content` is null); `tool_calls: [{id, type: "function",
  function: {name, arguments}}]` where the call id defaults to `"call_unknown"`
  when the engine emits none (`Router.swift:161-163`); `reasoning_content`
  present iff `--emit-reasoning` and the run produced reasoning text.
- `finish_reason` mapping (`Engine.mapStopReason`, `Engine.swift:1000-1008`):
  upstream `stop`→`"stop"`, `length`→`"length"`, `cancelled`→`"stop"`; any
  completion with tool calls reports `"tool_calls"` even if the underlying stop
  reason was `length`. Pinned by `CacheRestoreTrackerTests` (4 cases).
- `usage` **always present** non-streaming: `prompt_tokens`, `completion_tokens`,
  `total_tokens` (= prompt + completion), `prompt_tokens_details.cached_tokens`,
  plus Mei extensions when > 0 (`Router.usage`, `Router.swift:143-156`). Pinned
  by `OpenAITypesTests.testUsageContractFieldTypesAndArithmetic` and
  `testNonStreamingResponsesAlwaysIncludeUsage`.
- Absent vs upstream sample: no `refusal`, `annotations`, `system_fingerprint`,
  `service_tier`, `completion_tokens_details` (Mei has no tokenizer-level
  breakdown), `audio_tokens`.

### Streaming — `ResponseWriter.streamSSE` (`HTTPServer.swift:51-73`), `Router.sseFrame` (`Router.swift:235-269`)

- Headers: `content-type: text/event-stream`, `cache-control: no-cache`,
  `connection: keep-alive`, `x-accel-buffering: no` (`Router.sseHeaders`,
  `Router.swift:271-278`). Status 200.
- Frames are `data: <json>\n\n`: `object: "chat.completion.chunk"` with
  `choices[0].delta` carrying `content`, `reasoning_content` (iff
  `--emit-reasoning`), or `tool_calls: [{index, id?, type?, function:{name?,
  arguments?}}]` fragments. No `role` is ever emitted in a delta (upstream
  sends `role: "assistant"` in the first chunk) — a known cosmetic deviation.
- Terminal sequence (`Router.finishSSEData`, `Router.swift:180-207`): a finish
  chunk `delta: {}` + `finish_reason`, then — iff `stream_options.include_usage`
  — a usage chunk with `choices: []`, then `data: [DONE]`.
- Streaming usage counts are byte-identical in value to the non-streaming
  response for the same run: pinned by
  `OpenAItypesTests.testStreamingFinishUsageCountParityWithNonStreaming` and
  `testStreamingUsageAbsentWhenIncludeUsageFalse`.
- A generation error after headers are sent is emitted as an SSE `data:` frame
  carrying the error envelope with `code: "stream_error"`; the HTTP status
  stays 200 (`HTTPServer.swift:66-70`).

### Other routes

- `GET /v1/models` (also trailing slash) → 200, `object: "list"`,
  `data: [{id: <served-id>, object: "model", created, owned_by: "mei"}]`
  (`Router.swift:53,68-73`). Acceptance: `MeiAcceptanceTests.testModelsIdentity`.
- `GET /healthz`, `GET /health` → 200 `{"status":"ok"}` (`Router.swift:55-56`).
- `GET /v1/mei/status` → Mei extension status surface (memory, cache counters);
  not part of the OpenAI surface.
- `POST /v1/completions` (legacy text completions) exists — **out of P0
  scope** for this contract.

## 5. Status codes and error envelope (shipped)

Envelope shape — `APIErrorEnvelope`, `OpenAITypes.swift:523-531`:

```json
{"error": {"message": "<string>", "type": "<string>", "code": "<string|null>"}}
```

`code` is **omitted** when nil (optional encoding); `type` defaults to
`"invalid_request_error"` everywhere except the serializer fallback
(`ResponseSerializer.errorPayload`, `Router.swift:24-26`). Mei's envelope has
**no `param` field** — upstream's documented envelope carries
`message`/`type`/`param`/`code`.

| Status | When | Envelope details | Source |
|---|---|---|---|
| 200 | Success (all JSON routes and streams) | n/a | `Router.swift:53-128` |
| 200 + SSE error frame | Streaming generation error after headers sent | `type: invalid_request_error`, `code: "stream_error"` | `HTTPServer.swift:66-70` |
| 400 | JSON decode failure (missing `model`/`messages`, malformed body) | `type: invalid_request_error`, no code | `Router.swift:113-115` |
| 400 | Prompt empty after tokenization | `type: invalid_request_error`, `code: "engine_error"` | `Router.swift:111-112`, `EngineError.emptyPrompt` (`Engine.swift:29,594`) |
| 400 | Prompt exceeds `--context-cap` | message `"request exceeded context cap: N prompt tokens > CAP allowed"`, `type: invalid_request_error`, `code: "engine_error"` | `Router.errorStatus` (`Router.swift:132-138`), `Engine.swift:903-904,951-952` |
| 404 | Unmatched route | `type: invalid_request_error`, `code: "not_found"` | `HTTPServer.swift:139-143` |
| 413 | Body > 64 MiB | `type: invalid_request_error`, `code: "payload_too_large"` | `HTTPRequestLimiter` (`HTTPServer.swift:156-176`), `HTTPServer.swift:124-131` |
| 500 | Engine failure (`modelDirectoryMissing`, `modelNotLoaded`, `generationFailed`) | `type: invalid_request_error`, `code: "engine_error"` | `Router.swift:111-112` |
| 500 | Channel-level handler error | `type: invalid_request_error`, `code: "internal_error"` | `HTTPServer.swift:147-153` |
| 500 (payload) | Serializer encoding failure (theoretically unreachable for encodable DTOs) | literal `{"error":{"message":"encoding failure","type":"internal_error"}}` | `Router.swift:18-19` |

Deviations vs upstream (deliberate, local-server scope): **no authentication**
— no `Authorization: Bearer` check, so no 401/403 paths exist; **no 429/503**
(no rate limiting or quota); **no 422** (input validation is minimal by
design, §3); errors after a stream starts keep HTTP 200 with an SSE error
frame; unknown JSON fields never error.

## 6. `max_completion_tokens` versus `max_tokens` policy

**Upstream facts (pin §1):** `max_tokens` is deprecated in favor of
`max_completion_tokens`; the latter bounds visible + reasoning tokens; upstream
does not document the both-supplied conflict (§1 open ambiguity).

**Shipped (Mei 0.5.0):**
- Only `max_tokens` is decoded (`OpenAITypes.swift:228,244,299`).
- `max_completion_tokens` has **no code path anywhere** in `Sources/` or
  `Tests/` (verified by whole-tree search, 2026-09-23). Because unknown keys are
  ignored, a request that sends it is accepted silently and its value has zero
  effect — it neither errors nor influences generation.
- Effective per-request cap (`Engine.swift:941-943,986-988`):
  `maxTokens = min(request.maxTokens ?? config.maxTokensDefault, max(1, maxKVSize − promptTokens))`
  with `maxKVSize = contextCap + 4096` (`ServerConfig.swift:114`). A request
  with no `max_tokens` uses the server default (`--max-tokens`, default 32768,
  or the profile's measured cap). A prompt longer than `--context-cap` is
  rejected with 400 before this formula applies.

**Frozen policy (P0):**
- `max_tokens` is the **sole supported token-budget field**. When both fields
  are present, `max_tokens` governs and `max_completion_tokens` is ignored
  (deterministic, matching today's shipped decode).
- When only `max_completion_tokens` is present, the shipped behavior is the
  fallback to the server default (field inert). This fallback is currently a
  side effect of ignoring the field, not a tested promise.
- **Planned acceptance:** the two statements above (both-present precedence;
  only-`max_completion_tokens` falls back to default) must be pinned by decoder
  and/or black-box tests in a later unit before this policy can be called
  verified behavior. Implementing `max_completion_tokens` support (taking
  precedence per the upstream deprecation direction) is deferred and must land
  with its own acceptance test, not silently.
- Rationale for the divergence: Mei's P0 client (Hermes) sends `max_tokens`;
  the upstream deprecation prefers `max_completion_tokens`, so a future switch
  is expected, but it is a behavioral change that must be tested and dated.

## 7. Deferred / unsupported request fields (shipped: no code path; silently ignored)

Absent from the decoder → inert per the §3 unknown-field rule, unless noted:

- `max_completion_tokens` — see §6.
- `response_format` (`json_object`/`json_schema` structured outputs).
- `logprobs`, `top_logprobs`, `logit_bias`.
- `n` (multiple choices) — response is always one choice; `n` is ignored.
- `parallel_tool_calls`.
- `modalities`, `audio` (audio output).
- `web_search_options`; non-`function` tool types (pass-through without
  coverage, §3).
- `prediction` (predicted outputs), `store`, `metadata`, `service_tier`,
  `user`, `safety_identifier`, `prompt_cache_key`, `prompt_cache_options`,
  `prompt_cache_retention`, `moderation`, `verbosity`.
- Deprecated `function_call`/`functions`.
- Image/audio `content` parts (§3).

Request-side validation that upstream performs but Mei does not (shipped):
`temperature` not clamped to 0–2, `top_p` not clamped to 0–1, no stop-sequence
cardinality limit, no role whitelist, no `model`-vs-served-id check. These are
deliberate (single-model, local-loopback server) but are deviations.

## 8. Shipped versus planned — evidence

Shipped = current source at the §1 Mei pin. Test-pinned facts (run with
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
--filter <suite>` — the model-free suites; the black-box
`MeiAcceptanceTests` additionally require a live server and are part of the
parent's separate acceptance run):

- `OpenAITypesTests` — request decoding (full payload, content arrays,
  `stream_options.include_usage`, tool messages, tool_choice object path),
  the exact CoCore forced-tool canary shape (nested `function.name` pinned as
  `tool_choice_name`; `testCoCoreForcedToolCanaryPinsReportStatus`),
  response encoding shape, usage field types/arithmetic, usage parity between
  streaming and non-streaming, usage absence when `include_usage` false.
- `CacheRestoreTrackerTests` — `mapStopReason` finish-reason mapping (4 cases).
- `RouterSSEToolCallIndexingTests`, `ToolArgumentNormalizerTests` — streaming
  tool-call indexing and arguments normalization.

Not yet pinned by tests (planned acceptance, do not claim as verified):
§6 policy statements, `max_completion_tokens` inertness, unknown-field
ignorance, error-envelope/status matrix as a whole (no unit test asserts the
400/404/413/500 envelopes end to end), non-function tool pass-through.

## 9. Open ambiguities

1. **Upstream:** both `max_tokens` + `max_completion_tokens` in one request is
   undocumented upstream (o-series incompatibility is the only stated
   constraint); Mei's §6 policy is a deliberate local codification.
2. **Mei (resolved):** forcing one specific tool via the official nested
   `tool_choice` shape historically degraded to `tool_choice:"required"`
   without the name; `MessageMapping.additionalContext` now reads both a
   top-level `name` and the nested `function.name` and pins it as
   `tool_choice_name` (§3). Covered by the CoCore forced-tool canary
   regression test.
3. **Mei:** `request.model` is not checked against the served id; a client can
   send any `model` value (permissive by design for a single-model server, but
   a divergence from upstream's unknown-model error).