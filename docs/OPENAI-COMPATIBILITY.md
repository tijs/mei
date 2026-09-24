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
  HEAD `1e2f7a7a2eaec2e6ab3dbd793708ebaea46cfad0` (2026-09-23).
- Base URL: `http://127.0.0.1:8024/v1` (default; `--host`/`--port` reconfigurable).

**Contract revisions (dated):**

| Date | Commit | Change |
|---|---|---|
| 2026-09-14 | `950e8c2` | Pre-hardening behavior pinned as originally shipped. |
| 2026-09-23 | `e795592` | Freeze of this contract document; official-reference pin retrieved (§1). |
| 2026-09-23 | `1e2f7a7` | Request boundary hardened: strict DTO validation (`APIValidation`), `max_completion_tokens` alias + conflict policy, loud rejection of deferred platform fields, sampling range checks, tools/tool_choice validation, streaming preflight, probe-side fixes for the two failing P0 probe cases. §§3–9 below reflect this revision. |

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
  `seed` (deprecated/removed upstream; Mei still honors it — see §3), and the
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
   `reasoning_effort` (`low`/`medium`/`high`/`none`) passed to the engine.
5. Usage — `prompt_tokens`, `completion_tokens`, `total_tokens`,
   `prompt_tokens_details.cached_tokens`, plus Mei engine extensions
   (`tokens_per_second`, `prompt_tokens_per_second`, `prefill_ms`,
   `generate_ms`, `mei_memory_active_bytes/cache/peak`) when non-zero.
6. Identity/health — `GET /v1/models` (exact served model id), `GET /healthz`
   and `GET /health`.
7. The minimal legacy `POST /v1/completions` (text completion) path used by
   admission, with the same usage block and the same validation rules
   (non-streaming only).
8. Error envelope and status codes as specified in §5.

Everything else from the official reference is **deferred** (§7).

## 3. Request surface (shipped)

Decoder: `ChatRequest(json:)` — `Sources/MeiCore/OpenAITypes.swift:246-427`;
validation rules — `Sources/MeiCore/APIValidation.swift` (shared by
`/v1/chat/completions` and `/v1/completions`); router dispatch —
`Sources/MeiCore/Router.swift:54-74`.

| Field | Type | Shipped behavior / notes |
|---|---|---|
| `model` | string, **required** | Missing or empty → 400 (`invalidField`, `OpenAITypes.swift:317-319`). **Not** compared against the served id (single-model server; §9.3). |
| `messages` | array, **required** | Missing or empty → 400 (`OpenAITypes.swift:320-322`). Roles validated (`APIValidation.validateMessageArray`, `APIValidation.swift:169-225`): `system`/`user`/`assistant`/`tool` accepted; `developer` → 400 *deferred*; any other role → 400. `content` required for `system`/`user`/`tool`; `assistant` needs content **or** `tool_calls`; `tool` requires `tool_call_id`; `tool_calls` are only allowed on `assistant`. |
| `messages[].content` | string or array | Array of `{"type":"text","text":...}` parts joined with `"\n"`; a non-text part type (`image_url`, `input_audio`, ...) → 400 *deferred* (multimodal input not implemented); a text part without `text` → 400 (`FlexibleString`, `OpenAITypes.swift:164-196`). |
| `messages[].tool_call_id` | string | Required and non-empty for `role:"tool"` (else 400). |
| `messages[].tool_calls` | array | `{id?, function:{name, arguments}}`; `name` required (missing → 400), `arguments` defaults to `"{}"`; an empty `arguments` string → 400 (`APIValidation.swift:188-197`). `role:"assistant"` only. |
| `messages[].reasoning_content` | string | Passed through to the template (`MessageMapping.templateDictionary`, `MessageMapping.swift:9-47`). |
| `temperature` | number | Range-validated 0...2 (400 outside; `APIValidation.swift:89-94`). Request wins over server default 0.6 (`GenerationControlSelection.resolve`, `GenerationControls.swift:51-69`). |
| `top_p` | number | Range-validated 0...1 (400 outside); default 0.95. |
| `top_k` | integer | Mei extension; must be >= 1; default 20; `FlexibleInt` also accepts numeric strings/floats (truncated). |
| `min_p` | number | Mei extension; range-validated 0...1; default 0.0. |
| `max_tokens` | integer | See §6. `FlexibleInt` (`OpenAITypes.swift:211-223`). |
| `max_completion_tokens` | integer | See §6 — alias for `max_tokens`, or a hard conflict when both differ. |
| `stream` | boolean | Default `false`. `true` switches to SSE streaming. |
| `stop` | string or array | Normalized to `[String]` (`FlexibleStop`, `OpenAITypes.swift:225-237`); passed as `extraStopStrings`; an empty array means no stop. |
| `tools` | array | Function tools are the supported P0 case; each entry must be an object with `type:"function"` and a non-empty `function.name` (400 otherwise); other tool types (`code_interpreter`, `file_search`) → 400 *deferred* (`APIValidation.validateTools`, `APIValidation.swift:230-255`). |
| `tool_choice` | string or object | `"auto"`/`"none"`/`"required"`, a bare function name (legacy forced-tool form), or the OpenAI object form `{"type":"function","function":{"name":...}}` (validated `APIValidation.validateToolChoice`, `APIValidation.swift:262-297`). Template mapping: keywords pass through; a bare name or an object with a **top-level** `name` maps to `tool_choice:"required"` + `tool_choice_name`; the **nested** `function.name` is not extracted (documented degradation, `MessageMapping.additionalContext`, `MessageMapping.swift:65-98` — pinned by `MessageMappingTests`). |
| `repetition_penalty` / `presence_penalty` / `frequency_penalty` | number | Range-validated (>= 0; -2...2; -2...2); passed to the generator. |
| `seed` | unsigned int | Honored (`parameters.randomSeed`). Deprecated upstream; Mei keeps it for reproducible benchmark rows (§7). |
| `reasoning_effort` | string | Whitelisted `low`/`medium`/`high`/`none`; anything else → 400 (`OpenAITypes.swift:352-358`). Drives the engine's thinking decision (`Engine.resolveEnableThinking`, `Engine.swift:295`). |
| `stream_options.include_usage` | boolean | Only `include_usage` is recognized; unknown option keys → 400; requires `stream:true` (400 otherwise); non-boolean → 400 (`OpenAITypes.swift:380-401`). |
| Deferred platform fields (`response_format`, `logprobs`, `top_logprobs`, `prediction`, `store`, `metadata`, `service_tier`, `modalities`, `audio`, `functions`, `function_call`) | any | **Rejected loudly with 400** (`APIValidation.deferredChatFields`, `APIValidation.swift:52-64`; checked at `OpenAITypes.swift:359-362`). See §7. |
| `n` | integer | Only `n:1` accepted; any other value → 400 *deferred* (`OpenAITypes.swift:363-368`). |
| `parallel_tool_calls` | boolean | Only `true` accepted; `false` → 400 *deferred* (Mei always allows multiple tool calls per turn; `OpenAITypes.swift:369-374`). |
| `user` and any other unknown key | any | **Inert** — accepted and ignored (JSONDecoder is non-strict; only the enumerated deferred fields and named special cases reject). Pinned by `OpenAIRequestValidationTests.testUserFieldIsIgnored` and `testUnknownTopLevelFieldsAreInert`. |

Missing `model` or `messages`, or unparseable JSON, throws during decode → 400
(§5). All validation errors carry `type: invalid_request_error` and a message
naming the offending field.

## 4. Response surface (shipped)

Serializer: `JSONEncoder` with `.sortedKeys` — deterministic key order
(`Router.swift:8-13`).

### Non-streaming `POST /v1/chat/completions` — `Router.completionResponse` (`Router.swift:179-194`)

- `object: "chat.completion"`, `id: "chatcmpl-" + 24 lowercase hex`,
  `created`: unix seconds, `model`: exact served model id.
- `choices: [{ index: 0, message, logprobs: null, finish_reason }]`.
- `message`: `role: "assistant"`; `content` present unless the run produced only
  tool calls (then `content` is null); `tool_calls: [{id, type: "function",
  function: {name, arguments}}]` where the call id defaults to `"call_unknown"`
  when the engine emits none (`Router.swift:182-184`); `reasoning_content`
  present iff `--emit-reasoning` and the run produced reasoning text.
- `finish_reason` mapping (`Engine.mapStopReason`, `Engine.swift:1035`):
  upstream `stop`→`"stop"`, `length`→`"length"`, `cancelled`→`"stop"`; any
  completion with tool calls reports `"tool_calls"` even if the underlying stop
  reason was `length`. Pinned by `CacheRestoreTrackerTests` and
  `GenerationControlSelectionTests` (5 cases).
- `usage` **always present** non-streaming: `prompt_tokens`, `completion_tokens`,
  `total_tokens` (= prompt + completion), `prompt_tokens_details.cached_tokens`,
  plus Mei extensions when > 0 (`Router.usage`, `Router.swift:164-177`). Pinned
  by `OpenAITypesTests.testUsageContractFieldTypesAndArithmetic` and
  `testNonStreamingResponsesAlwaysIncludeUsage`.
- Absent vs upstream sample: no `refusal`, `annotations`, `system_fingerprint`,
  `service_tier`, `completion_tokens_details` (Mei has no tokenizer-level
  breakdown), `audio_tokens`.

### Streaming — `ResponseWriter.streamSSE` (`HTTPServer.swift:51-73`), `Router.sseFrame` (`Router.swift:267-305`)

- Headers: `content-type: text/event-stream`, `cache-control: no-cache`,
  `connection: keep-alive`, `x-accel-buffering: no` (`Router.sseHeaders`,
  `Router.swift:307-314`). Status 200.
- Frames are `data: <json>\n\n`: `object: "chat.completion.chunk"` with
  `choices[0].delta` carrying `content`, `reasoning_content` (iff
  `--emit-reasoning`), or `tool_calls: [{index, id?, type?, function:{name?,
  arguments?}}]` fragments. Content deltas carry `role: "assistant"` (upstream
  sends it once on the first chunk; Mei repeats it on every content chunk
  because the router is stateless and streaming clients merge the role
  idempotently — documented repetition, pinned by `RouterSSEFrameTests`).
  Reasoning and tool-call deltas carry no role.
- Terminal sequence (`Router.finishSSEData`, `Router.swift:212-239`): a finish
  chunk `delta: {}` + `finish_reason`, then — iff `stream_options.include_usage`
  — a usage chunk with `choices: []`, then `data: [DONE]`. The `data: [DONE]`
  event is emitted on **every successful stream**, with or without
  `stream_options` (the terminator is the data payload of the final event,
  never a bare `[DONE]` line).
- Streaming usage counts are byte-identical in value to the non-streaming
  response for the same run: pinned by
  `OpenAITypesTests.testStreamingFinishUsageCountParityWithNonStreaming` and
  `testStreamingUsageAbsentWhenIncludeUsageFalse`.
- A generation error after headers are sent is emitted as a single SSE
  `data:` frame carrying the error envelope with `code: "stream_error"`; the
  HTTP status stays 200 (`Router.errorSSEFrame`, `Router.swift:323-325`).
  **Defined error-stream behavior:** that error frame is terminal — no finish
  frame, no usage chunk, and **no `data: [DONE]`** follow it. `[DONE]` is the
  success terminator only; its absence after an error frame is correct behavior
  (clients treat the error frame, or EOF without `[DONE]`, as the end of a
  failed stream), and a `[DONE]` after an error frame is itself a contract
  violation. Pinned by `RouterSSEFrameTests`
  (`testErrorStreamFrameShape`, `testErrorStreamTerminalFrameIsNotDone`,
  `testErrorFrameAndSuccessTerminationAreDistinct`) and by
  `probe_mei.py --self-test`.
- Streaming requests are preflighted **before** the 200 headers go on the wire:
  a request that can never generate (over context cap, template-rejected
  message) fails with a clean JSON 400, not an error frame (`Router.swift:112-118`,
  `Engine.preflightChat`, `Engine.swift:312`).

### Other routes

- `GET /v1/models` (also trailing slash) → 200, `object: "list"`,
  `data: [{id: <served-id>, object: "model", created, owned_by: "mei"}]`
  (`Router.swift:76-81`). Acceptance: `MeiAcceptanceTests.testModelsIdentity`.
- `GET /healthz`, `GET /health` → 200 `{"status":"ok"}` (`Router.swift:63-64`).
- `GET /v1/mei/status` → Mei extension status surface (memory, cache counters);
  not part of the OpenAI surface.
- `POST /v1/completions` (legacy text completions) — shipped shape:
  `object: "text_completion"`, `id: "cmpl-"+24 hex`, one choice with `text` and
  `finish_reason`, and the same usage block (`Router.legacyCompletionResponse`,
  `Router.swift:198-205`). Non-streaming only: `stream:true` → 400
  (`Router.swift:134-141`).

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
| 200 | Success (all JSON routes and streams) | n/a | `Router.swift:54-74` |
| 200 + SSE error frame | Streaming generation error after headers sent | `type: invalid_request_error`, `code: "stream_error"`; terminal frame — no `[DONE]` follows | `Router.errorSSEFrame` (`Router.swift:323-325`) |
| 400 | JSON decode failure (unparseable body) | `type: invalid_request_error`, no code | `Router.swift:127`, `APIRequestError.invalidBody` |
| 400 | Request validation (missing/empty `model`/`messages`, unknown or `developer` role, missing content/tool_call_id, out-of-range sampling, max-token conflict, `stream_options` misuse, malformed tools/tool_choice, deferred platform fields, `n>1`, `parallel_tool_calls=false`, `/v1/completions` with `stream:true`) | `type: invalid_request_error`, no code; message names the field | `Router.swift:126-128`, `APIRequestError` cases (`APIValidation.swift:7-29`) |
| 400 | Prompt empty after tokenization | `type: invalid_request_error`, `code: "engine_error"` | `Router.swift:124-125`, `EngineError.emptyPrompt` (`Engine.swift:19,29`) |
| 400 | Prompt exceeds `--context-cap` | message `"request exceeded context cap: N prompt tokens > CAP allowed"`, `type: invalid_request_error`, `code: "engine_error"` | `Router.errorStatus` (`Router.swift:153-159`), `Engine.swift:312-320, 937, 985` |
| 404 | Unmatched route | `type: invalid_request_error`, `code: "not_found"` | `HTTPServer.swift:146-150` |
| 413 | Body > 64 MiB | `type: invalid_request_error`, `code: "payload_too_large"` | `HTTPRequestLimiter` (`HTTPServer.swift:165-183`), handler wiring `HTTPServer.swift:124-131` |
| 500 | Engine failure (`modelDirectoryMissing`, `modelNotLoaded`, `generationFailed`) | `type: invalid_request_error`, `code: "engine_error"` | `Router.swift:124-125` |
| 500 | Engine not loaded (router constructed without an engine) | `type: engine_error`, `code: "engine_error"` | `Router.swift:55-59` |
| 500 | Channel-level handler error | `type: invalid_request_error`, `code: "internal_error"` | `HTTPServer.swift:154-160` |
| 500 (payload) | Serializer encoding failure (theoretically unreachable for encodable DTOs) | literal `{"error":{"message":"encoding failure","type":"internal_error"}}` | `Router.swift:18-19` |

The 413 boundary and message text are pinned by
`OpenAIResponseShapeTests.testPayloadLimiter*`; the envelope shape and the
engine-status mapping by `OpenAIResponseShapeTests`; the full status matrix
end to end through a live socket is exercised by `tools/probe_mei.py` (live)
and `MeiAcceptanceTests` (needs a server).

Deviations vs upstream (deliberate, local-server scope): **no authentication**
— no `Authorization:` check, so no 401/403 paths exist; **no 429/503** (no rate
limiting or quota); **no 422** (input validation returns 400 by design, §3);
errors after a stream starts keep HTTP 200 with an SSE error frame; unknown
JSON fields are inert except the enumerated deferred fields (which reject).

## 6. `max_completion_tokens` versus `max_tokens` policy

**Upstream facts (pin §1):** `max_tokens` is deprecated in favor of
`max_completion_tokens`; the latter bounds visible + reasoning tokens; upstream
does not document the both-supplied conflict (§1 open ambiguity). The plan's
decision (note 01748433: "prefer it as an alias, reject conflicting
simultaneous values") is the codified local policy.

**Shipped (Mei 0.5.0, since 2026-09-23):** `APIValidation.resolveMaxTokens`
(`APIValidation.swift:140-156`):

- Either form alone is honored: `max_completion_tokens: 128` behaves exactly
  like `max_tokens: 128` (alias).
- Both forms present with **equal** values: accepted (identical semantics).
- Both forms present with **different** values: hard 400 `conflict` — the
  request is rejected even though either one alone would have been honored;
  silently keeping one would hide the incompatibility from the client.
- Neither form: server default applies (`--max-tokens`, default 32768, or the
  profile's measured cap).
- A value < 1 is a 400 (`invalidField`).

Effective per-request cap (`GenerationControlSelection.resolveMaxTokens`,
`GenerationControls.swift:97-102`):
`maxTokens = min(requested, max(1, maxKVSize − promptTokens))`
with `maxKVSize = contextCap + 4096` for the generation headroom window
(`ServerConfig.maxKVSize`). A prompt longer than `--context-cap` is rejected
with 400 before this formula applies.

**Pinned by:** `OpenAIRequestValidationTests` (`testMaxCompletionTokensAliasAlone`,
`testMaxTokensAloneStillWorks`, `testEqualMaxTokensFormsAccepted`,
`testConflictingMaxTokensFormsRejected`, `testZeroOrNegativeMaxTokensRejected`,
`testCompletionMaxCompletionTokensAlias`), `GenerationControlSelectionTests`
(max-token arithmetic), and live by `probe_mei.py` (`max_tokens_alias`,
`max_tokens_conflict`).

Rationale for the divergence: Mei's P0 client (Hermes) sends `max_tokens`; the
upstream deprecation prefers `max_completion_tokens`, so
`max_completion_tokens` is already a supported alias today and a future
client-side switch needs no server change.

## 7. Deferred / unsupported request fields (shipped: loud 400)

The following recognized OpenAI-platform features are **rejected loudly with
400** (`APIRequestError.deferred`), so a field can never look supported while
its semantics are dropped:

- `response_format` (structured outputs), `logprobs`, `top_logprobs`.
- `n` (any value other than 1; multiple choices).
- `parallel_tool_calls: false` (Mei always allows multiple tool calls per turn).
- `prediction`, `store`, `metadata`, `service_tier`, `modalities`, `audio`.
- Deprecated `function_call` / `functions` (use `tools` / `tool_choice`).
- `developer` message role (use `system`).
- Image/audio `content` parts (multimodal input), non-`function` tool types.
- `/v1/completions`: `echo`, `suffix`, `best_of`, `logprobs`, `top_logprobs`,
  `stream_options`, and `stream:true`.

Unknown keys **not** in any of these lists are inert (accepted and ignored),
matching OpenAI's behavior for client-side fields — including `user`,
`web_search_options` (no code path, inert), `safety_identifier`,
`prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`,
`moderation`, `verbosity`, and any future/private field.

Deviations vs upstream kept intentionally (single-model, local-loopback
server): `model` is not checked against the served id (§9.3); unknown JSON
fields never error (except the deferred list); reasoning/thinking is
template-driven.

## 8. Shipped versus planned — evidence

Shipped = current source at the §1 Mei pin. The model-free deterministic
suites run with
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --skip MeiAcceptanceTests`
(177 tests as of HEAD `1e2f7a7`; 183 with `MessageMappingTests` at this
revision):

- `OpenAIRequestValidationTests` (41) — request decoding and the whole §3
  validation matrix: required fields, roles, content parts, max-token policy,
  sampling ranges, stop/seed, `stream_options`, tools/tool_choice, deferred
  fields, legacy completions, unknown-field inertness.
- `OpenAITypesTests` (12) — decoding, response encoding, usage field
  types/arithmetic, streaming/non-streaming usage parity, usage absence when
  `include_usage` false.
- `OpenAIResponseShapeTests` (10, 12 with the payload-limiter tests) — error
  envelope, `errorStatus` mapping, `/v1/models` shape, usage arithmetic on the
  wire, chat/tool-call/legacy response shapes, reasoning emit flag, 413
  payload limiter.
- `RouterSSEFrameTests` (20) — content/reasoning deltas, role on content
  frames, finish frames, `[DONE]`, usage inclusion/omission, tool-call
  fragmentation, multiple tool indexes, interleaved frames, malformed and
  incomplete frames, error-stream contract.
- `RouterSSEToolCallIndexingTests` (2) — distinct streaming indexes for
  multiple tool calls.
- `ToolArgumentNormalizerTests` (16) — tool-argument schema coercion.
- `GenerationControlSelectionTests` (16) — request-over-server precedence,
  no-leak across requests, max-token arithmetic, reasoning-mode override
  precedence, stop-reason mapping.
- `CacheRestoreTrackerTests` (5) — cache-restore tracking + stop-reason mapping.
- `MessageMappingTests` (6) — template mapping of roles, tool_calls, and the
  `tool_choice` nested-name degradation (§3, §9.2).
- `ServerConfigParsingTests` (35), `SSMAnchorBoundariesTests` (9),
  `QuantizedRotatingKVCacheTests` (6) — runtime/config surfaces outside the API
  contract.

The live half is `tools/probe_mei.py`, the version-pinned P0 matrix gate:
`python3 tools/probe_mei.py --base-url http://127.0.0.1:8024/v1 --model <served-id> --tokenizer <model-dir> --context-cap 65536 --output artifacts/p0-matrix-<ts>.json`
(28 probes: identity, status, plain text, non-streaming/streaming tool calls,
tool-history replay, streaming/non-streaming parity, max-token alias/conflict,
deferred-field rejections, deterministic sampling override, reasoning toggles,
multi-index tools, tool_choice none, multi-turn tool loop, stream usage
omission, legacy completions shape + stream rejection, cache reuse incl.
growing transcript, exact/over context-cap chats and streams). `--self-test`
validates the probe's own assertion logic without a server
(`python3 tools/probe_mei.py --self-test`). The black-box Swift oracle
`MeiAcceptanceTests` additionally requires a live server
(`MEI_ACCEPTANCE_BASE_URL`, default `http://127.0.0.1:8024/v1`) and is part of
the separate acceptance run.

**Matrix evidence:** `artifacts/p0-matrix-qwen36-text-20260922T201844Z.json`
records a live run against `Qwen3.6-35B-A3B-4bit-textonly` at context cap
65536. Server behavior was correct on every probe — the two failures were
probe-side assertion bugs (the deterministic-override check accepted only the
`content` channel although a thinking model may answer entirely in
`reasoning_content`; the stream-terminator check compared raw lines instead of
`data:` payloads), and four rejection probes recorded their HTTP-status
evidence under the outcome key (record bug). All six were corrected in
`1e2f7a7` and are now covered by `probe_mei.py --self-test`. **A clean live
re-run on the current build is required to record a passing full-matrix
artifact and close plan unit 6** — it needs a live server + staged model and is
the one gate this document cannot close from the tree alone.

**Not yet pinned by tests (planned acceptance, do not claim as verified):**
the full status matrix through a live socket (§5 — 404/413/500 paths are
pinned at the component level, not end to end without a server); model/template
behavior under `reasoning_effort` toggles at generation time (the probe
asserts the request is accepted and the field reaches the engine; the
qualitative thinking change is model behavior).

## 9. Open ambiguities

1. **Upstream:** both `max_tokens` + `max_completion_tokens` in one request is
   undocumented upstream (o-series incompatibility is the only stated
   constraint); Mei's §6 policy (alias, or hard conflict when they differ) is
   the deliberate local codification, implemented and pinned.
2. **Mei:** forcing one specific tool via the official nested `tool_choice`
   shape `{"type":"function","function":{"name":X}}` does **not** extract the
   nested name — `MessageMapping.additionalContext` reads only a top-level
   `name`, so the nested form degrades to `tool_choice:"required"` (any tool).
   This is now pinned as shipped behavior (`MessageMappingTests`); whether the
   template could express "this specific tool" is a template-level question,
   not an API-shape one.
3. **Mei:** `request.model` is not checked against the served id; a client can
   send any `model` value (permissive by design for a single-model server, but
   a divergence from upstream's unknown-model error).