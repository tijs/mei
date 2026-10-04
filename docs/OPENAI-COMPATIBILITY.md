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

- Mei release pin: `ServerConfig.version = "0.7.0"` (`Sources/MeiCore/ServerConfig.swift:7`),
  planned release tag `v0.7.0` (2026-10-02).
- Base URL: `http://127.0.0.1:8024/v1` (default; `--host`/`--port` reconfigurable).
- Mei 0.7.0 engine pin: `tijs/vmlx-swift`
  `633fe166630ef04310aea7d5a1795555ab32970d`, pushed to the public fork. The
  released 0.6.1 binary used `fef563a5`; structured output requires the newer
  seam.

> **Mei 0.7.0 structured-output status.** `response_format` is shipped and
> enforced by token-level constrained decoding. The exact CoCore canary and
> the full 9-test live acceptance suite passed with `mlx-community/Qwen3-4B-4bit`.
> Structured-output compatibility remains checkpoint-specific: Qwen3.6
> (text-only and vision) and Ornith fail closed; they must not be advertised for
> schema jobs until their canaries pass. The shipped 0.7.0 release uses the
> `633fe166` vmlx seam; 0.6.1 used `fef563a5`. Full CoCore advisor
> Register-frame capability readback is not claimed here. Everything else
> below records the broader Chat Completions contract and deferred fields.

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
8. **Structured outputs (shipped in Mei 0.7.0)** — `response_format` on
   `/v1/chat/completions`: `text` (default), `json_object`, and strict
   `json_schema`, enforced by token-level constrained decoding, not prompt
   instructions. The supported schema subset is recursive (nested strict
   objects, arrays with `items` and `minItems`/`maxItems`, nullable scalar
   unions, enums on every scalar type, and numeric constraints —
   `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`/`multipleOf` —
   with exact decimal semantics; see §3/§4). Model-free tests cover
   decode/compile/mask/response paths; the exact CoCore canary passed in live
   buffered and SSE runs on `mlx-community/Qwen3-4B-4bit` — the only
   checkpoint that has passed so far; no other tested checkpoint has passed,
   and the shipped Qwen3.6 and Ornith profiles fail closed (§4, §8).

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
| `response_format` **(shipped in Mei 0.7.0)** | object | Chat-completions only. Decoded by `ResponseFormat.decode` (`Sources/MeiCore/ResponseFormat.swift:134-184`): `{"type":"text"}` (also absent/`null`) keeps the ordinary path byte-compatible; `{"type":"json_object"}` guarantees a syntactically valid JSON value; `{"type":"json_schema","json_schema":{name,strict,schema}}` accepts only `strict: true` with the recursive subset — root `type: "object"`; every object node declares `properties`, `required`, and `additionalProperties: false`; a property value may be a scalar (`string`/`number`/`integer`/`boolean`), a nullable union (`type: [scalar, "null"]`), a nested strict object, or an array (`items` required, same recursive value space, optional `minItems`/`maxItems`); `enum` is supported on every scalar type — string, number/integer (compared by exact decimal value), and boolean — and on nullable fields may include `null` (which then also decides whether null is accepted); `number`/`integer` scalars additionally accept `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, and `multipleOf` (finite JSON numbers, `multipleOf` strictly positive) with exact decimal semantics (schema numbers use their shortest round-trip decimal form, so `multipleOf: 0.1` accepts `0.3`); a schema whose declared constraints are unsatisfiable for the declared type (and cannot be null) is rejected. Everything else — constraints (`minLength`, `maxLength`, `pattern`, `format`, `uniqueItems`, `minProperties`, …), `$ref`/`oneOf`/`anyOf`/`allOf`, unions other than exactly one scalar plus `"null"`, non-strict forms, malformed envelopes — → 400 `param: "response_format"` **before generation**. Structured + non-empty `tools` is rejected (400); thinking is forced off. Not decoded on `/v1/completions` (still inert there). |
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
  stays 200 (`HTTPServer.swift:65-70`, `Router.streamErrorSSEData`,
  `Router.swift:318-320`).

### Structured outputs (`response_format`) — shipped in Mei 0.7.0

- Requests without `response_format` (or with `{"type":"text"}`) take the
  ordinary path byte-for-byte: no constraint processor is built and no
  tokenizer vocabulary work happens (`StructuredGeneration.plan` returns nil).
- `json_object` root semantics (pinned against the reference, re-retrieved
  2026-10-01): the grammar accepts any complete JSON value — object, array,
  string, number, boolean, `null` — a superset of the reference's stated
  guarantee ("JSON mode ensures that model output is valid JSON", "only that
  it is valid and parses without errors"). The reference's "must instruct the
  model to produce JSON / the API will throw an error if the string `JSON`
  does not appear in the context" safeguard is deliberately **not**
  replicated: CoCore's exact canary prompt contains no such instruction, and
  constrained decoding structurally prevents non-JSON output (an unterminated
  run fails closed as `engine_error`/`stream_error`, rather than being
  returned as a successful length-truncated response).
- Structured requests are compiled **before generation** (HTTP 400 on any
  unsupported construct, including for streaming requests — the SSE response
  has not started), and enforced token-by-token by `JSONGrammarLogitProcessor`
  riding the ordinary single-sequence, non-speculative decode path through the
  vmlx `additionalProcessor:` seam, composed **after** the built-in penalty
  processors. The token mask is a strict subset of the byte grammar: while the
  root value is incomplete it bounds whitespace runs to one whitespace-only
  token, so a checkpoint that prefers whitespace over structural bytes cannot
  spend the whole budget without making progress; after the root value
  completes, trailing whitespace is unbounded. Nothing the mask admits is
  rewritten, and the byte grammar's accepted language is unchanged. The
  response DTOs are unchanged (same `completionResponse` /
  SSE chunk shape as any other completion).
- Guarantee: content returned with `finish_reason: "stop"` is a complete JSON
  value — for `json_schema`, exactly the compiled value shape (no prose, no
  extra keys at any object level, every required key present at every level,
  nested object/array shapes enforced, enum values enforced on every scalar
  type, numeric constraints enforced exactly — `minimum`/`maximum`/
  `exclusiveMinimum`/`exclusiveMaximum`/`multipleOf` on `number`/`integer`,
  `minItems`/`maxItems` on arrays — nullable unions accepting only their
  scalar or null). The constraint is recursive: the grammar admits nested
  objects and arrays only along the compiled schema, so a nested key or item
  type the schema does not declare is masked out. EOS is masked until the
  root value is complete, so a normal stop cannot end an incomplete document.
- Numeric semantics are exact decimal: schema numbers enter through their
  shortest round-trip decimal form (`0.1` stays `0.1`, not the binary
  expansion), generated literals through their exact digits, so
  `multipleOf: 0.1` accepts `0.3`, `0.3 / 0.1 = 3`, and `1e-1` is exactly
  `0.1`. Prefix masking is exact: a number continuation (digit, fraction,
  exponent) is masked only when no completion of the prefix can satisfy the
  declared constraints, and the completed value is re-checked exactly at the
  number's terminator. A schema whose declared constraints no value of the
  declared type can satisfy is rejected with a 400 before generation.
- Failure semantics: a constraint failure (illegal token, all-illegal state,
  vocabulary mismatch) or a stop that contradicts the constraint (no complete
  root value while the response would report `stop`) fails the request — HTTP
  500 `engine_error` non-streaming; for streaming, the SSE error frame
  (`code: "stream_error"`) after the HTTP 200 head, never a success finish
  frame (`StructuredGeneration.postGenerationError`,
  `StructuredGeneration.swift:151-168`).
- An incomplete structured root fails closed regardless of the producer stop
  reason, including `length`: buffered requests return HTTP 500 with
  `engine_error`; streaming requests return an in-band `stream_error`, with no
  success finish frame or `[DONE]` after the error. Clients should retry with a
  larger budget when appropriate. The implementation never rewrites an
  incomplete root to a successful `stop` or `length` response.
- Thinking is forced off for structured requests: the request's
  `reasoning_effort` and the operator's server-side default cannot re-enable a
  reasoning preamble the grammar cannot start from
  (`StructuredGeneration.enableThinking`).
- Live evidence for Mei 0.7.0: `mlx-community/Qwen3-4B-4bit` at HF revision
  `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` passed the exact CoCore
  structured-output canary through both buffered and SSE Engine paths; the
  response content parsed to exactly the JSON object `{"status":"ok"}`;
  raw JSON whitespace is immaterial. The response had `finish_reason:
  "stop"` and six completion tokens in both runs. The live server for the
  original canary runs used `--enable-thinking false`, `--compiled-decode
  false`, and `--cache-reuse false` — the ordinary single-sequence,
  non-compiled decode path (thinking off; prefix cache disabled). This
  proves one real
  tokenizer/vocabulary/chat-template path, not all model families — no other
  checkpoint has passed: the Qwen3.6 text-only, Qwen3.6 vision, and Ornith
  profiles each fail the canary closed in the Mei 0.7.0 feature build
  (§8). The expanded numeric/array schema features (`minimum`/`maximum`,
  `multipleOf`, scalar enums, `minItems`/`maxItems`) also passed live buffered
  + SSE on this checkpoint.
- CoCore evidence: the merged attached-engine implementation at commit
  `0151475bf8c98de10a64cab51c23a46dd84a8fe1` reported readiness, tool canary
  pass, structured-output canary pass, and successful buffered/streaming proxy
  responses against that live Mei server. The full advisor connection and
  Register-frame capability readback were not run.
- Decode-path gate: structured requests ride the ordinary single-sequence,
  non-speculative path. An operator's `--compiled-decode true` does **not**
  apply to them (`StructuredGeneration.enableCompiledDecode`; text requests
  keep the configured value), Mei constructs no `DraftStrategy` anywhere, so
  speculative/MTP paths are unreachable for every request, and decode is
  batch size 1. Re-enabling any of these for structured requests requires its
  own correctness evidence first.

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
(`ResponseSerializer.errorPayload`, `Router.swift:24-26`). Mei 0.7.0 adds the
OpenAI-style `param` (also omitted when nil) for errors that name a
request field — currently only `response_format` errors — so upstream's
documented `message`/`type`/`param`/`code` envelope is now complete for those;
all pre-existing errors keep their previous bytes (no `param`).

| Status | When | Envelope details | Source |
|---|---|---|---|
| 200 | Success (all JSON routes and streams) | n/a | `Router.swift:53-128` |
| 200 + SSE error frame | Streaming generation error after headers sent | `type: invalid_request_error`, `code: "stream_error"` | `HTTPServer.swift:65-70` |
| 400 | JSON decode failure (missing `model`/`messages`, malformed body) | `type: invalid_request_error`, no code | `Router.swift:113-115` |
| 400 | `response_format` decode/validation failure (not an object, missing/wrong `type`, missing `json_schema`/`name`/`schema`, non-strict, structured + non-empty `tools`) | `code: "invalid_response_format"` (structured + tools: `"response_format_unsupported"`), `param: "response_format"` | `Router.errorResult` (`Router.swift:136-173`), `ResponseFormat.swift` |
| 400 | Schema outside the supported strict subset (unsupported keywords at any level, unions other than exactly one scalar plus `"null"`, arrays without a supported `items`, malformed shapes, duplicate/invalid names) | message prefixed `response_format.json_schema.schema: `; nested locations are named in the message (`at 'meta.id': …`), `code: "response_format_unsupported"`, `param: "response_format"` | `Router.errorResult`, `JSONSchemaCompiler.swift` |
| 400 | Prompt empty after tokenization | `type: invalid_request_error`, `code: "engine_error"` | `Router.swift:111-112`, `EngineError.emptyPrompt` (`Engine.swift:29,594`) |
| 400 | Prompt exceeds `--context-cap` | message `"request exceeded context cap: N prompt tokens > CAP allowed"`, `type: invalid_request_error`, `code: "engine_error"` | `Router.errorStatus` (`Router.swift:132-138`), `Engine.swift:903-904,951-952` |
| 404 | Unmatched route | `type: invalid_request_error`, `code: "not_found"` | `HTTPServer.swift:139-143` |
| 413 | Body > 64 MiB | `type: invalid_request_error`, `code: "payload_too_large"` | `HTTPRequestLimiter` (`HTTPServer.swift:156-176`), `HTTPServer.swift:124-131` |
| 500 | Engine failure (`modelDirectoryMissing`, `modelNotLoaded`, `generationFailed`) | `type: invalid_request_error`, `code: "engine_error"` | `Router.swift:111-112` |
| 500 | Structured-generation construction failure (unreadable `config.json`, no identifiable EOS, mismatched vocabulary) | `code: "engine_error"` | `Router.errorResult`, `StructuredGeneration.swift`, `TokenizerFragmentTable.swift` |
| 500 | Structured constraint failure, or a stop that contradicts the constraint (incomplete root value while the response would report `stop`) | message `structured output constraint failed: …`, `code: "engine_error"` | `StructuredGeneration.postGenerationError` (`StructuredGeneration.swift:151-168`), `Engine.swift:495,612,645` |
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

**Shipped (Mei 0.7.0):**
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
- `response_format` (`json_object`/`json_schema` structured outputs) — shipped
  on `/v1/chat/completions` in Mei 0.7.0: the field is decoded, compiled
  before generation, and enforced token-by-token by
  constrained decoding (the recursive subset plus enums on every scalar type,
  numeric constraints with exact decimal semantics, and array
  `minItems`/`maxItems`; structured + non-empty `tools` is rejected; thinking
  is forced off; see §3/§4). Model-free tests cover the contract, and one live
  checkpoint — Qwen3-4B — plus the CoCore attached-engine client passed the
  smoke canary; the shipped Qwen3.6 (text-only and vision) and Ornith profiles
  fail it closed and are not advertised for schema jobs (§8). Still absent
  from the `/v1/completions` DTO, so it remains inert there.
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

Structured-output suites (shipped in Mei 0.7.0; model-free core plus live
black-box probes):

- `ResponseFormatTests` — `response_format` decode/validation contract,
  including the exact CoCore structured-output canary request body; error
  status/code/param mapping.
- `JSONSchemaCompilerTests` — strict-subset compilation (recursive: nested
  objects, arrays with `items`, nullable scalar unions, nullable string
  enums), canonical constraint keys (flat-subset keys frozen), and the
  rejection matrix for unsupported constructs with nested paths.
- `SchemaMatrixTests` — the expanded matrix: numeric constraints
  (`minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`/`multipleOf`) on
  `number`/`integer`, enums on every scalar type, and array
  `minItems`/`maxItems`; exact-decimal validation; canonical keys for the new
  keywords; the invalid-value / unsatisfiable / enum-type-mismatch rejection
  matrix; and a brute-force consistency sweep that checks the grammar's
  accept/reject decision against an independent test-local oracle across
  thousands of integer and number literals (including exponent spellings).
- `JSONGrammarStateTests` — the byte-level automaton (JSON syntax, escapes,
  strict UTF-8, numbers, nesting, schema keys/enums/required/
  additionalProperties, nested objects/arrays, nullable unions, numeric
  bounds/`multipleOf`, scalar enums, array counts).
- `JSONGrammarProcessorTests`, `JSONGrammarLogitProcessorTests` — token-mask
  and lifecycle contract (fail-closed masking, EOS rules, reset, copies,
  completion recorded through the shared run record).
- `TokenizerFragmentTableTests` — the production tokenizer adapter
  (fragments, EOS union, fail-closed vocabulary handling).
- `StructuredGenerationTests` — Engine-seam construction, thinking-off policy,
  the compiled-decode gate, HTTP error mapping, pre-generation validation.
- `StructuredGenerationPipelineTests` — model-free end-to-end: the exact
  CoCore canary through the buffered and SSE response paths with a scripted
  constrained decoder, grammar-failure / incomplete-at-stop / fail-closed-length
  semantics, a scalar-type schema matrix, the recursive slice (nested
  object/array/nullable document through both response paths, nested mask
  boundaries), the schema-matrix slice (numeric bounds/`multipleOf`, scalar
  enums, and array counts through both response paths plus their mask
  boundaries), and the 400 mapping for remaining unsupported constructs.
- `CoCoreCanaryFixture` — the exact CoCore canary request/response oracle
  (mirror of the Rust source; used by the live probes below).
- `MeiAcceptanceTests` — live HTTP/SSE evidence including the exact CoCore
  structured canary, plain/tool pass-through, and buffered/streaming
  fail-closed truncation transport.

### Structured-output live model matrix

Live evidence is **per checkpoint**: a pass on one checkpoint is not evidence
for another model family, and a tokenizer/template preflight is not evidence at
all — only a completed canary run is. The released Mei 0.7.0 binary passed the
full `MeiAcceptanceTests` suite 9/9 on `mlx-community/Qwen3-4B-4bit` on
2026-10-02. Unreleased Mei candidate `ea5a67a` passed the same 9-test live
suite on Qwen3.6 text-only, Qwen3.6 vision, and aligned Ornith on 2026-10-04.
All candidate runs used the exact CoCore canary in both buffered and SSE form,
plus ordinary text/tool and fail-closed truncation checks. These results are
candidate-source evidence, not evidence for the shipped 0.7.0 binary or for a
current CoCore capability advertisement. The vision checkpoint was exercised
with text-only requests; image-conditioned structured output is unverified.
A checkpoint that fails its canary must remain unadvertised.

| Checkpoint (HF revision) | Result | Live evidence |
|---|---|---|
| `mlx-community/Qwen3-4B-4bit` (`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`) | **PASS** | Mei 0.7.0 release binary passed the full `MeiAcceptanceTests` suite **9/9** on 2026-10-02, including buffered and streaming structured canaries and both fail-closed truncation transports. Expanded schema features also passed live buffered + SSE: integer `count = 3` under bounds 3..3, number `ratio = 0.3` with exact `multipleOf: 0.1`, and `labels = ["alpha","beta"]` with enum items and item count 2; unsupported `pattern` → HTTP 400 `response_format_unsupported` before generation. |
| Qwen3.6-35B-A3B **text-only** (`Tostibrown/Qwen3.6-35B-A3B-4bit-textonly`, `693d7a0f4d0c1feb97d8e885ceb2c67d3eb98a56`; profile `qwen3.6-35b-a3b-text`) | **PASS on candidate** | Mei candidate `ea5a67a` passed live `MeiAcceptanceTests` **9/9** on 2026-10-04. Buffered and SSE exact CoCore canaries passed; ordinary text, both tool paths, and both fail-closed truncation transports also passed. CoCore `AttachedEngine` separately read back `ready=true structured_output=true tool_calls=false`; its buffered/SSE schema canary passed. Provider Register/PDS readback was not run. Structured path used ordinary single-sequence, non-compiled decode. |
| Qwen3.6-35B-A3B **vision** (`mlx-community/Qwen3.6-35B-A3B-4bit`, `38740b847e4cb78f352aba30aa41c76e08e6eb46`; profile `qwen3.6-35b-a3b`) | **PASS on candidate (text input only)** | Mei candidate `ea5a67a` passed live `MeiAcceptanceTests` **9/9** on 2026-10-04: buffered + SSE exact canaries, ordinary text, both tool paths, and both fail-closed truncation transports. CoCore `AttachedEngine` separately read back `ready=true structured_output=true tool_calls=false`; its buffered/SSE schema canary passed. Provider Register/PDS readback was not run. Request contained text only; image-conditioned structured output was not tested. |
| **Ornith 1.5 35B-A3B aligned** (`Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned`, `ddce5cd6e3d8bc720a5bac5a68c22f406f90403d`; profile `ornith-1.5-35b-a3b`) | **PASS on candidate** | Mei candidate `ea5a67a` passed live `MeiAcceptanceTests` **9/9** on 2026-10-04: buffered + SSE exact canaries, ordinary text, both tool paths, and both fail-closed truncation transports. CoCore `AttachedEngine` separately read back `ready=true structured_output=true tool_calls=true`; its buffered/SSE schema canary passed. Provider Register/PDS readback was not run. Structured path used ordinary single-sequence, non-compiled decode. |
| `mlx-community/Qwen3-8B-4bit` (`545dc4251c05440727734bcd94334791f6ab0192`) | **NOT RETESTED after fix** | Its earlier pre-fix live canary stalled on whitespace and failed closed. Candidate `ea5a67a` adds an anti-stall whitespace mask, but this checkpoint has not been re-run on the candidate; do not promote it yet. |
| `mlx-community/Qwen2.5-3B-Instruct-4bit` (`4f83f8f146fdf28b512a06562b671d7af4fab457`) | fail (not promoted) | Downloaded and exercised as a second family checkpoint; its live acceptance run did not pass the full structured canary class. Not promoted as structured-output evidence. |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` (`7f0dc925e0d0afb0322d96f9255cfddf2ba5636e`) | fail closed | Plain completion and fail-closed truncation transport passed live; the structured canary failed closed when the tokenizer path reached a state with no legal advancing token; tool-call canaries also did not pass. No structured-success claim. |
| `mlx-community/gemma-4-12B-it-4bit` (`73bcf09092aa277861d5a191b989b666f7f32e8f`) | fail closed | Attempted once with the exact strict CoCore canary (thinking disabled, `max_tokens` 64, buffered only): generation stopped before a complete JSON value → HTTP 500 `engine_error`; no streaming run was made. No structured-success claim; cache removed after the test. |

Checkpoint facts for the released Qwen3-4B row: `tokenizer_class:
Qwen2Tokenizer`; its chat template branches on `enable_thinking` and emits
`<think>`/`</think>` delimiters (added tokens 151667/151668); its `eos_token_id`
is 151645. The 2026-10-02 release canary used thinking=false, compiled-decode
false, and cache-reuse=false.

Candidate-matrix tokenizer facts (metadata preflight, followed by successful
runtime canaries): Qwen3.6 text-only reports `TokenizersBackend`, vocab size
248044 in tokenizer metadata, EOS 248046; its vision sibling has the same
`tokenizer.json` hash, vocabulary, special tokens, and chat template. Ornith
reports `Qwen2Tokenizer`, the same token-to-ID mapping and EOS 248046, but a
different tokenizer hash and chat template. On these three roots the exact
canary tokenizes as `[4754, 2738, 3147, 547, 8934]` (`{"`, `status`, `":"`,
`ok`, `"}`). With `enable_thinking=false`, the templates render an empty think
block before assistant generation. The candidate live runs used each model's
named profile, `--compiled-decode false`, and the ordinary single-sequence
structured path; no batch or speculative structured path was enabled. The
vision canary had text input only.

Not yet pinned by tests (do not claim as verified): §6 policy statements,
`max_completion_tokens` inertness, unknown-field ignorance, the complete
400/404/413/500 error-envelope matrix, and non-function tool pass-through.
The candidate Mei build passed structured canaries on Qwen3.6 text-only,
Qwen3.6 vision (text requests only), and aligned Ornith, as well as the
previously verified released Qwen3-4B checkpoint. Qwen3-8B has not been
retested after the anti-stall fix. The local CoCore LaunchAgent is not loaded
and its doctor reports the advisor offline; the configured CoCore model list
is still Qwen3-4B, so a current attached-agent capability readback for the
three new checkpoints remains unverified. The live Mei acceptance runs are
not CoCore advertisement evidence.

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