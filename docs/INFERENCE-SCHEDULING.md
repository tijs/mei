# Inference scheduling and admission

How Mei admits requests to its single execution slot: the bounded queue, the
priority classes, the overload refusals, the cancellation contract, and the
counters that make all of it observable. This is the Mei side of the
inference-scheduling plan for Peixinho (Kiem plan
`820814e1-d77a-4348-a50e-b85fcde1252e`, stages 3–4); Peix keeps semantic
priority, conversations, retries, and checkpoints — Mei stays topic-agnostic
plumbing.

## What is shipped, and what is not

Shipped in this slice:

- **Strict single-flight execution by default, unchanged.** One generation
  runs at a time, and slots are mutated only by the generations that hold
  them, so KV/prefix-reuse semantics stay deterministic. **Bounded overlap
  is opt-in only** (`--max-active-generations 2`): at most two generations
  run at once, never more. The runtime rejects any other value at parse
  time, and a process started without the flag behaves exactly as 0.8.0 did.
- **Bounded admission in front of the slots** (`RequestScheduler`): a bounded
  waiting queue with deterministic ordering and deterministic refusals.
- **Cancellable lifecycle**: a client disconnect (or any cancellation of the
  request task) removes a queued request without running it, and stops a
  running generation without a false success finish; each slot is released
  exactly once, after that generation's own cleanup.
- **Observability**: queue depth, priority mix, overload/cancellation
  counters, cancellation-to-slot-release timing, and the active/peak active
  generation counts on `GET /v1/mei/status`; per-run `priority` and
  `queue_wait_ms` in the `--request-log` JSONL.

Not in this slice: continuous batching, preemption of a running generation
(priority orders the queue and nothing else), more than two slots, and any
Peix-side scheduling surface.

## Request contract

### Priority class (Mei-local extension)

`POST /v1/chat/completions` and `POST /v1/completions` accept an optional
top-level `mei_priority` string:

| Value | Meaning |
| --- | --- |
| `"interactive"` (default, and what every request without the field gets) | A person or a synchronous agent turn is waiting. |
| `"background"` | Durable agent work that can wait. |

The field is namespaced deliberately: an unqualified top-level `priority`
already means something else in other OpenAI-compatible servers (vLLM reads
it as an integer scheduling priority), so Mei never reads it — a standard
`priority` is ignored like any other unknown field, and only `mei_priority`
selects the class.

A queued interactive request outranks every queued background request. Within
one class, requests run FIFO. A present but unknown value (or a non-string)
is rejected with HTTP 400 `invalid_priority`, `param: "mei_priority"` —
silently mis-scheduling a request is worse than a clear rejection. Priority
cannot preempt a generation that is already running: while one is active,
everything else waits or is refused.

**Starvation under continual arrivals.** Priority orders the queue and
nothing else, so a queued background request can be overtaken indefinitely
while interactive arrivals keep coming. The only bound is `--queue-wait-ms`:
with a finite budget the starved request is eventually removed and refused
with HTTP 503 `queue_timeout` (a retryable overload signal, not a slot),
while with `--queue-wait-ms 0` (unlimited) there is no bound at all — a
background request can wait for the entire duration of the interactive
burst. Interactive requests cannot starve each other: they are FIFO within
the class, and the running generation always finishes. Operators who need a
hard bound for background work should keep a finite `--queue-wait-ms` and
retry on `queue_timeout`; a true anti-starvation scheduler (aging, reserved
slots) is deliberately not part of this slice.

### Admission decisions

A request is admitted (given the slot) or refused before any generation work
starts:

| Condition | Behavior |
| --- | --- |
| Slot idle, no waiters | Admitted immediately; `queue_wait_ms = 0`. |
| Slot busy, queue below `--max-queue-depth` | Enqueued; runs when the slot frees, ordered interactive-first then FIFO. |
| Queue already at `--max-queue-depth` | Refused immediately: HTTP **429** `rate_limit_error` / `queue_full`. Never enqueued. |
| Waited longer than `--queue-wait-ms` | Removed from the queue and refused: HTTP **503** `server_error` / `queue_timeout`. Never started. |
| Request task cancelled while queued | Removed from the queue, `CancellationError`; the request never runs. No HTTP response is owed (the client is gone). |

Both refusals are retryable and name their exact condition, so a scheduler
(Peix) can distinguish "server busy, queue full" from "waited too long".
`--max-queue-depth 0` means *no queue*: anything arriving while the slot is
busy gets the 429 refusal. `--queue-wait-ms 0` means *wait forever* (the
pre-extension behavior).

### Defaults (provisional)

`--max-queue-depth` defaults to **32** waiting requests and `--queue-wait-ms`
to **600000** (10 minutes). These are provisional, documented defaults: the
plan's benchmark stage (stage 7) sets the final values against the measured
interactive-latency ceiling and memory reserve. The mechanism — bounds,
refusals, ordering, cancellation — is deterministic and tested regardless of
the values chosen.

## Bounded overlap (opt-in, two active generations)

`--max-active-generations N` is the number of generations that may run at
once. **`1` is the default and the only mode a process started without the
flag can be in** — strict single-flight, byte-compatible with 0.8.0. `2` is
the explicit opt-in bounded-overlap mode; any other value is rejected at
parse time (`--max-active-generations must be 1 … or 2`). There is no
continuous batching, no preemption, and no third slot.

With overlap enabled:

| Condition | Behavior |
| --- | --- |
| Arrival while fewer than 2 generations run and no request waits | Admitted immediately; `queue_wait_ms = 0`. |
| Arrival while 2 generations run (or a request waits) | Enqueued if below `--max-queue-depth`; runs when a slot frees, ordered interactive-first then FIFO. |
| A generation finishes / is cancelled | Its slot is released exactly once; the best queued request (if any) is granted that slot, never more than the configured maximum. |
| Queue at `--max-queue-depth` | HTTP 429 `queue_full`, unchanged. |
| Cancellation | Releases exactly its own slot; the other active generation is untouched. |

The overlap contract adds two guarantees on top of the single-flight one:

- **Never more than `maxActiveGenerations`**: the scheduler grants only while
  a slot is free and never lets an arrival overtake a waiting request; the
  `peakActiveGenerations` counter on `/v1/mei/status` is the audit trail.
- **Slot-exact release**: cancelling one of two active requests releases one
  slot; the surviving generation keeps running and its output is unaffected.

The overlap mode is an experiment with its own gates (isolation canaries,
≥20% aggregate throughput over a serialized pair, TTFT ceiling, memory
reserve, cancel-to-release latency; see the plan `820814e1` stages 5–8).

### Measured results (October 2026, 32-GiB Mac Studio, pinned vmlx)

Two profiles were tested against all gates with a purpose-built integration
client (`scripts/test_concurrent_generations_integration.py`, artifacts under
`~/.local/share/mei-concurrent-generations/artifacts/`):

| Gate | Qwen3.6-35B-A3B-4bit text-only | Qwen3-4B-4bit |
| --- | --- | --- |
| G1 isolation/lifecycle canaries | PASS | PASS |
| G2 aggregate tok/s ≥ +20% over serialized pair | **FAIL 0.899** (30 paired trials) | **FAIL 1.088** (30 paired trials) |
| G3 p95 TTFT ≤5 s and ≤1.20× alone | **FAIL 1.635×** | PASS 1.137× |
| G4 ≥4 GiB reclaimable at peak | **FAIL 3.36 GiB** | PASS 11.83 GiB |
| G5 cancel-to-release p95 ≤2 s | PASS | PASS |
| G6 one slot default, cap two | PASS | PASS |

**The ≥20% throughput gate is not reachable by interleaved single-sequence
execution on this runtime.** Both generations share one MLX default stream, so
a pair's GPU work is additive; only CPU gaps and per-request setup are
recoverable (measured sharing efficiency is slightly *negative* on the 35B and
caps near +12% on the 4B). A verified win needs continuous batching, which is
out of scope for this slice. Separately, the pinned vmlx revision is not safe
for two concurrent prompt prefills (intermittent SIGTRAP in the Qwen3.5-MoE
forward path — see `artifacts/concurrent-generations-rlog.md`, R5a); this
candidate therefore keeps prompt prefill on the engine actor in every mode.

**Do not enable overlap by default, and treat the flag as experimental and
unverified for throughput**: the measured evidence on both tested profiles is
negative for G2. Results do not generalize beyond the tested profiles.

## Cancellation contract

Cancellation enters Mei as cancellation of the request's task: the HTTP
handler's task is cancelled when the client disconnects (`channelInactive`),
and the streaming producer task is cancelled when the stream consumer goes
away.

- **Queued**: the waiter is removed from the queue and resumed with
  `CancellationError`. Nothing runs; no slot is consumed.
- **Running**: the vmlx generation task is cancelled. vmlx checks
  cancellation between prefill chunks and between decode tokens, then still
  runs its post-answer GPU drain and cache store — the slot is **not**
  released until that cleanup completes, so a cancelled generation can never
  overlap the next one.
- **Streaming**: a cancelled run terminates the stream **without a
  `.finish` event**. The client never sees `finish_reason: "stop"` for a
  truncated answer, and the request log records `finish: "cancelled"` once —
  including for structured requests, whose incomplete root is the expected
  result of the stop and is never surfaced as a structured-generation
  failure.
- **Response head**: a streaming response's headers are not sent until the
  engine produces its first event (the first chunk, tool call, or
  prefill-progress tick — a tick can start the head without producing a
  visible frame). Therefore **any** failure before that first event —
  an admission refusal, a pre-generation engine error, or a
  structured-constraint failure that produced no event — is delivered as a
  real HTTP status (429/503 for the admission refusals), exactly as for a
  non-streaming request, instead of a 200 with an empty or error body. Once
  the first event has been produced the status is committed and later
  failures travel in-band as `stream_error` frames. **Header-timeout
  implication**: because the head is withheld, a client's header/response
  timeout must cover the whole queue wait plus time-to-first-token (prefill);
  a client configured with a shorter timeout will abandon requests the
  server is still processing normally.
- **Timing**: the scheduler records the interval from the first observed
  cancellation of an active request to the slot release
  (`lastCancelToReleaseMs` / `maxCancelToReleaseMs`).

## Observability

`GET /v1/mei/status` gains a `scheduler` block:

```json
"scheduler": {
  "running": false, "activeGenerations": 0, "maxActiveGenerations": 1,
  "peakActiveGenerations": 0, "queueDepth": 0,
  "queuedInteractive": 0, "queuedBackground": 0,
  "maxQueueDepth": 32, "queueWaitTimeoutMs": 600000,
  "admitted": 0, "completed": 0, "failed": 0,
  "cancelledQueued": 0, "cancelledActive": 0,
  "rejectedQueueFull": 0, "rejectedTimeout": 0,
  "slotReleases": 0, "lastQueueWaitMs": 0, "maxQueueWaitMs": 0,
  "lastCancelToReleaseMs": null, "maxCancelToReleaseMs": null
}
```

`running` keeps its 0.8.0 meaning ("at least one generation active") and is
now exactly `activeGenerations > 0`. `activeGenerations` /
`maxActiveGenerations` / `peakActiveGenerations` are the overlap audit
surface: `maxActiveGenerations` is what the process started with (1 by
default), and `peakActiveGenerations` must never exceed it.

`slotReleases` is the slot-release signal: a client polling status sees it
advance exactly once per generation, after that generation's cleanup. (A
push-based release signal is deliberately not part of this slice; the plan's
stage-2 contract work decides whether Peix needs one.)

The `--request-log` JSONL gains `priority` (the class the run was admitted
under) and `queue_wait_ms` per run, which is where the stage-7 benchmark
reads queue wait and the class mix from.

## Implementation map

| Concern | Where |
| --- | --- |
| Queue, slots, ordering, bounds, timeouts, counters | `Sources/MeiCore/Scheduling.swift` (`RequestScheduler`) |
| Stream lifecycle / consumer-cancellation propagation | `Sources/MeiCore/CancellableStream.swift` (`makeCancellableStream`) |
| Slot lifecycle in the engine, vmlx task cancellation | `Sources/MeiCore/Engine.swift` |
| `mei_priority` field decoding | `Sources/MeiCore/OpenAITypes.swift` |
| 429/503 mapping, status surface | `Sources/MeiCore/Router.swift` |
| Deferred SSE begin + failure-status decision | `Sources/MeiCore/HTTPServer.swift` (`ResponseWriter.failureResponse`) |
| Flags | `ServerConfig.maxQueueDepth`, `queueWaitTimeoutMilliseconds`, `maxActiveGenerations` |

Tests: `Tests/MeiTests/RequestSchedulerTests.swift` (including the bounded
overlap section), `CancellableStreamTests.swift`, `AdmissionContractTests.swift`,
plus the scheduling-flag cases in `ServerConfigParsingTests.swift`. Live
acceptance + benchmark client:
`scripts/test_concurrent_generations_integration.py` (opt-in, model-backed;
`--self-check` runs offline).

## Relation to the rest of the plan

- Stage 1 (audit) informed this slice: the vmlx pin's `generateTask` checks
  `Task.isCancelled` per decode iteration and `LLMModel.prepare` checks it
  between prefill chunks, both on the pinned revision — that is the
  cancellation primitive this slice relies on.
- Stages 5–8 (isolation proof, bounded concurrency, benchmark, shipped
  default) are implemented as the **opt-in** overlap mode described above:
  the evidence gates are enforced by the live acceptance client, and the
  single-flight mode remains the default and the only mode without the
  explicit flag.
- Stage 9 (Peix end-to-end) consumes the contract here: `mei_priority` in
  the request body, 429/503 refusals to retry on, the status block, and the
  opt-in overlap decision for workloads that have passed the gates on their
  own profile.
