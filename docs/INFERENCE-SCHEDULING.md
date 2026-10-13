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

The overlap mode's original experiment used strict performance gates (isolation,
≥20% aggregate throughput over a serialized pair, TTFT ceiling, memory reserve,
cancel-to-release latency; see the plan `820814e1`). The initial measurements
below remain evidence, but the +20% throughput and ≤1.20×-alone TTFT ratio are
not hard acceptance gates for the revised Peix use case: Peix needs independent
agent tasks to overlap, and a modest slowdown may be an acceptable tradeoff.

### Initial measurements (October 2026, 32-GiB Mac Studio, pinned vmlx)

Two profiles were tested with the model-backed integration client; artifacts
are under `~/.local/share/mei-concurrent-generations/artifacts/`:

| Measurement | Qwen3.6-35B-A3B-4bit text-only | Qwen3-4B-4bit |
| --- | --- | --- |
| Isolation/lifecycle canaries | PASS | PASS |
| Aggregate tok/s ratio (two-slot vs serialized pair; 1.20 was the original target) | **0.899** (about 10% lower) | **1.088** (about 9% higher, below original target) |
| p95 TTFT alone → two-slot | 2.894 → 4.731 s (1.635×) | 2.813 → 3.198 s (1.137×) |
| Peak reclaimable memory | **3.36 GiB** (below reserve) | 11.83 GiB |
| Cancel-to-release p95 | PASS | PASS |
| Default one slot / cap two | PASS | PASS |

The existing acceptance client now has a `--mode peix-like` stand-in for
independent logical agents (same one-process launcher, no duplicate harness).
It runs a synchronized two-task pair, a staged four-agent burst at two active
slots with the remaining two verified queued before any release, and a
cancellation/peer-isolation case; then runs the identical task set serially.
Task/agent IDs, isolated histories/nonces, streaming, tool calls, structured
output, queue-wait sources, TTFT, makespan, throughput, memory, pressure, swap,
and cancellation evidence are saved outside the repo. The client validates
Mei-facing behavior only, not Peix persistence or orchestration.

### Peix-like live results (October 2026, 32-GiB Mac Studio)

| Model/config | Functional verdict | Interactive p95 TTFT | Reclaimable memory / swap | Burst makespan overlap vs serial |
| --- | --- | --- | --- | --- |
| Qwen3-4B-4bit | **PASS 10/10 runs**; two-active, queue depth 2 before release, mapping/isolation, cancel-peer all pass | 1.486 s (30 samples; serial 3.552 s) | overlap minimum 12.25 GiB; 0 warned, 0 swap growth | median serial/overlap 1.420× (range 1.406–1.453; 10 runs) |
| Qwen3.6-35B-A3B-4bit-textonly, default memory cap 26 GiB | **FAIL memory floor**; remaining overlap and task gates pass | 2.370 s (3 samples) | 3.32 GiB; 0 warned, 0 swap growth | 7.08 vs 9.64 s (1.363×) |
| Same 35B weights, `qwen3.6-35b-a3b-text` profile | **FAIL memory floor**; remaining overlap and task gates pass | 2.813 s (3 samples) | 3.43 GiB; 0 warned, 0 swap growth | 6.83 vs 9.84 s (1.440×) |
| Same 35B weights, 25-GiB memory cap | **FAIL memory floor**; remaining overlap and task gates pass | 2.172 s (3 samples) | 3.37 GiB; 0 warned, 0 swap growth | 6.61 vs 9.84 s (1.489×) |

The 4B results cover ten identical one-slot/two-slot stand-in comparisons and
30 interactive TTFT samples per arm. All ten functional/memory runs passed;
the old +20% throughput gate is not applied. Each 35B profile/cap result is one
exploratory run, not enough to establish a latency or throughput improvement.
None clears the 4-GiB reserve. The cap change only moved reclaimable memory by
about 0.05 GiB from the unprofiled baseline, and the named profile by about
0.1 GiB. No pressure warning or swap growth was observed in any 35B run; these
measurements do not establish memory pressure as the cause of its latency or
throughput behavior. The 4B is the only tested profile that currently clears
the workload's memory and interactive-latency gates. Results do not generalize
beyond these models and this device.

**Current working acceptance for the Peix use case:**

- One Mei process/model serves multiple independent task clients; at least two
  requests overlap, while requests above capacity queue. Every task completes
  and maps to the correct caller, with no cross-agent state leakage or crashes.
- One slot remains the default and opt-in overlap is capped at two. Cancellation
  releases only its own slot within 2 seconds p95; queued work and surviving
  tasks remain healthy. The stand-in cancellation-peer check is a functional
  sample; the existing 20-trial G5 test supplies the separate cancellation p95.
- Preserve the 4-GiB free+inactive/reclaimable memory reserve, no sustained
  yellow/red pressure or test-induced swap growth, and a 5-second absolute
  interactive p95 TTFT ceiling. User-approved exceptions must be explicit and
  recorded per model profile.
- Measure task-level TTFT, completion, queue wait, makespan, aggregate
  throughput, memory pressure, and swap against an identical one-slot run.
  Aggregate +20% gain and ≤1.20×-alone TTFT are reported, not required; a
  modest throughput regression can be acceptable if the multi-agent workflow
  works within the agreed task-level latency/resource envelope. Server request
  log queue waits are exact when task attribution is unique; watcher-derived
  waits are labeled estimates and disagreements are reported.
- Peix end-to-end persistence/orchestration remains unverified because its
  checked-out project is still design-only. The stand-in validates Mei-facing
  behavior only; it does not close Peix-specific durability or fallback gates.

No mixed-quant or lighter 27B alternative weights are staged for a same-device
comparison; their provenance/config records are not runnable model bundles. No
`Berger` model-handling component/model was found in the workspace or Kiem, so
that term is unresolved and cannot be tested without clarification. The staged
same-weights Qwen profile was tried as above; it does not make the 35B profile
pass memory reserve. A future optimization study should test a genuinely
staged alternative or changed model-management setting with paired workloads
and pressure/swap traces; do not claim memory causality from these runs.

The pinned vmlx revision also produced an intermittent SIGTRAP during an
experimental path that constructed two prompt prefills concurrently. That path
was reverted; the candidate serializes prefill on the engine actor. Any future
concurrent-prefill design requires a vmlx-side fix or re-pin and fresh crash and
isolation validation. See `artifacts/concurrent-generations-rlog.md`, R5a.

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
