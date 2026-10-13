# Bounded-overlap concurrency — improvement rounds (R1–R5) and acceptance log

Plan: Kiem `proj/mei` note `820814e1-d77a-4348-a50e-b85fcde1252e` (stages 5+).
Branch: `feat/concurrent-generations` (from `origin/main` @ `1a12ae8`, Mei 0.8.0).
Candidate: locally built release binary (identity at the end).
Client artifacts (logs, per-mode evidence JSON, request logs): outside the repo, under
`~/.local/share/mei-concurrent-generations/artifacts/`.

Gates (all required; none may be weakened):
- G1 isolation/lifecycle canaries (fixed greedy canaries vs single-flight references;
  nonce isolation; overload 429; restart; cache-reuse canary)
- G2 ≥30 paired counterbalanced trials/arm, aggregate completed tok/s ratio ≥ 1.20
- G3 p95 TTFT ≤ 5 s and ≤ 1.20× alone, for prompts ≤ 1000 input tokens
- G4 ≥ 4 GiB free+inactive+reclaimable at peak; no sustained yellow/red; no swap growth
- G5 cancel-to-release p95 ≤ 2 s; no orphaned generation
- G6 one slot stays the default; opt-in capped at two

Each round: identify the failing gate → make a grounded code/harness change → rerun
focused model-free tests → rerun the model-backed acceptance when resource-safe.

---

## R1 — first live canaries run on the primary target (Qwen3.6-35B-A3B-4bit text-only)

Command (artifacts dir `20261013-35b-canaries-r1/`):

    python3 scripts/test_concurrent_generations_integration.py \
      --binary <scratch>/release/mei \
      --model-dir ~/.local/share/local-model-bench/mei-models/Qwen3.6-35B-A3B-4bit-textonly \
      --served-model-id Tostibrown/Qwen3.6-35B-A3B-4bit-textonly \
      --mode canaries --artifacts-dir .../20261013-35b-canaries-r1

Result: **G1 PASS** — all checks green on the first attempt:

- initial state: `maxActiveGenerations=2`, idle, zero counters;
- single-flight references stable across two identical greedy runs (text/stream/
  tools/structured);
- all three two-in-flight pairs (buffered text ∥ streaming text; buffered tools ∥
  streaming tools; buffered structured ∥ streaming nonce) matched their single-flight
  references byte-for-byte (text, finish_reason, completion tokens, prompt tokens,
  tool calls); observed `activeGenerations=2` in every pair; peak never exceeded 2;
- nonce isolation held (each session saw exactly its own nonce; the other nonce never
  appeared, including in reasoning text);
- overload: with both slots busy and the queue at its bound, a third streaming request
  got HTTP 429 JSON `queue_full` before any SSE event, was never enqueued, and the
  active count was unchanged;
- restart: SIGTERM → clean exit → fresh process with zeroed counters; canary passed;
- cache-reuse canary (coordinator path ON, hybrid model, explicit `--kv-cache-dir`):
  two sessions × two turns, overlapped turn-by-turn (observed max active=2), both
  turn-1 and turn-2 outputs matched the serial reference byte-for-byte, nonces isolated.

Defects found in R1 (harness, not the server):

1. `comparable()` recorded buffered tool calls as `{name: null, arguments: null}`:
   the buffered OpenAI wire shape nests them under `function`, while the streaming
   assembly is flat. The comparison was therefore weaker than intended for the
   buffered tools canary (both sides null). Fix in R2.
2. The server's startup line still printed "admission: strict single-flight execution"
   when overlap was enabled — misleading operator output, no behavioral impact. Fix in R2.

## R2 — harness/UX fixes, focused re-verification

Changes:

1. `scripts/test_concurrent_generations_integration.py`: added `normalized_tool_call()`
   and used it in `comparable()` so both wire shapes compare equal; strengthened the
   tools canary to assert the call name (`add_numbers`) and the parsed arguments
   (`{"a": 17, "b": 25}`) for both buffered and streaming references; extended the
   self-check `comparable_case` to pin flat-vs-nested normalization.
2. `Sources/Mei/main.swift`: the admission startup line now says
   "bounded overlap (2 active generations, opt-in)" when
   `--max-active-generations 2` is set, and stays byte-identical otherwise.

Focused model-free verification: client `--self-check` 19/19;
`swift test --filter RequestSchedulerTests|ServerConfigParsingTests|AdmissionContractTests`
73 tests, 0 failures; release rebuilt from the same vmlx pin (`633fe166…`).

Live re-run (artifacts `20261013-35b-canaries-r2/`): **G1 PASS** again; the tools
canaries now record `add_numbers {"a":17,"b":25}` for both buffered and streaming
shapes; the startup line shows the bounded-overlap text. All other checks identical.

## R3 — first full benchmark on the 35B (30 paired trials/arm): G2/G3/G4 FAIL

Command: `--mode benchmark --trials 30 --warmup 2` (artifacts `20261013-35b-benchmark-r3/`).

Measured (paired, counterbalanced; 940-token interactive prompt, ≤1000-token gate scope):

| Gate | Result | Value | Threshold |
| --- | --- | --- | --- |
| G2 aggregate tokens/s | **FAIL** | serialized 40.49, concurrent 36.39, ratio **0.899** | ≥ 1.20 |
| G3 interactive TTFT | **FAIL** | p95 alone 2.894 s, p95 concurrent 4.731 s, ratio **1.635** | ≤ 5 s and ≤ 1.20× |
| G4 memory | **FAIL** | min reclaimable **3.36 GiB**, pressure 0/1356 warned, swap growth 0.0 MiB | ≥ 4 GiB |

Two-in-flight was verified throughout (`observed_max_active=2`, peak never >2), all
128 runs completed with `finish: length`, no stream errors, token counts identical
across arms (determinism guard held). Per-request decode in the concurrent arm drops
from 53.9 → ~21 tps (interactive) and ~35 tps (background): the GPU work of two
interleaved sequences is additive on the shared MLX stream, and the pair wall ends up
*longer* than the serialized pair.

Diagnosis (probes under `probe-r3b/`, `probe-timeline/`, `probe-actorblock/`):

1. **Engine-actor stall (dominant, fixable).** `/v1/mei/status` stalls for 1.888 s
   during a single 940-token request: the Engine actor is held by the synchronous
   `TokenIterator` construction (prompt prefill). A second concurrent request cannot
   even reach the scheduler until that stretch ends — the timeline probe shows the
   second admission ~2.2 s after the first, and its whole window shifts by that delay.
2. **Additive GPU work (structural).** Even with zero interference the two requests'
   GPU work must serialize on the shared MLX default stream: the serialized pair's
   GPU-busy time alone is ~8.5 s against the ≤7.2 s the 20% gate requires. Interleaved
   (non-batched) execution therefore cannot reach the gate on this model/device.

## R4 — grounded fix: deferred iterator construction (off-actor prefill) in overlap mode

Change (`Sources/MeiCore/Engine.swift`): new `startGeneration(...)` helper. Strict
single-flight (default) keeps the exact 0.8.0 path (eager `TokenIterator` construction
+ `generateTask`). When `--max-active-generations 2` is set, the three generation paths
(buffered chat, streaming chat, completions) construct the iterator *inside* the
producer task via vmlx's supported `generateTaskDeferred(makeIterator:)`, so a second
request's prefill no longer waits behind the first request's synchronous actor work.
Slot lifecycle, outputs, and cancellation semantics are unchanged; cancellation during
prefill is handled by the deferred loop itself (drain + cancelled info, no success
finish).

Focused verification: `swift build` clean; 73 focused tests pass; client self-check
19/19; live canaries re-run on the rebuilt binary (`20261013-35b-canaries-r4/`):
**G1 PASS** (all pairs matched references; observed max active 2; cache-reuse canary
green).

R4 result (12 paired trials, artifacts `20261013-35b-benchmark-r4/`): **G2/G3/G4 still
FAIL — the deferred change is measured-neutral.** serialized 40.81 tok/s, concurrent
36.20 tok/s, ratio **0.887** (R3: 0.899); TTFT p95 2.893 → 5.019 s, ratio **1.735**
(R3: 1.635); min reclaimable 3.45 GiB. The actor stall was real (probe
`probe-actorblock/`: 1.888 s status stall during a single 940-token request) but it is
not the dominant cost: with the stall removed, the concurrent arm is still ~11% slower
in aggregate. The interference is GPU/stream-level (two interleaved single-sequence
decodes on the shared MLX default stream lose ~13% aggregate throughput versus solo),
and the additive GPU work alone (~8.5 s) already exceeds the ≤7.2 s pair budget the
20% gate requires. Conclusion for the target model: **interleaved bounded overlap
cannot reach G2 on Qwen3.6-35B-A3B-4bit text-only on this device** — the remaining
levers are continuous batching (explicitly out of scope) or different hardware.

## R5 — fallback target: Qwen3-4B-4bit (authorized smaller staged model)

Same gates, same harness, same candidate binary; model
`/Users/tijs/.cache/mei/models/Qwen3-4B-4bit` (`mlx-community/Qwen3-4B-4bit`,
config sha256 `b5efdcf3b0035a3638e7…`, qwen3 dense 4-bit g64, 2.1 GB).

Harness adjustments made in R5 (model-agnostic canaries; not gate changes):

- `looks_like_pong()`: the trivial text canary asserts case-insensitive containment
  (the 4B answers "Pong."); the isolation guarantee remains the byte-for-byte
  reference comparison. Self-check extended (20/20).
- cache-reuse canary: the turn-2 assertion now requires (a) overlapped output ==
  serial reference byte-for-byte and (b) cross-session token exclusion. Probe
  `probe-4b-cache/` showed the 4B echoes the *earlier* token at turn 2 (both in the
  serial reference and overlapped — identical outputs), so "must contain its own
  turn-2 nonce" was a model-quality expectation, not an isolation property.
- Failed canary runs now write `canaries-partial.json` for diagnosis.

Results: see entries below. Interim (pre-revert deferred candidate — historical;
superseded by the final eager re-measurements in R5b):

- canaries: **PASS** (`20261013-4b-canaries-r5`, `-r5b`; after the two harness
  adjustments above).
- benchmark, 30 paired trials: G2 **1.103 FAIL**, G3 1.136 PASS, G4 12.68 GiB PASS
  (`20261013-4b-benchmark-r5/`).
- cancellation: **PASS**, server p95 23 ms (`20261013-4b-cancellation-r5/`).

### R5a — a G1-class crash found on the 35B overlap path (deferred prefill), and the revert

While setting up the 35B cancellation run, the candidate crashed intermittently: two
concurrent requests arriving as the **first** generation traffic of a fresh process
made the server exit with SIGTRAP and both connections close with no response
(1 of 6 fresh-server attempts; reproduced with a standalone probe). The macOS crash
report (`~/Library/Logs/DiagnosticReports/mei-2026-10-13-085233.ips`) shows two
threads inside `Qwen35GatedDeltaNet.callAsFunction` at the same time:

- faulting thread: `Swift runtime failure: Unexpectedly found nil while unwrapping an
  Optional value` at `Qwen35.swift:533` (`Qwen35GatedDeltaNet.callAsFunction`), reached
  from `TokenIterator.init` → `prepare` (i.e., a prompt prefill);
- second thread: data abort (translation fault) in
  `_NSSwiftProcessInfo.environment.getter` ← `NativeMTPVerifierStatePolicy.mode(for:)`
  ← `shouldStashVerifyInputs` ← the same `Qwen35GatedDeltaNet.callAsFunction` — the
  model's per-layer policy checks read `ProcessInfo.processInfo.environment` on every
  call, and two concurrent first forwards race on its lazy initialization.

Both crash paths are inside the pinned vmlx revision's model forward, not in Mei: the
vmlx Qwen3.5-MoE path is not proven safe for two concurrent prompt prefills. It never
fires in the eager (actor-serialized prefill) path — the first forward always happens
on one thread — and it did not fire in any of the 128 deferred-path benchmark runs or
dozens of canary pairs (the race window is the process's first concurrent forward).

Decision: **R4's deferred construction was reverted.** It was measured-neutral for
throughput (0.887 vs 0.899 on the 35B) and it opens a crash window on the pinned
dependency; the eager path keeps prompt prefill on the engine actor, so the opt-in
overlap mode cannot trigger the vmlx race via concurrent prefills. The vmlx race is
recorded as a primary-source blocker for any future concurrent-prefill design (it
needs a vmlx-side fix or a re-pin, not a Mei workaround).

Revert verification (this is where the harness earned its keep):

1. The first reverted rebuild accidentally dropped the Engine wiring that passes
   `config.maxActiveGenerations` into the scheduler limits, so the binary silently
   ran strict single-flight even with the flag. The live canary suite caught it
   immediately: `advertised maxActiveGenerations: expected 2, got 1`. Wiring
   restored, rebuilt.
2. Fresh-server + simultaneous-pair probe (both requests as the first generation
   traffic, 6 attempts) on the fixed eager candidate: **0 crashes, both requests
   admitted and streaming in every attempt**; the same probe on the deferred
   candidate reproduced the SIGTRAP crash. The eager path is therefore the safe
   configuration for this pinned dependency.

Consequently the 4B benchmark figures from R5 were re-measured on the reverted
(eager) candidate before being used for the gate verdict; the 35B numbers from R3
(eager) stand.

### R5b — final results (corrected eager candidate, all modes)

Artifacts (outside the repo, under `~/.local/share/mei-concurrent-generations/artifacts/`):

| Run | Artifacts dir | Result |
| --- | --- | --- |
| 35B canaries (G1) | `20261013-35b-canaries-final/` | **PASS** (7/7 checks) |
| 35B cancellation (G5) | `20261013-35b-cancellation-final/` | **PASS** (server p95 32 ms, client p95 248 ms; 20 cancels) |
| 4B canaries (G1) | `20261013-4b-canaries-final/` | **PASS** |
| 4B benchmark (G2/G3/G4) | `20261013-4b-benchmark-final/` | G2 **FAIL 1.088**, G3 PASS 1.137, G4 PASS 11.83 GiB |
| 4B cancellation (G5) | `20261013-4b-cancellation-final/` | **PASS** (server p95 25 ms, client p95 239 ms; 20 cancels) |
| single-flight default smoke | `final-singleflight-smoke/` | **PASS** (maxActiveGenerations=1, second request queued, peak=1) |

Final gate table (all gates required; none weakened):

| Gate | Qwen3.6-35B-A3B-4bit text-only | Qwen3-4B-4bit |
| --- | --- | --- |
| G1 isolation/lifecycle canaries | PASS | PASS |
| G2 aggregate tok/s ≥ +20% | **FAIL 0.899** (30 trials) | **FAIL 1.088** (30 trials; per-trial 1.074–1.121) |
| G3 p95 TTFT ≤5 s and ≤1.2× | **FAIL 1.635×** (2.894 → 4.731 s) | PASS 1.137× (2.813 → 3.198 s) |
| G4 ≥4 GiB reclaimable | **FAIL 3.36 GiB** | PASS 11.83 GiB |
| G5 cancel-to-release p95 ≤2 s | PASS (32 ms / 248 ms) | PASS (25 ms / 239 ms) |
| G6 default one slot, cap 2 | PASS (unit tests + live smoke) | PASS |

**Verdict: NO-GO for the ≥20% throughput gate.** Per the settled plan (“if that also
fails, stop rather than weaken thresholds”), the experiment stops here.

Why the gate is unreachable by this design (measured, not inferred):

- Two interleaved single-sequence generations share one MLX default stream: the
  GPU work of a pair is additive, so the only recoverable wall time is the serialized
  pair's CPU gaps and per-request setup. Measured sharing efficiency: aggregate
  decode while sharing ≈ 73–80 tps on the 4B (solo 80) and ≈ 47–54 tps on the 35B
  (solo 54) — sharing *loses* throughput, it does not gain it.
- The achievable ratio is bounded by (serialized pair wall) / (GPU-bound wall). On the
  4B that ceiling is ~1.12 and the measured value is 1.088; on the 35B the GPU-bound
  wall alone (~8.5 s) exceeds the ≤7.2 s the gate allows. Reaching +20% requires
  continuous batching (reading weights once per step for both sequences), which this
  slice explicitly excludes, or a runtime with genuinely parallel streams.
- The vmlx concurrency blocker (R5a) additionally means concurrent prompt prefills
  crash the pinned revision; the shipped candidate keeps prefill on the engine actor.

What this branch delivers: the opt-in bounded-overlap mechanism (default one slot,
capped at two, runtime-rejected above two), full model-free coverage, the live
acceptance/benchmark client, and this evidence. What it does **not** deliver: a
verified throughput win — the overlap mode must not be enabled by default, and the
measured negative result should be published with the flag if the flag ships at all.

Next blocker for a future attempt: (1) continuous batching in vmlx (out of scope
here), and (2) a vmlx-side fix for concurrent first forwards (the R5a crash), which
is a dependency change the plan rules out for this experiment.

