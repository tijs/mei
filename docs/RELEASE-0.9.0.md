# Mei 0.9.0 — opt-in bounded-overlap concurrency (experimental)

**Feature release — 2026-10-13**

Mei 0.9.0 adds **opt-in bounded-overlap concurrency**: `--max-active-generations 2`
lets at most two generations run at once in one server process — never more,
no continuous batching, no preemption. One generation at a time remains the
default: a process started without the flag behaves exactly as 0.8.0 did, and
values outside 1...2 are rejected at parse time. The mode is **experimental**
— it is not enabled by default, and its measured results do not generalize
beyond the tested model and device.

## What shipped

- `--max-active-generations` (default **1**, cap **2**): `1` is strict
  single-flight, byte-compatible with 0.8.0; `2` is the explicit opt-in
  bounded-overlap mode.
- The 0.8.0 scheduling contract is unchanged: the bounded queue
  (`--max-queue-depth`), `mei_priority` classes (interactive before
  background, FIFO within a class), overload refusals (HTTP 429
  `queue_full` / 503 `queue_timeout`), and end-to-end cancellation with
  slot-exact release. With overlap enabled, a second arrival is admitted
  immediately while one generation runs; further arrivals queue, and every
  slot release hands free capacity to the best queued request. Cancelling
  one of two active requests releases exactly its own slot — the surviving
  generation is untouched.
- `GET /v1/mei/status` gains `activeGenerations`, `maxActiveGenerations`, and
  `peakActiveGenerations` (the audit trail that the configured maximum was
  never exceeded); `running` keeps its 0.8.0 meaning
  (`activeGenerations > 0`).
- Prompt prefill stays serialized on the engine actor in every mode. An
  experimental deferred-prefill path was measured neutral on throughput and
  reverted after it exposed an intermittent SIGTRAP in the pinned vmlx
  revision when two prompt prefills ran concurrently; the shipped candidate
  cannot trigger that race.

Full contract: [docs/INFERENCE-SCHEDULING.md](INFERENCE-SCHEDULING.md).

## Acceptance evidence (Peix-like stand-in)

The `--mode peix-like` live client runs independent logical-agent tasks
against one Mei process — a synchronized two-task pair, a staged four-task
burst (two active, two verifiably queued before any slot release), and a
cancellation/peer-isolation case — then compares the identical task set
against one-slot serial mode.

| Model/config | Verdict | Interactive p95 TTFT | Reclaimable / pressure / swap | Burst makespan overlap vs serial |
| --- | --- | --- | --- | --- |
| `mlx-community/Qwen3-4B-4bit` | **PASS 10/10 paired runs** | 1.486 s (30 samples; serial 3.552 s) | ≥12.25 GiB / 0 warned / 0 swap growth | median 1.420× (range 1.406–1.453) |
| Qwen3.6-35B-A3B text-only, default cap | **FAIL memory reserve only** | 2.370 s (3 samples) | 3.32 GiB / 0 warned / 0 swap growth | 7.08 vs 9.64 s (1.363×) |
| same 35B weights, `qwen3.6-35b-a3b-text` profile | **FAIL memory reserve only** | 2.813 s (3 samples) | 3.43 GiB / 0 warned / 0 swap growth | 6.83 vs 9.84 s (1.440×) |
| same 35B weights, 25-GiB memory cap | **FAIL memory reserve only** | 2.172 s (3 samples) | 3.37 GiB / 0 warned / 0 swap growth | 6.61 vs 9.84 s (1.489×) |

- The 4B verdict is backed by ten full paired stand-in runs (30 interactive
  TTFT samples per arm); it is the only tested profile that clears the
  workload's 4-GiB memory reserve and 5-second absolute interactive p95 TTFT
  ceiling. The original strict gates (+20% aggregate throughput, ≤1.20×-alone
  TTFT) are reported, not required, for this task-flow objective; the earlier
  strict benchmark missed them on both profiles (ratios 0.899 on the 35B and
  1.088 on the 4B).
- The earlier strict-gate runs remain recorded in the rlog: isolation (G1),
  cancellation (G5), and the one-slot-default / cap-two guarantees (G6)
  passed on both profiles; the +20% aggregate-throughput gate (G2) failed on
  both; TTFT and memory (G3/G4) failed on the 35B and passed on the 4B.
- **The Qwen3.6-35B-A3B text-only profile failed the 4-GiB memory reserve in
  all three tested configurations (3.32 / 3.43 / 3.37 GiB) and is not
  accepted for this workload on the tested device.** No pressure warnings or
  swap growth were observed in any 35B run; memory pressure is not
  established as the cause of its behavior.
- **Peix end-to-end orchestration remains unverified** — the Peix checkout is
  still design-only, so the stand-in validates Mei-facing behavior only
  (task/result mapping, isolation, tools/structured output, cancellation,
  resources), not Peix durability or hosted-fallback gates. Those remain a
  future Peix-side E2E gate.
- Results do not generalize beyond the tested models and this device.

Full evidence: [artifacts/concurrent-generations-rlog.md](https://github.com/tijs/mei/blob/v0.9.0/artifacts/concurrent-generations-rlog.md)
in the repository (not bundled in the release tarball); client artifacts under
`~/.local/share/mei-concurrent-generations/artifacts/`.

## Verification

Model-free: the scheduling contract suites — `RequestSchedulerTests`,
`CancellableStreamTests`, `AdmissionContractTests`, and the
`--max-active-generations` cases in `ServerConfigParsingTests` — pass (78
tests, 0 failures) on the release candidate source, and the live client's
offline `--self-check` passes 29/29. `mei --version` reports `mei 0.9.0`;
`--max-active-generations` outside 1...2 is rejected at parse time (exit 2).

Live, model-backed (locally built release candidate binary): the Peix-like
stand-in results above on `mlx-community/Qwen3-4B-4bit` and
Qwen3.6-35B-A3B text-only, plus the isolation-canary (G1) and
cancel-to-release (G5) live runs on both profiles. Raw evidence:
[artifacts/concurrent-generations-rlog.md](https://github.com/tijs/mei/blob/v0.9.0/artifacts/concurrent-generations-rlog.md).

Structured-output model support is unchanged from 0.8.0 (carried from the
0.7.1 runtime source `ea5a67a`): the exact buffered and streaming CoCore
canaries passed on Qwen3.6 text-only, Qwen3.6 vision (text-only requests),
and aligned Ornith, as well as the earlier Qwen3-4B checkpoint.
Image-conditioned structured output on Qwen3.6 vision remains unverified,
and Qwen3-8B has not been retested. See
[OPENAI-COMPATIBILITY.md](OPENAI-COMPATIBILITY.md#structured-output-live-model-matrix)
and [COCORE.md](COCORE.md).

## Install

```bash
brew update
brew upgrade tijs/tap/mei
mei --version  # mei 0.9.0
```

Manual Apple Silicon release asset: `mei-0.9.0-macos-arm64.tar.gz` and its
`.sha256` companion on the GitHub release page.
