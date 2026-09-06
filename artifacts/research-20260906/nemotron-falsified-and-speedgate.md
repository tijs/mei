# FALSIFIED: proxy fix does NOT fix Nemotron (still 1/8); and the speed gate swings 1.02 -> 32.74 tok/s on KV cache state alone (2026-09-07)

# FALSIFIED: the proxy tool-call fix does NOT fix Nemotron — plus a reproducibility defect in the speed gate (2026-09-07)

Live re-run of `configs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B/mei.yaml` with
`needs_proxy: true` + `proxy_parser: qwen3_coder` (config_hash `d9be6d0097ea`),
chained automatically after the Qwen3.6 text-only benchmark.

## Result: no change. hermes_ops 1/8, exactly as before.

| | old (`e7a6209b765d`) | new, with proxy (`d9be6d0097ea`) |
|---|---|---|
| sanity | 2/2 | 2/2 |
| hermes_ops | **1/8** | **1/8** |

The proxy genuinely ran — `bench_local_proxy starting: upstream=…:8024
listen=…:8015 tool_call_parser=qwen3_coder` — and served requests. The config
wiring is correct. The fix simply does not apply.

## Why: the benchmark never hits the regime the parser recovers

`<function=` appears in **none** of the eight new result files. The model emits
only a prose preamble and stops — there is no tool-call syntax for any parser to
recover.

This was visible in my own isolation data and I read it wrong. Two distinct
failure regimes:

| condition | what the model emits |
|---|---|
| small system prompt + fat tools (~5.5–32k tok) | preamble **plus a bare `<function=…>` block** — recoverable, and what I built the fix for |
| **real Hermes system prompt + fat tools (~20–36k tok)** | **preamble only, no tool syntax at all** — nothing to recover |

The benchmark is the second row. I generalised from the first and claimed the
fix would "move hermes_ops well off 1/8". **That claim is now falsified by
direct test.** The parser work is still correct in itself — it parses all the
real captured forms and correctly refuses corrupted ones — it just addresses a
regime that does not occur in this harness.

So Nemotron's failure is a **model capability limit at this prompt shape**, not
a parser bug. With realistic varied context the threshold sits around 5–6k
tokens; Hermes sends ~22k, of which ~17k is tool schemas (its system prompt
alone is 4,739 tokens and *does* produce a valid call).

## Separate and more serious: the speed gate is not reproducible

Identical prompts, identical outputs, identical pass/fail — and:

| run | avg hermes_ops tok/s | avg ttft | speed gate (>= 4.0) |
|---|---|---|---|
| first (cold KV cache) | **1.02** | 84.6 s | **FAILED** → coding suites skipped |
| second (warm KV cache from run 1) | **32.74** | 1.8 s | **PASSED** → coding suites ran |

Per-task: `[0.64, 0.31, 0.07, 0.27, 4.09, 0.27, 1.73, 0.78]` vs
`[37.59, 27.31, 10.11, 24.62, 54.0, 25.79, 42.18, 40.29]`.

The only difference is that the config's persistent `--kv-cache-dir` was already
populated by the first run, so prefill was served from the disk KV tier. Since
`tokens_per_second = completion_tokens / wall_seconds` and this workload is
~99% prefill, the metric moves **32x** on cache state alone.

**Consequences:**
1. The 4.0 viability gate can pass or fail for the same config depending on
   whether a previous run left a warm cache. That is a reproducibility defect,
   not a Nemotron quirk — it applies to every config with a persistent
   `--kv-cache-dir`, i.e. all the Mei configs.
2. Nemotron's coding suites are running *now* on a gate pass that a cold run
   would not have produced. Any coding rows from this run must be labelled as
   warm-cache, and its recorded 32.74 tok/s must **not** be compared against
   other models' cold numbers.
3. Neither number is simply "wrong" — cold and warm are both real — but the gate
   needs to state which it measures. Suggested fix: clear the KV cache dir
   before a gated run, or record cache state alongside the gate row so the two
   are never compared.

## Actions

- **Revert `needs_proxy` to false** for Nemotron: it adds a proxy hop, disables
  `ttft_measurable` (`run_prompt_suite.py:165`), and demonstrably changes
  nothing. Keep the parser knowledge documented here in case the bare-`<function=`
  regime ever appears in a different harness shape.
- Nemotron stays **non-viable for this agent harness** until either its tool
  emission survives a ~22k-token tool payload, or Hermes's tool payload shrinks
  enough to stay under its threshold. The latter is testable and is the only
  remaining lever I can see.
- File the speed-gate reproducibility issue against the harness, not the model.

#proj/mei
