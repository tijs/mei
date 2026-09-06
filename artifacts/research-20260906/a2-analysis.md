# A2 analysis: 29% of every decode step is CPU graph-build, and compiled decode was closed on the one generation length where it loses (2026-09-06)

Analysis layered on the other agent's A2 run (note 851568fb, which recorded the
raw stage rows but did not interpret them). **No GPU used — this is arithmetic on
their artifacts.** Source:
`results-mei/A2-ornith-genprofile-20260906T170232Z/ornith/runtime/logs/ornith-genprofile-20260906T170232Z.log`.

## 1. The decode step is CPU-then-GPU, serialized

58-token short-decode row, Ornith, current pinned build (mei 67e897e):

| stage | avg ms/token |
|---|---|
| `decode.model_forward` (== `step_build`, CPU lazy graph construction) | **4.693** |
| `decode.async_eval_submit` (GPU execution) | **11.527** |
| `decode.sample` | 0.003 |
| `decode.token_item_sync` | 0.002 |
| **sum** | **16.220** |

16.220 ms implies 61.65 tok/s; the probe reported **61.70**. The two stages
account for the entire step, so **they do not overlap** — the GPU is idle for
4.693 ms of every token while the CPU builds the graph, and the CPU is idle
while the GPU runs.

**28.9% of every decode step is CPU-side graph construction with the GPU
stalled.** That is a bigger single lever than anything in the RUNBOOK's Phase C
(compile ~1.11x, F4/F5 ~1.05x).

Also note the baseline has moved: **61.70 tok/s**, not the 55.0 that all my
earlier projections were anchored to. The log shows
`[Qwen35] fused_gdn_decode_input_projections=active groups=[4]`, a GDN
input-projection fusion that is live on this pin and was not in the 55.0-era
measurements. **Every stacked-ceiling figure in the STATE OF PLAY and RUNBOOK
notes needs rebaselining against 61.70.** The Phase C levers are unchanged in
mechanism but their headline "55 -> 60.9" is now stale.

## 2. Compiled decode should be RE-OPENED — it was closed at 32 tokens

`decode.model_forward` at 4.693 ms/token is exactly what compiled decode
attacks. The 2026-09-02 artifact measured `decode.compiled_forward avg=2.183 ms`
against eager `model_forward avg~6.2 ms`, i.e. it removes ~2.5 ms/token of CPU
graph-build (using today's 4.693 as the eager figure).

That artifact concluded **"compiled-decode gate on 35B: CLOSED as a throughput
lever"**. But every row it measured generated **32 tokens** — and 32 tokens is
below the amortization breakeven for the one-time promote+trace tax.

Two independent readings of that tax from the same artifact:
- **Derived from its own numbers**: 32 tokens at 38.9 t/s (compiled) = 823 ms vs
  46.7 t/s (eager) = 685 ms, so the tax is ~**137 ms**.
- **As stated in the artifact's prose**: ~300-500 ms per request.

Breakeven at 2.51 ms/token saved: **55 tokens** (tax 138 ms) to **159 tokens**
(tax 400 ms).

Projected over generation length, short context:

| generated tokens | eager | compiled (tax 138) | compiled (tax 400) |
|---|---|---|---|
| **32** (what was measured) | 0.52 s | 0.58 s — **0.90x** | **0.62x** |
| 100 | 1.62 s | 1.51 s — 1.08x | 0.92x |
| 250 | 4.05 s | 3.57 s — 1.14x | 1.06x |
| 500 | 8.11 s | 6.99 s — **1.16x** | 1.12x |
| 1000 | 16.22 s | 13.85 s — **1.17x** | 1.15x |

**32 tokens is the single length at which compiled decode loses under both tax
estimates.** A Hermes coding turn routinely generates several hundred to a
thousand tokens. At those lengths compiled decode is worth **+12% to +17%** —
comparable to the entire Phase C stack, from a lever currently marked CLOSED.

## 3. Caveats — this is an inference, not a measurement

- `compiled_forward = 2.183 ms` comes from the 2026-09-02 run on an **older vmlx
  pin**; today's eager figure (4.693) is from the current pin. Mixing them is the
  weakest step in the chain. A fresh compiled-vs-eager profile on the current
  pin would settle it.
- The tax was measured **at 30k context**, where promote materializes the whole
  prefill KV/SSM into fixed Compilable buffers. At **short** context that tax
  should be substantially smaller, which would favour compiled decode further —
  so the table above is likely conservative for short-context requests.
- Conversely, the 2026-09-02 threshold guard exists precisely because the tax
  grows with context. At 30k+ the picture may still be negative even for long
  generations. **This finding is about short/medium context with long output,
  which is exactly the Hermes coding-agent shape.**
- Compiled decode requires `VMLX_ENABLE_UNSAFE_COMPILE=1`
  (`HardwareInfo.isCompiledDecodeSupported`), so it carries the same
  silent-corruption correctness gate as RUNBOOK C1: greedy temp-0
  token-for-token equality vs eager, not probe pass/fail.

## 4. Recommended change to the RUNBOOK

Add to Phase C, ahead of C2/C3:

> **C1b. Compiled decode at realistic generation length.**
> `--compiled-decode true` with `VMLX_ENABLE_UNSAFE_COMPILE=1`, short context,
> **max_tokens 500 and 1000** (not 32), 3 repeats, vs eager control at the same
> lengths. Also capture `decode.compiled_forward` on the current pin to replace
> the borrowed 2.183 ms figure.
> *Expect +12-17% if the analysis holds; 32-token rows will still show a loss
> and that is the predicted, not a contradicting, result.*

This does not reopen the 30k/80k conclusion, which stands. It reopens only the
short/medium-context, long-output case that was never measured.

#proj/mei
