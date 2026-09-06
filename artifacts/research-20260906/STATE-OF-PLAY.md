# STATE OF PLAY: hybrid-MoE optimization, consolidated (2026-09-06)

Index + current conclusions across the seven research notes filed today under
`proj/mei`. Read this first; it supersedes the estimates in the earlier notes
where they conflict.

## Companion notes
1. Findings F1-F6 (architecture, bandwidth budget, dead/disabled compile levers)
2. Microbenchmarks (F4/F5 stack; bit depth is not a speed dial)
3. Prioritized experiment plan P0-P6
4. Qwen3.6 vision tower / VLM load path
5. Nemotron-3.5-Lightning architecture analysis
6. BUILT: Qwen3.6-35B-A3B-4bit-textonly (published)
7. Decode-step budget, speculative economics, compile ceiling

Worktree `/Users/tijs/projects/mei-opt-research` (branch
`research/hybrid-superopt`), artifacts under `artifacts/research-20260906/`,
tools under `tools/`. Mei `main` untouched throughout.

## The offline decode-step model is validated

`tools/decode_step_model.py` predicts **18.11 ms/token** for the current
production config against a **measured 18.18 ms** — 0.4% error. That makes it a
usable digital twin: config- and graph-level levers can be pre-screened offline,
without a GPU window and without disturbing the benchmark owner. Every number
below comes from it or from the 3-repeat microbenchmarks.

## Realistic stacked ceiling (recalibrated)

Levers are **sub-additive where they overlap** — measured, not assumed: compile
alone gives 1.11x on the MoE block, the pre-fused gate+up repack alone 1.10x,
both together 1.165x rather than 1.22x.

| stage | ms/token | tok/s | cumulative |
|---|---|---|---|
| current production | 18.11 | 55.2 | — |
| + compile everywhere (P1) | 16.66 | 60.0 | 1.087x |
| + pre-fused gate+up repack (P3) | 16.43 | 60.9 | **1.102x** |

**~+10% is the honest config/graph-level ceiling.** Bit depth adds nothing in the
speed dimension (measured), and speculative decoding is closed (see note 7).

Projecting the saved 1.68 ms onto the real long-context rows — valid because the
saving is in projections, which do not grow with context:

| row | now | with the full stack |
|---|---|---|
| short decode | 55.0 | ~60.9 |
| 30k loaded decode | 47.5-50.3 | **~54.6** |
| 80k loaded decode | 35.7 | **~38.0** |

## Against the >= 30 tok/s long-context goal

**Ornith already meets it with margin** — 47.5-50.3 t/s at 30k and 35.7 t/s at
80k, both comfortably above 30 (`README.md` records the goal as EXCEEDED). The
stack above does not rescue a missed target; it **widens an existing margin**,
from ~1.6x to ~1.8x the goal at 30k and from 1.19x to 1.27x at 80k.

That reframes what the remaining work is for. The binding constraint on this
machine is **not** decode speed on the hybrid MoE models — it is:
1. **Memory**, which caps how much context fits (peak 25.73 GB @ 65k, 28.19 GB
   @ 100k). Context headroom, not tok/s, is what the 32 GB budget actually
   limits. This is where the expert-bank levers (P4 mixed 4/8-bit, P5 expert page
   residency) earn their keep, and why the 0.83 GiB vision-tower strip matters.
2. **The dense models**, where 30 t/s is genuinely unreachable — Qwen3.8-27B sits
   at 15.66 t/s, already 89% of its bandwidth floor. No stack of these levers
   changes that; it was correctly closed as a hardware ceiling.

## Where the remaining 3.8x actually lives

Achieved bandwidth across the step averages **29% of the 400 GB/s spec**. A
single large quantized matmul (`lm_head`) reaches 212 GB/s; the gathered MoE path
reaches 99 GB/s and the shared expert 44 GB/s on the same hardware. **The
headroom is small-M quantized matmul kernel efficiency**, which is MLX/vmlx
kernel work — outside Mei's configuration surface and outside the ~10% above.

Highest-ratio kernel target if that is ever pursued: the **shared expert**, at
11% of spec — 40 tiny 2048x512 matmuls per token, and the component `mx.compile`
helped most (1.32x). Batching or fusing it across layers is the best
effort-to-payoff ratio in the whole step.

One cheap untested kernel-level lever that IS reachable from config: production
passes `sortedIndices: false` to `gather_qmm` at decode (because
`doSort = indices.size >= 64` is false at top-8). Whether the sorted path
improves the 99 GB/s MoE figure at decode shapes is unmeasured.

## Closed / do not pursue

- **Speculative decoding** (DFlash, DSpark, Eagle3, native MTP): caps at 1.60x
  with perfect acceptance and zero draft cost; nets ~1.1x realistically. The
  premise (bandwidth-bound at T=1, so extra tokens ride free) does not hold for
  this model class. Matches the llama.cpp result the user already observed, and
  now has a mechanism. Revisit only at very long context, if ever.
- **5-bit and 6-bit quants**: slower AND larger than 4-bit (only 2/4/8 get fast
  kernels). Never "one step up" on this family.
- **Whole-graph compiled decode** (`--compiled-decode true`): already closed
  2026-09-02 on 35B evidence; the per-request promote+trace tax exceeds the win.

## Next actions, unchanged in order but recalibrated in expectation

P0 (profile the 22% remainder) and the load-path trace remain the right first
moves — both are ~10 minutes of GPU. P1 is still worth running because it is
free, but expect **~+8%, not the +45-70%** the upstream comment advertises for
small dense models. P4/P5 (memory) now matter more than P1/P2/P3 (speed),
because memory is what actually binds on this machine.

#proj/mei
