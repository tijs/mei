# The 4.7 ms/token CPU graph-build is Swift-side overhead, not inherent to MLX — and it is the single largest lever found (2026-09-06)

GPU-free analysis (graph construction submits no GPU work, so this ran safely
while the other agent's B1 leg owned the GPU). Tools:
`tools/graph_build_cost_bench.py`, `tools/full_graph_build_bench.py`.

## The measurement

A2 showed `decode.model_forward = 4.693 ms/token` — CPU-side lazy graph
construction, 28.9% of the 16.220 ms step, with the GPU idle throughout.

Building a **structurally equivalent** qwen3_5_moe decode graph in Python-MLX
(30 GDN + 10 attention + 40 MoE + shared experts + lm_head; tiny tensors, since
graph-build cost scales with op count not tensor size):

| | ms/token | us/op |
|---|---|---|
| Python-MLX, ~1484 ops | **0.788** | 0.53 |
| Swift `decode.model_forward` (measured) | **4.693** | 3.16 |

Both bind the same C++ MLX core. **Swift is ~6x the per-op cost.**

If Swift matched Python's per-op cost, the decode step would go
16.220 -> 12.31 ms: **61.7 -> 81.2 tok/s, a 1.32x speedup.** That is larger than
every Phase C lever combined (compile ~1.11x, F4/F5 ~1.05x).

## Sensitivity — the magnitude depends on an estimate, the existence does not

My 1484-op count is a structural estimate, not a trace of the real graph. It is
the weak link, so:

| assumed ops/step | implied Swift us/op | vs Python |
|---|---|---|
| 1484 (my estimate) | 3.16 | 6.0x |
| 3000 | 1.56 | 3.0x |
| 6000 | 0.78 | 1.5x |
| **8855** | **0.53** | **1.0x — break-even** |

For Swift to be merely *as* efficient as Python, the real graph would need
**8855 ops/step = 221 ops per layer**. My estimate is ~35/layer. Even a 4x
undercount leaves a 1.5-3x gap.

## Independent corroboration that does not depend on my op count at all

The 2026-09-02 compiled-decode artifact measured, on the real Swift model:

- eager `decode.model_forward` ~**6.2 ms/token**
- `decode.compiled_forward` **2.183 ms/token**

Compiled decode **replays a recorded graph instead of rebuilding it**. So the
~4.0 ms it removed *was* graph (re)construction, measured directly on the real
model. That independently confirms multiple ms/token of Swift-side per-step
construction cost, with no reliance on my structural estimate.

Two independent lines therefore agree: **~4 ms/token — roughly a quarter of the
decode step — is CPU-side graph rebuilding that produces the identical graph
every single token.**

## Why this is the most valuable thread

- It is bigger than anything else found: 1.32x vs Phase C's combined ~1.10x.
- It is **context-independent**. The graph is rebuilt per token regardless of
  prompt length, so the same ~4 ms is paid at 30k and 80k. Absolute savings
  transfer directly to the long-context rows, where they matter most:
  30k ~50 -> ~65 tok/s, 80k 35.7 -> ~43 tok/s on the same arithmetic.
- Two routes to it, one already half-built:
  1. **Compiled decode** — already exists, already removes ~4 ms/token. See the
     separate A2 analysis note: it was closed on 32-token rows, but breakeven
     against its promote+trace tax is 55-159 tokens and Hermes coding turns are
     far longer. **This is the cheapest path and needs no new code.**
  2. **Reduce Swift's per-op construction cost.** Likely suspects, all
     unverified: `@ModuleInfo(key:)` string-keyed property-wrapper lookups on
     every access, ARC retain/release traffic on `MLXArray` wrappers, protocol/
     existential dispatch through `Module`/`UnaryLayer`, and the per-layer
     `NativeMTPPhaseDiagnostics.enabled` and
     `JangPressCanonicalExpertAdvisor.shared.observe` calls (both cheap
     individually, both on every layer of every token).

## Next step

Instrument the Swift side to count actual ops per decode step and attribute the
4.693 ms. `MLXPRESS_GENERATION_PROFILE` already emits the stage totals; what is
missing is op-level attribution inside `model_forward`. Until that exists, the
**1.32x figure is an upper bound derived from an estimate** — but the existence
of a multi-ms rebuild cost is established by the compiled-decode delta and is
not in doubt.

Practical ordering consequence: **RUNBOOK C1b (compiled decode at realistic
generation length) is now the highest-value single experiment in the whole
plan**, ahead of C1. It attacks this directly, needs no code change, and its
only real risk is the silent-corruption correctness gate that C1 already
requires.

#proj/mei
