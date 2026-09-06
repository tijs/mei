# Offline decode-step budget, speculative-decoding economics, and compile ceiling (2026-09-06)

Three further offline measurements, all without loading the 20 GB model. These
substantially answer P0 (the uncharacterised 69% of the decode step) and settle
the DFlash/DSpark/Eagle3 question with numbers rather than intuition.

Tools: `tools/decode_step_model.py`, `tools/moe_multitoken_bench.py`,
`tools/compile_ceiling_bench.py` (worktree `research/hybrid-superopt`).
MLX 0.32.0; vmlx vendors 0.31.1, so ratios travel, absolutes are indicative.

## 1. Decode-step budget — the model reproduces 78% of the real step

All quantized projections at the exact staged shapes: 30 GDN layers, 10
full-attention layers, 40 MoE blocks, 40 shared experts, lm_head. NOT modelled:
the GDN recurrent scan, SDPA over the KV cache, norms, sampling.

| component | ms/token | share of real 18.18 ms |
|---|---|---|
| MoE routed x40 | 5.63 | 31% |
| GDN projections x30 | 4.23 | 23% |
| shared expert x40 | 1.61 | 9% |
| lm_head | 1.35 | 7% |
| full-attn projections x10 | 1.32 | 7% |
| **modelled total** | **14.14** | **78%** |
| unmodelled remainder | 4.04 | 22% |

So the earlier F4/F5 work targeted the largest single component (MoE, 31%), and
GDN projections are second at 23%. **The 4.04 ms remainder is the only part still
genuinely dark** — and it is where the GDN recurrent scan and SDPA live, so it is
also where long-context behaviour comes from. A real profile run (P0) is still
worth doing, but its scope is now one quarter of the step, not two thirds.

## 2. Speculative decoding: measured, and it does NOT pay here

Verifying T tokens in one forward, realistic per-token routing (each token
independently picks its own top-8 of 256):

| tokens/forward | ms/forward | ms/token | per-token speedup | distinct experts touched |
|---|---|---|---|---|
| 1 | 5.635 | 5.635 | 1.00x | 8 |
| 2 | 9.154 | 4.577 | 1.23x | 16 |
| 4 | 15.509 | 3.877 | 1.45x | 31 |
| 8 | 28.215 | 3.527 | **1.60x** | 58 |

And across the whole modelled step, not just MoE: total scales **5.00x for 8x
the tokens**, so the per-token speedup caps at **1.60x at T=8**.

**This confirms the llama.cpp/GGUF experience quantitatively, and explains it.**
The premise of speculative decoding is that the model is memory-bandwidth-bound
at T=1, so extra tokens ride along nearly free. This family is **not**
bandwidth-bound (finding F2: 3.8x below the ceiling) — it is kernel/dispatch
bound, so extra tokens buy real arithmetic that nothing was idle-waiting on.
Nothing in the step is flat in T; even lm_head scales 7.63x, and MoE additionally
touches ~7x more distinct experts at T=8.

Ceiling arithmetic: 1.60x is with **100% draft acceptance and zero draft cost**.
A realistic Eagle3-style scheme at ~65% acceptance with a draft head costing
10-20% of a full forward nets out around **1.1x or worse** — the same "cost
exceeds the win" result already seen on llama.cpp, for the same underlying
reason. **Recommendation: do not pursue DFlash/DSpark/Eagle3/MTP on this model
class.** The one caveat worth preserving: at long context SDPA over a large KV
cache is closer to flat in T, so the economics improve somewhat at 30k+. If it
is ever revisited, revisit it there, not at short context.

## 3. Compile ceiling — expect ~1.11x, not the upstream +45-70%

Eager vs one `mx.compile`d region per component, same shapes:

| component | eager ms | compiled ms | speedup |
|---|---|---|---|
| GDN proj x30 | 4.056 | 3.686 | 1.10x |
| full-attn proj x10 | 1.382 | 1.256 | 1.10x |
| MoE routed x40 | 5.704 | 5.176 | 1.10x |
| shared expert x40 | 1.619 | 1.230 | 1.32x |
| lm_head | 1.348 | 1.372 | 0.98x |
| **total modelled** | **14.109** | **12.719** | **1.11x** |

Holding the 4.07 ms unmodelled remainder constant: 18.18 -> 16.79 ms, i.e.
**55.0 -> 59.6 tok/s (+8%)**. If the remainder compiled equally well, ~61 tok/s.

**This recalibrates P1 sharply.** The upstream `HardwareInfo.swift` claim of
+45-70% was measured on gemma-4-e2b and qwen — small dense models where
elementwise ops are a large share of the step. On a 35B MoE the step is
dominated by quantized matmuls, which compile does not speed up. Expect **~+10%,
not +45-70%.**

P1 is still worth running (it is free, no code change) but it is **no longer the
biggest prize**, and the P0-P6 plan's ordering rationale should be updated
accordingly. Caveat kept honest: this proxy compiles only the projection graph;
vmlx compiles the true decode step including norms, rope, the GDN scan and
sampling, which is exactly the elementwise-dense 22% remainder. The real number
could land higher than 1.11x. It is unlikely to land near 1.5x.

## 4. Where the 3.8x headroom actually is: small-M quantized matmul kernels

Achieved memory bandwidth per component (bytes of weights read / measured time):

| component | MiB/token | ms | achieved GB/s | % of 400 GB/s spec |
|---|---|---|---|---|
| **lm_head** (one big matmul) | 272.8 | 1.348 | **212.2** | **53%** |
| GDN projections x30 | 544.0 | 4.056 | 140.6 | 35% |
| full-attn projections x10 | 146.3 | 1.382 | 111.0 | 28% |
| MoE routed x40 (gather_qmm) | 540.0 | 5.704 | 99.3 | 25% |
| shared expert x40 | 67.5 | 1.619 | 43.7 | 11% |
| **whole modelled step** | 1570.6 | 14.109 | **116.7** | **29%** |

This is the cleanest statement of the problem yet. **A single large quantized
matmul (`lm_head`) reaches 212 GB/s on this hardware. The same hardware, same
dtype, same quantization, running many small and gathered matmuls, reaches
44-141 GB/s.** The whole step averages 29% of spec.

So the 3.8x headroom from F2 is not hiding in graph structure, config, or
scheduling — **it is per-kernel efficiency of quantized matmul at batch size 1**,
worst for the gathered MoE path and the small shared-expert projections. That is
MLX/vmlx kernel work, not something Mei's configuration surface can reach.

Practical consequences:
- Every config-level lever (P1 compile ~1.11x, F4+F5 ~1.05x) is a few percent.
  Stacked and generous, they are worth perhaps **1.15-1.2x: 55 -> ~63-66 tok/s**.
  That is the realistic near-term ceiling. Anything beyond it needs kernel work.
- The one component already near the achievable ceiling is `lm_head`, which
  means shrinking it (vocab/precision) buys little; leave it alone.
- The **shared expert at 11% of spec is the worst offender per byte** — 40 tiny
  2048x512 matmuls per token. It is also the component compile helped most
  (1.32x). If any single kernel-level fix is worth requesting upstream, batching
  or fusing the shared expert across layers is the highest ratio.
- A genuine kernel-level lever worth testing when a GPU window opens: whether
  `gather_qmm` with `sorted_indices=True` (plus the `gatherSort`/`scatterUnsort`
  path already in `SwitchLayers.swift`) improves the 99 GB/s MoE figure. The
  production path passes `sortedIndices: false` at decode because
  `doSort = indices.size >= 64` is false at top-8. Untested at decode shapes.

#proj/mei
