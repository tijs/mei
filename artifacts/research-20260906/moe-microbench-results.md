# qwen3_5_moe decode microbenchmarks — measured 2026-09-06 (Sulaco M1 Max, 32 GB)

No model loaded; the routed-expert bank is rebuilt at the real staged shapes
with random weights (shapes/dtypes match, numerics irrelevant for kernel
timing). Machine was idle of inference workloads (`ps` checked before each run,
0 live inference processes). MLX **0.32.0** via `~/.cocore/python`; note the
Mei/vmlx stack vendors mlx **0.31.1**, so treat absolute ms as indicative and
the RATIOS as the result.

Shapes: E=256 experts, D=2048 hidden, H=512 moe_intermediate, top_k=8,
group_size=64, 40 layers chained into ONE graph with a single `mx.eval`
(matching how a real decode step is built lazily and evaluated once).

Scripts: `tools/moe_chain_bench.py`, `tools/moe_bitdepth_bench.py`.

## Methodology correction (recorded so it is not repeated)

The first version (`tools/moe_dispatch_bench.py`) called `mx.eval()` **per
layer**, paying 40 CPU<->GPU round-trips a real decode step never pays. It
reported 13.94 ms for the MoE blocks and showed `mx.compile` as a 0.79x
*regression*. Both were artifacts of the per-layer sync. Chaining all 40 layers
into one graph (the honest simulation) gives 5.64 ms and flips compile to a
1.11x *win*. The per-layer-eval script is kept only as the falsification record.

## Result 1 — F4 (compiled region) and F5 (pre-fused gate+up repack), 3 repeats

| leg | r1 | r2 | r3 | mean ms/token | vs current |
|---|---|---|---|---|---|
| split 3x `gather_qmm` (**current production path**) | 5.611 | 5.667 | 5.636 | **5.638** | 1.00x |
| pre-fused gate+up bank (F5 repack) | 5.097 | 5.110 | 5.120 | **5.109** | **1.104x** |
| split + per-layer `mx.compile` (F4) | 5.018 | 5.112 | 5.088 | **5.073** | **1.111x** |
| pre-fused + compiled (F4+F5) | 4.850 | 4.840 | 4.834 | **4.841** | **1.165x** |

The two levers are **largely independent and stack**: +10.4% and +11.1%
separately, +16.5% together.

Translating to the whole decode step (measured full-model short decode is
18.18 ms/token = 55.0 tok/s): the MoE routed blocks are **5.64 ms = 31%** of the
step. Saving 0.797 ms takes the step to ~17.38 ms:

> **F4+F5 project to ~57.5 tok/s, i.e. +4.6% end-to-end.** Real, cheap, and
> worth taking — but NOT the +45-70% that the upstream `HardwareInfo` comment
> claims for full compiled decode. Those are different levers; do not conflate.

## Result 2 — routed-expert bit depth barely moves decode speed, 3 repeats

| bits | expert bank, 40 layers | active MiB/token | mean ms/token | vs 4-bit |
|---|---|---|---|---|
| 2 | 9.38 GiB | 300 | **5.350** | 1.06x |
| 3 | 13.12 GiB | 420 | **5.936** | 0.95x |
| 4 (**production**) | 16.88 GiB | 540 | **5.655** | 1.00x |
| 5 | 20.62 GiB | 660 | **6.296** | 0.90x |
| 6 | 24.38 GiB | 780 | **6.791** | 0.83x |
| 8 | 31.88 GiB | 1020 | **5.376** | 1.05x |

(individual runs within +/-0.1 ms of each mean; highly reproducible)

Two conclusions, both actionable:

**(a) The bandwidth-bound hypothesis is dead for this block.** Active weight
traffic spans **3.4x** (300 -> 1020 MiB/token) while time spans only **1.27x**,
and not even monotonically. If decode were bandwidth-bound, 8-bit would be
~3.4x slower than 2-bit; it is in fact the *joint fastest*. Bit depth on this
architecture is a **memory-and-quality dial, not a speed dial** — the inverse of
the "lower bits = faster" heuristic that (correctly, for dense GGUF models)
drove this project's earlier quant decisions.

**(b) Only 2 / 4 / 8 bits get the fast kernels.** Ordering is
8 ~= 2 < 4 << 3 < 5 < 6. The power-of-two widths hit specialized `gather_qmm`
paths; 3/5/6-bit fall back to slower generic ones. **5-bit and 6-bit are the
worst possible choices here — they cost BOTH speed and memory.** Any future
"one quant step up" attempt on this family must go 4 -> 8 selectively, never
4 -> 5. (Compare: the sibling llama.cpp/GGUF track reasonably tried Q5_K_M for
Ornith and it OOM'd at 23.6 GB — on the MLX side 5-bit would additionally have
been ~10% *slower* than 4-bit.)

Practical recipe this implies: keep routed experts at **4-bit**, and spend any
memory headroom promoting selected tensors to **8-bit** (both are fast tiers).
The obvious first candidate is the **shared expert** — it is on the critical
path for every single token yet is only 1.69 MiB/layer (68 MiB across the whole
model), so 4 -> 8 bit costs ~68 MiB total. The checkpoint author already made
exactly this judgement for the router: `mlp.gate` and `mlp.shared_expert_gate`
are held at 8-bit in `config.json` while everything else is 4-bit.

## Caveats

- One layer's expert bank (432 MiB) is reused for all 40 chained calls. M1 Max
  L2 is 48 MB so reads still miss to DRAM, but true per-layer traffic is
  understated -> the MoE share of the step is a **lower bound**.
- Random weights: expert routing is fixed to indices 0..7 rather than
  data-dependent. Real routing scatters across all 256 experts, which can only
  make the gather *less* cache-friendly, again making these a lower bound.
- MLX 0.32.0 here vs 0.31.1 vendored in vmlx. Kernel selection for 3/5/6-bit
  could differ between versions; re-check the bit-depth ordering against the
  pinned vmlx before acting on it.
- These measure the MoE block ONLY. ~12.5 ms of the 18.18 ms step is elsewhere
  (30 GDN linear-attention layers, 10 full-attention layers, lm_head, norms,
  sampling) and is **not** characterized here. That is the top open question.
