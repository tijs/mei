# CORRECTION: sorted-index and group-size 'findings' were contention artifacts; B1 confirms 0.83 GiB vision saving (2026-09-06)

# CORRECTION: two apparent kernel-level findings were contention artifacts; plus B1's memory result (2026-09-06)

Recording a near-miss honestly, because both numbers looked like real findings
and neither survived a clean re-run.

## What happened

I ran a kernel-knob microbenchmark (sorted `gather_qmm` indices, group-size
sweep) without checking whether the machine was busy. The other agent had
started its B1 leg mid-run. The production baseline in that run read
**7.772 ms** against a clean **5.638 ms** measured earlier — 38% inflated — which
is what gave the contamination away.

## Artifact 1 — "sorted expert indices are 1.35x faster". They are not.

| leg | contended run | clean re-run |
|---|---|---|
| unsorted idx, `sorted_indices=False` (**production**) | 7.772 | **5.784** |
| sorted idx, `sorted_indices=True` | 7.678 (1.01x) | 5.604 (1.03x) |
| sorted idx, `sorted_indices=False` | 5.756 (**1.35x**) | 5.572 (**1.04x**) |

The 1.35x was **entirely contention**. The real effect is ~1.04x, inside noise.

The RUNBOOK listed "does the sorted `gather_qmm` path help at decode shapes?" as
a cheap unmeasured lever. **Answer: no.** Production already passes
`sortedIndices: false` at top-8 and that is the right call. Close the item.

Caveat: this leg still has a design confound — it compares one permutation to
its own sorted version, conflating "sortedness" with "which experts". A paired
N-draw version (`tools/moe_index_locality_bench.py`) is queued behind the GPU
guard. Given the clean delta is 1.04x, it is very unlikely to change the verdict.

## Artifact 2 — "group_size 32 is 2.6x slower". Also not real.

| group_size | scale+bias, 40 layers | contended | clean |
|---|---|---|---|
| 32 | 3.75 GiB | 15.476 | **5.641** |
| 64 (**production**) | 1.88 GiB | 5.915 | **5.602** |
| 128 | 0.94 GiB | 5.653 (1.05x) | **6.158 (0.91x)** |

Both the dramatic g32 penalty *and* the apparent g128 win were artifacts, and
g128 actually **reverses sign** on the clean run — 9% slower, not 5% faster.

Clean verdict: **g64 is essentially optimal.** g32 matches it on speed but
doubles scale/bias volume (+1.87 GiB) for nothing; g128 halves that volume but
costs 9% speed. Unlike bit depth — where 2/4/8 hit fast kernels and 3/5/6 do not
— group size offers no lever here. **Keep g64; close the item.**

## Process lesson, worth keeping

Both artifacts pointed the *same* direction as a plausible story (locality helps;
smaller groups thrash), which is exactly when a bad number is most likely to be
believed. The tell was not the result, it was the **control**: the production
baseline had moved 38% from a previously measured value. Every microbenchmark in
this project should carry a known-value control and refuse to run under
contention.

Both scripts now hard-refuse to start if `llama-server`, `/release/mei`,
`probe_mei` or `mlx_lm` is live, and print a post-run contention check.

## B1 memory result — clean confirmation

From the other agent's B1 legs (`B1-qwen36-textonly-vs-stock-20260906T171156Z`):

| leg | memory after load |
|---|---|
| stock Qwen 3.6 | 20,444,313,020 B (**20.44 GB**) |
| **Qwen 3.6 text-only** | 19,551,131,372 B (**19.55 GB**) |
| Ornith (A1, reference) | 19,551,131,108 B (19.55 GB) |

Delta stock -> text-only: **893,181,648 B = 0.83 GiB**, exactly the vision tower.
And the text-only build lands within **264 bytes** of Ornith's footprint —
expected, since after the strip the two have identical tensor counts and shapes.
The vision-strip hypothesis is confirmed on real runtime evidence.

### Open question B1 did NOT answer

Neither B1 leg set `VMLX_MODEL_FACTORY_TRACE=1`, so **no `[ModelFactory]` lines
were emitted** and factory selection for the text-only build is inferred, not
observed. Two readings both fit 19.55 GB:

- **(a)** VLM factory declines on the missing `preprocessor_config.json` (which
  is what A1 showed for Ornith, and which the strip removes) -> LLM path ->
  text-only **loses** `compileSeparatedDecode: true`.
- **(b)** VLM factory still succeeds, finds no vision tensors to load -> VLM path
  -> text-only keeps the compiled routed-MoE region **and** the memory saving.

(a) is much more likely given A1's Ornith evidence, but the difference decides
whether C2 is Ornith-only or applies to both. **Cheapest possible resolution:
one server start on the text-only build with `VMLX_MODEL_FACTORY_TRACE=1`** —
about two minutes. Worth doing before C2 is scoped.

## RESOLVED — clean paired re-run (3 reps, contention-verified before and after)

`tools/moe_index_locality_bench.py`, 6 independent top-8 draws per rep, each
timed unsorted **and** sorted (identical expert set, order only):

| rep | unsorted mean | sorted mean | ratio |
|---|---|---|---|
| 3 (representative) | 5.543 ms | 5.533 ms | **1.002x** |

Per-draw ratios: 1.02, 1.00, 1.00, 1.00, 1.00, 0.99. Pure noise (sd 0.058 /
0.016). **Expert index ordering has no effect.** The residual 1.03x seen in the
unpaired knob bench is index-*set* variance between different permutations, not
sortedness — exactly the confound the paired design was built to remove.
Sorted-indices item: **closed, negative.**

Group size, second clean rep: g32 5.664, g64 5.672, g128 5.560 — all within 2%.
Note g128 read 6.158 ms (0.91x) in the first clean rep and 5.560 ms (1.02x)
here, a 10% swing **between clean runs**, so even that is noise rather than a
small effect. Group size: **closed, no lever.** Keep g64.

## NEW and more important: the locality references prove the MoE gather is not bandwidth-bound

Same top-8 shape, varying only which experts are addressed:

| index pattern | distinct experts read | ms |
|---|---|---|
| contiguous `0..7` | 8 | 5.531 / 5.554 |
| stride-32 spread | 8 | 5.558 / 5.511 |
| **all-same expert `[0]*8`** | **1** | **5.566 / 5.528** |

Addressing **one** expert eight times instead of **eight distinct** experts —
one eighth of the routed weight bytes — costs **exactly the same time**.

This is a direct controlled falsification of the bandwidth hypothesis for this
kernel, replacing the inference drawn in F2 from the 3.8x headroom figure. The
`gather_qmm` cost at decode shapes is insensitive to how much expert weight it
actually touches, so it is bound by per-kernel overhead, not by DRAM traffic.

Two consequences:

- It independently confirms the STATE OF PLAY conclusion that the remaining
  headroom is **small-M quantized matmul kernel efficiency**, and does so by
  experiment rather than by arithmetic on achieved GB/s.
- It also means **no routing-side trick can help decode speed**. Expert
  affinity, routing skew exploitation, hot-expert pinning, cache-friendly expert
  layout — none of it can pay off in the speed dimension, because touching 1
  expert costs the same as touching 8. That does **not** touch RUNBOOK D1/D3,
  which are memory levers (keeping cold experts non-resident), but it does close
  off the "exploit routing skew for throughput" framing entirely. D1 remains
  worth doing for D3's sake; it is not a speed investigation.

#proj/mei
