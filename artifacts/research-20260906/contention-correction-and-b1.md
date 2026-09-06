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

#proj/mei
