# D3 ANSWERED: MLXPress reclaims nothing because weights are clean file-backed pages (143 MB footprint for a 19.5 GB model) — overturns the 93%-expert-bank memory framing (2026-09-07)

# D3 ANSWERED — and it overturns my own "93% is the expert bank" memory framing (2026-09-07)

Completion of the D3 experiment. The one-argument Mei change works and the cold
tier engages, but the lever has **nothing to reclaim**, for a reason that also
invalidates part of the earlier memory analysis.

## The measurement, done with the right instrument

Process RSS on Ornith (not MLX's allocator counter, which cannot see page
residency), across `MLXPRESS` and `MLXPRESS_FORCE_MODE`:

| MLXPRESS | forceMode | RSS after load | phys_footprint | RSS after 30k | after 20 s quiesce |
|---|---|---|---|---|---|
| 0 | soft | 222 MB | **143 MB** | 1.10 GB | 1.09 GB |
| 0 | force | 242 MB | 156 MB | 1.13 GB | 1.09 GB |
| 95 | soft | 240 MB | 150 MB | 1.09 GB | 1.09 GB |
| 95 | force | 239 MB | 149 MB | 1.13 GB | 1.13 GB |

**Identical within noise across every configuration.** `MLXPRESS=95` with
`force` (`msync(MS_INVALIDATE)`, the most aggressive setting available) reclaims
nothing relative to `MLXPRESS=0`.

## Why: there is nothing to reclaim

A 19.5 GB model shows **143–156 MB of phys_footprint** and ~1.1 GB RSS after a
30k-token request. The weights are **clean, file-backed mmap pages**. macOS does
not charge clean file-backed pages to a process's physical footprint, and can
evict them at any time under pressure **with no `madvise` involvement at all**.

MLXPress's cold tier exists to make routed-expert pages evictable. On this
configuration — `useMmapSafetensors` plus the aligned repack — they already are.
The lever is **redundant here**, not broken: it correctly indexed 10,240 experts
across 40 layers and advised 14.2 GiB cold at pct=95; the kernel simply had
nothing new to do.

## This corrects finding F6 / P5 / the STATE OF PLAY memory framing

I previously wrote, repeatedly, that "93% of the model (16.9 of 18.17 GiB) is the
routed expert bank, so any meaningful memory reduction has to come from there."
That is true of **weight bytes** and false of **resident cost**. The expert bank
is 93% of the weights and ~0% of the process's physical footprint.

So the D-phase premise was wrong: **D2 (mixed 4/8-bit) and D3 (expert residency)
target bytes that are not actually charged to memory.** Lowering routed-expert
precision would shrink the file and the page-cache working set, but it would not
move the number that constrains this machine.

What *does* constrain it is everything that is **not** clean file-backed:
- Metal/GPU command and compute buffers
- the KV cache and the SSM companion state (Ornith 30k: MLX-allocator active
  goes 18.79 -> 21.77 GB, peak 23.27 GB — that delta is real allocation)
- runtime-materialised duplicate banks, which is exactly why the +12.2 GiB fused
  gate+up bank was fatal and the 407 MiB GDN fusion is not

That also reframes the Ornith Q5_K_M OOM: it was not "the weights don't fit" but
"weights plus *dirty* runtime buffers exceeded the GPU working set".

**Practical consequence: stop treating the expert bank as the memory target.**
Context headroom is limited by KV/SSM state and compute buffers, so levers that
shrink those (KV quantization, window policies, smaller prefill buffers) are the
ones with headroom — not quantization of routed experts.

## What D3 delivered anyway

1. **The change is correct and worth keeping.** `jangPress: .default` at
   `Engine.swift:90` makes `MLXPRESS=N` live for the first time; without it the
   env var was silently inert on every Mei build to date. Anyone who later needs
   the tier (a bigger model, a machine where weights are *not* mmap-clean) now
   has it.
2. **No throughput cost.** 64.0–67.4 tok/s short and 40.4–40.6 at 30k across all
   settings — enabling it is free.
3. **A methodological correction worth keeping**: `mei_memory_active_bytes` /
   `peak` come from MLX's allocator and are **not** OS memory pressure. Every
   "peak 25.73 GB @ 65k" style figure in this project's notes is an allocator
   number. For anything about host memory, measure RSS or phys_footprint.

## Recommendation

Keep the branch and the change; do **not** enable `MLXPRESS=N` by default (no
benefit here, and `.default`'s auto-heuristic already declines to act unless the
bundle exceeds 50% of physical memory). Close D3 as **answered: redundant on
this configuration**, and close D2 as **mis-targeted** in its memory rationale —
its quality question remains open and legitimate, but it should not be sold as a
memory lever.

#proj/mei
