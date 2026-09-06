# The GDN input-projection fusion: precedent for C3, and a second undocumented duplicate bank (2026-09-06)

Static analysis of `Libraries/MLXLLM/Models/Qwen35.swift:255-326` in the pinned
fork, prompted by the A1/A2 server logs showing a line that does not appear in
any 55.0-tok/s-era artifact:

    [Qwen35] fused_gdn_decode_input_projections=active groups=[4]

No GPU used.

## What it does

`Qwen35GatedDeltaNet` lazily concatenates its four input projections —
`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` — into runs sharing a
quantization scheme, then issues **one** `quantized_matmul` instead of four and
splits the result. `groups=[4]` in the log means all four fused into a single
group on our checkpoints. Landed upstream as `e1f64fec` *"Decode perf: GDN
grouped input-projection fusion + MLA bf16 decode SDPA (#401)"*.

**This is very likely a large part of the baseline moving 55.0 -> 61.70 tok/s**,
alongside whatever else changed between pins. Worth attributing properly rather
than assuming, but it is the most visible new decode-path change in the logs.

## Finding 1 — it is the same runtime-concat pattern, with the same cost

`flushRun()` does `concatenated(...)` on weight/scales/biases and `MLX.eval`s
them, retaining the result for process lifetime. That is **structurally
identical** to `SwitchGLU.ensureFusedGateUp()`, the one that allocated a
permanent **+12.2 GiB** duplicate expert bank and had to be disabled via
`VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`.

Sized from the staged checkpoint shapes:

| | per GDN layer | x30 layers |
|---|---|---|
| weights (U32) | 12.06 MiB | 362 MiB |
| scales + biases (bf16, g64) | 1.51 MiB | 45 MiB |
| **total duplicate** | **13.57 MiB** | **407 MiB (0.40 GiB)** |

407 MiB is tolerable where 12.2 GiB was fatal, so this is **not** a bug and
should not be disabled — the decode win is real and the cost is affordable. But
two things are worth recording:

- **It has no cache-limit env guard.** The SwitchGLU fusion gained
  `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES` / `BENCH_NO_FUSED_GATE_UP` only after
  it blew the memory budget on a large MoE. This one has no equivalent escape
  hatch, so a future architecture with much larger GDN projections would hit the
  same class of failure with no way to opt out. Cheap defensive ask upstream.
- **It is ~0.40 GiB of the ~20.2 GB post-load footprint** — real, and currently
  unaccounted for in this project's memory arithmetic. Post-load active for
  Ornith reads 19.55 GB in the A1 log; that figure *includes* this bank, since
  the fusion happens on first forward, not at load. Peak-memory budgets should
  assume it.

## Finding 2 — this is in-tree precedent for RUNBOOK C3, and C3 should be broadened

C3 proposes repacking the MoE `gate_proj`+`up_proj` **pre-concatenated on disk**,
so `ensureFusedGateUp` becomes a zero-copy mmap view instead of a runtime
allocation — decode fusion at zero memory cost.

The GDN fusion proves the *runtime* half of that idea already ships and helps.
It also means the same disk-side treatment applies to a second site:

> **C3 should repack BOTH** the MoE gate+up bank **and** the GDN
> `in_proj_{qkv,z,b,a}` group. The GDN half reclaims the 407 MiB duplicate while
> keeping the measured decode win; the MoE half enables a fusion currently
> switched off entirely for memory reasons.

Combined, a single repack would deliver:
- MoE: +10.4% on the routed block (measured offline), memory-neutral instead of
  +12.2 GiB
- GDN: unchanged speed, **-407 MiB** resident

That materially improves C3's standing. It was ranked last in Phase C on a
~+1.5% marginal speed contribution; as a **memory** lever that also unlocks the
MoE fusion, it belongs alongside Phase D, which is where the binding constraint
actually is.

## Caveats

- The 407 MiB is computed from checkpoint shapes, not measured with the
  allocator tracer. `OSAURUS_MLX_MALLOC_TRACE=1` (the same instrument that
  root-caused the 12.2 GiB SwitchGLU bank) would confirm it directly and is
  cheap to add to any future GPU window.
- Attribution of 55.0 -> 61.70 to this commit is **plausible, not established**.
  Other changes landed between pins. A clean attribution would need the old pin
  re-measured, which is probably not worth a GPU window on its own.

#proj/mei
