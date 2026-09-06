# Hybrid-MoE super-optimization research — findings (2026-09-06)

Scope: `qwen3_5_moe` hybrid MoE family on Sulaco (M1 Max, 32 GB). Targets:
Ornith-1.5-35B-A3B (primary), Qwen3.6-35B-A3B (staged, gated), Nemotron-3.5-
Lightning-30B-A3B (not yet staged). CPU-side static analysis + checkpoint
arithmetic only — NO model was loaded and no GPU work was run (benchmark
machine shared with the Hermes benchmark owner). Work done in git worktree
`/Users/tijs/projects/mei-opt-research` on branch `research/hybrid-superopt`
(mei main untouched).

## F1 — Ornith 1.5 and Qwen 3.6 are the SAME architecture, parameter for parameter

Read from the two staged `config.json` files
(`~/.local/share/local-model-bench/mei-models/{Ornith-1.5-35B-A3B-MLX-4bit-aligned,Qwen3.6-35B-A3B-4bit}`):

| field | value (both) |
|---|---|
| model_type | qwen3_5_moe / qwen3_5_moe_text |
| num_hidden_layers | 40 |
| hidden_size | 2048 |
| num_experts / num_experts_per_tok | 256 / 8 |
| moe_intermediate_size | 512 |
| shared_expert_intermediate_size | 512 |
| full_attention_interval | 4 (-> 10 full-attn, 30 GDN linear-attn) |
| num_attention_heads / num_key_value_heads / head_dim | 16 / 2 / 256 |
| linear_num_value_heads / key_heads / key_head_dim / conv_kernel | 32 / 16 / 128 / 4 |
| vocab_size | 248320 |

Every field checked is identical. **Consequence: every runtime optimization,
profile default, and repack recipe validated on Ornith transfers to Qwen 3.6
with no re-derivation** — only a correctness re-gate is needed, not a new
optimization search. Both load through the same code path (see F4).

Nemotron-3.5-Lightning-30B-A3B is a DIFFERENT architecture (vmlx
`Libraries/MLXLLM/Models/NemotronH.swift`, which has its own
`JangPressCanonicalExpertAdvisor.observe` MoE site at :897). It is also a
hybrid MoE, so the *class* of findings below should apply, but every constant
must be re-derived. Do not assume transfer.

## F2 — Decode on this family is DISPATCH-bound, not bandwidth-bound (3.8x headroom)

Computed exactly from the staged safetensors headers (`tools/bandwidth_budget.py`
in the worktree), not estimated:

- Total tensor bytes: **18.17 GiB** / 1757 tensors.
- Active weight traffic per decode token (top-8 of 256 routed + all dense):
  **1592 MiB**.

Breakdown of that 1592 MiB per token:

| component | MiB/token | share |
|---|---|---|
| GDN linear-attn projections (30 layers x 18.13) | 544 | 34% |
| MoE routed experts (40 layers x 432 x 8/256) | 540 | 34% |
| lm_head | 273 | 17% |
| full attention (10 layers x 14.63) | 146 | 9% |
| shared expert (40 x 1.69) | 68 | 4% |
| router gates | 21 | 1% |

Bandwidth ceiling on M1 Max:

| assumed BW | ms/token | tok/s ceiling |
|---|---|---|
| 300 GB/s | 5.57 | 180 |
| 350 GB/s | 4.77 | 210 |
| 400 GB/s (spec) | 4.17 | 240 |

**Measured short decode: 55.0 tok/s = 18.18 ms/token — 3.8x above the
350 GB/s ceiling.**

Contrast with the dense comparator already measured in this project:
Qwen3.8-27B 4-bit runs at **89% of its bandwidth floor** (15.66 t/s vs a
~19-20 t/s floor) — i.e. at the wall, which is why its 30 t/s target was
correctly closed as a hardware ceiling (plan `0b87b76a` acceptance record).

**These hybrid MoE models are in a completely different regime from the dense
models this project benchmarked.** The dense conclusion ("we are at the memory
wall, optimization is exhausted") does NOT transfer. Corroborating profile
evidence already in the repo:
`artifacts/ornith-35B-compiled-decode-gate-20260902.md` records eager
`decode.model_forward avg ~6.2 ms/token` — that is *lazy graph construction on
the CPU*, a third of the entire 18.18 ms step budget spent before any GPU work
is submitted. Gemma4's matrix shows the same shape even more starkly
(`model_forward` 1.84-1.89 ms vs `decode.async_eval_submit` 134-140 ms).

Estimated per-token Metal dispatch count for this model: ~45-55 ops/layer
x 40 layers = **1800-2200 dispatches/token**. At 18.18 ms that is ~8-10 us per
dispatch — right at the Metal small-kernel dispatch floor. This is the
mechanism.

### F2 corollary — bit depth is nearly FREE in the speed dimension here

Because decode is 3.8x off the bandwidth floor, changing routed-expert bit
depth moves memory and quality but should barely move tok/s. This **inverts
the standard "lower bits = faster" heuristic** that (correctly) drove the
GGUF-side quant decisions in local-model-bench. For this family on this
hardware, bit depth is a memory-vs-quality dial, not a speed dial. Both
directions are worth testing:
- 4 -> 3 bit on routed experts: saves ~4.2 GiB resident, ~0 speed cost.
- Raising precision where quality matters (shared expert is on the critical
  path for EVERY token and is only 68 MiB of the whole model) is almost free:
  4 -> 8 bit on shared_expert costs ~68 MiB resident total.

Note the checkpoint author already reached this conclusion partially: the
router `mlp.gate` and `mlp.shared_expert_gate` are held at **8-bit** while
everything else is 4-bit (visible in `config.json`'s per-tensor quantization
overrides). Routing precision was judged worth paying for; shared-expert
precision was not, and that is worth revisiting.

## F3 — The MoE elementwise micro-fusions are OFF by default; upstream calls this "the single biggest local-decode speedup available"

`Libraries/MLXLMCommon/HardwareInfo.swift:49` — `isCompiledDecodeSupported`
returns false unless `VMLX_ENABLE_UNSAFE_COMPILE=1` (or
`MLXPRESS_ENABLE_UNSAFE_COMPILE=1`).

It gates, among others, three helpers directly in the MoE hot path
(`Libraries/MLXLMCommon/SwitchLayers.swift:15, 45, 57`):
`safeGeluApproximate`, `compiledSwiGLU`, `compiledGeGLU` — plus
`compiledSigmoidGate` and the Qwen35 shapeless micro-fusions
(`Libraries/MLXLLM/Models/Qwen35.swift:20`). With the gate off, `silu(gate)*up`
runs as two separate Metal dispatches per MoE layer per token instead of one
fused kernel; same for the shared-expert sigmoid gate.

The upstream comment at `HardwareInfo.swift:33-45` is explicit:

> "Performance impact of disabling: SIGNIFICANT for real decode, not the
> negligible micro-fusion cost this comment previously claimed. ... Local
> decode-throughput benchmarks put the compile-ON gain in the ~+45% to +70%
> range across gemma-4-e2b and qwen ... enabling it ... is the single biggest
> local-decode speedup available."

**The documented reason it is default-off is NOT primarily the Metal JIT bug.**
Per `HardwareInfo.swift:50-57` the blocker is Osaurus issue #1173: process-local
decode corruption observed after **loading one model and switching to another
in the same process**. The suggested fix is "clearing the MLX compile cache on
model swap".

**Mei is one-model-per-server-process by explicit design** (README: "The
project's scope is deliberately small: one model per server process"). Mei
never swaps models inside a process. **The stated blocker is structurally
unreachable in Mei's architecture.**

Residual risk is the separate macOS Tahoe Metal JIT bug (MLX #3329/#3201/#3256;
this machine is macOS 26.5.2 / build 25F84, i.e. Tahoe, so it is in scope).
Empirically it did not manifest: the 2026-09-02 Ornith run with
`VMLX_ENABLE_UNSAFE_COMPILE=1` passed 12/12 probe rows on each of 3 repeats
with no crash and no divergent output.

The prior worker observed **+4-5%** on the guard-skipped eager rows in that run
and explicitly flagged it as unproven/confounded, listing "a dedicated 3x A/B
(`VMLX_ENABLE_UNSAFE_COMPILE=1` alone, compiled-decode false)" as the one open
thread. **That A/B was never run.** It is the single highest-value pending
experiment, and F2 explains why it could be much larger than +4-5%: the earlier
observation was taken on 30k long-context rows where GDN recurrent cost
dominates, not on the short-decode path where dispatch overhead is the whole
story.

## F4 — A trusted compiled MoE region exists and is DEAD CODE for Ornith/Qwen3.6

`Libraries/MLXLMCommon/SwitchLayers.swift:64-142` defines
`Qwen4ExpCompiledRoutedSwitchGLU`: it fuses **all three `gatherQuantizedMM`
calls plus silu plus multiply into ONE compiled region** per MoE layer. It is
enabled by env default (`VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE`, default "1").

It is only reached when `SwitchGLU.compileSeparatedDecode == true`
(`SwitchLayers.swift:400`). That property defaults to **false**
(`SwitchLayers.swift:248`) and is passed `true` in exactly ONE place in the
whole tree:

    Libraries/MLXVLM/Models/Qwen35.swift:2184   compileSeparatedDecode: compileDecodeRegions

That is the **VLM** twin. The LLM construction site that Ornith and Qwen 3.6
actually use —

    Libraries/MLXLLM/Models/Qwen35.swift:870
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)      // <- no compileSeparatedDecode

— omits it, so it stays false.

Load path confirmed: `Libraries/MLXLLM/LLMModelFactory.swift:53` maps
`"qwen3_5_moe"` -> `Qwen35MoEModel` (weight_format is not "mxtq" for these
checkpoints, so the JANGTQ branch is not taken), and
`Libraries/MLXLLM/Models/Qwen35MoE.swift:38` declares
`Qwen35MoEModel: Qwen35Model`, which builds `Qwen35SparseMoeBlock` from
`Libraries/MLXLLM/Models/Qwen35.swift:851`. **So this compiled region has never
executed for either of our hybrid models.**

Every structural precondition of that region is satisfied by the staged
checkpoints (verified by reading the safetensors headers directly):

| precondition (SwitchLayers.swift:82-93) | Ornith staged checkpoint |
|---|---|
| `input.dim(-2) == 1` (single-token decode) | yes at decode |
| `indices.size < 64` | 8 (top-8) |
| `input.dtype == .bfloat16` | yes (config dtype bfloat16) |
| gate/up/down same groupSize, bits, mode | all affine, g64, 4-bit |
| gate/up/down `scales.dtype == .bfloat16` | **BF16 confirmed** |
| gate/up/down biases present and `.bfloat16` | **BF16 confirmed** |

Checkpoint header evidence (`model-00001-of-00004.safetensors`, dtype counts
BF16 336 / U32 137):

    layers.1.mlp.switch_mlp.gate_proj.weight  U32  [256, 512, 256]
    layers.1.mlp.switch_mlp.gate_proj.scales  BF16 [256, 512, 32]
    layers.1.mlp.switch_mlp.gate_proj.biases  BF16 [256, 512, 32]
    layers.1.mlp.switch_mlp.down_proj.weight  U32  [256, 2048, 64]

**Critically, this lever is INDEPENDENT of F3's unsafe-compile gate.** The
region is built with `vmlxTrustedCompile` (`SwitchLayers.swift:105`), and
`Source/MLX/Transforms+Compile.swift:301-318` documents that trusted functions
"compile even when the opt-in policy (`VMLX_ENABLE_UNSAFE_COMPILE`) is not set,
so validated hot paths keep their compiled fast path in production processes."

So F4 is a **low-risk, no-env-flag, one-line** change: pass
`compileSeparatedDecode: true` at `MLXLLM/Models/Qwen35.swift:870`. It is
already exercised in production by the VLM twin on the same block shape.

Caveat to check when it is run: the region is named for Qwen4-Exp, but all its
guards are structural (dtype/bits/groupSize/shape), not identity-based. Treat
the numerical-equivalence check as mandatory, not a formality.

## F5 — `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0` buys memory by forfeiting the decode fusion; a repack could give both

Current Ornith/Qwen3.6 launch config sets this to 0. Reason (proven, artifact
`artifacts/ornith-35B-fuse-gateup-eliminated-20260902.md`): `ensureFusedGateUp()`
concatenated gate+up into a **permanent +12.2 GiB duplicate** bank (40 x 256 MiB
+ 80 x 16 MiB), which pushed the process over 32 GB and collapsed decode to
4.02 / 0.094 t/s. Setting the limit to 0 was the correct fix and is what makes
the current 55 t/s possible.

But it is a trade, not a pure win. With `fusedGateUpWeight == nil`, decode takes
the fallback branch (`SwitchLayers.swift:477-484`) and issues **three separate
`gatherQuantizedMM` dispatches** per MoE layer per token. The code's own comment
(`SwitchLayers.swift:411-421`) states the fused path "is a net win for DECODE".
We are currently paying that cost to avoid the memory duplication.

**The duplication exists only because gate_proj and up_proj are stored as
separate tensors on disk and concatenated at runtime.** A repack that writes
them already concatenated along the output axis —
`[E, 2*hidden, in_packed]`, exactly the layout `ensureFusedGateUp` builds —
would let the fused decode path be a **zero-copy mmap view** of the checkpoint
instead of a runtime allocation. Decode fusion at zero memory cost.

Mei already owns an "aligned repack" step for file-backed NoCopy loading
(the staged dir is literally `...-MLX-4bit-aligned`), so the tooling and the
provenance discipline for producing a derived checkpoint already exist. This is
the most concrete "custom repack for our use case" lever available, and unlike
F3/F4 it is a *checkpoint* change, so it needs no upstream vmlx patch to be
adopted — only loader support for reading a pre-fused tensor.

## F6 — Memory is 93% routed-expert bank; that is the only place a real memory win can come from

Per-layer: routed expert bank **432 MiB** vs everything else in the layer
~20 MiB. Across 40 layers the routed bank is **16.9 GiB of the 18.17 GiB
total (93%)**. lm_head + embed_tokens together are ~605 MiB (3.3%).

So any meaningful reduction of the ~20.3 GB post-load / 25.7 GB peak @65k
footprint has to come from the expert bank. Candidate levers, in the order I
would try them:

1. **Lower bits on routed experts only** (4 -> 3): ~-4.2 GiB resident, ~0 speed
   cost per F2. Quality risk is the open question and is exactly what
   local-model-bench's suite is built to answer.
2. **Expert page residency.** 256 experts, top-8 per token. The aligned repack
   already loads file-backed NoCopy, so expert pages are file-backed and
   evictable by the OS. vmlx has an unfinished mechanism for exactly this:
   `JangPressCanonicalExpertAdvisor` (`Libraries/MLXLMCommon/Cache/`) resolves
   an `advise_experts` symbol and its own comment says "JangPress's production
   win is canonical mmap residency, not per-token readback". It is default-off
   and its per-token index readback is documented as a speed tradeoff. Worth a
   dedicated look: if expert routing is skewed, keeping only hot experts
   resident could cut the effective working set well below 16.9 GiB and make a
   *higher-precision* quant fit that otherwise would not.
3. **Role-aware mixed precision** (the APEX idea, ported to MLX): first/last
   layers and shared expert higher, middle routed experts lower. Per F2 the
   speed cost is ~0; this is purely a memory/quality search.

## What was NOT done and why

No model was loaded and no GPU work was run. The machine is shared with the
Hermes benchmark owner and `vm_stat` showed only ~1.7 GB free / ~10 GB inactive
during this session — loading a 20 GB model would have evicted the benchmark
owner's working set. All of the above is static analysis of source and
checkpoint headers, which is independently re-checkable without a GPU window.

Experiment plan for when the machine is free: see the companion Kiem note.

#proj/mei
