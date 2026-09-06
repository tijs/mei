# Hybrid-MoE super-optimization: prioritized experiment plan P0-P6 (2026-09-06)

> **[SUPERSEDED IN PART — 2026-09-06, same day]** The *ordering rationale* in
> this note is stale. It ranks P1 (`VMLX_ENABLE_UNSAFE_COMPILE`) as the single
> highest-value experiment on the strength of upstream's "+45-70%" claim. A
> later offline measurement the same day **falsified that for this model class**:
> `mx.compile` buys **1.11x on the modelled decode step (~+8% end-to-end)**, not
> +45-70% — the upstream figure was measured on small dense models where
> elementwise ops dominate, and does not transfer to a 35B MoE dominated by
> quantized matmuls. Speculative decoding is now **closed** (caps at 1.60x with
> perfect acceptance; nets ~1.1x). And because Ornith already exceeds the >=30
> tok/s long-context goal, **memory levers now outrank speed levers**.
>
> The *mechanics, commands, preconditions and gates* below remain correct and
> are still the reference. For the current ordering and expected values, follow
> the RUNBOOK note ("RUNBOOK: ordered next actions") and the STATE OF PLAY note.

# Hybrid-MoE super-optimization: prioritized experiment plan (queued 2026-09-06)

Companion to the findings note ("Hybrid-MoE super-optimization research:
findings") and the measurement note ("Hybrid-MoE microbenchmarks..."). Written
while the machine was shared with the Hermes benchmark owner, so everything
below needs a **clean GPU window** and has NOT been run.

**Coordination note:** these are deliberately NOT appended to plan
`0b87b76a`'s todo list. That plan was modified mid-tick once before and caused
a worker-renumbering incident (see note f2c81261). Wire them in deliberately
when the Mei track is picked back up, rather than racing an in-flight tick.

Common gates for every experiment below (inherited from plan `0b87b76a`):
3 representative repeats, not a single peak; model identity, streaming/
non-streaming parity, tool-call correctness, KV reuse and long-context survival
must all be preserved; record artifact paths + exact command + measured result
in a Kiem note; a lever that fails stays recorded, not silently dropped.

Baseline to beat (Ornith-1.5-35B-A3B-MLX-4bit-aligned, current production
config): short decode **55.0 tok/s**, 30k loaded decode **47.5-50.3 tok/s**,
post-load **19.55 GB**, peak **25.73 GB** @ 65k cap.

---

## P0 — Characterize the missing 12.5 ms/token. (measurement, not a lever)

**Why first:** the microbenchmarks show the MoE routed blocks are only
**5.64 ms of the 18.18 ms** decode step (31%). Every lever below targets that
31%. **We do not know what the other 69% is.** 30 of 40 layers are GDN
linear-attention (`Qwen35GatedDeltaNet`, Qwen35.swift:187) and GDN weights are
544 MiB/token — the single largest bandwidth component, larger than the routed
experts. Optimizing the MoE block while the GDN path is uncharacterized is
optimizing the wrong 31%.

Mei already has the instrument. Run short decode with the stage profiler and
get the per-stage split:

    MLXPRESS_GENERATION_PROFILE=1 ... scripts/start_mei_server.sh
    # then: tools/probe_load.py (32-token short decode), read
    # decode.model_forward / decode.async_eval_submit / per-stage rows in
    # ~/.local/share/local-model-bench/mei-runtime/logs/server.log

Deliverable: a per-component ms/token budget for the decode step (GDN vs
full-attn vs MoE vs lm_head vs sampling). Everything else re-prioritizes off
this. Cheap — one server start, no A/B.

## P1 — The never-run `VMLX_ENABLE_UNSAFE_COMPILE=1` dedicated A/B  [zero code change]

The single highest-value pending experiment, and it was explicitly queued by
the 2026-09-02 worker as "the one new thread worth a dedicated A/B later" and
never run.

- Upstream `HardwareInfo.swift:33-45` claims **+45% to +70%** decode from this
  and calls it "the single biggest local-decode speedup available".
- Its documented blocker (Osaurus #1173) is **model-switch corruption inside a
  process**. Mei is one-model-per-process by design — structurally unreachable.
- Residual risk is the macOS Tahoe Metal JIT bug (MLX #3329/#3201/#3256). This
  machine IS Tahoe (26.5.2 / 25F84). It did not manifest in the 2026-09-02 run
  (3 repeats x 12/12 PASS), but that run was not looking for silent numerical
  divergence.

Run: **`VMLX_ENABLE_UNSAFE_COMPILE=1` alone, `--compiled-decode false`** (the
2026-09-02 run confounded this with compiled-decode ON, which pays a
per-request promote+trace tax and lost). 3 cold repeats, short decode + 30k.

**Correctness is the gate here, not speed.** Because the failure mode is
"compiled eval returns zeros / silent corruption", a pass/fail probe is not
enough — require greedy (temp 0) **token-for-token output equality** against
the eager baseline on a fixed prompt set, plus probe_mei 12/12 and
probe_coding. If tokens diverge at all, the lever is dead regardless of tok/s.

If it holds: this is the largest single win available and needs no patch, only
a launcher env change + a documented risk acceptance.

## P2 — F4: `compileSeparatedDecode: true` on the LLM qwen3_5_moe path  [1-line vmlx patch]

Measured **+11.1%** on the MoE block -> **~+3% end-to-end** (see measurement
note). Low risk and, unlike P1, needs **no env flag**: the region is built with
`vmlxTrustedCompile`, which compiles even when `VMLX_ENABLE_UNSAFE_COMPILE` is
unset (`Source/MLX/Transforms+Compile.swift:301-318`).

Patch, in the Mei-maintained fork `tijs/vmlx-swift`:

    Libraries/MLXLLM/Models/Qwen35.swift:870
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts,
    +       compileSeparatedDecode: true
        )

This activates `Qwen4ExpCompiledRoutedSwitchGLU` (SwitchLayers.swift:64-142),
which is currently **dead code for every LLM-path MoE model** — only the VLM
twin (`MLXVLM/Models/Qwen35.swift:2184`) wires it. All of its structural
preconditions are already satisfied by both staged checkpoints (verified from
the safetensors headers: BF16 scales/biases, uniform affine g64 4-bit, top-8).

Gate: the region is named for Qwen4-Exp. Its guards are structural, not
identity-based, but **numerical equivalence vs eager is mandatory** — same
greedy token-equality check as P1. Re-pinning the fork revision also requires
re-running the whole acceptance suite per Mei's own `docs/VMLX-FORK.md` rule.

Note P1 and P2 partially overlap (both fuse MoE elementwise work). Measure P2
**with P1 off** so the attribution is clean, then measure the combination.

## P3 — F5: pre-fused gate+up checkpoint repack  [checkpoint change + loader support]

Measured **+10.4%** on the MoE block, and it **stacks with P2 to +16.5%**
(-> ~+4.6% end-to-end for P2+P3).

Today `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0` is set because
`ensureFusedGateUp()` builds a **+12.2 GiB permanent duplicate** of the gate+up
bank at runtime (proven root cause,
`artifacts/ornith-35B-fuse-gateup-eliminated-20260902.md`). That fix is correct
and must stay — but it forfeits the decode fusion, which is why production runs
the 3-dispatch fallback.

The duplication only exists because gate_proj and up_proj are stored
**separately on disk** and concatenated at load. Repack them pre-concatenated
along the output axis (`[E, 2*H, in_packed]` — exactly the layout
`ensureFusedGateUp` builds) and the fused path becomes a **zero-copy mmap
view**: decode fusion at zero memory cost.

Mei already owns an aligned-repack step and the staged dir is literally
`...-MLX-4bit-aligned`, so the tooling, staging allowlist and
`conversion-provenance.json` discipline all exist. Work: (a) repack tool,
(b) loader support for a pre-fused tensor, (c) provenance + published-artifact
handling per the plan's quantization policy.

## P4 — Mixed 4/8-bit custom quant  [checkpoint change]

From the bit-depth sweep: **only 2/4/8 bits get fast `gather_qmm` kernels**;
3/5/6-bit fall back to slower generic paths and are the worst of both worlds.
And a 3.4x swing in active bytes/token produced only a 1.27x swing in time, so
**precision is nearly free in the speed dimension** on this architecture.

Therefore: keep routed experts at 4-bit; spend headroom promoting selected
tensors to **8-bit** (never 5-bit). First candidate is the **shared expert** —
on the critical path for every token, only 1.69 MiB/layer, so 4->8 bit costs
**~68 MiB total** across the whole model. The checkpoint author already applied
exactly this reasoning to the router (`mlp.gate` / `mlp.shared_expert_gate` are
8-bit in `config.json`).

Gate: this is a **quality** experiment, so speed probes are insufficient — it
must go through local-model-bench's real suite (sanity / hermes_ops / coding),
which is what that harness exists to answer.

**Corollary worth recording for the GGUF track too:** the sibling llama.cpp
lineup tried Ornith Q5_K_M and it OOM'd at 23.6 GB. On the MLX side 5-bit would
additionally have been ~10% *slower* than 4-bit. "One quant step up" is the
wrong move on this family in both stacks.

## P5 — Expert page residency  [memory lever, highest ceiling, least certain]

93% of the model (16.9 of 18.17 GiB) is the routed-expert bank: 256 experts,
only 8 read per token. The aligned repack already loads **file-backed NoCopy**,
so expert pages are file-backed and OS-evictable.

vmlx has a half-built mechanism for exactly this:
`JangPressCanonicalExpertAdvisor` (`Libraries/MLXLMCommon/Cache/`) resolves an
`advise_experts` symbol; its own source comment says "JangPress's production
win is **canonical mmap residency**, not per-token readback", and its per-token
index readback is documented as a deliberate speed tradeoff kept default-off.

If real routing is skewed, holding only hot experts resident could cut the
effective working set well below 16.9 GiB — which would let a
**higher-precision quant fit that otherwise would not**, feeding back into P4.
Highest upside of anything here, but the least characterized: start by just
**measuring routing skew** (which experts are actually selected over a real
Hermes coding transcript) before touching residency. That measurement is cheap
and is a prerequisite for judging the lever at all.

## P6 — Nemotron-3.5-Lightning-30B-A3B staging + gate

Already an open todo on plan `0b87b76a` (last unchecked item). Note it is a
**different architecture** (`Libraries/MLXLLM/Models/NemotronH.swift`, own MoE
site at :897), so the *class* of findings here should apply but **every
constant must be re-derived** — do not assume Ornith/Qwen3.6 transfer.

For Qwen 3.6 the opposite is true: it is byte-for-byte the same architecture as
Ornith 1.5 (verified field by field), so P1-P5 transfer directly and need only
a correctness re-gate, not a new optimization search.

---

## Re-derivable in ~30 min without a GPU

All of the static findings can be re-checked cheaply if anything above is
doubted:

- `tools/bandwidth_budget.py`-style safetensors header arithmetic (per-token
  active traffic, per-component split, expert-bank share).
- `grep -n compileSeparatedDecode` across vmlx -> exactly one `true` call site,
  and it is the VLM one.
- `Libraries/MLXLLM/LLMModelFactory.swift:53` -> confirms qwen3_5_moe loads
  `Qwen35MoEModel` on the LLM path.
- `Source/MLX/Transforms+Compile.swift:301-318` -> confirms trusted compile
  ignores the unsafe-compile opt-in.

Scripts live in the research worktree
`/Users/tijs/projects/mei-opt-research` (branch `research/hybrid-superopt`,
`tools/moe_chain_bench.py`, `tools/moe_bitdepth_bench.py`,
`artifacts/research-20260906/`). Mei `main` was not touched.

#proj/mei
