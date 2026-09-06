# RUNBOOK: ordered next actions, Phase A-E (2026-09-06) — AUTHORITATIVE ordering

> ## UPDATE 2026-09-06, after A1/A2 landed — read this before Phase C
>
> Three changes from analysing the other agent's A1/A2 artifacts (notes
> "A1 load-path", "A2 ANALYSIS", "BIGGEST LEVER", "GDN input-projection fusion"):
>
> **1. Baseline moved: 61.70 tok/s, not 55.0.** The current pin has
> `fused_gdn_decode_input_projections` active. Every "55 -> X" projection in this
> runbook and in STATE OF PLAY is stale; the mechanisms are unchanged.
>
> **2. NEW C1b is now the highest-value single experiment in the plan — run it
> before C1.** A2 showed `decode.model_forward` = 4.693 ms/token of CPU graph
> rebuild with the GPU idle: 28.9% of the step, paid identically at every context
> length. Compiled decode already removes ~4 ms of it (2026-09-02 measured
> `compiled_forward` 2.183 vs eager ~6.2). It was closed on **32-token** rows —
> the one length below its promote+trace breakeven of 55-159 tokens.
>
> > **C1b. Compiled decode at realistic generation length.**
> > `--compiled-decode true` + `VMLX_ENABLE_UNSAFE_COMPILE=1`, **short context**,
> > `max_tokens` **500 and 1000** (not 32), 3 repeats, eager control at the same
> > lengths. Capture `decode.compiled_forward` on the current pin to replace the
> > borrowed 2.183 ms figure.
> > *Expect +12-17%. A 32-token row will still lose — that is the predicted
> > result, not a contradiction.* Same greedy temp-0 token-equality gate as C1.
> > Does NOT reopen the 30k/80k conclusion, which stands.
>
> **3. C3 should repack the GDN input projections too, and it is a memory lever.**
> `Qwen35GatedDeltaNet` already runtime-concatenates `in_proj_{qkv,z,b,a}` (the
> `groups=[4]` log line), holding a permanent **~407 MiB** duplicate with no
> cache-limit env guard. Repacking those pre-fused on disk reclaims that outright
> while keeping the measured win — alongside the MoE gate+up half. That makes C3
> a memory lever, not a ~1.5% speed lever, so it belongs with Phase D priority.

# RUNBOOK: ordered next actions for hybrid-MoE optimization (2026-09-06)

**This is the authoritative ordering.** The P0-P6 plan note and the Stage 0-3
sequencing note keep the correct mechanics, commands and gates, but their
*rankings* are stale and both now carry a superseded banner pointing here. Read
STATE OF PLAY for the evidence behind these numbers.

Everything below needs a clean GPU window (no llama-server, no other Mei, no
benchmark owner running). Standing gates from plan `0b87b76a` apply to every
step: 3 representative repeats, model identity / streaming parity / tool-call
correctness / KV reuse / long-context survival preserved, and a Kiem note with
exact artifact paths, command, config, result and remaining uncertainty.

---

## Phase A — diagnostics. ~15 min GPU total. Do these first; they are cheap and they redirect everything after.

- [x] **A1. Settle the load path.** `VMLX_MODEL_FACTORY_TRACE=1` on a server
  start for **both** Ornith and stock Qwen 3.6. Prints
  `[ModelFactory] <type> failed: ...` per declining factory, so the winner is
  unambiguous.
  *Answers:* whether Qwen 3.6 takes the VLM path (and therefore already has
  `compileSeparatedDecode: true`, which would make C2 an Ornith-only change).
  *Blocks:* B1 interpretation and C2's entire premise.

- [x] **A2. Profile the unmodelled 22%.** `MLXPRESS_GENERATION_PROFILE=1` on a
  short-decode leg (`tools/probe_load.py`), then read the stage rows in
  `~/.local/share/local-model-bench/mei-runtime/logs/server.log`.
  *Answers:* what the 4.04 ms/token outside the projection model is — GDN
  recurrent scan vs SDPA vs norms vs sampling. The offline twin covers the other
  78% already (predicts 18.11 ms vs 18.18 measured).
  *Why it still matters:* that remainder is where long-context cost grows, so it
  drives whether 80k decode can be improved at all.

---

## Phase B — the artifact that already exists. ~1 h GPU.

- [ ] **B1. Qwen3.6 text-only vs stock Qwen 3.6.** Built, verified and published
  as `Tostibrown/Qwen3.6-35B-A3B-4bit-textonly`; staged locally at
  `mei-models/Qwen3.6-35B-A3B-4bit-textonly`. Not yet loaded by Mei even once.
  *Run:* loadability + `probe_mei` + short decode + 30k, 3 repeats, vs the stock
  bundle as control.
  *Expect:* ~0.83 GB less resident **if** A1 showed the VLM path; unchanged
  memory if it was already on the LLM path (which would falsify the hypothesis —
  a useful result either way).
  *Gate:* must pass the same acceptance the stock bundle passed on 2026-09-06
  (note 9e140ea7) before it is described anywhere as usable.
  *Then:* add a `configs/model-lineup.json` entry with
  `class: mei-produced-conversion`, `hf_repository`, `hf_url`, `hf_revision`,
  `model_card`, `provenance_file`. **Deliberately not added yet** so nothing can
  silently stage or serve an unbenchmarked artifact.

---

## Phase C — speed levers. ~1.10x stacked, sub-additive. ~3 h GPU.

Expected total: short decode 55.2 -> ~60.9 tok/s; 30k 50 -> ~54.6; 80k 35.7 -> ~38.0.
Each is small; they are worth taking together, not individually impressive.

- [ ] **C1. `VMLX_ENABLE_UNSAFE_COMPILE=1` alone**, `--compiled-decode false`.
  Zero code change. 3 cold repeats, short + 30k.
  *Expect ~+8%*, NOT the upstream +45-70% (measured: `mx.compile` gives 1.11x on
  the modelled step; the upstream figure came from small dense models).
  *Gate — this one is about correctness, not speed:* the failure mode is **silent
  numerical corruption**, so require greedy temp-0 **token-for-token equality**
  against the eager baseline on a fixed prompt set. Probe pass/fail is not
  sufficient. A leg 20% faster that diverges by one token is a FAIL.
  *Risk note:* the documented blocker (Osaurus #1173, in-process model switching)
  is structurally unreachable in Mei's one-model-per-process design; the residual
  risk is the macOS Tahoe Metal JIT bug, and this machine is Tahoe (26.5.2).

- [ ] **C2. `compileSeparatedDecode: true`** at `MLXLLM/Models/Qwen35.swift:870`
  in the `tijs/vmlx-swift` fork. One line. Uses `vmlxTrustedCompile`, so it needs
  **no** env flag. Measure with C1 OFF for clean attribution, then with C1 on.
  *Expect ~+3% end-to-end* (+11% on the MoE block alone).
  *Cost:* fork re-pin + full acceptance re-run per `docs/VMLX-FORK.md`.
  *Skip entirely if* A1 shows Qwen 3.6 already on the VLM path AND you only care
  about Qwen 3.6 — it is then already active there and this is Ornith-only.
  *Gate:* same token-equality check as C1 — the region is named for Qwen4-Exp and
  its guards are structural, not identity-based.

- [ ] **C3. Pre-fused gate+up repack.** Store gate_proj+up_proj concatenated on
  disk as `[E, 2*H, in_packed]` so `ensureFusedGateUp` becomes a zero-copy mmap
  view instead of a +12.2 GiB runtime duplicate. Needs a repack tool plus loader
  support. *Expect +10% on the MoE block, ~+1.5% on top of C1/C2* (measured
  sub-additive: compile 1.11x, repack 1.10x, together 1.165x).
  Reuse the `strip_vision_tower.py` scaffolding — per-tensor sha256, 8-byte
  aligned data start, provenance — it already does this class of rewrite.

---

## Phase D — memory levers. Higher value than Phase C on this machine.

Ornith already **exceeds** the >=30 tok/s long-context goal (47.5-50.3 at 30k,
35.7 at 80k). What actually binds on 32 GB is how much context fits: peak
25.73 GB @ 65k, 28.19 GB @ 100k. So context headroom, not tok/s, is the real
constraint — and 93% of the model is the routed expert bank.

- [ ] **D1. Measure routing skew first.** Log selected expert indices over a real
  Hermes coding transcript. Cheap, and it is a **prerequisite** for judging D3 at
  all — if routing is uniform across 256 experts, page residency cannot help.

- [ ] **D2. Mixed 4/8-bit quant.** Keep routed experts at 4-bit; promote selected
  tensors to 8-bit. **Never 5- or 6-bit** — measured slower *and* larger than
  4-bit (only 2/4/8 hit fast `gather_qmm` kernels). First candidate: the shared
  expert, on the critical path for every token but only ~68 MiB across the whole
  model. Bit depth costs ~nothing in speed here (3.4x byte swing -> 1.27x time
  swing), so this is purely a memory/quality dial.
  *Gate:* this is a **quality** experiment — it must go through
  local-model-bench's real suite (sanity / hermes_ops / coding), not speed probes.

- [ ] **D3. Expert page residency.** `JangPressCanonicalExpertAdvisor` in vmlx
  resolves an `advise_experts` symbol and its own comment says the production win
  is "canonical mmap residency". Default-off, per-token readback documented as a
  speed tradeoff. Highest ceiling here, least characterised. Only after D1.

---

## Phase E — new model tracks (independent of A-D).

- [x] **E1. Nemotron-3.5-Lightning stage + gate.** Plan `0b87b76a`'s last open
  todo. Use `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit`.
  **C1-C3 do not transfer**: its routed experts are fc1/fc2 with no gate
  projection, so there is nothing to fuse, and its linear layers are Mamba2 not
  GDN. It is the smallest of the three (16.55 GiB) with only 6 KV-carrying
  attention layers — **the best memory headroom in the lineup**, which matters
  most given Phase D's framing.
  *Free win:* turn on the built-in `NemotronHLayerProfiler` during the
  short-decode leg to get the per-component budget A2 has to reconstruct by hand.
  *Optional first:* build a Mamba2 offline twin with the same method
  (`tools/decode_step_model.py`) before spending GPU on it.

- [ ] **E2. Qwen 3.6 full benchmark suite** once B1 passes — it has a loadability
  gate (note 9e140ea7) but zero benchmark rows.

---

## Closed — do not re-open without new evidence

- **Speculative decoding** (DFlash / DSpark / Eagle3 / native MTP): caps at 1.60x
  at T=8 with perfect acceptance and zero draft cost; nets ~1.1x realistically.
  The premise (bandwidth-bound at T=1) does not hold — this family is
  kernel-bound. Matches the llama.cpp result already observed. Revisit only at
  very long context, if ever.
- **5-bit / 6-bit quants**: slower AND larger than 4-bit on this family.
- **Whole-graph `--compiled-decode true`**: closed 2026-09-02 on 35B evidence;
  per-request promote+trace tax exceeds the win.
- **MLX-backend investigation inside local-model-bench** (vllm-mlx / oMLX):
  closed 2026-08-25; Mei is the separate, live effort.

## Beyond this runbook: the remaining 3.8x is kernel work

The step averages **29% of spec bandwidth**. `lm_head` (one big matmul) reaches
212 GB/s; the gathered MoE path 99 GB/s; the shared expert 44 GB/s — same
hardware, same dtype. The headroom is **small-M quantized matmul kernel
efficiency**, i.e. MLX/vmlx kernel work, outside Mei's config surface and outside
Phase C's ~10%. Highest-ratio target if ever pursued: batching or fusing the
**shared expert** across layers (11% of spec, 40 tiny matmuls/token, and the
component `mx.compile` helped most at 1.32x). One cheap config-reachable probe:
production passes `sortedIndices: false` to `gather_qmm` at decode (because
`doSort = indices.size >= 64` is false at top-8) — whether the sorted path
improves the 99 GB/s figure at decode shapes is unmeasured.

#proj/mei
