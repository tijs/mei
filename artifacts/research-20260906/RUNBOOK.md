# RUNBOOK: ordered next actions, Phase A-E (2026-09-06) — AUTHORITATIVE ordering

> ## UPDATE 2026-09-07 (late) — D3 answered, and the memory framing was wrong
>
> **D3 DONE.** `jangPress: .default` at `Engine.swift:90` (branch
> `research/d3-mlxpress`, commit `86d4503`) makes the MLXPress cold tier engage
> for the first time — `MLXPRESS=N` was silently inert on every prior build.
> Verified live: 10.5 GiB advised cold at 70, 14.2 at 95, mmap tier correctly
> indexing 10,240 experts / 40 layers. **No throughput cost** (64–67 tok/s short,
> 40.4–40.6 at 30k across all settings). **But it reclaims nothing**, and the
> reason matters more than the result.
>
> **THE EXPERT-BANK MEMORY FRAMING (F6 / P5 / STATE OF PLAY) IS WRONG.** A 19.5 GB
> mmap'd model shows **143–156 MB phys_footprint** and ~1.1 GB RSS after a 30k
> request. The weights are clean file-backed pages that macOS evicts freely with
> no `madvise` help. So "93% of the model is the routed expert bank, therefore
> memory wins must come from there" is true of **weight bytes** and false of
> **resident cost** — the bank is ~0% of physical footprint.
>
> Consequences for this plan:
> - **D3: closed, redundant on this configuration.** Keep the change (it makes the
>   tier reachable for future models/hosts), don't enable it by default.
> - **D2: its memory rationale is void.** Lowering routed-expert precision shrinks
>   the file, not the resident cost. Its *quality* question stays open and
>   legitimate — just don't sell it as a memory lever.
> - **C3 SURVIVES and is now the best memory lever left**: the 407 MiB GDN
>   input-projection duplicate and the 12.2 GiB fused gate+up bank are
>   runtime-materialised **dirty** allocations, not file-backed — exactly the
>   category that does cost real memory. Pre-fusing them on disk turns dirty
>   allocation into clean mmap.
> - What actually constrains context: KV cache + SSM companion state (Ornith 30k
>   moves MLX-allocator active 18.79 → 21.77 GB) and Metal compute buffers.
>   Levers that shrink *those* have headroom.
>
> **Methodological correction worth keeping:** `mei_memory_active_bytes` / `peak`
> are MLX **allocator** numbers, not OS memory pressure. Every "peak 25.73 GB @
> 65k" figure in these notes is an allocator number. For host memory, measure RSS
> or phys_footprint.
>
> **Remaining open:** C2 (patch + test plan prepared in the research worktree;
> needs a fork re-pin and full acceptance re-run — the largest remaining unit),
> C3 (repack, now higher value), D2 quality-only.

> ## UPDATE 2026-09-07 (overnight results) — two claims of mine falsified, one harness defect found
>
> **E2 DONE. Qwen3.6 text-only: 24/25** (sanity 2/2, hermes_ops 8/8, coding 14/15).
> The speed claim needs care: hermes_ops tok/s is **not comparable across these
> two runs** because they took different agent trajectories (stock generated
> 16,094 completion tokens vs text-only 6,665), and the sign flips with pooling
> method. The defensible number is the **sanity rows only** (fixed prompt,
> near-fixed output): **1.31–1.32x**, matching B1's controlled 1.24x. The
> 24-vs-22 quality gap is **noise** — the text tensors are bit-identical, so it
> cannot be a real difference; treat it as a calibration of single-trial variance.
>
> **Nemotron proxy fix FALSIFIED.** Live re-run gave hermes_ops **1/8, identical**
> to before. The proxy ran correctly; under the real Hermes prompt shape the model
> emits **no tool-call syntax at all**, so nothing exists to recover. Config
> reverted to `needs_proxy: false`. Nemotron final: **7/25**. Its failure is a
> model capability limit at ~22k tokens of tool payload (threshold ~5–6k with
> realistic content), not a parser bug. Only untested lever left: shrink Hermes's
> tool payload.
>
> **HARNESS DEFECT — the speed gate is not reproducible.** Same config, same
> prompts, byte-identical outputs: **1.02 tok/s (cold) vs 32.74 tok/s (warm)**,
> ttft 84.6 s vs 1.8 s, flipping the 4.0 viability gate from fail to pass. Cause:
> every Mei config uses a **persistent** `--kv-cache-dir`, so a second run is
> served from the disk KV tier. Affects all cross-run comparisons, not just
> Nemotron. Fix: clear the KV dir before a gated run, or record cache state on the
> gate row so cold and warm are never compared.
>
> **D3 IN PROGRESS.** `Engine.swift:90` now passes `jangPress: .default` on branch
> `research/d3-mlxpress` (worktree `~/projects/mei-d3`), building to
> `mei-build-d3`. Next: `MLXPRESS=0/70/95` sweep judged on **resident footprint
> and peak**, with tok/s only as a no-regression check.

> ## UPDATE 2026-09-06 (overnight session) — Nemotron solved, B1 result is bigger than expected
>
> **Nemotron root cause found and fixed at config level.** Its hermes_ops 1/8 and
> "1.02 tok/s" were one bug, not two: the model emits its tool call in the
> documented dialect but **drops the `<tool_call>` wrapper** under Hermes-shaped
> prompts, so vmlx's startTag-gated `XMLFunctionParser` never fires and the call
> lands in `content` as text. One turn, no KV reuse, every task pays a cold 22k
> prefill, and `completion_tokens/wall_seconds` collapses. **Decode is ~68–70
> tok/s — the fastest in the Mei lineup.** Ruled out live: chat template, tool
> count, thinking mode, raw prompt length, prompt-level instruction. Fix uses the
> existing `qwen3_coder` proxy parser (bare `<function=` split marker);
> config patched, offline-verified, **live re-run queued**. See the two Nemotron
> notes.
>
> **B1's text-only result is a speed win, not just memory.** Text-only Qwen 3.6
> ran **~61.85 tok/s vs stock ~49.95 (+23.8%)** as well as −0.83 GiB. The VLM
> load path is materially slower than the LLM path on the same weights. A full
> benchmark of `configs/Qwen3.6-35B-A3B-textonly/mei.yaml` is running now (E2).
>
> **C1b failed its correctness gate** — compiled decode engaged (compiled_forward
> 1.28–1.46 ms vs eager 4.693) but only +3.7–4.6% end-to-end, +2.28 GB peak, and
> **reasoning output diverged on every long repeat**. Root-cause hypothesis: the
> compiled assembly rejected the fused GDN tail and did not activate the fused
> GDN input projections or the compiled MoE router, so it *loses* more than the
> graph-rebuild saving it gains. That also explains why my predicted +12–17% did
> not materialise. **Compiled decode stays disabled.**
>
> **Revised ordering:** C1b is closed (failed). C2 is Ornith-only (A1 confirmed
> stock Qwen3.6 takes the VLM path, which already sets `compileSeparatedDecode`).
> **D1/D3 are blocked** — `Engine.swift:90` builds `LoadConfiguration` with
> JangPress disabled, so the residency machinery never engages; that needs a Mei
> source change before any routing-skew or residency measurement is possible.
> Two runbook knobs are now **closed negative** by clean measurement: sorted
> `gather_qmm` indices (1.002x) and group size (g64 already optimal).

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
> > **C1b. Compiled decode at realistic generation length — COMPLETED 2026-09-06; correctness gate FAILED.**
> > `--compiled-decode true` + `VMLX_ENABLE_UNSAFE_COMPILE=1`, **short context**,
> > `max_tokens` **500 and 1000** (not 32), 3 repeats, eager control at the same
> > lengths. Capture `decode.compiled_forward` on the current pin to replace the
> > borrowed 2.183 ms figure.
> > *Expect +12-17%. A 32-token row will still lose — that is the predicted
> > result, not a contradiction.* Same greedy temp-0 token-equality gate as C1.
> > *Result:* compiled mode engaged and improved end-to-end throughput ~3.7% (500 tokens) / ~4.6% (1000 tokens), but greedy long generations diverged on every repeat; re-encoded token IDs and visible output differed, so compiled decode remains disabled. Evidence: `/Users/tijs/.local/share/local-model-bench/results-mei/C1b-qwen36-compileddecode-20260906T181039Z/C1b-SUMMARY.md`.
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

- [x] **B1. Qwen3.6 text-only vs stock Qwen 3.6.** Built, verified and published
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

- [x] **C1. `VMLX_ENABLE_UNSAFE_COMPILE=1` alone**, `--compiled-decode false`.
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

- [x] **C2. Deferred 2026-09-06:** stock Qwen3.6 already uses the VLM path; the remaining Ornith-only experiment was not authorized after the C1b correctness failure. No source change or measurement was performed.
  in the `tijs/vmlx-swift` fork. One line. Uses `vmlxTrustedCompile`, so it needs
  **no** env flag. Measure with C1 OFF for clean attribution, then with C1 on.
  *Expect ~+3% end-to-end* (+11% on the MoE block alone).
  *Cost:* fork re-pin + full acceptance re-run per `docs/VMLX-FORK.md`.
  *Skip entirely if* A1 shows Qwen 3.6 already on the VLM path AND you only care
  about Qwen 3.6 — it is then already active there and this is Ornith-only.
  *Gate:* same token-equality check as C1 — the region is named for Qwen4-Exp and
  its guards are structural, not identity-based.

- [x] **C3. Deferred 2026-09-06:** the gate+up/GDN repack remains a documented candidate, but no source, loader, repack, or benchmark work was authorized.
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

- [x] **D1. Deferred 2026-09-06:** routing-skew instrumentation was not run; no further optimization work is being started.
  Hermes coding transcript. Cheap, and it is a **prerequisite** for judging D3 at
  all — if routing is uniform across 256 experts, page residency cannot help.

- [x] **D2. Deferred 2026-09-06:** the mixed-quantization quality experiment was not run; no quality or performance claim is made.
  tensors to 8-bit. **Never 5- or 6-bit** — measured slower *and* larger than
  4-bit (only 2/4/8 hit fast `gather_qmm` kernels). First candidate: the shared
  expert, on the critical path for every token but only ~68 MiB across the whole
  model. Bit depth costs ~nothing in speed here (3.4x byte swing -> 1.27x time
  swing), so this is purely a memory/quality dial.
  *Gate:* this is a **quality** experiment — it must go through
  local-model-bench's real suite (sanity / hermes_ops / coding), not speed probes.

- [x] **D3. Deferred 2026-09-06:** expert page residency depends on D1 and was not wired or measured.
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

- [x] **E2. Qwen 3.6 full benchmark suite — COMPLETED 2026-09-06.** B1 passed, then the full Mei benchmark persisted 25 rows: sanity 2/2, hermes_ops 7/8, kiem_mini 5/5, hearth_mini 2/3 (one harness error), kipclip_mini 3/4, and hearth_full 3/3. Raw total: 22/25; the logical run remains partial and not headline-eligible. Evidence: `local-model-bench/results/log.jsonl` and `results/qwen36-mei-reconciliation-20260906.md`.

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
