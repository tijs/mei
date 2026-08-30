# Review: upstream vmlx-swift bf8b31995 "Optimize Ornith 35B compiled decode regions (#346)" — backport decision (session G, 2026-08-30)

Machine: sulaco, M1 Max g13s, 32 GiB. Engine: mei on vmlx-swift pinned aeb5e21c +
Mei fork series 0001-0005. This is a CPU-side source review; no Metal touched.

## Exact source / revisions

- Commit: `bf8b31995195fffd833968658f14c707317eaa70`
  (osaurus-ai/vmlx-swift main, author Eric Jinho Jang, 2026-08-29T20:18-0700,
  PR #346). Fetched via `git fetch https://github.com/osaurus-ai/vmlx-swift.git main`
  into the pinned checkout (read-only research; the checkout stays on aeb5e21c).
- Files changed (5): `Libraries/MLXLMCommon/Qwen4ExpFusedAffineMoE.swift` (186),
  `Libraries/MLXVLM/Models/Qwen35.swift` (288),
  `Tests/MLXLMTests/Qwen35VLMGatedDeltaTests.swift` (172),
  `Tests/MLXLMTests/Qwen4ExpFusedAffineMoETests.swift` (65),
  `docs/CONTINUOUS_BATCHING_FAMILY_TRIAGE_2026_08_29.md` (41).
- Functional changes (the remainder of the diff is formatting churn):
  1. `Qwen35Language.shouldCompileDecodeRegions(_:environment:)` (MLXVLM Qwen35.swift,
     after line ~1107 in the new file): env override `VMLINUX_QWEN35_COMPILE_DECODE_REGIONS`
     (default ON only when the config exactly matches the 35B topology:
     modelType qwen3_5_moe_text, hidden 2048, layers 40, fullAttentionInterval 4,
     numExperts 256, numExpertsPerTok 8, moeIntermediateSize 512, linearNumKeyHeads 16,
     linearNumValueHeads 32, linearKeyHeadDim 128, linearValueHeadDim 128). When true,
     `GatedDeltaNet(args, fuseDecodeInputProjections: true)` and
     `SparseMoeBlock(args, layerIdx:, compileDecodeRegions: true)`; `compiledDecodeTail`
     now keys off `fuseDecodeInputProjections` and threads `sigmoidGate:` (silu vs
     sigmoid gate mode) into `Qwen4ExpCompiledGDNInputs.callTail`.
  2. `Qwen4ExpFusedAffineMoE`: adds an `ornith35Shape` (input 2048, expert 512,
     routes 8) alongside qwen4ExpShape (2560/640/10), adds 5-bit pack support and
     64-bit code masks, shape-parameterized kernels.
- Upstream context: main HEAD at fetch time = 33d1b6fa9 (2026-08-30,
  session-F record), 31 commits past the pin per fresh rev-list; bf8b31995 is an
  ancestor of main. No commit between the pin and main HEAD touches the MLXLLM
  text-path files (Libraries/MLXLLM/Models/Qwen35.swift, Qwen35MoE.swift);
  post-bf8 compiled work is VLM-scoped again (d8fd7010, c1162b43 = Qwen3.8 MTP
  verifier rows in the MLXVLM class). The text path is untouched upstream.

## Which class actually serves Mei's models (resolution evidence, pinned tree)

`Libraries/MLXLLM/LLMModelFactory.swift:51-57`:
- `"qwen3_5"`      -> `Qwen35Model` (MLXLLM/Models/Qwen35.swift:1377) — TEXT class
- `"qwen3_5_moe"`  -> `Qwen35MoEModel` (MLXLLM/Models/Qwen35MoE.swift:38, subclass of
  Qwen35Model) — TEXT class
- `"qwen3_5_text"` -> `Qwen35TextModel` (MLXLLM/Models/Qwen35.swift:1038) — TEXT class

Mei loads via `loadModelContainer` (Sources/MeiCore/Engine.swift:87) -> MLXLLM
factory. Measured 9B artifact (config.json text_config): model_type `qwen3_5_text`,
num_hidden_layers 32, full_attention_interval 4, NO num_experts -> dense
`Qwen3NextMLP` per MLXLLM/Models/Qwen35.swift:767-776. So the 9B executes
Qwen35TextModelInner/Qwen35TextModel (text path). Blocked 35B artifact:
model_type `qwen3_5_moe` (text model_type qwen3_5_moe_text) -> Qwen35MoEModel
(text path), NOT the MLXVLM vision `Qwen35` class (MLXVLM/Models/Qwen35.swift:2740)
that PR #346 modified.

The MLXVLM vision `Qwen35` class is reached only for image-capable artifacts and
is not used by any currently Measured-or-blocked Mei artifact.

## Decision: NOT backportable as a minimal default-off patch

1. The model-side compiled-region enablement (`GatedDeltaNet(fuseDecodeInputProjects:)`,
   `SparseMoeBlock(compileDecodeRegions:)`, `compiledDecodeTail` gate-mode change) lives
   entirely in the MLXVLM vision `Qwen35` class and its private
   `Qwen4ExpCompiledGDNInputs` machinery. Mei's 9B (`Qwen35TextModel`) and 35B
   (`Qwen35MoEModel`) both resolve to MLXLLM text-path classes that share none of
   that machinery (their own `Qwen35GatedDeltaNet` at MLXLLM/Models/Qwen35.swift:187
   and `Qwen35SparseMoeBlock` at :692 have no compiled regions; the only compiled op
   on the text path today is `compiledSigmoidGate` at :16-23). Patching the pinned
   tree with the commit as-is would be a no-op for the A/B matrix.
2. The shared-kernel change (`Qwen4ExpFusedAffineMoE` ornith35Shape) cannot engage on
   the 9B (dense MLP, no experts; grep: the reducer is only wired through
   `SwitchLayers.swift:232-237`, Qwen4Exp family) and its 35B target shape
   (2048x512 topk-8) is exactly the memory-blocked artifact (~19.5GB weights +
   working-set thrash, documented in artifacts/blocked-35B-measured-20260830.json) —
   unmeasurable until residency is solved.
3. Porting compiled GDN/MoE regions INTO the text path would be a NEW fork
   (from-scratch implementation of the VLM machinery against the pinned text
   classes), not a minimal backport: new region graphs interacting with
   quantizedMM, BF16 decode contract, RotatingKVCache, and the disk restore
   path. Per workstream-5 discipline (kernel/metallib A/B only if profiling
   justifies it), that port is gated on MLXPRESS_GENERATION_PROFILE evidence that
   per-op dispatch / GDN overhead dominates the 9B decode step — no such evidence
   exists yet (no measurements have run this cycle).
4. Claimed numbers (their doc: ~25 -> ~94 tok/s on 35B rows, 27.3GB peak footprint)
   are NOT apples-to-apples: different model class (VLM), different artifact, their
   own compiled path with its own KV footprint. Not copied anywhere in Mei.

## Transferable elements recorded

- The topology-scoped default-off env-override pattern
  (`VMLINUX_QWEN35_COMPILE_DECODE_REGIONS`, semantics: override beats topology
  match; only the exact 35B shape gets it by default) is the right shape for any
  future Mei model-side compile knob: default off, env override, topology-scoped,
  plus stderr engagement lines. Mei's existing compiled-decode-threshold flag
  (patch 0003) already follows the same default-off philosophy.
- Reference implementation to port IF profiling so justifies it:
  `Qwen4ExpCompiledGDNInputs` (MLXVLM/Models/Qwen35.swift:90-~335 at bf8b31995)
  front/tail regions with shared weight inputs and per-shape region keys; a text-path
  port would need its own parity tests against the eager GatedDelta path.
- PR #346's test additions (Tests/MLXLMTests/Qwen35VLMGatedDeltaTests.swift) are
  VLM-scoped as well; nothing reusable for the text path without porting.

## Follow-up state

- Next measured candidates remain those already staged in
  scripts/run_measurement_cycle.sh (phase B: llama.cpp ceiling fp16/q8 KV without
  --spec-type; phase A: cliff matrix; phase C: kv8/kv4/compiled16/combined/window
  cells; phase D: anchor A/B), gated on an uncontended window.
- If/when phase-A profiling shows GDN dispatch dominating 9B decode, revisit this
  review and design the text-path compiled-GDN port (with parity + acceptance +
  30K/80K survival gates). Do not port blind.

Recorded by: Mei session G (2026-08-30). Sources: pinned checkout aeb5e21c,
fetched bf8b31995 diff, 9B/35B local MLX artifacts' config.json, LLMModelFactory
resolution. No memory/correctness risk introduced: no runtime change landed from
this review.