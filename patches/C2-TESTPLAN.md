# C2 test plan — `compileSeparatedDecode: true` on the LLM qwen3_5_moe path

## Preconditions verified (source-read, 2026-09-06)

- **Reachable**: `LLMModelFactory.swift:53` maps `qwen3_5_moe` -> `Qwen35MoEModel`
  (weight_format is not `mxtq` for our bundles), which builds
  `Qwen35SparseMoeBlock` at `MLXLLM/Models/Qwen35.swift:851`.
- **Structural preconditions all satisfied** by the staged checkpoints, read
  from the safetensors headers: bf16 scales AND biases, uniform affine g64
  4-bit across gate/up/down, top-8 so `indices.size = 8 < 64`, bf16 input.
- **No env flag needed**: `vmlxTrustedCompile` compiles regardless of
  `VMLX_ENABLE_UNSAFE_COMPILE`.
- **Does NOT conflict with the GDN input-projection fusion.** Both
  `Qwen4ExpCompiledRoutedSwitchGLU.call` (SwitchLayers.swift:82) and
  `fusedDecodeInputs` (Qwen35.swift:332) gate on `!CompiledDecodeTrace.isActive`.
  Compiled decode is disabled in production (and failed its own correctness gate
  in C1b), so that flag is false and both optimizations are live together.
  **This check exists because I got it wrong for C1b** — I costed compiled
  decode's graph-rebuild saving without noticing it disables the GDN fusion.
  Never project a stacked gain on this codebase without grepping the guards.

## Expected effect

+11.1% on the MoE routed block measured offline (3 repeats, chained 40-layer
decode graph), which is ~31% of the decode step -> **~+3% end-to-end**. Modest;
worth taking only alongside the other Phase C work, not on its own.

Baseline to beat has MOVED: **61.70 tok/s** short decode on the current pin
(A2), not the 55.0 the earlier notes used.

## Cost

Changing the fork revision forces a full acceptance re-run per Mei's own
`docs/VMLX-FORK.md` rule, and `Package.resolved` must be re-pinned. That is the
dominant cost of this unit, not the one-line edit.

## Gate — correctness first, speed second

The region is named for Qwen4-Exp. Its guards are structural (dtype, bits,
groupSize, shape), not identity-based, so it *should* be safe for Ornith — but
"should" is exactly what C1b's divergence punished.

1. **Greedy temp-0 token-for-token equality** against an eager control on a
   fixed prompt set, short AND long output (500+ tokens). A leg that is faster
   and diverges by one token is a FAIL. C1b's divergence first appeared around
   reasoning position 988, so a 32-token probe would not have caught it.
2. `probe_mei` full matrix (streaming + non-streaming tool calls, KV reuse,
   exact context cap and over-cap rejection).
3. `probe_coding` 4/4.
4. Only then: 3 cold repeats short decode + 30k, against the 61.70 baseline.

## Rollback

Single-line revert plus restoring `Package.resolved`. Keep the prior pin hash
recorded in the Kiem note before starting.
