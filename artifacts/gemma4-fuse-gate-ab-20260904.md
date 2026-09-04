# Gemma4-26B fuse-gate A/B: VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 (todo 0b87b76a#12 leg) — 2026-09-04

Unit: continuing optimization loop (#12) — the last untested Mei-side lever on the
Gemma 30k-decode gap. The 2026-09-04 lever matrix root-caused the gap to
~135 ms/step GPU eval and closed kv-quant (crash), compiled-decode (silent
no-op), and window-gate (no valid config) as Mei-side levers. What it never
tested: the SwitchGLU fused gate+up concat cache disable env that gave the
Ornith-35B (same SwitchGLU MoE machinery) its +12 GB elimination / 3x decode
win on 2026-09-02 — gemma-4-26b-a4b-it-4bit uses `@ModuleInfo(key:
"switch_glu")` (vmlx Libraries/MLXLLM/Models/Gemma4Text.swift:434), so
`ensureFusedGateUp()` (SwitchLayers.swift:282) runs per SwitchGLU layer's
first forward.

Window: clean for every leg — no foreign inference processes before/during,
port 8024 Mei-owned, disposable runtime bases
`mei-runtime-gemma4-30k-fuseoff/r{1,2,3}`, local-model-bench READ-ONLY during
the run (its configs updated after, see below).

Config (identical to the recorded fuse-ON family rows except the ONE env var):
generic profile, prefill-step 256, context cap 65536, kv-bits none, disposable
disk KV, port 8024, release binary at mei-build (main 3253d07/HEAD f706620,
vmlx fork 318a4e68), model mlx-community/gemma-4-26b-a4b-it-4bit staged at
mei-models. Each repeat: fresh disposable KV removed before start, cold
server, `probe_long_context --lengths 30000 --max-tokens 32` (fresh fill +
strict-extension reuse decode). Env under test: `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`
(0 => cache limit 0 => fused bank never built).

## Measured (n=3, cold; vs fuse-ON family rows from 2026-09-04 leg1 + pref256)

| leg | fresh 30k decode t/s | reuse decode t/s | 30k prefill pps | peak mem |
|---|---|---|---|---|
| fuse-ON baseline (leg1) | 7.099 | 7.395 | 229.2 | 27.23 GB |
| fuse-ON recorded family (pref256 r3 + cdec rows) | 7.431 / 7.503 / 7.564 | 7.354 / 7.520 / 7.524 | 266.0 | 27.23 GB |
| fuse-OFF r1 | 16.961 | 15.272 | 318.06 | 18.76 GB |
| fuse-OFF r2 | 23.266 | 19.506 | 349.09 | 18.76 GB |
| fuse-OFF r3 | 23.327 | 23.833 | 349.44 | 18.76 GB |
| fuse-OFF mean (n=3) | 21.185 (sd 3.68) | 19.537 (sd 4.28) | 338.9 (sd 18.1) | 18.76 GB |
| fuse-OFF r2/r3 mean | 23.297 (sd 0.04) | 21.670 (sd 3.06) | 349.3 | 18.76 GB |

Deltas (vs fuse-ON mean 7.27 fresh / 7.37 reuse / ~247 pps / 27.23 GB):

- fresh decode **+191%** (r2/r3 stable band +220%; r1 = first-run transient,
  the only repeat that immediately followed the swift build)
- reuse decode **+165%**
- 30k fresh prefill **+37%**
- peak memory **-8.47 GB (-31%)**, flat 18.76 GB across all 3 repeats

All rows `status: passed` with full checks; reuse rows cached all 30000 prefix
tokens (disk-tier KV reuse intact).

Artifacts: artifacts/probe-30k-gemma-fuseoff-r{1,2,3}-20260904T*Z.json
(committed); server stage dumps mei-runtime-gemma4-30k-fuseoff/r{1,2,3}/logs/
(disposable).

## Mechanism

Peak drop 27.23 -> 18.76 GB == the retained fused gate+up bank (26 MoE layers
x per-layer concat, same machinery as the Ornith trace: SwitchLayers.swift
`ensureFusedGateUp` builds a concatenated `[E, 2*hidden, in_packed]` weight +
scales + (biases) bank on first forward and retains it). The decode-token cost
moves with it: 30k fresh decode 7.1 -> 23.3 t/s (per-step GPU eval 135 ms ->
~43 ms) with no Mei source change. This is the same env that is REQUIRED in
the fitted Ornith-35B config (plan/README/manifest), so the win is
architecture-family-consistent, not a fluke: the fused decode path is a
measured LOSS on both MoE families on this Metal machine despite the source
comment claiming a decode win (the concat+gatherQMM fused bank wins only when
the per-layer copy cost is amortized, which never happens on B=1 decode with
quantized packed weights; memory pressure compounds it).

## Cross-engine position

30k loaded decode: fuse-ON 7.4 vs llama.cpp APEX-I-Quality 37.08 t/s (5.0x
GGUF-faster) -> fuse-OFF 21.2-23.3 vs 37.08 t/s (~1.6x GGUF-faster). Gap
closed 3.1x by one env var; the residual ~1.6x is the documented vmlx
SDPA/long-KV kernel-efficiency matter (5 full-attention layers at 30k), still
outside the Mei config surface. Peak 18.76 GB vs GGUF RSS ~21 GB-class —
Gemma now fits with 13 GB headroom on the 32 GB machine.

## Correctness re-gate on the fitted config (same window, same env)

Artifacts: artifacts/probe-mei-gemma4-fuseoff-regate-20260904T064811Z.json +
artifacts/probe-coding-gemma4-fuseoff-regate-20260904T065233Z.json; server
stage dump mei-runtime-gemma4-fuseoff-regate/ (disposable, MLXPress profile
ON).

- probe_mei FULL 12/12 PASS: models_identity, mei_status, plain_completion,
  tool NON-STREAMING add_numbers(15,27) exact ints, tool STREAMING same,
  streaming/non-streaming parity, cache_repeat (prefix reused),
  cache_growing turn1 + turn2_reuses_slot, context_exact_cap 65536 FILLED
  (286.2 pps, 231.8 s, active 20.06 GB, PEAK 21.79 GB, 1 completion token),
  context_over_cap 65537 rejected HTTP-400.
- probe_coding 4/4 PASS (swift_fibonacci / python_json_sum / sql_users_query /
  shell_rename families; generations at 50.7 + ~17-18 t/s rows).
- Mechanism rows (MLXPress generation profile, server.log): fuse-OFF
  `decode.async_eval_submit` avg 17.08-17.72 ms/step (4 generations, 730/182/
  185/... tokens) vs fuse-ON 134-140 ms/step measured 2026-09-04 — 7.7x
  GPU-eval reduction; `decode.model_forward` 1.88-2.01 ms unchanged (CPU graph
  build unaffected, as expected). The fused gate+up bank was the per-step
  decode cost on this MoE, exactly as on Ornith-35B.

## Config propagation (authorized isolated Mei bench files)

- ~/projects/local-model-bench/configs/Gemma-4-26B-A4B/mei.yaml: launch
  command now exports VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0; settings +
  known_gaps refreshed with these numbers (30k decode row, lever status).
- ~/projects/local-model-bench/configs/Ornith-1.5-35B-A3B/mei.yaml: launch
  command now serves the -aligned repack dir with the same env (the fitted
  config); verified earlier this tick that the plain dir + no env was still
  wired there (would have reproduced the +12 GB materialization at benchmark
  time). settings entries added for both.
- configs/model-lineup.json: Gemma entry carry-forward note updated from
  PENDING to these measured rows (this tick).

## Verdict

Gemma4 26B decode at 30k: 7.4 -> 21.2-23.3 t/s (3-repeat), prefill 247 ->
339 pps, peak 27.23 -> 18.76 GB — the biggest Gemma win measured, achieved
with one upstream env var, no Mei source change, gates re-checked (12/12 +
coding re-gate artifacts on the same config, see
probe-mei-gemma4-fuseoff-regate-*.json + probe-coding-gemma4-fuseoff-regate-*.json).
The env is REQUIRED for Gemma's fitted config and is added to the bench launch
path. #12 stays OPEN (Ornith primary rows still active + other models); the
Gemma #12 gap is now a measured ~1.6x-vs-llama.cpp residual with the
Mei-side lever list closed.