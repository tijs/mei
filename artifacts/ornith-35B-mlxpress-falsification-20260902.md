# Ornith-35B MLXPress=70 cold-routing engagement FALSIFICATION (2026-09-02)

Question (plan todo 1 remaining item): can the `MLXPressRouter` cold-routing
advice path be engaged for the 35B via `MLXPRESS=70`, for a 3-repeat A/B?

## Verdict: NOT ENGAGEABLE in the current Mei binary. Candidate closed as
vacuous until a Mei source change wires the policy.

## Evidence

### Source (pinned vmlx-swift aeb5e21c + Mei HEAD 4c99e3f)
- Mei `Sources/MeiCore/Engine.swift:90` calls
  `loadModelContainer(..., loadConfiguration: LoadConfiguration(useMmapSafetensors: config.useMmapSafetensors))`.
- `LoadConfiguration` memberwise init defaults `jangPress: JangPressPolicy = .disabled`
  (vmlx `Libraries/MLXLMCommon/Cache/LoadConfiguration.swift:220`).
- `JangPressPolicy.resolve` (LoadConfiguration.swift:667-672) returns `.disabled`
  for the `.disabled` case WITHOUT consulting `MLXPRESS` env. The env is only
  consulted under `.auto(envFallback: true)` / `.enabled`.
- `JangPressActivation.activate` (JangPressActivation.swift:41) early-returns
  `.none` when `options.enabled == false`.
- `JangPressCanonicalExpertAdvisor.configure` (JangPressCanonicalExpertAdvisor.swift:125)
  sets `enabled = mmapEnabled && options.enabled && options.backend == .mmap && ...`,
  so with options.disabled the router is off; `resolveAdviseExpertsSymbolLocked`
  is only attempted when `enabled == true` -> `symbol=missing` is a consequence,
  not a build defect.

### Runtime probe (this run)
Server: aligned 35B, VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0, cap 100032,
memory-limit 30GB, fresh kv-cache-dir. Env: MLXPRESS=70 MLXPRESS_ROUTER_ADVICE=1
MLXPRESS_ROUTER_DEBUG=1 MLXPRESS_GENERATION_PROFILE=1.

Server log at load:
```
[MLXPressRouter] enabled=false async=true warmAdvice=false hotPerLayer=77 maxIndices=32 maxPending=512 symbol=missing
mei: prefix cache enabled (paged in-memory + disk); topology layers=40 kvLayers=10 mambaLayers=30 companion=ssm restore=disk-backed
mei: model loaded
mei: memory after load: active 19551131108 cache 0 peak 0 bytes; limit 30000000000
```

So the router is OFF and the cold-advice symbol is never resolved even though
`mlx_safetensors_mmap_advise_experts` IS exported by the shipped binary
(`nm -gU build/release/mei | grep mlx_safetensors_mmap_advise_experts` ->
`T _mlx_safetensors_mmap_advise_experts`).

## Consequence
- The planned "MLXPress=70 cold-routing A/B with 3 clean repeats" cannot be run
  through the normal Mei launch: any A/B would be vacuous (both legs identical,
  router off). This mirrors the earlier compiled-decode falsification control
  (VMLX_ENABLE_UNSAFE_COMPILE absent -> silent pure-eager) — same class of
  env-gate unreachability, now proven for MLXPress.
- Engaging it would require a Mei source change (e.g. read MLXPRESS env or add a
  config flag into `LoadConfiguration(useMmapSafetensors:)`). Per plan rule,
  generic compiler/kernel ideas stay default-off until measured; the env-only
  measurement is impossible in the current binary, so the candidate is closed
  as "not engageable in current build", NOT closed as "measured and slower".
- The +4-5% thread from the compiled-decode session (guard-skipped eager rows
  under VMLX_ENABLE_UNSAFE_COMPILE=1) is a SEPARATE, still-credible experiment
  (dedicated A/B, compiled decode off) and is run this session.

## Evidence captured from live server log

```
30:[MLXPressRouter] enabled=false async=true warmAdvice=false hotPerLayer=77 maxIndices=32 maxPending=512 symbol=missing
```

(The `hotPerLayer=77` comes from `hotPerLayerDefault(compressPct:70, numRoutedExperts:256, topK:8)`; the advisor DID parse the routed-expert facts — the config's `text_config.num_experts=256` is correctly detected — but `options.enabled=false` from the `.disabled` policy still keeps the router off. Note also line 50/58 of the same log: `[Qwen35] fused_gdn_decode_input_projections=active groups=[4]` from the fork's decode-path commit, confirming MLXPress's non-engagement is unrelated to the GDN path.)
