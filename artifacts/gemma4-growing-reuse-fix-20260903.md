# Gemma4 growing-transcript prefix-extension reuse FIXED in vmlx (todo 0b87b76a#7, blocker (b)) — 2026-09-03

Unit: Gemma 4 26B-A4B rotating-layer (25/30) prefix-extension restore gap —
`cache_growing_turn2` returned cached_tokens=0 on both cache tiers.

## Root cause (vmlx-level, confirmed by live trace)

The growing-transcript probe sends turn-1 (793 tokens: system+user1+gen
suffix) then turn-2 (824 tokens: turn-1 minus the gen suffix, plus
assistant reply + user2 + gen suffix). The ONLY stored key that turn-2
contains as a strict prefix is the **gen-suffix-stripped boundary**
(turn-1's prompt stripped back to the end of the last real user message,
786/6174-heuristic positions). The full-prompt (793/6174), N-1 seed
(792/6173) and post-answer (805/6179) keys all fail the content-address
check on turn-2 because turn-2 replaces the gen-suffix token run with
different tokens.

That stripped boundary was persisted exclusively when `coordinator.isHybrid`
(SSM hybrids: Qwen3.8/Ornith pass growing reuse on the disk tier). Gemma4's
cache is 30x RotatingKVCache (all-rotating recurrent backbone; admission
classifies it `isPagedIncompatible` — `cacheCanUsePagedWithRotatingCompanion`
=false because it has no pageable KV layers), so:
- `hybridStripBoundaryIndex` (Evaluate.swift) required `isHybrid` → returned
  nil for Gemma4 despite computing heuristic=786/6174/817;
- BatchEngine.finishSlot's gen-suffix-stripped store branch was
  `isHybrid`-gated → never fired.

Result (fresh disposable disk KV, unpatched binary): turn-1 stores
793/792/805; turn-2 probes 823→805→793→792→193→…→18 all `noRow` →
`MISS all tiers` → full 824-token prefill 4.74s → cached_tokens=0.

## Fix (vmlx fork commit 318a4e68, ON TOP of pinned 91fed8be, NOT pushed)

`Libraries/MLXLMCommon/Evaluate.swift` + `BatchEngine/BatchEngine.swift` +
SpecDec call sites:

1. `hybridStripBoundaryIndex` now takes the live cache and admits
   `cacheHasStandaloneRotatingWindowState(cache)` topologies (standalone
   rotating/SWA: Gemma3/4, Mistral SWA) in addition to hybrid SSM and the
   rotating paged-companion mix. Pure dense/paged topologies stay excluded
   (their paged tier already matches mid-stream prefixes).
2. BatchEngine.finishSlot stripped-store branch gate updated identically.
3. Env-gated diagnostics added (VMLX_CACHE_FETCH_TRACE prints
   `[vmlx][cache/admit]` classification; VMLX_STRIP_BOUNDARY_TRACE prints
   isHybrid/companion flags).

Delivery note: SwiftPM currently consumes this via a LOCAL edit
(`swift package --scratch-path ~/.local/share/local-model-bench/mei-build
edit vmlx-swift --path /Users/tijs/projects/vmlx-swift`); Mei's
Package.resolved/Package.swift pin stays at remote 91fed8be. A user push of
fork 318a4e68 is required to make the pin permanent (no autonomous GitHub
writes). The local edit survives subsequent start_mei_server.sh builds.

## Verification (Sulaco M1 Max 32 GB, port 8024, generic profile, cap 65536,
prefill 64, kv-bits none, fresh disposable disk tier, likewise for baseline)

| Leg | Baseline (91fed8be) | Fixed (318a4e68) |
|---|---|---|
| cache_growing_turn1 (793 tok, cold) | cold prefill 5.6 s, cached=0 | identical 5.68 s, cached=0 |
| cache_growing_turn2 (824 tok) | FAILED: cached_tokens=0, 4.74 s prefill (MISS all tiers) | **PASS: cached_tokens=786, 1.11 s** (restore 786 + prefill 38) |
| cache_repeat_2 (6174 tok) | cached=6173, 0.65 s | cached=6167 (stripped key preferred over N-1), 0.83 s, PASS |
| parity_stream_vs_nonstream | PASS | PASS |
| plain_completion / models_identity / mei_status | PASS | PASS |
| tool legs | structural PASS / strict-schema FAIL (Gemma string-typed args) | unchanged (blocker (c), user go/no-go) |
| probe_mei overall | failed (blocker b) | failed only on blocker (c) |

Trace (fixed run, turn-2 second hit): `[vmlx][cache/fetch] HIT disk
boundary=786 remaining=38` / later `boundary=817 remaining=7` (turn-2's own
stripped key takes over), `[vmlx][cache/disk-store] count=786` present.

## Correctness gate: cache-ON == cache-OFF (byte-identical)

Same turn-2 prompt, temp=0, max_tokens=16:
- cache-ON (restored 817): content `'cache-reuse-ok'`, sha256
  `98417da6c3883fa9c8eaed9e2d7affd0bd676203e5218d4293760f0ca4f6a16c`
- cache-OFF (cold fresh process, 824-token prefill): content
  `'cache-reuse-ok'`, sha256 identical → **MATCH**.
Rotating ring state restored at the stripped boundary is a faithful Markov
resume; the re-feed/trim fast path is unchanged for full hits.

## Cross-model no-regression

- Qwen3.8-27B-4bit on the SAME binary (hybrid topology, disjoint branch):
  probe_mei --skip-context OVERALL PASSED — growing turn-2 cached=823,
  cache_repeat cached=6207, tool non+stream PASS, parity PASS.
- Mei unit suite: 54/59 pass (same 5 live-server MeiAcceptanceTests
  failures as every prior run; the tool-arg acceptance failures mirror
  Gemma blocker (c)).

## Artifacts

- artifacts/probe-mei-gemma4-growing-fix-20260903T0*.json (full legs, fixed)
- artifacts/probe-mei-gemma4-growing-fresh-baseline-20260903T0*.json
  (unpatched, fresh cache: turn-2 FAIL cached=0)
- artifacts/probe-mei-qwen38-4bit-growfix-noregress-20260903T0*.json
- server traces: ~/.local/share/local-model-bench/mei-runtime/logs/server.log
  (VMLX_CACHE_FETCH_TRACE + VMLX_STRIP_BOUNDARY_TRACE runs)

## Status

Blocker (b) CLEARED. Todo 0b87b76a#7 stays OPEN: blocker (c) tool
strict-schema string-args needs user go/no-go; Heretic long-context +
Ornith matrix legs + GGUF comparisons still pending. Follow-ups: push fork
318a4e68 (user), disk-guard OS-temp disposable-cache cleanup gap (launch
cleanup does not cover /var/folders/.../T/mei-kv-cache — caused stale-cache
contamination of one baseline run before manual removal; candidate todo #13
report item).