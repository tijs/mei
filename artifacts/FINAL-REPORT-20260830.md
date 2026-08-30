# Mei optimization session report — 2026-08-30

Status: MEASUREMENTS PENDING. Machine contended through sessions B–F
(other agent's llama-server on 8017 + active run_bench/run_prompt_suite
runners + cocore; reclaimable memory 1–4GB vs the 10GB floor). The
bounded foreground cycle (scripts/run_measurement_cycle.sh --max-wait-min 5)
is the only measurement path and exits 3 on gate expiry, writing
artifacts/sweep-*.json, artifacts/llama-ceiling-*.json,
artifacts/acceptance-variant-*.json, artifacts/survival-variant-*.json,
artifacts/probe-diverging-chat-*.json when a clean window opens. Session E
made a bounded gate attempt at 17:11Z; session F made the second
(2026-08-30T17:20-17:25Z, --phase B, exit 3; boundary in
artifacts/cycle-gated-boundary-20260830T172015Z.txt).

## Session F (continuation) deliverables (all committed)
- probe_diverging_chat.py --ssm-anchor-boundaries K A/B label + self-test
  gate (798ed36): the patch-0005 A/B is now complete, deterministic and
  self-tested; artifacts record server_config.ssm_anchor_boundaries,
  per-request cached_tokens / prefill_ms / TTFT, deterministic content
  tails and tool names.
- start_mei_server.sh MEI_SSM_ANCHOR_BOUNDARIES wiring + cycle phase D
  (anchors=4 vs default cells, fresh KV per cell, bounded readiness poll,
  server-log anchor evidence, MLXPRESS_GENERATION_PROFILE=1 on every cell
  server incl. phase C's probe servers) (9100d43).
- Independent verification, all green: patch-0005 default-off (source +
  binary), apply_vmlx_patches --reset byte-exact + idempotent on both
  checkouts at the pinned rev, 34/34 non-Metal tests, release binary
  surface, sweep/ceiling CLI surfaces, llama-server 10470 flags, GGUF
  provenance (PROVENANCE OK), digest fixture verdicts unchanged.
- Upstream research refresh: vmlx-swift main = 33d1b6fa9 — NEW
  bf8b31995 "Optimize Ornith 35B compiled decode regions (#346)"
  (model-side compiled-region enablement; their ~25 -> ~94 tok/s 35B
  claim on a 27.3GB-footprint machine; NOT vendored — re-pin-level change
  against a memory-blocked target; recorded as the candidate next
  compiled-decode experiment); mlx-swift-lm head = 37688d2c still throws
  on RotatingKVCache.toQuantized (Mei's 0001-0002 remain the only live
  hybrid rotating-KV quant); llama.cpp HEAD = 6d1479c1 (brew 10470);
  FreeToken HEAD = 4b94bdc3.

## CPU-side deliverables (sessions B–E, all committed)
- Measurement pipeline complete and pre-flight validated: sweep driver
  (salted fresh prompts, strict-extension reuse repeats, per-row
  contention labels, fresh-KV-per-cell, 40K-chat transcript fixed for
  transformers>=5, MLXPRESS_GENERATION_PROFILE capture), llama-ceiling
  driver (same 45K prompt/40K chat pattern, no --spec-type, provenance
  gate), digest tools (summarize_rows, gate_report with deterministic
  PIVOT/CEILING/MET verdicts — median crash fixed session E), bounded
  foreground supervisor, single-instance lockfile.
- Fork 0001–0005 applied byte-exactly to both checkouts and reproducible
  via scripts/apply_vmlx_patches.sh --reset:
  - 0001–0003: QuantizedRotatingKVCache (real 4/8-bit affine hybrid KV),
    quantized rotating disk store (dequant-at-store determinism), compiled
    decode threshold (skip promote+trace past N tokens).
  - 0004: --max-kv-window bounded-ring probe (correctness-bounded).
  - 0005 (session E): --ssm-anchor-boundaries K — SSM companion anchors at
    early role-turn boundaries, default off, unit-covered, TTFT lever only.
- probe_diverging_chat.py (patch-0005 evidence): hardened with
  deterministic transcript/schema/output checks; --self-test PASS.
- llama_ceiling.py: --provenance-only verified against the staged official
  GGUF (sha256 70c11219… == ornith-ai/Ornith-1.5-9B-GGUF@abdd624b;
  arch qwen35, ctx 262144, MTP head present, --spec-type stays off) —
  tested live under contention, exit 0, no server launched.
- Non-Metal test suites green: 34/34 Swift (ServerConfigParsing 11,
  SSMAnchorBoundaries 9, OpenAITypes 8, CacheRestoreTracker 6) + probe
  self-tests + provenance negative tests + gate-report regression fixture.

## Commits (Mei, this report's session range)
| hash | purpose |
|---|---|
| 1fc1e24 | cliff-characterization + llama.cpp ceiling drivers; methodology notebook |
| 5dd5a17 | GGUF metadata validator; measurement-cycle orchestrator v1 |
| 5ffe11e | cycle gate: runner-aware + reclaimable-memory floor |
| da3c669 | FORK 0001-0003: QuantizedRotatingKVCache, disk-store, compile threshold; unit tests |
| 07bd5ba | FORK 0004: --max-kv-window bounded-ring probe |
| 296ec88 | sweep: reuse repeats strictly extend the prior prompt |
| 97cff09 | cycle: 80K survival probes in 131K-cap cells |
| 0c4ea67 | sweep: family-salted fresh prompts (no cross-row prefix reuse) |
| c65cafb | artifacts digest tool summarize_rows.py |
| fb37a23 | per-row contention labels in both drivers |
| 897bc0e | final-report template |
| 0eb76c8 | sweep: request timeout 5400->2400s |
| c1a7ac7 | cycle: single-instance lockfile |
| e37daa5 | cycle: window16-compiled gate candidate cell |
| de7dc03 | cycle: gate wait cap 6h -> 24h |
| ff33467 | log: session-B research record + MTP-head finding on the ceiling GGUF |
| 6d9c8c5 | digest: group fresh/reuse medians per context length |
| f9a432e | log: bandwidth model + per-variant expectations |
| 53742de | tools: bounded foreground measurement policy (no detached gate waiters) |
| 15d53f8 | tools: diverging-chat probe (patch-0005 evidence), fork-flag config tests, cycle-pipeline fixes |
| 066ae8c | tools: probe hardening, llama_ceiling provenance gate, digest fixes (session E) |
| dec9c8f | FORK 0005: SSM anchor-boundary plumbing (default off) + Mei flag + unit tests (session E) |
| f0b018c | docs: patch-0005 design record, README, session-E log (session E) |
| 798ed36 | tools: diverging-chat probe --ssm-anchor-boundaries A/B label + self-test gate (session F) |
| 9100d43 | cycle: phase D diverging-chat A/B cells; MLXPRESS profile on all cell servers (session F) |
| (pending) | measurement results + optimization log + final numbers |

## Research sources / revisions
- vmlx-swift pinned aeb5e21c195d8519609488ef75a25ce7e48d8f88
  (osaurus-ai/vmlx-swift; origin/main 8 commits newer: batch capacity/
  position fixes #331/#335, tool parser pin #330 — none touch KV quant,
  compiled decode, or SSM anchors).
- KVCache.swift:2070 maybeQuantizeKVCache (affine skips rotating before
  patch 0001); AttentionUtils.swift:94 quantized dispatch;
  Evaluate.swift setupCompiledDecode promote+trace;
  SSMReDerive.swift:438-443 reDeriveAndStoreSSMStatesAtPromptBoundaries
  (additionalBoundaries param — patch 0005 threading), :493-498 "Keep the
  LARGEST boundaries"; Evaluate.swift:2760 + BatchEngine.swift:3174 store
  call sites. Same upstream gap on rotating-KV quantization confirmed in
  lmstudio-ai/mlx-engine#31 and ml-explore/mlx-swift-lm main.
- llama.cpp build 10470 (brew, commit 34af94cd9), arch qwen35 GGUF v3,
  official ornith-ai/Ornith-1.5-9B-Q4_K_M pinned by content sha256
  70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6
  (repo default revision abdd624b12ebf020b767fff532ff44fe552b28c3).
  MTP head PRESENT (qwen35.nextn_predict_layers=1, blk.32.nextn.*) —
  ceiling runs WITHOUT --spec-type draft-mtp; spec-engage verified via
  timings predicted_n vs evaluated_n once artifacts land.
- Level1Techs (forum.level1techs.com/t/253917) hypothesis controls:
  temp 0 everywhere, fixed template, deterministic output + tool-call
  schema checks per variant, acceptance + 30K/80K survival per variant.
- FreeToken (github.com/FlashML-org/FreeToken; arXiv:2608.16157):
  semantic anchors -> patch 0005 (landed, default off); bandwidth-adaptive
  execution -> allocator/cache-budget sweeps (phase A); expert residency
  -> 35B KV-budget experiments (blocked artifact unchanged).

## Hypotheses -> outcomes
- H-1 cliff shape: [TBD sweep artifacts — phase A gated]
- H-2 kv quant: [TBD kv8/kv4 cells — phase C gated]
- H-3 compiled threshold: [TBD compiled16/combined cells — phase C gated]
- H-4 hardware ceiling: [TBD llama-ceiling artifacts — phase B gated;
  driver + provenance gate validated under contention, never launched]
- H-5 window probe: [TBD window8k/window16k cells — phase C gated]
- H-6 ssm anchors (patch 0005): [plumbing landed default-off; effect TBD —
  probe_diverging_chat + phase-A ssm-rederive rows gated]

## Short-context (median/min/max)
[TBD — historical eager baseline short ≈ 28.1 tok/s; compiled 47.2 tok/s
short-context (2026-08-29, disabled by default due to long-prefill tax)]

## 40-50K loaded fresh / reuse (median/min/max)
[TBD — historical: fresh45 5.2 tok/s (prefill ~273s), reuse45 13.24 tok/s
(cached 45,000; prefill 10.2s), chat40k 10.3 @33K]

## TTFT / prefill / decode / memory / cache
[TBD per cell once artifacts land]

## Best combined configuration / rollback configuration
[TBD] — rollback remains: eager fp16, no compiled decode, no kv quant,
no window, ssm anchors off (fork flags default = upstream behavior).

## >=40 at loaded context: met / disproven / blocked
BLOCKED ON MEASUREMENT (machine fully contended through sessions B–F;
zero uncontended GPU windows). No claim either way until phase B/C rows
land. Session-E gate attempt: exit 3 at 17:16:27Z (reclaimable 4GB vs
10GB floor); session-F gate attempt: exit 3 at 17:25:16Z (--phase B,
reclaimable 3GB, Muse-Glimmer-30B + hermes_ops suite active).

## Remaining work / next experiment
1. In a clear window: scripts/run_measurement_cycle.sh --phase B
   (llama.cpp ceiling, fp16 KV + q8 KV, provenance-gated, no --spec-type)
   FIRST per the session-F ordering, then --phase A,C,D
   (A = cliff characterization, C = kv8/kv4/compiled16/combined/window
   cells, D = patch-0005 diverging-chat A/B); digest with
   summarize_rows.py + gate_report.py; fill the TBD sections.
2. Precisely staged commands (all under the cycle; direct forms):
   - ceiling fp16:  python3 tools/llama_ceiling.py --gguf
       ~/.local/share/local-model-bench/mei-models/gguf/Ornith-1.5-9B-Q4_K_M.gguf
       --alias ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M --repeats 3 --chat-40k
       --output artifacts/llama-ceiling-fp16kv-<ts>.json
   - ceiling q8 KV: same + --kv-cache-type-q8
   - anchors A/B:   scripts/run_measurement_cycle.sh --phase D
       (anchors-default cell then anchors4 cell; compare
       divergence.gap_tokens / turn4_prefix_restored across
       artifacts/probe-diverging-chat-anchors-{default,4}-<ts>.json)
3. First new-variable experiment when the window opens: llama-ceiling
   phase B to set the hardware ceiling before the fork variants.
4. Candidate compiled-decode experiment (research-recorded, not yet
   scheduled): backport upstream bf8b31995 (#346, model-side compiled
   region enablement for the Ornith/Qwen3.5 topology) to the pinned tree
   and A/B on the full matrix — only if phase-B/C rows point at
   model-side compile regions as the binding cost.