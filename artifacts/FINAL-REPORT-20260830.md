# Mei optimization session report — 2026-08-30

Status: MEASUREMENTS PENDING. Machine contended through sessions B–E
(other agent's llama-server on 8017 + active run_bench/run_prompt_suite
runners + cocore; reclaimable memory 1–4GB vs the 10GB floor). The
bounded foreground cycle (scripts/run_measurement_cycle.sh --max-wait-min 5)
is the only measurement path and exits 3 on gate expiry, writing
artifacts/sweep-*.json, artifacts/llama-ceiling-*.json,
artifacts/acceptance-variant-*.json, artifacts/survival-variant-*.json when
a clean window opens. Session E made one bounded gate attempt
(2026-08-30T17:11-17:16Z, exit 3; boundary in
artifacts/cycle-gated-boundary-20260830T171126Z.txt).

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
BLOCKED ON MEASUREMENT (machine fully contended through sessions B–E;
zero uncontended GPU windows). No claim either way until phase B/C rows
land. Session-E gate attempt: exit 3 at 17:16:27Z (reclaimable 4GB vs
10GB floor).

## Remaining work / next experiment
1. In a clear window: scripts/run_measurement_cycle.sh --phase A,B,C
   (bounded foreground, exit 3 if gated); digest with summarize_rows.py +
   gate_report.py; fill the TBD sections.
2. First new-variable experiment when the window opens: llama-ceiling
   phase B (fp16 KV + q8 KV) to set the hardware ceiling before the fork
   variants; then kv8 -> kv4 -> compiled16 -> combined -> window cells.
3. Patch-0005 A/B: probe_diverging_chat.py --ssm-anchor-boundaries 4 vs
   default on the first uncontended server; effect is TTFT, not decode.