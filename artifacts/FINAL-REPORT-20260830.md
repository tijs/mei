# Mei optimization session report — 2026-08-30

Status: IN PROGRESS — numbers appended as the gated measurement cycle
(scripts/run_measurement_cycle.sh) produces artifacts. Raw evidence:
artifacts/sweep-*.json, artifacts/llama-ceiling-*.json,
artifacts/acceptance-variant-*.json, artifacts/survival-variant-*.json.

## Commits (Mei, clean at c0a5152 -> HEAD)
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
| (pending) | measurement results + optimization log |

## Research sources / revisions
- vmlx-swift pinned aeb5e21c195d8519609488ef75a25ce7e48d8f88
  (osaurus-ai/vmlx-swift; origin/main 8 commits newer: batch capacity/
  position fixes #331/#335, tool parser pin #330 — none touch KV quant or
  compiled decode).
- KVCache.swift:2070 maybeQuantizeKVCache (affine skips rotating);
  KVCache.swift:1017 RotatingKVCache.toQuantized fatalError;
  AttentionUtils.swift:77 attentionWithCacheUpdate dispatch;
  KVCache.swift:228 QuantizedKVCacheProtocol;
  Evaluate.swift:2063+ setupCompiledDecode (promote+trace after prefill,
  buffer = promptOffset + maxTokens + 8); Evaluate.swift:3049 store guard;
  TQDiskSerializer serialize/restore (.rotating records);
  CompilableRotatingKVCache (BatchEngine) — compile-needs-mask + fixed
  buffers, quantized attention lacks mask integration (deep-merge gap).
- llama.cpp build 10470 (brew), arch qwen35 GGUF v3 (verified via
  tools/gguf_meta.py: Ornith-1.5-9B-Q4_K_M, 442 tensors).

## Hypotheses -> outcomes
- H-1 cliff shape: [TBD sweep artifacts]
- H-2 kv quant: [TBD kv8/kv4 cells]
- H-3 compiled threshold: [TBD compiled16/combined cells]
- H-4 hardware ceiling: [TBD llama-ceiling artifacts]
- H-5 window probe: [TBD window8k/window16k cells]

## Short-context (median/min/max)
[TBD]

## 40-50K loaded fresh / reuse (median/min/max)
[TBD]

## TTFT / prefill / decode / memory / cache
[TBD per cell]

## Best combined configuration / rollback configuration
[TBD]

## >=40 at loaded context: met / disproven / blocked
[TBD]

## Remaining work / next experiment
[TBD]