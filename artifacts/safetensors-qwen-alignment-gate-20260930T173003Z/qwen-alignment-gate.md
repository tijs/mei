# Qwen safetensors alignment gate — 2026-09-30

**Verdict: approved. Both Qwen active roots were promoted to stage-time aligned artifacts; original misaligned roots were preserved as timestamped backups. Load-time healer remains disabled.**

## Qwen3.6-35B-A3B-4bit

- Active launcher path: `/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.6-35B-A3B-4bit`
- Preserved misaligned source: `/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.6-35B-A3B-4bit-misaligned-20260930T171531Z`
- Alignment audit: **1,485 → 0** unaligned tensors; source unaligned bytes **19,020,243,584 → 0**.
- Post-load active memory: **20,444,312,596 → 24,328,474,566 bytes** aligned → misaligned (**+3,884,161,970, 19.0%**).
- Short decode: **47.73 → 9.45 tok/s** aligned → misaligned (**5.05x aligned speedup**).
- Same binary/metallib: `d15286efca67b349c1dec31e32f44f8a815680927982e357c6e5836c215e8e53` / `0f6bdc29ff9afb81369eb5ca6f6561e0fdbb63228db0d4520c91e3c281738da3`.
- Payload hashes: all 4 shard payloads equal after header-only repack.
- Misaligned MLX realignment log: 4 shard events.
- Aligned `local-model-bench` acceptance: **passed** identity, plain completion, streaming parity, and streaming/non-streaming native tools (context/cache legs intentionally skipped in this bounded promotion probe).

## Qwen3.8-27B-4bit

- Active launcher path: `/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.8-27B-4bit`
- Preserved misaligned source: `/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.8-27B-4bit-misaligned-20260930T171531Z`
- Alignment audit: **945 → 0** unaligned tensors; source unaligned bytes **14,043,394,624 → 0**.
- Post-load active memory: **16,054,543,612 → 24,741,637,100 bytes** aligned → misaligned (**+8,687,093,488, 54.1%**).
- Short decode: **15.39 → 4.00 tok/s** aligned → misaligned (**3.85x aligned speedup**).
- Same binary/metallib: `9acb4d80366589a2f136a7ff0fa6b12f425196d6289e8cb196fe9125fd3ad68a` / `0f6bdc29ff9afb81369eb5ca6f6561e0fdbb63228db0d4520c91e3c281738da3`.
- Payload hashes: all 3 shard payloads equal after header-only repack.
- Misaligned MLX realignment log: 3 shard events.
- Aligned `local-model-bench` acceptance: **passed** identity, plain completion, streaming parity, and streaming/non-streaming native tools (context/cache legs intentionally skipped in this bounded promotion probe).

## Promotion and reproducibility

The existing local-model-bench configs were not modified. Their existing base model paths now resolve to the aligned directories; the original files remain available at the timestamped `-misaligned-20260930T171531Z` paths. Each active directory contains `MEI_ALIGN_MANIFEST.json` and an adjacent aligned provenance sidecar.

Raw gate output is under `qwen36-gate/`, `qwen38-gate/`, and `local-model-bench/`; machine-readable results are in `qwen-alignment-gate.json`.
