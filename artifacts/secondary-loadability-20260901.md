# Staged secondary MLX candidates: loadability evidence (2026-09-01, clean window)

Companion to `vmlx-compat-matrix-20260901.md`, which marked each secondary as
`staged_complete-loadability-pending`. This window was clean (only Mei on 8024,
no foreign inference). Each model was tested one at a time on the isolated
Mei server; only one Mei server process at a time, and only Mei's own pid/port 8024
was managed (no foreign process was touched).

Load probe = model identity + `/v1/mei/status` + hello + short decode (probe_load.py).
Tool call = forced `add_numbers` via probe_mei's `validate_add_call` (non-stream).

## 1. Qwen3.8-27B-4bit (mlx-community/Qwen3.8-27B-4bit, dense qwen3_5, reg 4-bit)
- **LOADABLE: yes.** `artifacts/load-Qwen3.8-27B-20260901T165715Z.json`, probe exit 0.
  Topology layers=64 kvLayers=16 mambaLayers=48, companion ssm, disk-backed restore.
  Served id `mlx-community/Qwen3.8-27B-4bit`.
- Memory after load: active 24,741,637,100 B; during decode active 25.5 GB.
  **Perf-memory-constrained** (dense 27B working set): hello 5.93 tok/s,
  short_decode 6.15 tok/s, prefill hello 3.93 s. Loadable + correct, not clean-decode
  fast on 32 GB (same unified-memory ceiling as the 35B, smaller but still dense).
- Tool call: **passed**, `add_numbers(15,27)` schema validated (17.3 s, memory-light decode).
- Status: `loadable-accept-passed-perf-memory-constrained` (regular 4-bit comparator; NOT
  GGUF-UD-Q5; no MTP/speculative, compared without --spec-type).

## 2. gemma-4-26b-a4b-it-4bit (mlx-community, gemma4, reg 4-bit)
- PLACEHOLDER — tested next; results appended below.

## 3. Qwen3.8-27B-Uncensored-MLX-4bit (orcarouter, Heretic lineage, reg 4-bit, separate source)
- PLACEHOLDER — tested last.