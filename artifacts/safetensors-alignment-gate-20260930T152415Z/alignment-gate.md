# Safetensors alignment gate — 2026-09-30

## Verdict

**Approved: align at stage time; keep load-time original-file healing disabled.**

The same real Ornith 1.5 35B payload was served through the same Mei binary, same `mlx.metallib`, same settings, and separate cold processes. Only safetensors header alignment differed.

- Aligned post-load active memory: **19,551,131,124 bytes**
- Misaligned post-load active memory: **24,278,561,190 bytes**
- Misaligned overhead: **4,727,430,066 bytes (24.2%)**
- Aligned short decode: **52.63 tok/s**
- Misaligned short decode: **9.56 tok/s**
- Misaligned decode delta: **-81.8%** relative to aligned

The misaligned arm's local-model-bench probe passed model identity, plain completion, streaming/non-streaming parity, and both native tool-call paths. The slowdown is therefore attributable to the mmap realignment copies and memory pressure, not a missing-template or correctness failure.

## Artifact isolation

- Binary SHA-256: `8b3bec40a1b345f120b3eb79d5be5c5cbf70279bad2d9bd7c04f39897cf44ac6`
- Metallib SHA-256: `0f6bdc29ff9afb81369eb5ca6f6561e0fdbb63228db0d4520c91e3c281738da3`
- Both hashes were identical in both arms.
- All four Ornith shard payload SHA-256 values were identical; only header padding differed.

## Scope

The audit covers aligned/misaligned Ornith, Qwen3.6 base/text-only, Qwen3.8, and retired Ternary-Bonsai staged roots. The current benchmark configs serve aligned Ornith and Qwen3.6 text-only, but still point Qwen3.6 base and Qwen3.8 at misaligned roots; those roots should be repacked before treating their tuned results as release-quality.

The historical 2026-09-06 in-place heal occurred on Mei 0.2.0/vmlx `91fed8be`, where healing was default-on. Current Mei 0.6.0/0.6.1 uses vmlx `fef563a5`, where healing is opt-in and additionally requires direct-send authorization; Mei supplies neither authorization nor an automatic heal path.

## Reproduction

Raw evidence is in `benchmark/`. Machine-readable evidence is `alignment-gate.json`. The live arms used `local-model-bench/runner/start_mei_server.sh`, `stop_mei_server.sh`, and `probe_mei.py`, with Mei's load probe for memory/timing fields.
