# Four-model Mei MLX benchmark — leg 1: Ornith-1.5-35B-A3B-MLX-4bit (2026-09-04)

Status: DONE (leg 1 of 4). Committed evidence: local-model-bench `d5504f3` (results + config snapshot `22dac305d46e` + 30 transcripts).

## Leg definition (exact runtime configuration)

- Engine: Mei (native Swift/MLX server), isolated backend, port 8024, model dir `~/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned`.
- Served ID: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` (identical served ID as the repo's vllm-mlx leg on 8018).
- Config: `configs/Ornith-1.5-35B-A3B/mei.yaml` @ snapshot `22dac305d46e` (verbatim copy in `results/configs/22dac305d46e.yaml`), runner git sha `622d0ad7`, context cap 65536, prefill step 512, max-tokens 32768, temperature 0.6, top-p 0.95, top-k 20, emit-reasoning true, cache-reuse true, disk KV tier `mei-runtime/kv-ornith-35B`, env `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0`.
- Artifact: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` @ `19504d912fa8fc7622bf6b1de3db5d5d890b1f02` (19.508 GB, 4 safetensors shards; aligned repack dir, 0 unaligned tensors).

## Results (clean evidence)

| suite | result |
|---|---|
| sanity | 4/4 |
| hermes_ops | 12/16 (gate PASS 75%; noisy-result + multi-step-chain fail identically on both waves) |
| kiem_mini | 4/5 |
| hearth_mini | 3/3 |
| kipclip_mini | 4/4 |
| hearth_full | 3/3 |
| **coding total** | **14/15 = 93%** |

Leaderboard (rank 6, composite 0.773): `4/4 sanity · 12/16 hermes_ops · 14/15 coding · 15.3 tok/s harness-measured decode · avg 170 s/task · 13.5 turns`.

## Evidence-integrity correction (15 rows)

The first coding wave (12:46–12:47Z, all tasks) produced 0.4 s hermes-exit-1 rows whose transcripts are 268-byte shells: `Unknown provider 'bench-mei'` — hermes's `custom_providers` entry did not exist yet, so no backend request was ever issued. These 15 rows are NOT model evidence. They were marked `harness_error=true` with a grade_output note (the leaderboard's sanctioned exclusion, `build_leaderboard.py:313`), keeping the raw row data while removing them from scoring. The valid wave-2 rerun (13:02–13:46Z) has the same `config_hash`/`runner_git_sha`. No historical rows were touched; the log grew append-only (2087 → 2137 rows).

## Cross-engine comparison (same suite, same settings discipline)

- Mei MLX 4-bit: coding 14/15 (93%), decode 15.3 tok/s (harness-measured).
- llama.cpp GGUF Q4_K_M sibling rows: coding 95–96%, decode 30.0–30.9 tok/s.
- vllm-mlx 4-bit rows (historical): coding 38–62%, decode 2.1–2.4 tok/s.
- Coding quality: Mei ≈ GGUF brother; the gap is decode speed (vmlx serving-layer throughput, already characterized in the Qwen 30 t/s hardware-ceiling acceptance).

## Build-provenance finding (vmlx pin drift)

The bench-server binary (`mei-build/arm64-apple-macosx/release/mei`, built 2026-09-04 00:10) embeds 26 source-path references to the LOCAL clone `/Users/tijs/projects/vmlx-swift` and zero to any checkouts path: `start_mei_server.sh`'s scratch dir resolves vmlx-swift from the local working tree (HEAD `318a4e68` = pinned fork `91fed8be` + 1 cache-fix commit "persist gen-suffix-stripped boundary"), not from the remote pin. The local clone was NOT modified. Mei's own `.build/checkouts/vmlx-swift` is at the exact pin `91fed8be`. A bench build also rewrote `~/projects/mei/Package.resolved` (dropping the vmlx-swift remote entry — restored by the fresh pinned build below).

Fix: `swift build -c release --scratch-path ~/.local/share/local-model-bench/mei-build-pinned-91fed8be --package-path ~/projects/mei` resolves `https://github.com/tijs/vmlx-swift.git` at exactly `91fed8be` (verified in build log). All four live `mei.yaml` configs gained a `known_gaps` provenance entry documenting the local-clone build identity (2026-09-04 state).

## Blockers / notes for later legs

- Pre-existing runner test failures: 2/210 in the oMLX integration module (`test_omlx_integration.py`), reproduced on pristine HEAD via stash — unrelated to this data commit; environmental/mock-drift, not caused by Mei work.
- Next legs: Qwen3.8-27B (port 8025), Heretic Qwen3.8-27B-Uncensored (8026), Gemma-4-26B-A4B (port 8027, `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0` + prefill 256). All four staged model dirs + provenance JSONs present.