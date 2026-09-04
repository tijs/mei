# Four-model Mei MLX gate matrix — consolidated status (todo 0b87b76a#17 leg) — 2026-09-04

Single-source status for the four benchmark finalists + the Ornith-9B proxy.
Every row cites committed artifacts; nothing here re-measures. Numbers are the
best/canonical measured row on the machine (Sulaco, M1 Max 32 GB) as of this
tick. "Fitted config" = the documented safe per-model launch (see the
per-model bench yamls in local-model-bench and the notes in
configs/model-lineup.json).

## Per-model summary

| model / artifact (pin) | quant | fitted config essentials | 30k decode fresh/reuse | 30k prefill | short decode | peak mem (30k / cap) | vs GGUF ref |
|---|---|---|---|---|---|---|---|
| Ornith-1.5-35B-A3B (ornith-ai official 4-bit, rev 19504d91; aligned repack dir) | 4-bit affine g64, 8-bit gates | aligned dir + VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 + disk KV + prefill 512 + cap 90000 (bench: 65536) | 47.5-50.3 t/s (30k eager; 3x repeats) | ~392 pps (512-step; 64/128/1024/2048 all worse) | 55.0-55.9 t/s 3x | 25.73 GB @65k / 27.72 GB @90k / 28.19 GB @100k | GGUF Q4_K_M: MLX +8.7% short, +18% @30k; prefill -23% (llama.cpp wins chunked prefill) |
| Qwen3.8-27B (mlx-community 4-bit, rev 3e6447f0) | 4-bit affine g64 | generic + disposable disk KV + prefill 64 + cap 65536 | 11.61 / 11.76 t/s (raw path, 30k) | 55.8 pps (chat 30k) | 15.656 t/s 3x (sd 0.06) | 18.87 GB short-peak / 31.70 GB raw 65k-cap fill | UD-Q5_K_M: MLX +68.6% short, +40-54% @30k; GGUF +48% prefill; 30 t/s ceiling ACCEPTED (plan record 2026-09-02) |
| Qwen3.8-27B 5-bit (Mei-produced, published f592c6fb) | 5-bit affine g64 bf16 | same as 4-bit | 6.73 t/s @30k | 55.1 pps | 13.108 t/s 3x | 21.86 GB / 34.64 GB @65k (over phys, constrained) | parity artifact; NOT UD, no equivalence claim; slower than 4-bit |
| Qwen3.8-27B-Uncensored / Heretic (orcarouter 4-bit, rev 14963e70) | 4-bit affine g64 | generic + disk KV + prefill 64 + cap 65536, port 8026 | 11.896 (sd 0.058) / 11.842 (sd 0.054) t/s | 56.21 pps | ~15.1-15.8 t/s | 24.71 GB @30k | Q5_K_M: MLX +1.55x @30k; base-Qwen ceiling record transfers; lineage gate passed (NOT base Qwen) |
| Gemma 4 26B-A4B (mlx-community 4-bit, rev 0d77464e) | 4-bit affine g64 | generic + VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 + disk KV + prefill 256 + cap 65536 | 21.185 mean (r2/r3 23.30) / 19.537 mean | 338.9 pps (r2/r3 349.3) | 51.3 t/s | 18.76 GB @30k / 21.79 GB @65k-cap fill | APEX-I-Quality: MLX 21.2-23.3 vs 37.08 t/s @30k (~1.6x GGUF-faster); fuse-ON was 5.0x — residual = vmlx long-KV SDPA kernel efficiency (blocker, vmlx-side) |
| Ornith-1.5-9B (ornith-ai official 4-bit) | 4-bit affine g64 | proxy, dense qwen3_5 | (30k not the proxy focus) | — | ~56 t/s aligned | fits | GGUF Q4_K_M staged; proxy for 35B-family optimization |

## Gate status (todo 0b87b76a gates)

| gate | Ornith-35B | Qwen3.8-4bit | Heretic | Gemma4 |
|---|---|---|---|---|
| load / identity | PASS (12/12 acceptance incl. identity) | PASS (12/12 rawfix) | PASS (10/10) | PASS (12/12 re-gate) |
| streaming/non-streaming parity | PASS | PASS | PASS | PASS |
| tool calls (schema-aware ints) | PASS (a=15,b=27 non+stream) | PASS (12/12) | PASS | PASS (12/12 re-gate) |
| coding | PASS 4/4 | PASS 4/4 | PASS 4/4 | PASS 4/4 (re-gate) |
| short context | PASS | PASS 3x | PASS | PASS |
| long context (30k/80k/100k) | PASS 30k/80k/90k/100k (3x @80k, 100k) | PASS 30k-65k (peak 31.7 GB raw) | PASS 30k (3x) | PASS 30k + 65k exact-cap fill (re-gate) |
| KV reuse | PASS (disk tier, 80000/90000/100032 restored) | PASS (6207/6212 + raw 30000/30001) | PASS (30000/30001 3x) | PASS (cache_repeat + growing-turn re-gate) |
| memory | fits (28.19 GB @100k < 30 GB limit) | fits (31.70 GB peak raw 65k; tight) | fits (24.71 GB) | fits with headroom (21.79 GB @65k) |
| repeatable speed | PASS 3x (0.1 sd @80k) | PASS 3x (sd 0.06) | PASS 3x (sd 0.06) | PASS 3x (+ sd noted) |
| 30 t/s goal | EXCEEDED short (55) and @100k (31.7); 47.5-50.3 @30k | ACCEPTED CEILING ~15.7 (plan record 2026-09-02) | transferred ceiling ~11.9 | ~23.3 @30k best (no 30 t/s claim; residual vmlx kernel blocker recorded) |

## Provenance & quant (all recorded, pins immutable)

- configs/model-lineup.json is the manifest: per-model pinned revisions, quant
  recipes, GGUF reference blobs (sha256), artifact_audit class, publication
  handles (Tostibrown/Qwen3.8-27B-5bit-affine-g64 @ f592c6fb — the only
  Mei-produced published artifact; all others upstream, referenced not
  relabeled).
- Mei-produced = Qwen3.8-27B 5-bit affine g64 (mlx_lm.convert 0.31.3/0.32.0,
  source rev 1d4bf0f2, tree digest 77d181b3; NOT-UD/non-GGUF provenance).
- Aligned repacks = Mei tools/align_safetensors.py header-pad (payload
  bit-identical, 0 unaligned): Ornith-35B-aligned only.
- GGUF references: Ornith Q4_K_M (ornith-ai), Qwen3.8 UD-Q5_K_M (unsloth,
  no --spec-type), Heretic Q5_K_M (trohrbaugh), Gemma APEX-I-Quality (mudler).
  All behavioral comparisons; quant families differ across engines and are
  called out, never claimed bit-equivalent.

## Rollback-safe defaults (Mei source defaults UNCHANGED this tick)

- No Mei source/config-file default changed by this tick's work: the two
  measured env gates (VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0 for
  Ornith-35B-aligned and Gemma4) are carried by the bench launch commands and
  documented as REQUIRED per fitted config; compiled-decode, KV-quant, unsafe
  env, MLXPress routing all stay default-off (each has a measured-recorded
  verdict). Gemma's arch-scoped prefill 256 default (2026-09-03) unchanged.

## #14 readiness (updated 2026-09-04 2nd pass)

All four finalists pass correctness/provenance gates and have 3-repeat speed
rows; optimization loop #12 per-model levers are closed with measured records
(Ornith: wins documented; Qwen: ceiling accepted; Heretic: transfer accepted;
Gemma: fuse-gate win + vmlx kernel residual blocker) — todo #12 CHECKED OFF this
tick (Kiem note c8b1adcd; evidence spot-verified from raw probe files, see note).

Concrete #14 run plan (8 configs, one server at a time, each a full
sanity+hermes_ops+coding battery; rows append to local-model-bench
results/log.jsonl, never overwrite; results-mei/ acceptance records live in
~/.local/share/local-model-bench/results-mei/):

1. Ornith-1.5-35B-A3B/mei.yaml (aligned dir + fuse-off env, port 8024)
2. Ornith-1.5-35B-A3B/gguf.yaml (ornith-ai Q4_K_M, temp 0.6 — parity OK)
3. Qwen3.8-27B/mei.yaml (4-bit, port 8025)
4. Qwen3.8-27B/gguf-unsloth-ud-q5-64k.yaml (UD-Q5_K_M — quant-family caveat)
5. Qwen3.8-27B-Uncensored/mei.yaml (Heretic 4-bit, port 8026)
6. Qwen3.8-27B-Uncensored/gguf-heretic-q5.yaml (trohrbaugh Q5_K_M — same caveat)
7. Gemma-4-26B-A4B/mei.yaml (fuse-off env, prefill 256, port 8027)
8. Gemma-4-26B-A4B/gguf-apex-i-quality.yaml (mudler APEX-I-Quality — same base
   model, DIFFERENT quant recipe from Mei 4-bit affine; call out in comparison)

BLOCKER (2026-09-04): run_bench.py hard disk gate = 30 GB free
(bench_common.MIN_FREE_DISK_GB_BEFORE_LAUNCH); measured 14.3 GB free (97% full).
~16+ GB must be freed; safe candidates are documented-disposable Mei runtime
scratch (mei-runtime/kv-cache* ≈ 25.4 GB, mei-runtime-* stage dumps ≈ 18.7 GB)
— deletion is outside autonomous authority, awaiting Tijs go-ahead. GGUF cache
blobs (75 GB) NOT candidates. Next tick: re-check df; if >= 30 GB free, launch
config pair #1 (full Kiem note dbdd632a with commands and details).