# Mei optimization log (9B path) — 2026-08-29 evening

Machine: sulaco, Apple Silicon g13s, 32 GiB. Model: ornith-ai/Ornith-1.5-9B-MLX-4bit
(hybrid qwen3_5_moe: 32 layers, 8 attention KV layers, 24 mamba/GatedDelta layers).
Serving: mei (vmlx-swift pinned aeb5e21c), prefill step 512, temp 0, 64 max tokens
unless stated. Decode figures are engine usage.tokens_per_second. Every row was a
separate HTTP request against the live server; exact-token prompts are " hello"*N.

## Baseline (eager decode, fp16 KV, cache-reuse+disk tier)

| context tokens | decode tok/s | prefill tok/s | notes |
|---|---|---|---|
| 512      | 27.1 | 115   | prefill incl. per-request fixed cost |
| 4096     | 22.8 | 232   | |
| 16384    | 23.4 | 248   | |
| 33175    | 5.3  | 305   | decode cliff between 16K and 33K |
| 45000 (fresh)  | 6.4  | 549   | 45K already partly in disk cache from earlier bench |
| 45001 (reuse)  | 14.3 / 13.3 / 22.6 | restore 4063-5767 | variance across repeats; prefill_ms=7.8-11s = disk restore of 45K |

Benchmark row set (bench_mei 45K): short 28.1; tool ns/stream 27.7/27.7;
long_loaded_fresh 5.2 (64 tok, 298s total); long_loaded_reuse 13.2 (cached=45000);
long_chat_40k 10.3 @33K prompt (correct "cache-ready", no collapse).

Interpretation: decode at 45K loaded sits in the bare-mlx band (10.75-15.7 @80K
per local-model-bench) — the server is NOT collapsing below library capability,
but 40 tok/s at 40-50K is not reachable with the default eager path.

## Experiment 1: enableCompiledDecode = true (--compiled-decode true)

| context tokens | decode tok/s | prefill tok/s | notes |
|---|---|---|---|
| 512      | 47.2 | 227   | +74% decode vs eager |
| 16384    | 34.8 | 234   | +49% decode vs eager |
| 33175    | n/a  | ~13   | prefill collapsed ~20x (29184→30720 over ~2min) |
| 45000+   | n/a  | n/a   | aborted; long prefill unusable |

Verdict: compiled decode is a short-context win and a long-context prefill
regression on this pin. Flag retained (default off) as a documented A/B lever;
NOT used for the primary loaded-context benchmark.

## Decision record
- Eager decode + fp16 KV is the default config for all artifact runs.
- kv-bits 4 and prefill-step-size A/B runs follow in this log if executed.