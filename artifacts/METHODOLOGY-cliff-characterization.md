# Mei cliff characterization — hypothesis notebook

Working document for the five workstreams. Artifacts are timestamped JSON +
server logs under artifacts/; this file records hypotheses, exact commands,
and result interpretation. Machine: sulaco, M1 Max g13s, 32GB unified,
recommended Metal working set ~26.8GB, no concurrent backend allowed for
official numbers (contention boundary recorded per run).

## 0. Model facts (from ornith-ai/Ornith-1.5-9B-MLX-4bit/config.json)

- model_type qwen3_5_text; hidden 4096; 32 layers; full_attention_interval 4
  -> attention layers at 3,7,11,...,31 (8 layers), GatedDelta (SSM) for the
  other 24; kv heads 4; head_dim 256; vocab 248320; max_position 262144;
  no sliding_window in config -> attention layers are full-context.
- Mei's RingKV: `RotatingKVCache(maxSize: maxKVSize=contextCap+4096, keep: 4)`
  with maxKVSize 69632 at the default 64K context cap. At a 45K prompt the
  ring never rotates; decode attention scans all 45K keys per token.
- vmlx pinned revision aeb5e21c (8 upstream commits behind origin/main, none
  touching KV quant or compiled decode).

## 1. Hypotheses under test

H-1 (cliff shape): eager decode tok/s falls with context because every
decode step attends over the full stored KV (9B: 8 layers x 4 kv heads x 256
dim; fp16). From the 2026-08-29 log: 512 -> 27.1, 4K -> 22.8, 16K -> 23.4,
33K -> 5.3, 45K fresh -> 6.4, 45K reuse -> 13.2. The sweep adds profile
splits (decode.model_forward vs decode.sample, prompt.prepare_total) and
allocator state per row.

H-2 (kv quant): quantized attention KV (4/8-bit affine, QuantizedRotatingKV
cache) reduces per-step memory/bandwidth; at 45K the KV read is ~half the
per-token working set read; expect decode gain bounded by attention's share
of step time. NOTE upstream: `maybeQuantizeKVCache` affine path only converts
KVCacheSimple; RotatingKVCache.toQuantized() was an explicit fatalError; the
`guard !needsCacheQuantization` in the coordinator store path refused caching
for ANY kvBits request. Both fixed in patches/0001-0003.

H-3 (compiled decode): the promote+trace setup sizes Compilable buffers to
promptOffset + maxTokens and traces the full-length attention graph right
after prefill -> multi-minute prefill tax at 33K+ (measured ~20x prefill
collapse, and 45K unusable). Thresholding the setup (skip when offset >
--compiled-decode-threshold) keeps the short/medium compiled win (47.2 tok/s
short, 34.8 at 16K) without the long-context tax (patches/0003).

H-4 (hardware ceiling): llama.cpp with equivalent 9B Q4_K_M GGUF at the same
45K prompt bounds what ANY engine can do on this GPU for full-context decode.
Gate: >=40 -> fork strongly; 25-39 -> hardware ceiling; <25 -> pivot to reuse
latency + prefill throughput. llama.cpp KV is quantized-on by default (q8_0
zeRO?), and llama.cpp Metal kernels are the fastest known reference.

H-5 (window lever, later): a smaller maxKVSize would bound attention to a
sliding window and make decode nearly context-independent — but truncates
what attention can see for a full-attention model (quality cliff). Only
test after H-1..H-4 numbers land, and only as a documented correctness-
bounded probe, not a default.

## 2. Profile env and endpoints used

- MLXPRESS_GENERATION_PROFILE=1 -> per-generation stage dump on stderr:
  prompt.prepare_total, prompt.model_prepare, prompt.sample,
  decode.model_forward, decode.compiled_forward, decode.sample, etc.
  (Evaluate.swift MLXPressGenerationProfileState; dumpAndReset at
  generation end with total ms + count + avg per stage).
- GET /v1/mei/status -> active/cache/peak bytes, limits, device info.
- usage.prefill_ms / generate_ms / tokens_per_second / cached_tokens from
  the engine's GenerateCompletionInfo (Evaluate.generateTask).

## 3. Exact commands (rerun)

```bash
# characterization sweep (baseline fp16, contexts 512..45K, cell matrix)
MEI_MODEL_DIR=... python3 tools/sweep_mei.py --model-dir ... --model-id ... \
  --contexts 512,4096,16384,33175,45000 --prefill-steps 512 \
  --ssm-rederive true --kv-bits none --repeats-45k 3 --chat-40k \
  --output artifacts/sweep-cliff-<ts>.json

# hardware ceiling (llama.cpp, own port 8074)
python3 tools/llama_ceiling.py --gguf ~/.local/.../mei-models/gguf/Ornith-1.5-9B-Q4_K_M.gguf \
  --alias ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M --repeats 3 --chat-40k \
  --output artifacts/llama-ceiling-<ts>.json

# patch series reproduction
bash scripts/apply_vmlx_patches.sh --reset && bash scripts/apply_vmlx_patches.sh
```

## 4. Variant matrix definition (one variable at a time)

| variant | flags | status |
|---|---|---|
| baseline fp16 | (defaults) | current pin |
| compiled, no threshold | --compiled-decode true | known long-prefill tax (regression) |
| compiled, threshold 16384 | --compiled-decode true --compiled-decode-threshold 16384 | patch 0003 |
| kv8 | --kv-bits 8 | patch 0001+0002 |
| kv4 | --kv-bits 4 | patch 0001+0002 |
| compiled-threshold + kv8 | both | combined |
| ssm off | --ssm-rederive false | A/B of the re-derive pass |
| prefill step | --prefill-step-size 512/2048/4096 | prefill-shape sweep |
| cache limit | --cache-limit-bytes 2G/8G | allocator-pool sweep |