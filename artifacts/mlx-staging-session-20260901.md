# Mei session artifact — Ornith-first, four-model MLX staging (2026-09-01)

Kiem plan `0b87b76a` (Ornith-first Mei optimization and four-model MLX parity).
This session was run under a fully Metal-occupied window: the foreign
`llama-server` (Ornith-1.5-35B-A3B Q4_K_M, port 8017, alias
`bench-f51c7e72e2ad`, PID 3968) owns the GPU working set and MUST NOT be
stopped. Per plan gate, no Mei server was launched, no second backend ran,
and no GPU measurement row was taken. All work this session is CPU-side
reproducible staging, provenance correction, tooling, and tests.

## Machine state (measured)
- M1 Max, 32 GB unified; foreign llama-server resident ~10.9 GB RSS; vm_stat
  free pages ≈ 7.2K/2048K, swap used 4.8/6 GB → 35B MLX (~19.5GB) memory-blocked.
- `cocore agent serve` (present last audit) has exited; only llama-server +
  marktplaats MCP remain.
- No Mei GPU measurement ran; nothing contaminated; no foreign process touched.

## Key finding — Heretic MLX lineage now has a native 4-bit (manifest correction)
Prior audit/README/manifest claimed `orcarouter/Qwen3.8-27B-Uncensored-MLX`
ships "only a 2-bit directory". Verified today that the repo is a **single
commit `14963e70f886455cf93090ac95bdbf4c8730cbe1` (2026-08-27 “Initial
commit”) == the already-pinned revision**, and it contains `2-bit`, `4-bit`,
`6-bit`, `8-bit`, and `mtp` subdirectories.
- `4-bit/config.json`: `model_type: qwen3_5`, `quantization {bits:4,
  group_size:64, mode:affine}`, `Qwen3_5ForConditionalGeneration` (VLM).
- 3 shards, 16,054,541,599 B total; same `chat_template.jinja` (8952 B),
  `tokenizer.json` (19,989,325 B), `vocab.json` (6,722,759 B) as the gated
  source `orcarouter/Qwen3.8-27B-Uncensored` → lineage is the Uncensored
  derivative, distinct from base `Qwen/Qwen3.8-27B`.
- Source repo `orcarouter/Qwen3.8-27B-Uncensored` is **gated (auth required)**,
  so a source-based conversion path is impractical; the existence of the
  first-party 4-bit MLX removes the prior conversion-blocker. THIS IS NOT a
  silent substitution: it is the same-author Uncensored lineage, pinned to the
  original commit, 4-bit (tool-reliable), and recorded separately from base
  Qwen3.8. Stage requires only `allow_patterns=["4-bit/*"]`.

## Staging (reproducible, GPU-independent) — in progress
New `tools/stage_mlx_checkpoint.py` downloads a pinned revision into the
isolated `mei-models/` root as real files, verifies safetensors shard
byte-completeness against the model's own index, records a
`*.provenance.json`, and quant/arch metadata. It never claims loadability
(that is a separate GPU-gated step).

| MLX checkpoint | pinned revision | target (mei-models/) | quant (config) | status |
|---|---|---|---|---|
| `mlx-community/Qwen3.8-27B-4bit` | 3e6447f082e89cc7f0bc6e5441afd38dfce760ff | Qwen3.8-27B-4bit | 4-bit affine g64 (`qwen3_5`) | staged-complete (loadability GPU-pending) |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 0d77464eeb233a2da68ebf9d7dc4edaac7db956d | gemma-4-26b-a4b-it-4bit | 4-bit affine g64 (`gemma4`) | staged-complete (loadability GPU-pending) |
| `orcarouter/Qwen3.8-27B-Uncensored-MLX` → 4-bit/ | 14963e70f886455cf93090ac95bdbf4c8730cbe1 | Qwen3.8-27B-Uncensored-MLX-4bit | 4-bit affine g64 (`qwen3_5`) | staging-in-progress |

Verified on-disk: Qwen3.8-27B-4bit = 16,054,541,349 B (config_sha
`14b65a0e…`); gemma-4-26b-a4b-it-4bit = 15,341,205,776 B (config_sha
`419e13a2…`). orcarouter 4-bit (target 16,054,541,599 B) download continues in
background and will be verified on completion. Provenance files:
`<target>.provenance.json` next to each staged directory.

Provenance corrections:
- Manifest `weight_bytes: 4,665,462,000` for `mlx-community/Qwen3.8-27B-4bit`
  is **erroneous** — measured on-disk safetensors > 11 GB (3 shards
  `model-00001..00003-of-00003`), consistent with a full 27B 4-bit, not 4.7 GB.
  The 4.66 GB figure was an estimate common to both Qwen and Gemma entries;
  both are corrected with measured values on completion.
- Gemma 26B-A4B 4-bit actual size likewise being measured (not 4.5 GB assumption).

## Tests (CPU, non-Metal)
- `swift test --filter 'ServerConfigParsingTests|SSMAnchorBoundariesTests|
  OpenAITypesTests|CacheRestoreTrackerTests|QuantizedRotatingKVCacheTests'`:
  non-Metal suites passed 0 failures; the KV-precision test that touches Metal
  errors on missing metallib as expected while the foreign server owns Metal.
- `configs/model-lineup.json` remains the source of truth; updated alongside.

## Next (GPU-gated)
When the foreign owner releases Metal: verify loadability of the staged 4-bit
checkpoints (Qwen3.8, Gemma4, Heretic-4bit) + run Ornith 9B proxy and 35B
acceptance/optimization matrix with >=3 repeats. Loadability is NOT asserted
here.
