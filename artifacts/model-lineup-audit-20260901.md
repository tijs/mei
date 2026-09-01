# Model lineup audit — four-candidate MLX parity (2026-09-01)

Session executing Kiem plan `0b87b76a` (Ornith-first Mei optimization and
four-model MLX parity). This artifact records the measured cache/provenance
state and the MLX selections; it is durable evidence distinct from transient
runtime state.

## Machine state (measured at start)

- Apple M1 Max, macOS 26.5.2, 32 GB unified memory.
- Foreign workload still owns Metal and MUST NOT be stopped:
  - `llama-server` PID 3968: `--hf-repo ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M`
    on 127.0.0.1:8017, alias `bench-f51c7e72e2ad`, ctx 65536, chat-template
    `configs/Qwen3.8-27B/chat_template.jinja`. Resident RSS ≈ 15.7 GB. This is
    the live Ornith 35B GGUF reference. `/v1/models` responds.
  - `cocore agent serve` PID 3873, resident RSS ≈ 11 GB.
- **GPU measurements are blocked this session** by the resident reference
  (plan gate: never run two model servers concurrently; no GPU-contention wait
  > 5 min). CPU-side work proceeded. Any decode/prefill row taken now would be
  invalid because the 35B GGUF and its KV cache hold the Metal working set.

## Local staged models (usable)

| repo | arch | quant | weights | staged path | status |
|---|---|---|---|---|---|
| ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit | qwen3_5_moe | 4-bit affine g64, gates 8-bit | 19,509,024,201 B (4 shards) | mei-models/… | complete, blocked-on-memory |
| ornith-ai/Ornith-1.5-9B-MLX-4bit | qwen3_5 | 4-bit affine g64 | 5,038,161,163 B | mei-models/… | complete, validated |

Note: `~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-MLX-4bit`
holds only a **partial/re-incomplete** download (4 `.incomplete` weight shards)
plus metadata; it is NOT the usable copy and MUST NOT be deleted or mistaken
for it. The staged copies under `mei-models/` are the complete ones.

## Cached GGUF references (all complete, integrity-checked)

| reference | quant | arch/size | MTP | blob SHA-256 (HF cache) | bytes |
|---|---|---|---|---|---|
| ornith-ai/Ornith-1.5-35B-A3B-GGUF → Ornith-1.5-35B-Q4_K_M.gguf | Q4_K_M | qwen3_5_moe | — (reference, ctx 65536) | 42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f | 21,713,463,040 |
| unsloth/Qwen3.8-27B-GGUF → UD-Q5_K_M | Q5_K_M | qwen35 / 27B / 65 blk | yes (nextn=1) | 2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd | 19,771,509,664 |
| mudler/gemma-4-26B-A4B-it-APEX-GGUF → APEX-I-Quality | APEX-I-Quality | gemma4 / 26B-A4B / 30 blk | none | 472828ccd00bcf52d6ca72e97d49526fd254371a90ae6a77505409e6e2bf3304 | 20,576,637,248 |
| trohrbaugh/Qwen3.8-27B-heretic-ara-gguf-Q5 | Q5_K_M | qwen35 / 27B / 65 blk | yes (nextn=1) | e79fdc96668747e3d629568582209b3bfab3c3a8496f8b90f7098a47238556a4 | 19,704,559,264 |

- `tools/gguf_meta.py --check-mtp` confirms the two Qwen3.8 references carry an
  MTP/Next-N head (nextn_predict_layers=1); llama.cpp comparators must run
  WITHOUT `--spec-type`. Gemma APEX-I-Quality has no MTP head; arch `gemma4`
  may need llama.cpp >= 10470.
- Only the APEX-I-Quality Gemma GGUF is cached (Compact/mmproj blobs absent).
- Ornith 9B staged GGUF `Ornith-1.5-9B-Q4_K_M.gguf` = 5,780,090,816 B (digest
  abdd624b verified in session G).

## MLX selections (exact pinned revisions, inventory-only — no new downloads)

| candidate | MLX repo | pinned revision | bits/weight | decision |
|---|---|---|---|---|
| Ornith-1.5-35B-A3B Base Q4 (primary) | ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit | HF snapshot 19504d912fa8 (staged complete) | 4-bit, g64 | use as-is; blocked-on-memory |
| Qwen3.8-27B (secondary) | mlx-community/Qwen3.8-27B-4bit | 3e6447f082e89cc7f0bc6e5441afd38dfce760ff | 4-bit, 4,665,462,000 B | regular 4-bit MLX comparator (NOT UD-Q5) |
| Gemma 4 26B-A4B APEX-I-Quality | mlx-community/gemma-4-26b-a4b-it-4bit | 0d77464eeb233a2da68ebf9d7dc4edaac7db956d | 4-bit, 4,514,678,350 B | fit-on-32GB default; 5/6/8-bit only if headroom |
| Qwen3.8 Uncensored/Heretic | orcarouter/Qwen3.8-27B-Uncensored-MLX | 14963e70f886455cf93090ac95bdbf4c8730cbe1 (source orcarouter/Qwen3.8-27B-Uncensored @ 404ea47a) | only a 2-bit dir exists | **blocker**: needs reproducible 4-bit conversion from orcarouter abliterated source; no silent substitution of base Qwen3.8 |

## Validation run (CPU-side)

- `swift test --filter 'ServerConfigParsingTests|SSMAnchorBoundariesTests|OpenAITypesTests|CacheRestoreTrackerTests'`
  → 34/34 passed, 0 failures (same baseline as session G).
- `configs/model-lineup.json` parses; 5 models present.

## Files changed (this session)

- `configs/model-lineup.json` — rewrote to 5 entries with exact pinned
  revisions, byte sizes, blob digests, quant settings, staged paths, status.
- `README.md` — models section expanded to four-candidate lineup; added
  "MLX quantization strategy" section.
- this artifact (`artifacts/model-lineup-audit-20260901.md`).
