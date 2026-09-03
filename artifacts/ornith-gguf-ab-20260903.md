# Ornith GGUF/llama.cpp reference A/B vs Mei MLX (todo 0b87b76a#8 leg) — 2026-09-03

Unit: todo 8 (common matrix per loadable model vs finalized GGUF/llama.cpp
references) — Ornith reference legs. Mei HEAD 8ec69d3 + tooling fix
(arch-aware GGUF header hygiene, see below). Clean window: zero foreign
inference processes before/during runs (contention gate re-checked before
each launch; no row flagged `contended_during_row`), Mei-owned ports
8076/8077/8078/8079. local-model-bench READ-ONLY (runtime logs under
`~/.local/share/local-model-bench/mei-runtime-ornith{9,35}-ceiling/` are
Mei-owned disposable dirs).

## References (content-pinned)

| model | GGUF | sha256 (== HF LFS blob oid) | size | arch | blocks | MTP |
|---|---|---|---|---|---|---|
| Ornith-1.5-35B-A3B | ornith-ai/Ornith-1.5-35B-A3B-GGUF Q4_K_M rev 12393612fd4f | `42739874cc…d41f` | 21,713,463,040 B | qwen35moe | 41 | present (nextn=1) |
| Ornith-1.5-9B | ornith-ai/Ornith-1.5-9B-GGUF Q4_K_M rev abdd624b12 (Mei-owned copy) | `70c112196e…e8fab6` | 5,780,090,816 B | qwen35 | 33 | present (nextn=1) |

llama.cpp: brew build 10470 (commit 34af94cd9), runs WITHOUT `--spec-type`
draft-mtp (single-token decode — comparable to Mei's no-MTP baseline).
Context set by explicit server `--ctx-size 65536` (headers carry
context_length 262144 for both; runtime gate is the launch arg).

Tooling fix (committed with this leg): `tools/gguf_meta.py` + `tools/llama_ceiling.py`
gate/record were hardcoded to the `qwen35.*` key prefix and exact arch
equality, which rejected this family's `qwen35moe` arch and would have
rejected `gemma4`. Now arch is a family-prefix gate (qwen35 matches
qwen35/qwen35moe; gemma matches gemma4), keys are arch-aware with bare
fallbacks, and a missing `context_length` header key is recorded as a note
(the sha256 content pin remains the FATAL exact-identity gate).

## Measured — GGUF/llama.cpp side (3 repeats for short decode; single-shot rows else)

### Ornith-1.5-35B Q4_K_M (artifacts/gguf-ceiling-ornith35-q4-20260903.json, gguf-ref-ornith35-q4-20260903.json)

- short decode 3x fresh (max_tokens 32): 50.555/50.630/50.538 → mean **50.57 t/s** (sd 0.048); chat row 50.55
- peak RSS: 23.02 GB (probe rows up to 24.55 GB during tool-call row)
- coding 4/4 PASS (swift_fibonacci 50.29, python_json_sum 50.67, sql_users_query 50.80, shell_rename 50.78 t/s)
- native tool call `add_pair{"a":15,"b":27}` PASS (single-shot, temp 0)
- 30k long-context: fresh fill 29,703 tok @ **510.6 pps** (59.6 s), decode at 30k loaded **39.86 t/s**; slot-cache reuse 29,699/29,703 cached, decode 39.89 t/s; RSS 23.74 GB
- identity limerick: no refusal cues (content-empty thinking-lock on claiming, same family behavior as noted for Qwen/Heretic; engine-consistent)

### Ornith-1.5-9B Q4_K_M (artifacts/gguf-ceiling-ornith9-q4-20260903.json, gguf-ref-ornith9-q4-20260903.json)

- short decode 3x fresh: 34.572/34.565/34.567 → mean **34.57 t/s** (sd 0.004); chat row 34.60
- peak RSS 8.25 GB (probe rows 8.9–10.1 GB)
- coding 4/4 PASS (~34.5–34.6 t/s)
- native tool call `add_pair{"a":15,"b":27}` PASS
- 30k long-context: fresh fill 29,703 tok @ **303.8 pps** (99.6 s), decode at 30k loaded **29.81 t/s**; reuse 29,699/29,703 cached, decode 29.82 t/s; RSS 8.90 GB
- identity limerick: no refusal cues

## A/B vs Mei MLX

### Primary target: Ornith-1.5-35B (Mei MLX = aligned repack + fuse-off + disk-KV, artifacts/ornith-35B-{aligned-repack,fuse-gateup-eliminated,longctx-80k-90k}-20260902*.md)

| row | GGUF Q4_K_M | Mei MLX 4-bit | delta |
|---|---|---|---|
| short decode (≤64 tok, 3x) | 50.57 t/s (sd 0.05) | 55.0 t/s (ready 3x: 54.99/55.00/55.00; limerick row 56.4) | MLX **+8.7%** |
| 30k loaded decode | 39.86 | ~47 (doc trend: ~55 short → ~47 @30k → ~35 @80k) | MLX **+18%** |
| 30k prefill pps | 510.6 | ~400 (30k in 76.5 s) | GGUF **+28%** |
| peak memory (32 GB machine) | 24.55 GB | 25.73 GB (ctx 65536 acc) / 26.49 GB (80k fill) | parity, both < 32 GB |

- coding 4/4 PASS both engines; tool `add_pair(15,27)` PASS both; identity: no
  refusal cues both.
- Direction note: unlike Qwen3.8/Heretic (where MLX beat llama.cpp by
  +41%/+82% short decode), Ornith-35B GGUF is within ~9% of Mei MLX on short
  decode and 28% FASTER on prefill — the MoE 35B is a different balance than
  the dense 27B. Both engines far below the 30 t/s target only at ≥80k
  context; at 30k and short context both exceed 30 t/s.
- Mei MLX rows are from the 2026-09-02 evidence set (commit 893e53d +
  longctx md); not re-run this tick (no overlap policy).

### 9B proxy (Mei MLX = the 2026-08-29/30 documented baseline)

| row | GGUF Q4_K_M | Mei MLX 9B (old baseline) | note |
|---|---|---|---|
| short decode | 34.57 t/s | 28.1 t/s | GGUF **+23%** vs old Mei-9B baseline |
| loaded decode | 29.8 t/s @30k | 13.24 t/s @45k (reuse) | different depths; approximate only |

The Mei 9B baseline predates the 35B optimization work; the 9B MLX proxy is
not the primary target and no claim is made from this row beyond
"GGUF engine is not a ceiling constraint for the 9B path".

## Uncertainty

- 30k decode rows are single-shot (fresh/reuse pair), not 3 repeats; short
  decode rows are 3 cold repeats (sd ≈ 0.05 / 0.004).
- GGUF KV-reuse = llama.cpp internal slot cache (cache_n 29,699); Mei
  disk-KV tier is a different mechanism — recorded, never claimed equal.
- Mei MLX 35B numbers were measured 2026-09-02 on the fitted env-gated
  config; GGUF rows measured today on stock llama.cpp defaults (f16 KV).

## Todo status

`0b87b76a#8` STAYS OPEN. Remaining legs: (a) Gemma 4 APEX-I-Quality GGUF
reference row (cached, blob sha `472828cc…` verified 09-02, file present)
+ its MLX comparison — queued next; (b) Gemma tool strict-schema
string-args acceptance — USER go/no-go, blocked autonomously. With the
Gemma GGUF row done, every loadable model will have its same-suite GGUF
reference; only the user gate remains inside todo 8.

## Reproducibility

```bash
# 35B ceiling (short 3x); 9B identical with its own pin/port
GGUF35=$HOME/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-GGUF/snapshots/\
12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf
MEI_RUNTIME_BASE=$HOME/.local/share/local-model-bench/mei-runtime-ornith35-ceiling \
python3 tools/llama_ceiling.py --gguf "$GGUF35" \
  --alias ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M --mode short --port 8076 \
  --ctx-size 65536 --max-tokens 32 --repeats 3 \
  --gguf-sha256 42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f \
  --gguf-repo ornith-ai/Ornith-1.5-35B-A3B-GGUF \
  --gguf-revision 12393612fd4f730ff5aadc23e9b8f9648aa49ceb --arch qwen35 \
  --output artifacts/gguf-ceiling-ornith35-q4-20260903.json

# adapter legs; server (2.06 [~190 s] log total incl. 30k fill):
llama-server --model "$GGUF35" --host 127.0.0.1 --port 8077 --ctx-size 65536 \
  --temp 0 --top-p 0.95 --top-k 20 --alias ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M \
  --parallel 1 --no-webui --metrics
python3 tools/gguf_ref_probe.py --base-url http://127.0.0.1:8077/v1 \
  --model ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M --pid <llama-server pid> \
  --output artifacts/gguf-ref-ornith35-q4-20260903.json
```

Artifacts: `gguf-ceiling-ornith{35,9}-q4-20260903.json`,
`gguf-ref-ornith{35,9}-q4-20260903.json`, this md. Servers killed at leg
end; ports free for the next worker.