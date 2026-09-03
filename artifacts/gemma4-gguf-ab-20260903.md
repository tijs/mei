# Gemma 4 26B-A4B GGUF/llama.cpp reference A/B vs Mei MLX (todo 0b87b76a#8 leg) — 2026-09-03

Unit: todo 8 (common matrix per loadable model vs finalized GGUF/llama.cpp
references) — Gemma reference leg. Mei HEAD 55c865a. Clean window (no
foreign inference processes re-checked before launch; Mei-owned ports
8076/8077). local-model-bench READ-ONLY (runtime logs under
`~/.local/share/local-model-bench/mei-runtime-gemma-ceiling/` are owned
disposable dirs).

## Reference GGUF (content-pinned)

`mudler/gemma-4-26B-A4B-it-APEX-GGUF` rev `e79ec24ceca5` file
`gemma-4-26B-A4B-APEX-I-Quality.gguf`: sha256 `472828ccd0…3304` == HF LFS
blob oid (blob sha re-verified 2026-09-02 as the finalized APEX-I-Quality
reference), 20,576,637,248 B, loaded from the Hugging Face cache snapshot.
Header: arch `gemma4`, 30 blocks, context_length 262144, file_type 18
(Q6_K — the APEX-I-Quality flavor is a 6-bit GGUF; the Mei MLX side is
4-bit affine g64 per quantization policy, so bit depth differs and the
A/B below is engine-vs-reference-flavor, not bit-equal), no MTP head.
llama.cpp brew build 10470 (commit 34af94cd9); context = explicit
`--ctx-size 65536`.

## Measured — GGUF/llama.cpp side (artifacts/gguf-ceiling-gemma4-apexq-20260903.json, gguf-ref-gemma4-apexq-20260903.json)

- short decode 3x fresh (max_tokens 32): 47.705/47.869/47.716 → mean **47.76 t/s** (sd 0.090); chat row 47.67
- peak RSS: 22.43 GB (probe rows up to 23.98 GB on the tool-call row)
- coding 4/4 PASS (swift_fibonacci 46.97, python_json_sum 46.90, sql_users_query 47.69, shell_rename 47.70 t/s)
- native tool call `add_pair{"a":15,"b":27}` PASS — llama.cpp's gemma4 path emitted proper JSON integers (contrast: Mei MLX gemma emits string-typed args `{"a":"15","b":"27"}` per its template → the recorded strict-schema FAIL, user go/no-go)
- 30k long-context: fresh fill 29,413 tok @ **231.1 pps** (128.6 s), decode at 30k loaded **37.08 t/s**; slot reuse 29,412/29,413 cached, decode 37.17 t/s; RSS 23.82 GB
- identity limerick: no refusal cues (content-empty thinking-lock consistent with the other engines this family set)

## A/B vs Mei MLX (MLX = mlx-community/gemma-4-26b-a4b-it-4bit, disk-KV config,
rows from artifacts/gemma4-26b-common-matrix-20260903.md + probe files, 2026-09-03 morning)

| row | GGUF APEX-I-Quality (Q6_K) | Mei MLX 4-bit | delta |
|---|---|---|---|
| short decode (≤64 tok, 3x) | 47.76 t/s (sd 0.09) | 51.35 t/s (sd 0.08; 51.35/51.43/51.27) | MLX **+7.5%** |
| 30k loaded decode | 37.08 / 37.17 (reuse) | n/a — 30k decode t/s not recorded in the Gemma MLX longctx artifact (gap; queued for the next matrix/optimization leg or the final four-model comparison) | — |
| 30k prefill | 231.1 pps | ~141 pps (212.9 s fill) | GGUF **+64%** |
| peak memory | 24.0 GB (tool row) | 24.08 short / 27.58 @6.2k / 30.13 @65k-cap | parity; fits 32 GB both |

- coding 4/4 PASS both engines.
- tool: GGUF JSON-int args PASS; Mei MLX string args = probe strict-schema FAIL
  (model-faithful per Gemma's template; coercion needs USER go/no-go —
  unchanged blocker (a) of todo 8).
- identity: no refusal cues both.
- Direction matches the Ornith-35B finding: llama.cpp prefill markedly faster,
  Mei MLX decode slightly faster; both far from the dense-Qwen pattern where
  MLX beats GGUF by +41-82% on decode.

## Uncertainty

- 30k rows single-shot pairs (not 3 repeats); short decode 3 cold repeats.
- Different bit depths by design (Q6_K reference vs 4-bit Mei quant) —
  decode deltas are flavor-level, not bit-equal.
- GGUF KV-reuse = llama.cpp slot cache; Mei = disk-KV tier; recorded, not
  equated. Mei MLX rows measured this morning (2026-09-03); GGUF rows today.

## Todo status

`0b87b76a#8` — with this leg, EVERY loadable model now has its same-suite
GGUF/llama.cpp reference row (base Qwen3.8 UD-Q5_K_M: 9.286 t/s ceiling
09-02; Heretic trohrbaugh Q5_K_M: 8.619 t/s + full adapter suite 09-03;
Ornith 9B/35B Q4_K_M: full rows + adapter suites 09-03; Gemma APEX-I
06-bit: full rows + adapter suite 09-03). Remaining inside todo 8: only
(b) Gemma tool strict-schema string-args acceptance — USER go/no-go
(blocked autonomously). Todo 8 can close after that user decision and an
explicit plan-side record; measurable legs are otherwise complete.

## Reproducibility

```bash
GGUF=$HOME/.cache/huggingface/hub/models--mudler--gemma-4-26B-A4B-it-APEX-GGUF/snapshots/\
e79ec24ceca5060b55b9a267c565b0b0843f3678/gemma-4-26B-A4B-APEX-I-Quality.gguf
MEI_RUNTIME_BASE=$HOME/.local/share/local-model-bench/mei-runtime-gemma-ceiling \
python3 tools/llama_ceiling.py --gguf "$GGUF" \
  --alias mudler/gemma-4-26B-A4B-it-APEX-GGUF:APEX-I-Quality --mode short --port 8076 \
  --ctx-size 65536 --max-tokens 32 --repeats 3 \
  --gguf-sha256 472828ccd00bcf52d6ca72e97d49526fd254371a90ae6a77505409e6e2bf3304 \
  --gguf-repo mudler/gemma-4-26B-A4B-it-APEX-GGUF \
  --gguf-revision e79ec24ceca5060b55b9a267c565b0b0843f3678 --arch gemma \
  --output artifacts/gguf-ceiling-gemma4-apexq-20260903.json
llama-server --model "$GGUF" --host 127.0.0.1 --port 8077 --ctx-size 65536 \
  --temp 0 --top-p 0.95 --top-k 20 --alias mudler/gemma-4-26B-A4B-it-APEX-GGUF:APEX-I-Quality \
  --parallel 1 --no-webui --metrics
python3 tools/gguf_ref_probe.py --base-url http://127.0.0.1:8077/v1 \
  --model mudler/gemma-4-26B-A4B-it-APEX-GGUF:APEX-I-Quality --pid <llama-server pid> \
  --output artifacts/gguf-ref-gemma4-apexq-20260903.json
```

Artifacts: `gguf-ceiling-gemma4-apexq-20260903.json`, `gguf-ref-gemma4-apexq-20260903.json`,
this md. Server killed; port released.