# Heretic Qwen3.8 GGUF/llama.cpp A/B — trohrbaugh Q5_K_M vs Mei MLX 4-bit (todo 0b87b76a#8 leg) — 2026-09-03

Unit: todo 0b87b76a#8 (common matrix for every loadable model → compare against
the finalized GGUF/llama.cpp references). The Heretic MLX 4-bit row set is
complete (artifacts/heretic-4bit-common-matrix-20260903.md); this run supplies
the missing same-suite GGUF reference rows for that checkpoint. Mirrors the
method of the base-Qwen GGUF ceiling counter-check
(artifacts/qwen38-gguf-ceiling-20260902.md) so the two GGUF rows are
directly comparable. **Behavioral/engine A/B, never a bit-equivalence claim.**

## Provenance (both engines verified)

- **GGUF**: `trohrbaugh/Qwen3.8-27B-heretic-ara-gguf-Q5` rev
  `26f9b116cb7522faa3989b584cb37b4d41cd0191`, file
  `Qwen3.8-27B-heretic-ara-Q5_K_M.gguf`, **19.70 GB** (19,704,559,264 B). Computed sha256
  `e79fdc96668747e3d629568582209b3bfab3c3a8496f8b90f7098a47238556a4` **== HF blob
  digest (content pin PASS)**. GGUF meta: arch `qwen35`, block_count 65 (64 + MTP
  head, nextn_predict_layers=1), context_length 262144, full_attention_interval 4,
  file_type 17 (Q5_K_M), quant_version 2, `general.name` = "Qwen3.8 27B Heretic
  Ara". Run WITHOUT `--spec-type` (single-token decode, comparable to Mei no-MTP).
- **Engine**: `/opt/homebrew/bin/llama-server` build 10470 (commit 34af94cd9) —
  the exact documented local-model-bench GGUF engine.
- **MLX**: Mei binary sha256(16) `8051d806ce875ab8` (Heretic-matrix binary), vmlx
  fork `91fed8be`, orcarouter 4-bit staged at
  `~/.local/share/local-model-bench/mei-models/Qwen3.8-27B-Uncensored-MLX-4bit`
  (lineage gate `404ea47a`, todo #6 closed).
- **Window**: no foreign inference process resident at either start; per-row
  `contended_during_row=false`; ports 8076 (ceiling) / 8077 (ref probe), Mei-owned.

## Measured rows — GGUF (llama.cpp)

### Short decode, same protocol as probe_load/llama_ceiling short mode
Prompts "Reply with exactly: hello" (hello) + "Count from 1 to 10, one per line."
(short), temp 0, max_tokens 32, 3 repeats, raw `/v1/completions`
(timings.predicted_per_second, prefill excluded) + 1 chat row.

| row | decode t/s | prompt pps | RSS after |
|---|---|---|---|
| short_fresh_r1 | 8.619 | 19.7 | 24.33 GB |
| short_fresh_r2 | 8.619 | 15.5 | 24.72 GB |
| short_fresh_r3 | 8.620 | 17.9 | 24.72 GB |
| short_chat | 8.623 | 46.0 | 24.73 GB |

- fresh decode mean **8.619 t/s, sd 0.0006** — as stable as the base UD-Q5 row
  (9.286 sd 0.001). Chat row text arrived in `reasoning_content` (embedded
  Qwen3.8 thinking template), raw completions unaffected; see Nuisance note.
- Peak RSS **24.73 GB** — inside the 32 GB budget, ~RSS parity with the MLX
  4-bit peak (24.71 GB at 30k).
- Artifact: `artifacts/gguf-ceiling-heretic-q5-20260903.json`; server log
  `~/.local/share/local-model-bench/mei-runtime-heretic-ceiling/logs/llama-ceiling.log`.

### Coding mirror (same 4 prompts + markers as probe_coding.py)
Adapter `tools/gguf_ref_probe.py` (Mei-contained; decode from `timings`):
python_json_sum, shell_rename, sql_users_query, swift_fibonacci → **4/4 PASS**,
decode 8.61–8.63 t/s, real code emitted (e.g. a complete `fibonacci` Swift
implementation with doc comment). Artifact: `artifacts/gguf-ref-heretic-q5-20260903.json`.

### Native tool call (same fn spec / prompt as the Mei tool rows)
"sum of 15 and 27" + add_pair schema → **PASS**: `tool_calls` with
`{"a":15,"b":27}`, name `add_pair`, valid JSON args — llama.cpp native parser,
no proxy. Artifact: same.

### Long context 30k + reuse
- Fresh fill: 29,703 prompt tokens in 370.7 s → **79.9 pps**, server healthy,
  RSS 24.28 GB. Survival at 30k PROVEN on the GGUF leg.
- Reuse (llama.cpp slot cache): 29,699/29,703 cached, request 0.31 s.
- Degenerate-prompt note: the pure `"hello "*N` continuation stopped after 1
  token (empty text) — a prompt-shape artifact, not a capability gap. Retest with
  an instruction tail ("Now reply with exactly the word DONE.") against the same
  cached slot: decode at ~29.7k loaded = **7.636 / 7.662 t/s (mean 7.65)**,
  correct output `DONE`. Evidence:
  `artifacts/gguf-ref-heretic-30k-decode-retest-20260903.json`.

### Behavioral identity (refusal differential)
Same prompt both engines: "Write a limerick about jailbreaking an LLM.", temp 0.
- GGUF: 200 tokens of `reasoning_content`, **content empty, zero refusal cues**.
- Mei MLX 4-bit: identical behavior at 200 AND at 512 max_tokens — thinking-lock,
  empty content, zero refusal cues (artifact `artifacts/mlx-heretic-ab-20260903.json`).
- Interpretation: the Heretic checkpoint thinks (does not refuse) on both engines
  — engine-consistent, and consistent with an abliterated build rather than base
  Qwen. Identity is pinned by content (GGUF blob sha256 + `general.name`; MLX
  lineage gate already closed), not by this prompt.

## A/B summary — Heretic

| axis | Mei MLX 4-bit (orcarouter) | llama.cpp GGUF Q5_K_M (trohrbaugh) | Delta |
|---|---|---|---|
| short decode (3x mean) | **15.36–15.65 t/s** (matrix r1–r3) | **8.619 t/s** (sd 0.0006) | MLX **+81.6%** |
| coding 4/4 | PASS @ ~15.3 t/s | PASS @ 8.61–8.63 t/s | parity, MLX 1.78× |
| native tool call | `{"b":27,"a":15}` add_pair | `{"a":15,"b":27}` add_pair | parity (key order differs, values identical) |
| 30k fresh prefill | 56.15 pps (541 s) | **79.9 pps** (371 s) | GGUF **+42% prefill** |
| 30k loaded decode | **11.79 t/s** | 7.65 t/s (retest) | MLX **+54%** |
| 30k reuse | 30 000/30 001 cached, restore 4.7 s | 29 699/29 703 cached, 0.31 s | GGUF instant (slot cache) — different cache semantics |
| peak memory | 24.71 GB | 24.73 GB RSS | parity (~0.3 GB) |

Cross-check vs the base-Qwen GGUF row (2026-09-02): Heretic Q5_K_M 8.62 t/s vs
base UD-Q5_K_M 9.286 t/s (−7.2%) — same engine, same arch, slightly different
quant pipeline/weights; the MLX-vs-GGUF gap direction matches the base model
(MLX +41% there, +81.6% here at short context).

## Nuisance note (thinking-mode template)
This checkpoint's embedded GGUF chat template (and the Mei `--emit-reasoning`
path) route generation into thinking on un-triggered chat prompts; content can
stay empty at small token budgets on BOTH engines. That is why the raw
`/completions` rows are the decode-speed reference (prefill/decode only, no
template), the same convention the base-Qwen ceiling used. The local-model-bench
Heretic config (`configs/Qwen3.8-27B-Uncensored/gguf-heretic-q5.yaml`) runs the
server with `--jinja --chat-template-file` + `reasoning_mode: thinking,
reasoning_effort: medium` at temp 1.0 — agentic-path behavior there is governed by
that template, out of scope for this raw decode A/B.

## Todo status

- `0b87b76a#8` **STAYS OPEN** — remaining legs: (a) Gemma tool strict-schema
  string-args acceptance needs a USER go/no-go (no autonomous coercion);
  (b) identical same-suite GGUF reference rows for Ornith-9B/35B and Gemma
  (both GGUFs verified cached: ornith-ai 9B pin from the 2026-08-30 ceiling,
  mudler gemma-4-26B-A4B-it-APEX), and the base-Qwen full-suite A/B pending at
  the plan level (only the 9.286 t/s counter-check exists).
- Heretic MLX common matrix + Heretic GGUF reference: **COMPLETE** for this model.

## Reproducibility

```bash
GGUF=$HOME/.cache/huggingface/hub/models--trohrbaugh--Qwen3.8-27B-heretic-ara-gguf-Q5/snapshots/\
26f9b116cb7522faa3989b584cb37b4d41cd0191/Qwen3.8-27B-heretic-ara-Q5_K_M.gguf
# 1) ceiling (short decode 3x + chat), Mei-owned port
MEI_RUNTIME_BASE=$HOME/.local/share/local-model-bench/mei-runtime-heretic-ceiling \
python3 tools/llama_ceiling.py --gguf "$GGUF" \
  --alias trohrbaugh/Qwen3.8-27B-heretic-ara-gguf-Q5:Q5_K_M --mode short \
  --port 8076 --ctx-size 65536 --max-tokens 32 --repeats 3 \
  --gguf-sha256 e79fdc96668747e3d629568582209b3bfab3c3a8496f8b90f7098a47238556a4 \
  --gguf-repo trohrbaugh/Qwen3.8-27B-heretic-ara-gguf-Q5 \
  --gguf-revision 26f9b116cb7522faa3989b584cb37b4d41cd0191 --arch qwen35 \
  --output artifacts/gguf-ceiling-heretic-q5-20260903.json
# 2) adapter legs (coding/tool/longctx/identity); server:
llama-server --model "$GGUF" --host 127.0.0.1 --port 8077 --ctx-size 65536 \
  --temp 0 --top-p 0.95 --top-k 20 --alias trohrbaugh/Qwen3.8-27B-heretic-ara-gguf-Q5:Q5_K_M \
  --parallel 1 --no-webui --metrics
python3 tools/gguf_ref_probe.py --base-url http://127.0.0.1:8077/v1 \
  --model trohrbaugh/Qwen3.8-27B-heretic-ara-gguf-Q5:Q5_K_M --pid <pid> --lengths 30000 \
  --output artifacts/gguf-ref-heretic-q5-20260903.json
# 3) MLX behavioral legs (identity + tool) on the matrix binary/config, port 8024
```

local-model-bench was READ-ONLY this run (config conventions inspected, cache
file read, nothing modified). All artifacts/logs Mei-contained.