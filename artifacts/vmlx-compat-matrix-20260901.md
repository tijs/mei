# Pinned-vMLX compatibility / loadability matrix — four-model MLX parity (2026-09-01)

Executing Kiem plan `0b87b76a` (Ornith-first Mei optimization and four-model
MLX parity). CPU-side, source-backed audit of every staged MLX candidate against
the **pinned vmlx-swift revision** Mei resolves. **No speculative runtime changes
were made this session**; this artifact records loadability where it can be
established by source facts, and marks GPU-gated load/generate checks as pending.

## Build of record

- Mei HEAD: `34bc84e` (unchanged from prior verified gates).
- vmlx-swift pinned revision: **`aeb5e21c195d8519609488ef75a25ce7e48d8f88`**
  (Package.resolved, `location = https://github.com/osaurus-ai/vmlx-swift.git`).
- Verified checkout present at
  `.build/checkouts/vmlx-swift` and scratch
  `~/.local/share/local-model-bench/mei-build/checkouts/vmlx-swift`, each at
  `git rev-parse HEAD == aeb5e21c195d8519609488ef75a25ce7e48d8f88`
  (last commit `2026-08-28 17:09:26 -0700`).
- MeV patch series `patches/0001..0005` applied to BOTH checkouts (sentinel
  `QuantizedRotatingKVCache` present in each `Libraries/MLXLMCommon/KVCache.swift`).

## vmlx factory registrations (source: `Libraries/MLXLLM/LLMModelFactory.swift`)

Text LLM factory dispatch (model_type → type/model):
- `qwen3_5` → `Qwen35Model` (L51).
- `qwen3_5_moe` → `Qwen35MoEModel` (L52–70); **conditional on `weight_format`**:
  `"mxtq"` routes to `Qwen35JANGTQModel`, otherwise `Qwen35MoEModel`. Ornith
  35B is plain 4-bit affine (no mxtq) → non-JANG path applies.
- `qwen3_5_text` → `Qwen35TextModel` (L71).
- `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text` →
  `Gemma4TextModel` (L40–43).

VLM factory (source: `Libraries/MLXVLM/VLMModelFactory.swift`):
- `qwen3_5` → `Qwen35` (L127), `qwen3_5_moe` → `Qwen35MoE` (L128),
  `gemma4` → `Gemma4` (L151), plus `gemma4_unified` (L152).

Model structs present at pinned rev:
- `Libraries/MLXLLM/Models/Qwen35.swift`:
  `Qwen35TextConfiguration`, `Qwen35TextModelInner`, `Qwen35TextModel`,
  `Qwen35Model` (L33/934/1038/1377). All conform to `LLMModel, KVCacheDimensionProvider`.
- `Libraries/MLXLLM/Models/Qwen35MoE.swift`: `Qwen35Configuration` (L15) + MoE model.

## vmlx tool-call format inference (source: `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`)

- `qwen3_5`, `qwen3_5_moe`, `qwen35` (prefix) → `.xmlFunction` (L421–424). Ornith
  35B (`qwen3_5_moe`) AND Qwen3.8/Heretic/9B (`qwen3_5`) all resolve to the Qwen
  XML `<tool_call>` envelope used by the qwen3_coder-style template.
- `gemma4*` (prefix) → `.gemma4` (L373–375, `Gemma4ToolCallParser`, L245).

## vmlx chat-template fallbacks (source: `Libraries/MLXLMCommon/ChatTemplates/ChatTemplateFallbacks.swift`)

- `gemma4WithTools` (L79) selected before `gemma4Minimal` (L37) when the model's
  native template throws (L1711–1717).

## Staged candidates — staged/dir state (measured, paths exact)

All under `~/.local/share/local-model-bench/mei-models/`. All `staged_complete`,
verified by `tools/stage_mlx_checkpoint.py --verify-only` in prior sessions.

| candidate | dir | config model_type | txt_config.model_type | architectures | shards | tokenizer.json | chat_template.jinja | codebook preproc |
|---|---|---|---|---|---|---|---|---|
| Ornith-1.5-35B-A3B-MLX-4bit | `Ornith-1.5-35B-A3B-MLX-4bit/` | `qwen3_5_moe` | `qwen3_5_moe_text` | `Qwen3_5MoeForConditionalGeneration` | 4 | yes | yes | none |
| Ornith-1.5-9B-MLX-4bit | `Ornith-1.5-9B-MLX-4bit/` | `qwen3_5` | `qwen3_5_text` | `Qwen3_5ForConditionalGeneration` | 1 | yes | yes | none |
| Qwen3.8-27B-4bit | `Qwen3.8-27B-4bit/` | `qwen3_5` | `qwen3_5_text` | `Qwen3_5ForConditionalGeneration` | 3 | yes | yes | preprocessor_config.json + processor_config.json |
| gemma-4-26b-a4b-it-4bit | `gemma-4-26b-a4b-it-4bit/` | `gemma4` | `gemma4_text` | `Gemma4ForConditionalGeneration` | 3 | yes | yes | preprocessor_config.json + processor_config.json |
| Qwen3.8-27B-Uncensored-MLX-4bit | `Qwen3.8-27B-Uncensored-MLX-4bit/` | `qwen3_5` | `qwen3_5_text` | `Qwen3_5ForConditionalGeneration` | 3 | yes | yes | preprocessor_config.json + processor_config.json |

## Compatibility verdict (source-backed)

- **ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit — `qwen3_5_moe`: SUPPORTED by pinned vmlx.**
  The top-level `model_type` matches L52. Non-mxtq weight_format → `Qwen35MoEModel`,
  the MoE (mixture-of-experts, weight_format kv-style expert routing) text path.
  Tool format `.xmlFunction` (L421). **Primary OTG target: loadability + acceptance + KV
  reuse are GPU-gated (see Blockers).**
- **ornith-ai/Ornith-1.5-9B-MLX-4bit — `qwen3_5`: SUPPORTED.** L51 dense `Qwen35Model`.
  This is the validated 9B optimization proxy.
- **mlx-community/Qwen3.8-27B-4bit — `qwen3_5`: SUPPORTED at the LLM text layer** (L51).
  Card/config is a **vision-language** bundle (arch `Qwen3_5ForConditionalGeneration`,
  processor/preprocessor present). Mei's server serves text chat only; the VLM
  factory (`Libraries/MLXVLM`) exists for the same model_type but is not wired into
  Mei's text `Engine.load`. Text-only loader is compatible; image/video input is not
  an advertised Mei path. **Not UD-Q5; regular 4-bit comparator only (manifest note).**
- **mlx-community/gemma-4-26b-a4b-it-4bit — `gemma4`: SUPPORTED at the LLM text layer**
  (L40, `Gemma4TextModel`). Tool format `.gemma4` (L373). Also a vision-language
  bundle (arch `Gemma4ForConditionalGeneration`, processor present) — same caveat:
  text chat compatible, VLM media path not advertised by Mei.
- **orcarouter/Qwen3.8-27B-Uncensored-MLX — `qwen3_5`: SUPPORTED at the LLM text
  layer** (L51), same as base Qwen3.8. Kept distinct (separate source checkpoint
  and pinned revision), **not a silent substitution** for base Qwen3.8.

### Architecture-conditional notes (no speculative code changes made)

- vmlx MoE gate/expert 8-bit exceptions in the Ornith-35B manifest
  (`mlp gate + shared_expert_gate` at 8-bit) are a **quantization** property read from
  the safetensors/quantization metadata, not a new arch; the `qwen3_5_moe` factory
  handles mixed-bit MoE experts via its standard config. This is a load-time fact and
  was already present in the verified manifest; not asserted by live load this session
  (GPU-gated).
- Neither Qwen 3.5 LLM model enables MTP/spec-decode by default in Mei; the GGUF
  reference Qwen3.8/Heretic headers carry MTP heads (mtp_head=true) but Mei's MLX
  loader has no MTP reuse path — MTP stays **default-off** and any llama.cpp
  comparator must run WITHOUT `--spec-type` (manifest note, unchanged).

## Measured rows

- **GPU loadability / decode / prefill rows: NONE taken this session.** Gate blocked
  (see next section). No invented metrics.
- CPU-side rows: staging self-test `RESULT: PASS (0 failure(s))`;
  Swift non-Metal suite **34/34 passed, 0 failures**
  (`ServerConfigParsingTests|SSMAnchorBoundariesTests|OpenAITypesTests|CacheRestoreTrackerTests`).

## Blockers

1. **Metal gate still blocked.** Foreign owners present and MUST NOT be stopped:
   - `llama-server` PID 3968: `--hf-repo ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M`
     on 127.0.0.1:8017, alias `bench-f51c7e72e2ad`. Physical RSS ~ **9992 M**
     (measured) — grown this session.
   - `cocore agent serve` PID 3873.
   - vm_stat: free ≈ **1 GB**; reclaimable steady-state remains below the 10 GB
     uncontended floor (matches prior gate-boundary cycles; foreign llama-server's
     35B GGUF working set ~10 GB is not reclaimable without stopping it).
   Plan gate: never run two model servers concurrently; no GPU-contention wait > 5 min.
   Per the bounded-gate discipline, polling was stopped after one check.
2. Runs blocked by that: every candidate's **load + acceptance + streaming/tool.call +
   KV-reuse + long-context** phases — the conclusion "staged-complete ≠ loadable" is
   unchanged until a clean window appears. Optimizing Ornith-35B then 9B then
   secondary models one-at-a-time remains the sequenced plan.
3. Acceptance tests (`MeiAcceptanceTests`) and the Metal-backed
   `QuantizedRotatingKVCacheTests` fail to run while no Mei server is up / metallib is
   unavailable under the foreign Metal owner — expected environmental failures, not code
   regressions; non-Metal 34 suites are green.

## Files

- This artifact: `artifacts/vmlx-compat-matrix-20260901.md`.
- No source/launcher/manifest edits; no speculative runtime changes; no commits.

## Suggested next (gated on a clean Metal window)

1. Start Mei (35B) via `scripts/start_mei_server.sh` (defaults target Ornith-35B),
   run `tools/probe_load.py` → acceptance (stream/non-stream parity, tool calls,
   KV-reuse, long-context) before any perf.
   On exact memory failure, switch `MEI_MODEL_DIR` to the 9B proxy.
2. Then Qwen3.8-27B, then Gemma4, then Heretic, one at a time, recording
   loadability separately from Ornith optimization results.
3. Any perf claim requires ≥3 clean repeats with dated artifacts.
