# Mei optimization log — 2026-08-30 (session B: gated measurement cycle + research)

Machine: sulaco, M1 Max applegpu_g13s, 32 GiB unified, Metal recommended
working set 26.8GB. Model: ornith-ai/Ornith-1.5-9B-MLX-4bit (hybrid
qwen3_5: 32 layers, 8 attention KV layers, 24 GatedDelta/SSM layers).
Engine: mei on vmlx-swift pinned aeb5e21c + Mei patch series 0001-0004.

## Session start state (hard boundary)
- Mei working tree clean at de7dc03 (HEAD), fork patches 0001-0004 applied
  to both checkouts (in-repo .build and scratch mei-build), release binary
  rebuilt at 10:20Z with all fork flags (verified via `mei --help`:
  --compiled-decode-threshold and --max-kv-window present).
- Detached measurement cycle ALIVE: pid 29951, lock /tmp/mei-cycle.lock held,
  nohup log /tmp/mei-cycle-nohup.log. It gates on: (1) no active
  run_fixture_suite|run_bench.py runners, (2) >=10GB reclaimable memory.
- Machine contended at session start and throughout the research phase:
  other agent's local-model-bench suite (kipclip_mini via llama-server
  Devstral-Small on port 8017, ~25GB RSS) + 5 runner processes. Reclaimable
  memory floor read 0GB at 16:00Z. Per the repository boundary: no Mei GPU
  measurement while contended; the detached cycle owns the measurement
  window when it opens. This session did NOT launch any concurrent backend.

## Research record (exact sources / revisions)

### vmlx-swift (pinned engine)
- Pinned revision aeb5e21c195d8519609488ef75a25ce7e48d8f88
  (osaurus-ai/vmlx-swift origin/main, "Emit .info before cache persistence
  (#329)"). Local origin ref shows 8 newer commits, none touching KV
  quant / compiled decode / rotating caches: 5084ae81 (#335 batch-capacity
  provenance), 2a84b4d7, acc9c125 (#331 qwen35 batch position), ff7b05a3,
  295f8c5c, 8999e6d0, d2510444 (#330 tool_parser pin), aeb5e21c (#329).
- KVCache.swift:1017 (pinned) `RotatingKVCache.toQuantized` was an explicit
  fatalError; `maybeQuantizeKVCache` affine path only handled KVCacheSimple
  (KVCache.swift:2070). Both fixed by Mei patches 0001/0002/0003.
- Upstream gap is industry-wide: lmstudio-ai/mlx-engine#31 documents the
  same limitation ("KV cache quantization is only available for KVCache and
  not RotatingKVCache; we would need to refactor our implementation to use
  KVCache"); ml-explore/mlx-swift-lm KVCache.swift retains the same
  "RotatingKVCache quantization not yet implemented" fatalError on main.
  Mei's QuantizedRotatingKVCache (preserving ring semantics + temporal
  ordering) is an independent implementation of the alternative path.

### llama.cpp ceiling targets (workstream 4)
- llama-server Homebrew build 10470, commit 34af94cd9, AppleClang 21
  (same binary local-model-bench uses; Mei-only port 8074 in
  tools/llama_ceiling.py).
- GGUF verified with gguf PyPI reader (installed to /tmp/ggufpkg for the
  check; not a repo dependency): version 3, 442 tensors, arch qwen35,
  general.file_type 15, quantize.imatrix dataset present, 33 blocks
  (32 base + MTP block 32), full_attention_interval 4, head_count_kv 4,
  key/value_length 256, context_length 262144. Quant split: Q4_K 223,
  Q6_K 35, F32 184 tensors (mixed-precision official conversion).
- **MTP-HEAD FINDING (comparator hygiene)**: this GGUF CONTAINS an MTP
  head — metadata qwen35.nextn_predict_layers = 1 and tensors
  blk.32.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight + a 33rd
  block. Mei's baseline (Ornith-1.5-9B-MLX-4bit, no-MTP MLX artifact) has
  no speculative head. llama.cpp exposes `--spec-type draft-mtp`; the
  ceiling MUST NOT pass it, and the ceiling row records whether llama.cpp
  engaged any spec path (timings predicted_n vs evaluated_n + server log
  lines). tools/llama_ceiling.py does NOT set --spec-type, so the default
  (none) applies; a separate labeled --spec-type draft-mtp row is optional
  and excluded from the >=40 gate decision.
- tokenizer.chat_template present (Qwen-style tool-call XML envelope
  format); tokenizer.ggml.add_bos_token false; eos 248046.
- Fallback recorded: bartowski/Ornith-1.5-9B-GGUF NOT needed (official
  conversion appears to be the staged artifact; provenance: license link
  and imatrix dataset path point to ornith-ai).

### Level1Techs article (hypotheses, not facts)
https://forum.level1techs.com/t/why-your-local-llm-feels-dumber-than-it-is/253917
(thr3e, 2026-08-16; HN discussion 49402232). Claims adopted as controls:
(1) sampler/template settings change behavior -> every Mei row records
temperature 0 (server + payload), fixed chat template, no nucleus drift;
(2) attention backend/kernel choices can change logits -> kernel/metallib
A/B only with a hypothesis (workstream 5, deferred until profiling says
so); (3) KV quantization can accumulate long-context divergence and break
tool calls -> every kv variant runs acceptance + tool-call schema check
(probe_mei.py validate_add_call) + 30K/80K survival probes; (4) weight
quant / fused-GEMM fidelity trades -> llama.cpp ceiling is a reference
measurement, not a quality endorsement; (5) realistic long-context
agentic/tool workloads -> 40K chat pattern included in every cell.

### FreeToken (architecture research only; not a dependency)
https://github.com/FlashML-org/FreeToken ; paper arXiv:2608.16157
("FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive
Execution"). NVIDIA/CUDA-PyTorch target; do NOT port. Transferable ideas
mapped to Apple/MLX:
- Semantic anchor checkpoints for recurrent state/KV after agentic context
  edits (tool/thinking blocks) -> maps to Mei's SSM re-derive + disk
  restore; candidate experiment: checkpoint GatedDelta state at template
  suffix boundaries so diverging edits re-derive only the suffix (currently
  the coordinator falls back to full prefill — always correct).
- Bandwidth-adaptive CPU/GPU execution + full-layer double-buffered prefill
  streaming -> on unified memory this becomes allocator/cache-budget and
  prefill-chunk scheduling experiments (prefillStepSize sweep already in
  phase A).
- Global LRU expert caching/residency -> relevant to the 35B MoE
  (well-known blocked artifact): on unified memory, translate to which
  expert tensors stay resident vs page-in under the Metal working set;
  no claim that any patch unblocks 35B weights without independent
  measurement.

## Fork state (committed before this session; working tree clean)
- da3c669  FORK 0001-0003: QuantizedRotatingKVCache (real 4/8-bit affine
  hybrid KV), quantized rotating disk store (dequant at store,
  quant(dequant(quant(x)))==quant(x) determinism), compiled-decode
  threshold (skip promote+trace when prefill offset exceeds threshold).
- 07bd5ba  FORK 0004: --max-kv-window bounded-ring probe (attention scans
  at most the ring; correctness-bounded, NOT a production config).
- Unit validation: 20/20 tests incl. fp16-vs-8bit attention parity
  (QuantizedRotatingKVCacheTests). Re-run deferred to an uncontended
  window (Metal-touching).

## Measurement plan status
- The detached cycle (phases A/B/C) produces:
  artifacts/sweep-cliff-baseline|prefillsteps|ssm|cachelimit-<ts>.json,
  artifacts/llama-ceiling-fp16kv|q8kv-<ts>.json,
  artifacts/sweep-variant-{kv8,kv4,compiled16,combined-kv8-compiled16,
  window8k,window16k,window16-compiled,kv8-80k,survival80k}-<ts>.json,
  artifacts/acceptance-variant-*.json, artifacts/survival-variant-*.json,
  cycle-{A,B,C-*}-boundary-*.txt, plus per-cell server logs with
  MLXPRESS_GENERATION_PROFILE dumps under mei-runtime/logs/.
- At session end: NO measurement artifacts had landed yet (cycle still
  gated). Contention blocker recorded above.

## Pre-analysis: bandwidth model and per-variant expectations (16:45Z)

M1 Max unified bandwidth ~400GB/s. Per decode step the engine streams
weights (~5.0GB Q4 for the 9B) + attention KV reads:
  fp16 KV at 45K: 8 attn layers x 45K pos x 4 kv-heads x 256 dim x 2B x 2
  (K+V) = ~1.47GB  -> total ~6.5GB/step -> ~62 tok/s BW bound (eager
  measured 5.2-6.4 -> eager is LATENCY/dispatch-bound, not BW-bound).
  q8 KV at 45K: ~0.74GB; q4 KV: ~0.37GB (minor BW lever unless eager
  overhead is reduced).
  16K window: KV reads ~0.52GB (fp16) -> ~5.5GB/step -> ~73 tok/s bound;
  compiled decode at 16K measured 34.8 tok/s (2026-08-29) = ~48% of BW
  bound (plausible real-world Metal efficiency).

Variant expectations (to be checked against measured rows):
- kv8/kv4 (eager): small decode gain unless attention share dominates;
  the real value is memory (KV 1.47GB -> 0.74/0.37GB) and enabling the
  disk tier + 80K survival at a fixed memory envelope. Acceptance must
  stay green; 30K/80K survival is the long-context divergence gate.
- compiled16 (thresholded): short/16K win kept (~34.8-47.2 band),
  no long-prefill tax; 33K/45K rows expected to fall back to eager.
- combined kv8+compiled16: kv quant reduces the traced-graph KV size;
  expected <= compiled16 alone unless attention dominates within 16K.
- window8k/window16k (eager): attention bounded -> decode nearly
  context-independent BUT full-attention model loses old context: 30K/80K
  survival + 40K chat are expected to FAIL coherence checks; throughput
  probe only, never a production config.
- window16-compiled (THE >=40 gate candidate): promote+trace O(16K), so a
  45K offset traces cheaply; compiled replay at 16K width. Bandwidth model
  says 30-45 tok/s achievable if Metal efficiency holds; MUST be reported
  with its correctness caveat (window truncation), and does NOT satisfy
  the gate unless acceptance + 30K/80K survival stay green.
- llama.cpp ceiling (fp16kv and q8kv): same 45K prompt, no --spec-type
  draft-mtp (MTP head present in the GGUF but unused); timings
  predicted_n vs evaluated_n must show single-token decode to confirm no
  auto spec engage. 25-39 tok/s => hardware ceiling band; >=40 => fork
  justified.

35B generalization: nothing in this session changes the 35B blocked
status (weights ~19.5GB + 45K KV + activations vs 26.8GB recommended
working set). kv quant WOULD shrink the 35B KV (~3.6GB fp16 -> ~0.9GB
q4), which is a necessary (not sufficient) step toward a 35B resident
run; only an independent 35B measurement may claim unblocking.