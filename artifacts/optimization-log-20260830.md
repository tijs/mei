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
  conversion is the staged artifact; provenance: license link and imatrix
  dataset path point to ornith-ai).
- **Exact provenance pin (verified 16:10Z)**: local sha256
  70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6
  == blob sha256 of Ornith-1.5-9B-Q4_K_M.gguf in
  huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF default revision
  abdd624b12ebf020b767fff532ff44fe552b28c3 (repo lastModified
  2026-08-24T02:45:24Z). The staged GGUF is the OFFICIAL artifact, pinned
  by content hash regardless of download-date drift.

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

## Probe-coverage honesty note (what survival/acceptance can and cannot
## assert for window cells)
- probe_long_context gates on: HTTP 200, exact prompt-token counts,
  non-empty completion, decode >= 1.0 tok/s, full-prefix cache reuse.
  It does NOT assert semantic coherence. A windowed cell (attention
  truncated to 8K/16K) can PASS survival while being semantically blind
  to pre-window context, because the filler prompt is repetitive and the
  model continues it plausibly. Window cells are therefore bounded by
  CONSTRUCTION (documented truncation), not by probe failure.
- probe_mei's tool-call/parity checks run at ~300-token prompts — they
  cannot detect long-context truncation either; their role is per-variant
  correctness at the KV-quantization level (divergence would break
  exact-arg tool calls), which is their intended gate here.
- chat_40k rows in sweep_mei assert only decode >= 1.0; the bench_mei
  driver additionally checks content_tail 'cache-ready' (long_chat
  transcript check). Where content is compared across variants, raw
  choice content is not stored in sweep rows — content-level divergence
  across kv8/kv4/compiled cells is assessed via the acceptance and
  survival artifacts + raw transcripts in the probe logs, not via sweep
  rows.
- The Level1Techs-derived rule stands: a tok/s win that regresses tool
  calls, reasoning stability, or long-context survival is rolled back;
  window cells are the documented exception (throughput probe only).
## Session end boundary (16:40Z)
- Watcher proc_2d028b62138b (8 samples, 16:04-16:39Z): foreign_runners=6
  and reclaimable=1GB at every sample; the other agent's llama-server
  (Devstral-Small, port 8017) plus its active suite runner held the
  machine for the entire session. NO Mei artifact of any kind was
  produced; no Mei measurement ran; nothing was contaminated.
- The detached cycle (pid 29951, <24h gate) remains alive and owns the
  window; phases A/B/C run automatically when the machine clears. All
  drivers were pre-flighted and validated this session (salt units 35/35,
  chat-40k transcript 44,002 tokens post-fix, digest + gate_report dry
  runs, GGUF provenance pinned, MTP comparator rule recorded).

---

# Session D (2026-08-30 evening, 18:47-19:10Z) — CPU-side deliverables + pipeline fixes

## Contention state
- Machine contended throughout: other agent's llama-server on port 8017
  (Devstral-Small Q4_K_M, then Muse-Glimmer-30B Q4_K_M), cocore agent
  serve, run_bench.py wrappers, run_fixture_suite hearth_full. Reclaimable
  memory 1-3GB (floor is 10GB). Gate sample written to
  artifacts/cycle-gate-sample-20260830T164750Z.txt. No Mei GPU
  measurement ran; no foreign backend touched.

## CPU-side deliverables (per the continuation directive)
1. NEW tools/probe_diverging_chat.py — the patch-0005 evidence generator:
   5-turn tool-calling transcript; run A = strict growth (requests 1..5
   extend), run B = identical turns 1..4, turn 5 diverges (place_order ->
   cancel_order). Records cached_tokens / prefill_ms / TTFT (streaming
   first-delta) per request; summary decides growth-anchors-work vs
   gap_confirmed. `--self-test` validates transcript invariants WITHOUT a
   server (no Metal): SELF-TEST PASS exit 0. Measurement run is client-
   side and gated on an uncontended server window.
2. NEW Tests/MeiTests/ServerConfigParsingTests.swift — 10/10 passed
   (swift test --filter ServerConfigParsingTests, non-Metal): fork-flag
   plumbing incl. rollback-default guard. Covers patches 0001 (kv-bits),
   0003 (compiled-decode-threshold), 0004 (max-kv-window), ssm-rederive,
   kv-cache-dir, memory limits, error paths.
3. Patch-0005 source anchors re-verified: additionalBoundaries param
   exists at SSMReDerive.swift:443; derived boundaries come from
   iterator-internal sharedPromptAdditionalBoundaries (Evaluate.swift:2760,
   BatchEngine.swift:3174) = cachePrefixTokenCounts + [hybridStripBoundary]
   — NOT operator-configurable; a fork patch must thread a new knob into
   both call sites. Kept DESIGN-ONLY: design note gates implementation on
   probe evidence + phase-A ssm-rederive rows. No speculative runtime
   change landed.
4. Found + fixed a real pre-window pipeline bug: phase C's
   run_variant_cell passed --compiled-decode-threshold to sweep_mei.py
   which had no such argument (would abort the first clean window). Also
   sweep cells did not record compiled_threshold/max_kv_window, and both
   digest tools read cell config via cell.get("config", {}) which sweep
   cells do not have. All fixed; digest pipeline validated end-to-end on
   a clearly-labeled schema fixture (artifacts/digest-schema-test-
   fixture-20260830T1650Z.json): correct family medians + verdicts
   (PIVOT for 13.2-shaped baseline, MET for 41.5-shaped window16-compiled).
5. llama_ceiling.py gate validated: with foreign processes resident it
   prints FATAL and exits 2 WITHOUT launching (run attempted with
   --gguf /dev/null; no llama-server on 8074 was ever started).
6. GGUF provenance re-verified: sha256
   70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6 of
   Ornith-1.5-9B-Q4_K_M.gguf == pinned official blob
   (ornith-ai/Ornith-1.5-9B-GGUF@abdd624b); gguf_meta confirms arch
   qwen35, file_type 15, context 262144, MTP head PRESENT
   (nextn_predict_layers=1) — comparator keeps --spec-type off.
7. Patch series re-verified end-to-end: scripts/apply_vmlx_patches.sh
   --reset re-applied 0001-0004 byte-exactly to BOTH checkouts (the
   scratch mei-build checkout had partially drifted and was repaired);
   release binary still exposes the full fork flag surface.

## Commits (session D)
- (listed in the final report table)

## Session-D end boundary
- GPU measurements remain PENDING (zero Mei sweep/ceiling artifacts
  exist). Official numbers remain the 2026-08-29 baseline: short 28.1,
  fresh45 5.21 (prefill 273.3s), reuse45 13.24 (cached 45000, prefill
  10.2s), chat40k @33,175 tokens (prefill 210s); acceptance-9B-coordinator
  green (10/10 probes).
- Next: bounded foreground gate attempt (--phase A,B,C) in a clear
  window; digest with summarize_rows.py + gate_report.py; fill the
  FINAL-REPORT numbers.

---

# Session C (2026-08-30 evening, 16:40-17:30Z) — policy alignment + CPU-side validation

## Contention state at session start
- The session-B detached cycle pid 29951 was DEAD (lock /tmp/mei-cycle.lock
  released; nohup log ends 16:39:30Z — it did not survive the new
  fixture-suite launch). It produced no artifacts (correct per boundary).
- Machine fully contended at 16:40Z and throughout: other agent's
  llama-server unsloth/Devstral-Small-2507-GGUF:Q4_K_M on port 8017,
  ~25GB RSS, plus run_fixture_suite.py --suite hearth_full (pid 45729) and
  two run_bench.py wrappers (pids 24929/24930); free pages ~4000x16KB
  (~67MB); reclaimable GB floor 0-1. No Mei GPU measurement ran; no
  foreign backend was touched.
- Policy change adopted (per the shared-GPU contention cap): NO detached
  gate waiters anymore. Gate polls are foreground and bounded; see
  commit 53742de below.

## Commits (Mei)
| hash | purpose |
|---|---|
| 53742de | tools: bounded foreground measurement policy + fresh-KV-per-cell hygiene |

## Fork correctness review (source-level + numeric, no GPU)
1. **temporal-order parity**: patch 0001's static `temporalOrder` mirrors
   the fp16 ring's private reorder exactly (KVCache.swift:714 fp16:
   `[..<keep, idx..., keep..<idx]` == 0001 static reorder) — the quantized
   attention span equals the fp16 ring's temporal span by construction.
2. **Dispatch safety**: AttentionUtils.swift:94 routes
   `QuantizedKVCacheProtocol` caches through `updateQuantized` +
   `quantizedScaledDotProductAttention`; `QuantizedRotatingKVCache.update`
   stays a fatalError safety net that the hot path can never reach
   (verified: no other caller in KVCache/Evaluate/AttentionUtils).
3. **Quant determinism (numeric)**: affine quant(dequant(quant(x)))==quant(x)
   verified exact (max |Δcode| 0) over bits {4,8} x groupSize {32,64,128}
   x 20 random trials; min/max (quant grid) preserved by dequant. The
   dequantize-at-store / requantize-on-restore disk design (patch 0002)
   is deterministic. (numpy 2.5.2, mei-runtime venv; no Metal touched.)
4. **Mamba/QSA isolation**: maybeQuantizeKVCache skips MambaCache,
   CacheList, QSAKVCache and TurboQuant by construction; only
   KVCacheSimple + RotatingKVCache are replaced.
5. **Documented minor divergence (accepted)**: the quantized ring trims to
   exactly maxCacheSize immediately after each append, while fp16
   updateConcat transiently holds maxCacheSize+S-1 during multi-token
   prefills (the "+S-1 context floor"). At prefill step 512 vs 65536 ring
   the effect is <1% of a window edge and only concerns prefill-chunk
   attention, never decode — fidelity-minus, not a correctness break.

## Upstream research refresh (exact revisions)
- osaurus-ai/vmlx-swift-lm LIVE main head = 4546a5d720e7
  (2026-05-15, "fix(dsv4): render DSML tools in fallback template");
  commits since the aeb5e21c pin that touch caches are all
  disk-materialization/restore hardening (ad1d2319 sync-before-persist,
  495ac32b restore disk prefix hits for growing prompts, 34a86398 store
  growing chat answer boundaries, d8c2bb22 materialize token iterator
  disk restores ...) — none touch KV quantization or compiled decode.
  Mei's fork remains the only live implementation of hybrid rotating-KV
  quant + thresholded compiled decode on this pin. Re-pinning is
  deliberately NOT done (would re-run the whole acceptance suite).
- ml-explore/mlx-swift-lm main KVCache.swift (fetched 2026-08-30):
  RotatingKVCache.toQuantized now THROWS "RotatingKVCache quantization is
  not implemented because its temporal ordering requires dedicated
  handling" (line ~911) instead of fatalError; KVCacheSimple.toQuantized
  (line ~503) and maybeAffineQuantizeKVCache (line ~2586) still handle
  only simple leaves. Gap confirmed on BOTH upstreams.
- llama.cpp brew build 10470 (commit 34af94cd9, AppleClang 21) supports
  --cache-type-k/v (q8_0 KV), --spec-type none|...|draft-mtp,
  --no-cache-prompt, --metrics, --parallel. The ceiling driver passes NO
  --spec-type (default none) so MTP stays off vs Mei's no-MTP baseline;
  expected single-token decode verified via timings predicted_n vs
  evaluated_n at digest time.
- Level1Techs forum.level1techs.com/t/253917 (thr3e, 2026-08-16) fetched
  and claims verified by reading (CUDA/vLLM study on hybrid Qwen3.6-27B):
  (1) attention-backend choice → top-1 logit divergence that GROWS with
  context (bit-identical within a backend across runs); (2) KV-quant
  ALONE (weights/activations fixed) accumulates divergence past ~40K —
  directly motivating Mei's per-variant acceptance + 30K/80K survival
  gates and the tool-call schema checks; (3) W8A16 > FP8 > FP4 on token
  flips — weight-quant fidelity trades; (4) their baseline disables MTP
  like Mei's comparator. All treated as hypotheses, already encoded in
  the drivers (temp 0, family-salted fresh prompts, strict-extension
  reuse, tool-call schema, survival probes).
- FreeToken (github.com/FlashML-org/FreeToken, main @ 2026-08-30 fetch,
  524 files) — architecture research with exact cites:
  README.md:20-21 (bandwidth-adaptive CPU-GPU co-execution q* policy,
  double-buffered prefill streaming, global LRU expert caching, semantic
  anchor checkpoints for recurrent state + KV);
  python/freetoken/checkpoint/convert.py (HF safetensors -> FTW with
  offload-expert banks, self-contained checkpoints) -> maps to Mei's
  kv-cache-dir tier + a candidate GatedDelta anchor checkpoint at chat-
  template suffix boundaries so diverging agentic edits re-derive only
  the suffix (currently full-prefill fallback, always correct);
  python/freetoken/attention/dsv4_compress.py (paged compressed-KV pool,
  per-window compress-state ring, radix resume by value) -> the disk-tier
  reuse/restore design parallels; python/freetoken/daemon/ (lifecycle
  receipts) -> not transferable (Metal/MLX native constraint).
  arXiv:2608.16157 abstract verified: "continuously maps computation and
  model state onto the resources actually available" — on Apple unified
  memory this is allocator/cache-budget control (--memory-limit-bytes,
  --cache-limit-bytes sweeps in phase A) and prefillStepSize chunk
  scheduling, NOT PCIe-style offload. No port; no 9B gate changes.

## Measurement status (unchanged — still zero artifacts)
- No sweep/cliff/ceiling/acceptance/survival artifact exists yet. All
  official numbers remain those of bench-9B-45000-20260829T2338Z.json
  (eager fp16 baseline: short 28.1, fresh45 5.2, reuse45 13.2, chat40k
  10.3 @33K) and the compiled-decode A/B rows in optimization-log-
  20260829.md (47.2 short / 34.8 @16K, long-prefill tax).
- The gate poll (1 minute, foreground, bounded) wrote
  artifacts/cycle-gate-sample-20260830T164306Z.txt and
  artifacts/cycle-gated-boundary-20260830T164306Z.txt and exited 3 —
  the exact new-policy behavior. Machine remained contended at session
  end (reclaimable ~1GB, hearth_full fixture suite active).

## Session C end boundary (17:30Z)
- No Mei GPU measurement ran this session; nothing was contaminated; no
  foreign process was stopped, restarted, or reconfigured.
- Next: run the bounded foreground cycle (scripts/run_measurement_cycle.sh
  --max-wait-min 5 --phase A|B|C) in a clear window; digest with
  summarize_rows.py + gate_report.py; fill FINAL-REPORT numbers.

## Addendum (17:45Z): next-experiment design committed
- artifacts/design-anchor-ssm-0005.md records the FreeToken-derived
  candidate: the pinned engine ALREADY has semantic anchors
  (captureCleanSSMStateInline §440 + per-boundary re-derive with
  ssmMaxEntries=50 LRU), but stores only the LARGEST boundaries (prompt
  END); mid-transcript agentic edits therefore fall back to full prefill.
  Patch 0005 would thread --ssm-anchor-boundaries K into the existing
  additionalBoundaries param (SSMReDerive.swift:443). Gated on a new
  diverging-chat probe (tools/probe_diverging_chat.py) + phase-A rows —
  TTFT lever, not a decode tok/s lever.

---

# Session E (2026-08-30 evening, 18:12-19:30Z) — CPU-side: probe hardening + provenance gate + patch-0005 plumbing + gate attempt

## Contention state
- Machine contended at session start and end: other agent's llama-server on
  port 8017 (Muse-Glimmer-30B Q4_K_M, ~22GB RSS), run_bench.py and
  run_prompt_suite.py active, cocore agent serve resident; ~65MB free pages.
  No Mei GPU measurement ran; no foreign backend touched.

## CPU-side deliverables (per the continuation directive)

### 1. probe_diverging_chat.py — independently reviewed, hardened, self-tested
- `--self-test` PASS (exit 0), and a NEGATIVE test pass: the validator
  catches a broken tool_call_id cross-reference, missing required argument
  keys, non-JSON tool results, and a broken growth chain.
- NEW deterministic checks: `validate_messages` per transcript (role
  sequencing user->assistant(tool_calls)->tool->assistant(closing);
  tool_call_id must reference a pending assistant tool_call; exactly one
  call per assistant message; arguments JSON-parses and matches the
  per-tool key schema exactly; tool-result JSON), `build_deterministic`
  (two builds byte-identical), `turn5_user_contents_differ`,
  `all_transcripts_schema_valid` (all ten transcripts).
- Artifact now records: sampler settings (temperature 0, max_tokens), exact
  tool schema, system prompt, per-request content tails and called tool
  names (deterministic output controls per Level1Techs-derived rule),
  per-run sizes.
- Status gate strengthened: ANY errored request now fails the probe (a hole
  in run B previously read as "gap confirmed" via b_r5 cached=0).

### 2. llama_ceiling.py — same-model official 9B GGUF command/provenance/digest path
- `--provenance-only` mode validates WITHOUT launching llama-server:
  sha256 of the staged GGUF == pinned official blob
  (ornith-ai/Ornith-1.5-9B-GGUF@abdd624b,
  70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6),
  header facts via gguf_meta (arch qwen35, context 262144, file_type 15,
  MTP head PRESENT nextn_predict_layers=1 — recorded, --spec-type stays
  off), llama-server binary present (build 10470, commit 34af94cd9).
  Result: PROVENANCE OK exit 0.
- Negative tests pass: wrong pinned digest rejected; missing file rejected;
  meta-only path OK. A digest mismatch is FATAL (exit 2, no measurement).
- Full runs now embed the provenance block in the artifact header.

### 3. patch 0005 — SSM anchor-boundary plumbing, default OFF, unit-covered
- patches/0005-ssm-anchor-boundaries.patch (delta over 0001-0004, generated
  via the checkout's git index; full-stack tree byte-identical after
  `apply_vmlx_patches.sh --reset` on BOTH checkouts).
- Engine: `GenerateParameters.ssmAnchorBoundaries: [Int] = []`; both SSM
  store paths union it (Evaluate.swift storeCacheAfterGeneration via
  cacheInitParameters; BatchEngine.swift finishSlot via slot.parameters);
  engine-side validation unchanged (0 < b <= prompt len, Set-deduped;
  anchor at any offset is the exact state for that prefix — misplaced
  anchors can only be suboptimal, never incorrect).
- Mei: `--ssm-anchor-boundaries K` (default 0); SSMAnchorBoundaries.compute
  derives the first K user-message start offsets from the request's OWN
  rendering path (same tokenizer/tools/context) with an additivity
  self-check -> non-additive transcripts fall back to [] with a logged
  warning, never a wrong offset.
- Tests: SSMAnchorBoundariesTests 9/9 + ServerConfigParsingTests 11/11
  (incl. new parse test + default-off guard) — 34/34 non-Metal total.
- NO performance claim: TTFT lever only; measurement gated on the probe +
  an uncontended window (design-anchor-ssm-0005.md updated to PLUMBING
  LANDED).

### 4. Patch-series hygiene
- apply_vmlx_patches.sh updated to 0001-0005 (patch list + sentinels);
  `--reset` re-applies the full series byte-exactly to both checkouts
  (verified: re-applied trees match the reference diff).

### 5. Tool fixes / gate tooling
- gate_report.py: fixed statistics.median([]) crash when no row reports
  prefill_ms (+ regression fixture test; verdicts unchanged: PIVOT 13.2 /
  MET 41.5 on the session-D schema fixture).
- sweep_mei.py: removed a dead ternary that always produced `ctx_N_fresh`.
- run_bounded.py: verified documented behavior (125 usage/help, propagates
  child rc; 0/124).

### 6. Release binary
- Rebuilt (scratch mei-build) with the full 0001-0005 fork surface; --help
  now shows --ssm-anchor-boundaries.

## Commits (session E)
- (listed in the final report / git log; all Mei-only)

## Session-E end boundary
- GPU measurements remain PENDING (zero Mei sweep/ceiling artifacts exist).
  Official numbers remain the 2026-08-29 baseline: short 28.1, fresh45 5.21
  (prefill 273.3s), reuse45 13.24 (cached 45000, prefill 10.2s), chat40k
  10.3 @33K; acceptance-9B-coordinator green (10/10).
- Gate attempt for this session: see artifacts/cycle-gate-sample-<ts>.txt /
  cycle-gated-boundary-<ts>.txt (bounded foreground poll, exit 3 expected
  while Muse-Glimmer-30B holds the machine).

---

# Session F (2026-08-30 evening, 19:16-19:40Z) — probe A/B completion + verification + gate attempt

## Contention state
- Machine contended throughout: other agent's llama-server on port 8017
  (Muse-Glimmer-30B Q4_K_M, ~21.8GB RSS), run_bench.py wrapper, active
  run_prompt.py / run_prompt_suite.py (hermes_ops), cocore agent serve.
  Reclaimable memory floor 3GB (free pages ~4K x 16KB). One bounded
  foreground gate attempt (--phase B, --max-wait-min 5): exit 3 at
  17:25:16Z; boundary artifacts
  cycle-gated-boundary-20260830T172015Z.txt + cycle-gate-sample-*.txt.
  No Mei GPU measurement ran; no foreign backend touched.

## Session-F priority items (continuation directive)

### 1. probe_diverging_chat.py --ssm-anchor-boundaries A/B path — completed
Gap found: the probe had NO --ssm-anchor-boundaries surface, so an
anchors=4 run and a default run produced indistinguishable artifacts, and
the design-doc A/B (design-anchor-ssm-0005.md evidence item 2) was
unexecutable end-to-end. Added (commit 798ed36):
- `--ssm-anchor-boundaries K` (default 0): LABEL of the server-side launch
  config under test, recorded verbatim in artifact
  server_config.ssm_anchor_boundaries; usage block documents the
  orchestrator-driven A/B contract (probe is client-side; server must have
  been launched with the matching flag).
- --self-test now gates `anchors_label_non_negative` (negative label
  FAILS loudly). SELF-TEST PASS (default and anchors-4 label), negative
  label -> SELF-TEST FAIL. Probe still records cached_tokens / prefill_ms
  / TTFT (streaming first-delta) per request, deterministic per-request
  content tails + tool names.

### 2. Server launch env wiring (commit 9100d43)
- start_mei_server.sh: MEI_SSM_ANCHOR_BOUNDARIES -> --ssm-anchor-boundaries
  (previously IMPOSSIBLE to launch the server with anchors on; the A/B was
  not expressible).
- run_measurement_cycle.sh: NEW phase D = diverging-chat A/B cells:
  anchors-default (anchors=0) + anchors4 (anchors=4), each with fresh KV
  dir, cell-scoped runtime base/log, bounded 300s readiness poll before
  the client-side probe, per-cell server-log ssm-anchor evidence artifact,
  MLXPRESS_GENERATION_PROFILE=1.
- MLXPRESS_GENERATION_PROFILE=1 also added to phase-C run_variant_cell
  server envs (sweep_mei set it for its own cells; the cycle's
  acceptance/survival probe servers had no profile capture).
- Verified the engine's profile hook at pinned
  Libraries/MLXLMCommon/Evaluate.swift:1307-1358: env-gated
  ([MLXPressGenerationProfile] ... count/total/avg-ms rows to stderr),
  no other MLXPRESS hook exists in the pinned tree.

### 3. Independent verification (all green, CPU-side)
- patch 0005 default-off: GenerateParameters.ssmAnchorBoundaries: [Int] = []
  (patches/0005, line 33) and Mei CLI default 0 (ServerConfig.swift:69) —
  default == upstream behavior exactly; release binary --help shows
  --ssm-anchor-boundaries.
- apply_vmlx_patches.sh --reset: both checkouts (in-repo .build +
  mei-build scratch) repaired from pristine pinned tree aeb5e21c and
  re-applied 0001-0005 byte-exactly; second run idempotent (sentinel
  check "already applied" both).
- Non-Metal tests: 34/34 (ServerConfigParsing 11, SSMAnchorBoundaries 9,
  OpenAITypes 8, CacheRestoreTracker 6).
- Release binary surface (19:10Z build): kv-bits, compiled-decode,
  compiled-decode-threshold, max-kv-window, ssm-anchor-boundaries all
  present.
- Phase CLI surfaces parse exactly as staged: sweep_mei.py
  (--contexts --prefill-steps --ssm-rederive --cache-limit-gb --kv-bits
  --compiled --compiled-decode-threshold --repeats-45k --chat-40k),
  llama_ceiling.py (--kv-cache-type-q8 --chat-40k --provenance-only).
- llama-server build 10470 (commit 34af94cd9, AppleClang 21) confirmed:
  --cache-type-k/v (q8_0 available), --spec-type none|...|draft-mtp —
  ceiling keeps spec off.
- GGUF provenance re-verified: sha256 70c11219…e8fab6 == pinned repo
  blob (ornith-ai/Ornith-1.5-9B-GGUF@abdd624b); PROVENANCE OK exit 0.
- Digest pipeline regression: gate_report verdicts PIVOT (baseline-shaped
  13.2) / MET (41.5-shaped window16-compiled) on the schema fixture —
  unchanged.

### 4. Upstream research refresh (exact heads, 2026-08-30)
- osaurus-ai/vmlx-swift main = 33d1b6fa9 (37 commits past pin aeb5e21c).
  NEW relevant commit since sessions B/C: bf8b31995 "Optimize Ornith 35B
  compiled decode regions (#346)" — model-side compiled-region enablement
  for the Qwen3.5-VL architecture (their doc: enabling the topology-scoped
  compiled GDN + MoE decode regions was the causal change behind ~25 ->
  ~94 tok/s on their 35B rows, env opt-out, footprint 27.3GB peak). NOT
  vendored: it patches model classes/tests on a main 37 commits past the
  pin (re-pin-level change) and targets the memory-blocked 35B; recorded
  as the candidate next compiled-decode experiment (backport region
  enablement to the pinned tree, A/B on the full matrix) if measurement
  points at model-side compile regions. All other new commits (MTP/spec
  draft, batch-position #331, batch-capacity #335, Qwen3.8 parser #330)
  are out of Mei scope. No upstream RotatingKVCache-quant commit exists.
- ml-explore/mlx-swift-lm head = 37688d2c: KVCache.swift:911-914
  RotatingKVCache.toQuantized STILL throws ("temporal ordering requires
  dedicated handling") — Mei's QuantizedRotatingKVCache (patch 0001-0002)
  remains the only live implementation of hybrid rotating-KV quant.
- ggml-org/llama.cpp HEAD = 6d1479c1 (brew build 10470 = 34af94cd9).
- FlashML-org/FreeToken HEAD = 4b94bdc3 (README:20-21 semantic-anchor /
  bandwidth-adaptive cites unchanged; patch 0005 mapping stands).

## Session-F outcome
- Measurements remain PENDING (zero Mei sweep/ceiling artifacts — machine
  fully held by the other agent through the entire session). Official
  numbers remain the 2026-08-29 baseline; nothing was contaminated.
- The patch-0005 A/B path is now COMPLETE end-to-end (probe label +
  self-test gate + server env wiring + cycle phase D + profile capture +
  per-cell evidence artifacts) and executable in the first uncontended
  window with:
    scripts/run_measurement_cycle.sh --phase D
- Next GPU work when a clean window opens: --phase B (llama.cpp ceiling,
  fp16 + q8 KV, provenance-gated, no --spec-type) BEFORE --phase A/C/D;
  digest with summarize_rows.py + gate_report.py.
