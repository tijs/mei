# Flint 40B-A2.5B: experimental source path

**Default off; model-free validation is not real-checkpoint admission.**
No weights, server, generation benchmark, or quality/speed claim accompanies
this implementation. It does not add a measured model profile or change any
other architecture's defaults.

## Scope and provenance

- Mei base: `6b632ebea3b2b843085ad4f37a0e75b31df3e4d7`.
- Unchanged vmlx-swift pin: `633fe166630ef04310aea7d5a1795555ab32970d`.
- Architecture reference: `Blaizzy/mlx-vlm` v0.7.10,
  `8c061ef516e7f8b584c0d5b64f26b50d5806303b`, complete
  `mlx_vlm/models/flint/` directory. MIT attribution is in
  [`THIRD-PARTY-NOTICES.md`](../THIRD-PARTY-NOTICES.md).
- First intended artifact (not downloaded/admitted):
  `mlx-community/flint-mixed-3-4bit@b1c305855ae7c3464813815be4dbd544b22c36aa`.
  Root `model_type=flint`, `architectures=[FlintForCausalLM]`.
  The config and index were inspected without downloading weight shards.
- Supported serialization is **pre-stacked standard MLX affine**, default
  4-bit/group64, 141 routed projection overrides at 3-bit and an 8-bit
  embedding override. Router tensors have `.weight`, not `.scales`.
  Both equivalent `quantization` and `quantization_config` maps are retained
  and decoded by the pinned `BaseConfiguration`; conflicts are errors.
  Raw quantization declarations are checked first: custom pack fields and
  modes cannot be silently ignored or defaulted to affine by that decoder.
- **Not supported:** `OsaurusAI/Flint-40B-A2.5B-JANGH2`, `tq2_packed` /
  `tq2_scales`, JANGTQ, individual/unstacked expert exports, tied heads,
  non-default RoPE scaling, or a different router/activation convention.
  Source support for ordinary affine 3-bit tensors does not imply support
  for any custom 2/3/4-bit packed format.

## Explicit gate and launch plumbing

A future authorized local launch must explicitly pass `--enable-flint true`.
The default, or `--enable-flint false`, refuses a Flint bundle **before**
allocator policy, tokenizer/weight loading or factory fallback and names the
opt-in in its error. The gate is checked on every `Engine.load`, even after
an earlier enabled load registered the global `flint` creator.

`MEI_ENABLE_FLINT=true` forwards the flag through Mei's own launcher. Inspect
argv without building, starting a process, creating runtime paths, or requiring
a model:

```bash
MEI_ENABLE_FLINT=true MEI_MODEL_DIR=/path/to/flint \
  scripts/start_mei_server.sh --dry-run
```

This document is not permission to stage weights or launch that argv. Disk,
memory, single-GPU-owner and user-authorization gates remain independent.
No changes to `local-model-bench` or CoCore are required for this source phase;
a benchmark harness with its own launcher still needs its own explicit
forwarding/config review before use.

The app registers a creator through `LLMTypeRegistry.shared.registerModelType`.
It uses the existing `loadModelContainer(from:using:configuration:loadConfiguration:)`
entry point; the dependency revision and sources are unchanged. Unknown model
types remain the factory registry's errors, not aliases to a similar MoE.

## Numerical and weight contract

- Q/K projections in activation dtype, per-head Q/K RMSNorm, then optional
  non-traditional RoPE, exactly following `no_rope_layer_indices`.
- Full and sliding masks, with learned per-head `self_attn.softmax_offset`
  supplied to the pinned SDPA `sinks:` path. Sliding layers use
  `RotatingKVCache(maxSize: 513, keep: 0)`, global layers `KVCacheSimple()`;
  generic `maxKVSize` must not turn global layers into rotating caches.
  `--kv-bits` is refused for Flint before loading: the pinned quantized KV
  update API is different, and its public attention helper cannot pass sinks.
  No sink-dropping or silent KV-option fallback is installed.
- Layers 0 and 1 are dense SwiGLUs; the other 47 use 256 routed experts,
  top-six routing and an always-on, ungated shared expert.
- Router projection occurs in activation dtype, then **sigmoid in FP32**.
  Correction bias affects selection only. Gather mixture weights from the
  **unbiased** sigmoid scores and divide by their sum plus `1e-20`.
  Reduce routed outputs in FP32, cast to activation dtype, add shared output.
  There is no routed scale, softmax router, expert grouping or shared gate.
- Preserve checkpoint floating-point parameter dtypes; project the final
  normalized hidden states in FP32. Existing public `Linear`, `Embedding`
  and `SwitchGLU` projection modules retain the exact tensor namespaces.
  The optional fused gate/up bank duplication is disabled for this model.
- The sanitizer preserves stacked `.weight`/`.scales`/`.biases` keys, norms,
  sinks and unknown tensors. It strips the optional `language_model.` wrapper,
  maps the reference's two router-bias aliases, and supplies zero expert bias
  only when absent alongside the router weight (as in the selected export).
  Colliding aliases are preserved, not overwritten. Strict `.noUnusedKeys`
  remains in force; unknown keys are never dropped to force acceptance.

The inspected index has 1,751 keys, 49 sinks, 423 routed projection/quant
keys, and no router scales or expert-bias tensors. Key-schema checks are not
proof that real checkpoint arrays bind or execute correctly.

## Parsing and authored generation defaults

Explicit parser overrides are `.glm4` and `think_xml` only. The pinned GLM
parser recognizes `<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`.
Its scalar JSON fragment path can return numeric-looking text; Mei's existing
schema-aware API normalizer types numeric arguments without changing strings.

The pinned generation loop uses `ReasoningParser.forPrompt`, which starts in
content when the actual prompt tail does not prefill `<think>`. It then
recognizes the model's self-emitted `<think>…</think>`. A stamp-only parser
would incorrectly start in reasoning; tests deliberately exercise the actual
prompt-aware API. No think opener/closer, effort toggle, forced stop, template
rewrite or token bias is added.

For this path `ServerConfig` reads the bundle's `generation_config.json`:
selected defaults are temperature **1.0**, top_p **0.95**, top_k **20**.
Explicit CLI/programmatic sampler assignments and per-request overrides win.
The launcher omits unset sampler flags rather than silently passing 0.6.
It still permits explicit `MEI_TEMPERATURE`, `MEI_TOP_P`, `MEI_TOP_K`, including
zero. Parser registration supplies no generation/EOS override; the factory
continues to read the bundle's generation/EOS configuration.

## Model-free checks and remaining gates

`FlintTests.swift` covers configuration, gating, registration,
sanitization, routing/reduction, defaults and parsing. `FlintSyntheticTests.swift`
uses tiny generated tensors: a scalar attention oracle (QK norms, optional
RoPE, learned sinks, full/sliding masks and chunked caches), actual public
registry instantiation and the **pinned real affine loader**, including mixed
3/4/8-bit module substitution, unquantized router, tensor identity and strict
unknown-key failure. These tests do not read downloaded model weights.
`scripts/test_flint_launcher.py` exercises the real launcher's dry-run path;
it also runs from the existing `scripts/test_start_mei_server_flags.sh` check.
The gate's model-type sniffing preserves the pinned loader's JSON5 support
for other architectures. Parser tests use the bundle's native `<|assistant|>`
prompt tail, not a Qwen-style prompt or a forced thinking marker.

Use an isolated scratch path and explicit scratch `TMPDIR`:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export TMPDIR="$HOME/.hermes/cache/scratch"
BUILD="$TMPDIR/flint-mei-verify/build" # independent task scratch, not another run's cache
swift test --parallel --scratch-path "$BUILD" --filter Flint
python3 scripts/test_flint_launcher.py
bash scripts/test_start_mei_server_flags.sh
# All model-free suites; no live server or staged-tokenizer matrix:
swift test --parallel --scratch-path "$BUILD" \
  --skip MeiAcceptanceTests --skip StructuredOutputModelMatrixTests
swift build -c release --product mei --scratch-path "$BUILD"
git diff --check
```

Independent source-phase verification on 2026-10-23 used Xcode Swift 6.4 and
an APFS clone at `flint-mei-independent-20261023/build`; the supplied
`flint-mei-source-20261022/build` was preserved read-only. SwiftPM and tests
ran under network denial with `--disable-sandbox --disable-automatic-resolution`
(the first option disables only SwiftPM's nested sandbox). The selected
artifact's config, generation config and 1,751-key index were re-fetched as
text at the exact revision and matched the earlier metadata byte-for-byte;
no weight shards were fetched. All 25 resolved dependency checkout SHAs matched
the unchanged lock, including the exact vMLX pin above.

Behavioral RED was reproduced for the non-Flint JSON5 regression (one test,
exit 1), and for three review regressions (eight failing assertions, exit 1):
custom quantization fields/modes, malformed/conflicting RoPE fields and
unsupported quantized KV. After narrow fixes, the focused Flint suite passed
26 tests and the complete permitted model-free parallel suite passed 401 tests
with no failures or skips (both exit 0). The two named live-server and staged
tokenizer matrix suites were explicitly excluded, not counted as passing.
The earlier implementation RED log is a compile-time missing-implementation
failure, not numerical or real-checkpoint evidence.

The launcher tests were also replayed against the exact Mei base script in
scratch (five failures, exit 1), then passed against the candidate (five tests,
exit 0). The existing launcher contract passed all six checks, including 35
CLI-flag validation invocations against the freshly built release binary;
those invocations use a nonexistent model path and never start a server.
`swift build -c release --product mei` completed successfully (exit 0).
The actual arm64 release artifacts from that independent build are:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `out/Products/Release/mei` | 92,742,128 | `cee07e54e2d8f72e197045ade171942efffcc267b50ff13765f296484e1be624` |
| Cmlx bundle `default.metallib` | 2,422,136 | `0f6bdc29ff9afb81369eb5ca6f6561e0fdbb63228db0d4520c91e3c281738da3` |

Both were emitted during the independent build, not copied into place as
substitute artifacts. vMLX's clean submodule revisions were MLX
`c0a51a085b9252dc6469afe1aed772b1f538c9b0` and mlx-c
`2d783ac38713458eae2067ffff9ef8ebbff2ec70`. Full command/exit logs, xUnit
results, text-metadata hashes and build identity are retained under
`$HOME/.hermes/cache/scratch/flint-mei-independent-20261023/`.

Still required before any loadability/support/benchmark claim: exact artifact
integrity/alignment and resource admission; fresh executable/metallib provenance;
real strict weight binding; tokenizer/template replay; logit/output parity with
the pinned reference; prefill across the 513-token sliding boundary and long
context; plain/multi-turn generation; buffered/streaming reasoning and tools;
cache-off/on correctness and memory behavior. Real launch, full benchmark
suite and any staging exception belong to a separately authorized phase.
