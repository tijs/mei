# Flint JANGH2 — experimental Mei adapter

Default off. This path supports the explicit Flint Hadamard-32/odd-cubic v2
contract, with per-projection 2/3/4-bit LSB bitstreams, F16 row scales, BF16
routers/norms/sinks and affine8/group64 ordinary linears. It is distinct from
legacy JANGTQ and the existing standard-affine Flint path.

The vmlx dependency is pinned to published cache-fix revision
f6510fc384bcddb5db9efc96e1dd5ab3afc49346. No editable dependency is required.
The MIT projection/H32 kernel reference is osaurus-ai/vmlx-swift
1a084b45d2496002673c11a8d849627a212db03d; see THIRD-PARTY-NOTICES.md.

## Staging and admission

Preserve the immutable original bundle. Run:

```
python3 tools/prepare_flint_jangh.py ORIGINAL_DIRECTORY ADAPTER_DIRECTORY
mei --model-dir ADAPTER_DIRECTORY --enable-flint true --enable-flint-jangh true ...
```

The adapter retains the original config in `mei_jangh_config.json`. Its ordinary
config marks every custom bank `false` for generic quantization; the Mei module
consumes the original tq2_packed/tq2_scales keys directly. No weight payload is
renamed, requantized, dequantized permanently, filtered or substituted. Shards,
tokenizer, generation config and template are links to the preserved original.
Original config digest and adapter provenance are checked before load, and the
full architecture key/shape/dtype set is checked against all real headers.
The generic loader's noUnusedKeys guard still applies. Missing ordinary weights
cannot leave initialized values behind. Legacy runtime sidecars are refused.

Codebooks, map aliases, every projection's width and rotation, metadata version,
header length, offsets, byte count, dtype-natural alignment and overlaps are
validated. The adapter is not a label-based fallback to affine execution.
Sampling/template/EOS behavior comes from the bundle; no thinking markers or
hidden sampler fixes are introduced. Current native work uses compiled decode
off and requests a 512 MiB reusable-buffer cache cap; verify the actual
`/v1/mei/status` cache limit rather than trusting launch arguments. Live model/KV
allocations are not artificially capped to present a smaller footprint.

## Evidence required before a support verdict

Model-free tests cover default-off gates, malformed/contradictory metadata,
all-width numerical projections against an independent scalar oracle, 3-bit
cross-word packing, H32 normalization/involution, and the actual generic loader
with mixed-bank tiny fixtures. Python staging tests validate the complete
published artifact's 1,610 header shapes/dtypes and negative mutations.

For real-pack numerical parity, use raw prompt `The capital of France is`,
tokenized with the exact bundle tokenizer and no added special tokens. Compare
the last-position full-vocabulary FP32 logits with an independent Python
full-forward implementation of the pinned mlx-vlm Flint architecture and
jjang-ai/jangq@6dd1268e864a36bf40ee176c7e98b67336df14f2 packed primitives,
using the same artifact and the exact same C++/Metal backend build. The
Python wheel and Swift library can share a version number while differing in
math compilation. Link the pinned Swift Cmlx objects into a private Python
reference environment and use the same Metal library; record both hashes.
Keep both full-forward cache paths disabled for this comparison. Require
identical greedy top token and maximum
absolute logit difference <=0.05. Tolerance permits FP32 H32/reduction ordering;
no tolerance is increased automatically after a failed comparison. This is a
short forward parity gate, not proof of cache or long-context parity.

Then require buffered/streaming answers and native tools, tool-result
continuation, cold/warm/growing/diverging cache proof, context admission and
physical-footprint/host-pressure/swap telemetry. Missing proof remains partial.
Do not treat publisher BF16 fidelity numbers as this runtime's parity oracle.
Native runs must be serialized and supervised; stop at critical pressure,
sustained <=10% memory free, swap >=16 GiB, disk <10 GiB, or deadline/stall.
No full benchmark or generalized support claim follows from load-only success.

## Initial validation boundary (2026-10-25, historical)

The exact pinned pack loaded without realignment or dtype expansion. The final
independent full-forward comparison matched all 150,144 logits exactly (maximum
absolute difference 0; unchanged tolerance 0.05; identical greedy token 12834).
This proves the packed projection/model graph on the matched backend for that
short prompt. Initial wheel-backend comparisons failed and remain in the
benchmark evidence; they are not counted as passing runs. Exact stored F32
2-bit levels matter: recomputing those levels from the cubic coefficients
introduced a one-ULP discrepancy.

The existing short HTTP probe passed 10/10, including native tools in both
transport modes and repeated/growing-prefix cache counters. Tool-result
continuation also passed. A stronger diverging-transcript check found a
cold/warm greedy-answer mismatch (`ORANGE` versus `ORANGE` twice), despite the
same request and a reported 653-token cache hit. The initial word-presence
assertion was too weak; the separate stronger assessment records this as a
failure. Root cause is not established. Do not treat cache-hit counters as
correctness proof, or advertise complete native support while this remains.

This lane remains experimental and default off. No ranked benchmark or general
mixed-bit support verdict follows from these results. Full evidence is under
`local-model-bench` worktree `local-model-bench-flint-gate-20261022`, artifacts
`flint-jangh2-contract-20261025` and `flint-jangh2-native-20261025`.

The long-context admission run completed 65,535 input + 1 output token,
65,536 input + 1 output token, and rejected a 65,537-token prompt. The configured
context cap is a **prompt** cap, not a combined input/output limit. The initial
64K request took 716 seconds; extending its cached prefix took 2.45 seconds.
Host pressure stayed normal, swap remained 2,557.19 MiB, and minimum reported
free memory was 25 percent (a source/test build also ran during this phase).
The sampled process physical footprint peaked at 9.18 GB; report it beside MLX
allocator bytes and host pressure, because it does not by itself account for
all mmap/Metal residency.

Status inspection caught an existing allocator-policy bug: a cache-only limit
was ignored unless an explicit memory limit was also provided. Consequently
the long-context run's actual buffer-cache limit was 32,641,751,449 bytes,
despite the requested 536,870,912 bytes. The engine now applies the two explicit
limits independently before and after model loading. Preserve that distinction
when comparing the historical run to a profile using the corrected cap.

## Cache replay fix verified locally (2026-10-25)

The duplicated-answer failure above is resolved in the isolated runtime worktree
`vmlx-flint-cache-20261025`. An exact paged prompt hit restored every layer at N,
then the replay path trimmed only full-attention layers to N-1. Wrapped sliding
rings remained at N, so the final prompt token was forwarded twice in those
layers. The state trace showed full KV at 653 and every sliding ring at 654.

The runtime patch applies the replay-boundary exclusion to paged rotating
companions as well as disk entries. It also selects a shorter complete partial
boundary when longer unsealed full blocks would hide it. Dense lookup behavior
is unchanged; absent a safe captured boundary, lookup misses. Regressions cover
block/window boundaries, both store orders, missing seeds, growing/diverging
prefixes, dense controls and block-reference release. No sampler/template,
quantization or model-payload changes are involved.

Final validation: 418 offline Swift passes plus one explicit real-pack skip;
separate actual-pack numerical test passed with all 150,144 logits identical.
Seven strict HTTP replay/conversation checks passed with real 653/731-token
hits, ten standard HTTP/tool checks passed, and tool-result continuation plus a
following turn passed. The seven cache-disabled controls returned the same
visible answers. Hidden reasoning/token counts differ between cold cache modes;
full-trace equivalence between those profiles is not claimed. The old 64K run
has not been repeated on this patch and corrected buffer-cache cap.

The initial evidence used a local editable dependency. The reviewed runtime
fix is now published at f6510fc384bcddb5db9efc96e1dd5ab3afc49346 and both
Package.swift and Package.resolved pin it. A clean remote-pin resolution passed
419 offline tests with one opt-in real-pack skip (420 total). The absolute
Packages/vmlx-swift development symlink was removed before publication. The source patch and all raw evidence are in
`local-model-bench` artifacts `flint-jangh2-cache-debug-20261025`; earlier failed
artifacts remain unchanged. Keep the feature default off until the remaining
admission coverage and integration/release gates are complete. No Peixinho
benchmark or ranked score was produced by these native tests.
