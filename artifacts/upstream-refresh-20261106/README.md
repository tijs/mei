# Upstream refresh qualification evidence

## Current candidate and remaining gates

The current frozen candidate is Mei `57b620b352a7f479b4403cc1ea0c4b3fae06994b`
with vMLX `9dd023d3299b16e7695470911c0a7ed36b7656c4`. Its executable SHA-256 is
`20657dac37d08beb7b5f449c484472f601c30c018d25cc435e4368b3fcc03396`;
the source-built Metal SHA-256 is
`5be6b742d5017c31a8db5b87cdf90265c8219867039eb871b42a737908aebc61`.
Later documentation and contribution-proof commits do not change that artifact.

The full offline gate passes 495 XCTest cases (two skipped) and four Swift
Testing cases. The same candidate passes all 13 native checks on each of Qwen,
Ornith and Groot, including streaming/buffered tools, history/cache reuse and
the 65,536-token context boundary. All 42 model files were rehashed unchanged.
Qwen and Ornith each had one warning-pressure sample, below the unchanged
sustained stop threshold; Groot remained normal. During Groot's existing long
prefill, all 64 passive status requests succeeded (maximum observed 21.13 ms).

Exact profiles, logs, resource samples and preservation receipts live in the
[benchmark qualification campaign](https://github.com/tijs/local-model-bench/tree/main/artifacts/mei-upstream-qualification-20261106).
Balanced 0.10.0/0.11.0 speed, greedy-output and live HTTP qualification is still
running. Stable remains **0.10.0**. Publishing 0.11.0, verifying Homebrew delivery
and obtaining fresh four-model scored results remain open. Native admission is
not a scored-task result or a throughput improvement claim.

## Contribution index

Each proposal has its own preceding problem issue. The PRs are proposals, not
claims of upstream acceptance.

| Problem | Issue → PR | Focused evidence |
|---|---|---|
| Repeated process-environment copies in decode diagnostics | [745](https://github.com/osaurus-ai/vmlx-swift/issues/745) → [746](https://github.com/osaurus-ai/vmlx-swift/pull/746) | Preserve current/legacy flag behavior |
| Bounded allocator rejects useful replacement shapes | [4841](https://github.com/ml-explore/mlx/issues/4841) → [4842](https://github.com/ml-explore/mlx/pull/4842) | Three baseline assertion failures; six fixed cases and six neighboring cases |
| Raw completions lose literal generation text | [747](https://github.com/osaurus-ai/vmlx-swift/issues/747) → [750](https://github.com/osaurus-ai/vmlx-swift/pull/750) | Raw-text path and stop propagation |
| Tensor-parallel target omits its direct MLXRandom dependency | [748](https://github.com/osaurus-ai/vmlx-swift/issues/748) → [749](https://github.com/osaurus-ai/vmlx-swift/pull/749) | Explicit target dependency |
| Restored rotating wrappers can share mutable state | [751](https://github.com/osaurus-ai/vmlx-swift/issues/751) → [752](https://github.com/osaurus-ai/vmlx-swift/pull/752) | Negative isolation assertions and positive public-path proof |
| Retained prefix components lack a shared byte allowance | [753](https://github.com/osaurus-ai/vmlx-swift/issues/753) → [754](https://github.com/osaurus-ai/vmlx-swift/pull/754) | 16 cases, including I/O serialization and zero-byte capability |
| Replay can select an incomplete or shortened safe boundary | [755](https://github.com/osaurus-ai/vmlx-swift/issues/755) → [756](https://github.com/osaurus-ai/vmlx-swift/pull/756) | Six real-Metal cases |
| Request processors need to compose with native schema constraints | [757](https://github.com/osaurus-ai/vmlx-swift/issues/757) → [758](https://github.com/osaurus-ai/vmlx-swift/pull/758) | 14 XCTest and six Swift Testing cases |
| Optional cache copies continue under host pressure | [759](https://github.com/osaurus-ai/vmlx-swift/issues/759) → [760](https://github.com/osaurus-ai/vmlx-swift/pull/760) | Five policy cases; no induced-pressure claim |
| Sentinel-only paged cache advertises unavailable persistence | [761](https://github.com/osaurus-ai/vmlx-swift/issues/761) → [762](https://github.com/osaurus-ai/vmlx-swift/pull/762) | One negative failure, then five passing cases |

The no-coordinator/auxiliary snapshot guards and quantization alias handling
already exist upstream and were excluded as duplicates. Fork-specific prompt
snapshot ownership/budget policy was not proposed as a broad independent patch
without evidence that it benefits upstream's different ownership model.

The sections below retain the chronology and original identities of earlier
attempts. Statements that a gate was pending describe that artifact at that time;
they do not replace the current status above.

## Initial integration gate

Candidate vMLX: `e408c6ebdabb9d2dca382605fb6cb510c09e9996`.
It integrates Osaurus `effcd981`, Apple MLX Swift 0.33.0, MLX core 0.32.4
(`ecc27d0b4b5d221cfe3dd5da50ab0e55d1a869d6`) and the merged C ABI
(`a69a7c5c08691025d38325b200d1cccdeff9236e`). Crypto resolves to 5.1.1;
SwiftSyntax resolves to 604.0.0. The package requires Swift tools 6.3.

[Qualification summary](qualification-summary.json) selects 398 passing
model-free tests across 22 groups, plus 10 C++ core tests with 336 assertions.
This includes compiler lifetime/lock ordering, materialized arrays, host-read
races and failures, safetensors loading, quantization, schema generation,
rotating replay, shared retention accounting, raw text and request processors.

The initial failures remain alongside their subsequent checks. The summary
explains the serialized host-read test, consumed-snapshot lifetime assertions
and source-contract updates. The tests still exercise explicit concurrency,
real stores and cold/warm output parity.

These are integration gates, not model benchmark results or a stable-release
qualification. Version-reporting macros were corrected to 0.32.4 after the
focused unit runs. A fresh build from the published pin, the full Mei suite,
real-model native/cache/context checks and performance qualification must pass
before promotion. No new stable release or throughput improvement is claimed
by this evidence.

The scripts record the exact filtered suites and preserve per-suite logs.
Their absolute paths describe this qualification host; they are not portable
install instructions.

First atomic contribution: [problem #745](https://github.com/osaurus-ai/vmlx-swift/issues/745)
and [resolving PR #746](https://github.com/osaurus-ai/vmlx-swift/pull/746),
head `79cc30f71bf6244c98e08bf01c2ab6672f9b879e`.

## Mei storage/residency adapter

The follow-up vMLX pin `d0e3d45404f9f19ea8dae2a3fca44edfff939870`
prevents post-answer history replay when an exact prompt snapshot was deliberately
omitted. The initial full suite caught this integration regression; its failed
log is preserved alongside the passing rerun.

[Adapter summary](mei-adapter-summary.json): 491 XCTest cases, two skipped,
zero failures, plus three Swift Testing cases. The native acceptance class was
excluded because no model server was running. Eight Mei launcher tests and
17 benchmark launcher tests passed separately.

Mei now uses owned weights with an explicit bounded residency request. Obsolete
mmap=true and non-off cold-tier settings fail before loading. The status field
reports the requested residency policy, never an unobserved grant. Canonical
hybrid stripped checkpoints may already be durable after prefill; tests verify
that entry and stable stores instead of retaining an unused in-flight copy.
Real-model loading, memory, cache/context and speed gates remain outstanding.

## Upstream allocator contribution

[Issue #4841](https://github.com/ml-explore/mlx/issues/4841) was opened before
[PR #4842](https://github.com/ml-explore/mlx/pull/4842). The clean upstream-base
regression failed three assertions before the fix, then passed all six; six
neighboring tests passed 22 assertions. The [proof manifest](core-upstream-contribution/manifest.json)
records exact source and logs. This small contribution contains only allocator
admission and its regression; the full Osaurus integration is not proposed to Apple.

The build-cache cleanup manifest records removal of reproducible intermediates
from idle owned builds. Compiled Products, weights and benchmark evidence were
preserved.

## Corrected 0.11.0 candidate

The next published pin `05cca5906d7a8ebe626f88b1dc31b4dd3afded60` adds the
missing direct MLXRandom dependency to MLXDistributedTP. The fresh offline run
passes 492 XCTest cases (two skipped) and four Swift Testing cases, with no
missing-direct-dependency warning. [Candidate summary](mei-011-candidate-summary.json).
The first release build passed but reported that warning; its log is retained.
The corrected release build and real-model gates remain pending.

Small issue-first upstream proposals now also cover [raw completions #750](https://github.com/osaurus-ai/vmlx-swift/pull/750),
[direct dependency #749](https://github.com/osaurus-ai/vmlx-swift/pull/749), and
[rotating restore isolation #752](https://github.com/osaurus-ai/vmlx-swift/pull/752).
The [rotating proof](rotating-restore-contribution/manifest.json) separates a
negative exact-upstream-function probe (12 failed isolation assertions) from
the positive public restore path across standalone and composite layouts.
These are open contributions, not claims of upstream acceptance.

## Media-independent prefix replay regression

A new negative consumer probe reproduced a missing safe hit: a shorter
media-independent prefix was shortened again by the replay-seed rule. The
[regression summary](media-prefix-regression/summary.json) preserves one failed
case before the fix and the full passing rerun after it:493 XCTest cases (two
skipped), zero failures, plus four Swift Testing cases. The updated published
pin is `3fa3d7edbc98b31cad4e5589a35e58783c7561fe`. Only full-request probes
now drop a final token; shorter probes already leave an unprocessed suffix.

The earlier Qwen native passes belong to the05cca590 frozen binary. They remain
under their original identity; the corrected candidate requires a fresh build
and qualification. No stable release or benchmark score is claimed here.

## Rebuilt artifact and live model admission

The47fdf020 candidate release build passed; exact binary/Metal identities and
native results are published in [local-model-bench](https://github.com/tijs/local-model-bench/tree/main/artifacts/mei-upstream-qualification-20261106).
The new Ornith artifact passed13 native probes including65536-token context
with default residency=true. There was one warning-pressure sample, below the
unchanged sustained stop threshold; all11 model hashes remain unchanged. Two
residency-off attempts stopped on sustained pressure after successful load.
A read-only cache-isolation retry did not resolve that condition. These are
admission diagnostics; neither speed improvement nor stable readiness is claimed.
Qwen/Groot rebuilt-artifact native and balanced throughput gates remain open.

Additional atomic proposals are being qualified through test-only local SwiftPM
consumers on clean upstream bases. The fixture preparation script records exact
source revisions and test-source hashes. It does not change Mei release pins.

## Clean-base contribution qualification and review fix

Four additional proposals build against clean upstream effcd981. The focused
consumer proofs are separate from the integrated Mei suite:

| Proposal | Passing cases | Proof |
|---|---:|---|
| Shared retained-byte allowance |14 XCTest | [manifest](shared-budget-contribution/manifest.json) |
| Complete rotating replay boundary |6 XCTest | [manifest](rotating-replay-contribution/manifest.json) |
| Request-local processor, native schema outermost |14 XCTest +6 Swift Testing | [manifest](request-processor-contribution/manifest.json) |
| Kernel-pressure optional-store veto |5 XCTest | [manifest](store-pressure-contribution/manifest.json) |

The correctness review found that the new paged materialization needed the
existing global cache-I/O gate. Two real-Metal regressions reproduced stores
completing while that gate was held, without inducing a crash. Both fail before
the fix and all14 budget cases pass afterward. The integrated vMLX pin is now
9dd023d3299b16e7695470911c0a7ed36b7656c4. Mei passes495 offline XCTest cases
(two skipped) plus4 Swift Testing cases; [logs and scope](paged-io-review-fix/summary.json).

A new frozen release artifact and real-model gates are required for this pin.
The earlier47f/3fa native passes keep their original identity. The contribution
proofs use source-built Metal for the exact clean base; no full upstream package
suite, scored-task improvement or all-family memory guarantee is claimed.

## Published atomic proposals

The additional issue-first proposals are [shared optional byte budget#754](https://github.com/osaurus-ai/vmlx-swift/pull/754)
(issue753), [safe replay#756](https://github.com/osaurus-ai/vmlx-swift/pull/756)
(issue755), [request processors#758](https://github.com/osaurus-ai/vmlx-swift/pull/758)
(issue757), and [optional-store pressure veto#760](https://github.com/osaurus-ai/vmlx-swift/pull/760)
(issue759). Each is independent against effcd981. The byte-budget API is newly
proposed upstream; it was already present in the Mei fork, so it must not be
described as repairing an existing upstream maxCacheBytes setting.

The historical no-coordinator/auxiliary snapshot guards already exist upstream
and are not proposed again. A remaining independent sentinel-only capability
bug has [negative and positive proof](boundary-capability-contribution/manifest.json):
five cases, one failure before the data-block check and zero failures afterward.
The integrated fork already contains this check, so it does not change the
new57b620b/9dd023d3 release artifact.

Upstream CI explicitly marks repository-wide formatting advisory. The observed
lint job recursively rewrites unrelated baseline Swift/CMake files; no wholesale
formatting was applied. Linux SwiftPM/container builds observed successful;
queued Mac/CUDA jobs are not claimed green. Focused local proof remains bounded
as described in each manifest.
