# Upstream refresh qualification evidence

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
