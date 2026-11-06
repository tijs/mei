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
