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
