# Mei 0.11.0 — release preparation

**Unreleased candidate. Stable remains 0.10.0.** The balanced observable/API
gates pass, but Groot has lower candidate throughput in both process pairs.
Stable promotion is held while that difference is investigated. Fresh four-model
task results will follow the published release; earlier scores do not describe
this candidate.

Mei 0.11 integrates the refreshed vMLX/MLX dependency chain while preserving
Mei's cache budgets, raw completions and strict structured-output behavior.
The candidate pins vMLX `9dd023d3299b16e7695470911c0a7ed36b7656c4`, integrating
Osaurus `effcd981`, Apple MLX Swift 0.33.0 and MLX core 0.32.4. Compatible
dependencies include Swift Crypto 5.1.1 and SwiftSyntax 604.0.0.

## Upgrading weight-storage settings

Upstream now loads weights into owned storage. Remove `--load-mmap true` and
non-off MLXPress settings from existing launch commands; these obsolete requests
fail before loading with migration guidance. Explicit `--load-mmap false` and
`--mlxpress off` remain accepted.

`--keep-model-in-memory true|false` requests the runtime's bounded weight-residency
lease, defaulting to true. The runtime can reduce or decline that lease based on
available headroom. Status reports the requested policy separately from allocator
counters. False opts out of the lease while keeping owned weight storage; it
does not unload the model or promise lower physical memory use. Earlier mmap
and alignment speed measurements remain historical. See
[runtime policies](RUNTIME-POLICIES.md) for the complete memory/cache controls.

## Correctness through the integration

The integration avoids replaying history after an answer when an unused exact
prompt snapshot was deliberately omitted. Independent stable and prefill-published
checkpoints remain usable. Shorter media-independent prefixes retain their full
safe boundary; only a full-request probe drops the replay-seed token.

Paged KV and companion materialization now use the existing global cache-I/O
gate. Real-Metal regressions first reproduced both paths completing while the
gate was held, then passed after serialization. Host-read/compiler adaptations
retain explicit ownership, failure and lock-order tests. Request-local
processors run before authoritative native schema masks, with constraint failure
and completion propagated through composites.

Mei keeps its tested strict schema implementation. Native upstream grammar,
concurrent checkpoint loading and speculative decoding have additional contract
and model-specific requirements; this refresh does not enable them implicitly.
Inherited Qwen/Gemma improvements and these integration boundaries are explained
in [the fork guide](VMLX-FORK.md).

## Qualification completed so far

The exact frozen candidate is Mei source
`57b620b352a7f479b4403cc1ea0c4b3fae06994b`, executable SHA-256
`20657dac37d08beb7b5f449c484472f601c30c018d25cc435e4368b3fcc03396`, and
source-built Metal SHA-256
`5be6b742d5017c31a8db5b87cdf90265c8219867039eb871b42a737908aebc61`.

- Full offline suite: 495 XCTest cases, two skipped, zero failures; four Swift
  Testing cases pass. Live-server acceptance is a separate gate.
- Native admission: Qwen, Ornith and Groot each pass 13 checks, including
  streaming/buffered tools, history/cache reuse and the 65,536-token context
  boundary. All 42 model files rehash unchanged.
- Qwen and Ornith each have one warning-pressure sample below the unchanged
  sustained stop threshold. Groot remains normal. Separate older residency-off
  attempts stopped under sustained pressure and remain preserved.
- Passive status observation during Groot's existing 65,536-token prefill:
  64 successful requests, maximum observed latency 21.13 ms. This is bounded
  responsiveness evidence, not a universal worst-case guarantee.

The balanced comparison uses two fresh processes per release and five measured
repeats per context per process, in ABBA order for each model. Cold observations
and warmups are separate; nine prebuilt live HTTP tests run against each process.
No previous timings or selected best repeats are substituted. All twelve cells
pass observable parity and 108 live API cases, with 120 measured requests.
Qwen/Ornith have no mean-throughput decline in these pairs; Groot does. The
component cause is unresolved, so no stable-readiness verdict is claimed.

The follow-up phase screen retains the frozen runtimes and numeric profiles.
The refreshed backend sends diagnostics to private unified logging by default;
capturing enabled phase counters in `server.log` also requires the existing
`VMLX_DIAGNOSTICS_STDERR=1` opt-in. An initial capture mismatch was preserved,
and fresh matched arms explicitly select that destination. Profiled observations
will not replace the untraced measurements.

Raw profiles, outputs, status/resource samples, post-run hashes and owned-cache
cleanup receipts are in the
[benchmark qualification campaign](https://github.com/tijs/local-model-bench/tree/main/artifacts/mei-upstream-qualification-20261106).
Native admission, deterministic observable text, short synthetic decode timing,
held-out task quality and real Peixinho app acceptance remain separate claims.

## Upstream contributions

The overlap audit excluded fixes already present upstream. Ten small proposals
were published, each after a problem issue, covering allocator admission,
diagnostic lookups, raw text, a missing target dependency, rotating isolation and
replay, shared cache-byte accounting, processor composition, optional-store
pressure and sentinel-only cache capability. See the
[contribution index and proof scopes](../artifacts/upstream-refresh-20261106/README.md).
These are proposed contributions, not claims of upstream acceptance.

## Remaining release work

Finish and review all balanced qualification cells, resolve any reproduced
regression, publish the final source/artifact identity, then verify GitHub assets
and Homebrew delivery. After release, rerun Qwen, Ornith and Groot on the released
Mei artifact plus the pinned Logics-SWE llama.cpp control, seeds 17 and 29, using
the frozen household/Python benchmark. Publish all reviewed attempts and current
weighted-average comparisons before declaring the broader upgrade complete.
