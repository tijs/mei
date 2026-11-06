# Mei 0.11.0 — release preparation

**Unreleased candidate. Stable remains 0.10.0.** A previous candidate passed
observable/API checks but showed a repeatable Groot slowdown. A scoped batching
retune and a reproduced NoPE configuration fix are now built; fresh native and
untraced qualification is running. Promotion remains held until those results
are reviewed. Fresh four-model task results follow the published release.

Mei 0.11 integrates the refreshed vMLX/MLX dependency chain while preserving
Mei's cache budgets, raw completions and strict structured-output behavior.
The candidate pins vMLX `710741556966841b98d863220bda2d034671a58e`, integrating
Osaurus `1bcd74ad`, Apple MLX Swift 0.33.0 and MLX core 0.32.4. Compatible
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

## Qualification status

The new frozen runtime is source `588413434e9c5cc417e3a6da1bf9cef90598ba32`,
executable SHA-256
`195e0d43b09c25fa485c0955d9beda5a4ac40a04961791a97cc45ef399e9622e`,
with source-built Metal SHA-256
`5be6b742d5017c31a8db5b87cdf90265c8219867039eb871b42a737908aebc61`.
The final release build and full offline495-case suite (two skips, no failures)
plus four Swift Testing cases pass. Selected synthetic parser/cache/width/NoPE
tests pass62 XCTest cases (one skip) plus132 Swift Testing cases. A five-input
negative control reproduces malformed NoPE declarations before validation.

Fresh native checks on Groot/Qwen/Ornith and an untraced twelve-cell ABBA speed,
observable-parity and live-API comparison are in progress. Each version has two
fresh processes and five measured requests per short/30,000-token context per
process; cold/warmups are separate. Old timings are not substituted. Profiles
retain weights, precision, budgets, cache and resource guards. Storage migration
is part of this aggregate runtime comparison, which does not isolate every patch.

The previous57b/9dd artifact passed13 native checks per model and all120 timing /
108 live-API observations, but both Groot candidate processes were slower than
both controls. Those attempts remain historical. A same-binary profiled50/200 MiB
command-budget diagnostic supports commit frequency as a contributor; it does
not replace fresh untraced release qualification. The retune preserves correct
storage-byte accounting and explicit overrides, changing only roomy g13s GPUs.
Optional speculative paths and Ripple65B retain separate live-model qualification
gates; this release does not implicitly enable them or change the benchmark roster.

[Current and historical runtime evidence](https://github.com/tijs/local-model-bench/tree/main/artifacts/mei-upstream-qualification-20261106).

## Upstream contributions

The overlap audit excluded fixes already present upstream. The first ten small proposals
were published, each after a problem issue, covering allocator admission,
diagnostic lookups, raw text, a missing target dependency, rotating isolation and
replay, shared cache-byte accounting, processor composition, optional-store
pressure and sentinel-only cache capability. See the
[contribution index and proof scopes](../artifacts/upstream-refresh-20261106/README.md).
Additional issue-first proposals cover the scoped batching retune (draft corePR54,
vMLX issue763 because core issues are disabled) and malformed NoPE declarations
(vMLX issue764/PR765). Two test-only proposals cover width policies (issue766/PR767)
and tool/prompt-retention boundaries (issue768/PR769), for14 total. These are
proposed contributions, not upstream acceptance.

## Remaining release work

Finish and review all balanced qualification cells, resolve any reproduced
regression, publish the final source/artifact identity, then verify GitHub assets
and Homebrew delivery. After release, rerun Qwen, Ornith and Groot on the released
Mei artifact plus the pinned Logics-SWE llama.cpp control, seeds 17 and 29, using
the frozen household/Python benchmark. Publish all reviewed attempts and current
weighted-average comparisons before declaring the broader upgrade complete.
