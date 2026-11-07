# Mei 0.11.0 — release preparation

**Unreleased candidate. Stable remains 0.10.0.** The Apple-main integration
passed offline, native and API qualification before the attention selector
change. The new production artifact passed the offline suite and all three
native gates. Live API and performance qualification is in progress. Publication
and fresh four-model task results remain open.

Mei 0.11 integrates the refreshed vMLX/MLX dependency chain while preserving
Mei's cache budgets, raw completions and strict structured-output behavior.
The candidate pins vMLX `3cb7240a39d92475345c6c1d0c9057da31a2f289`, integrating
Osaurus `1bcd74ad`, Apple MLX Swift main `5f117a2e` (release0.33.0 plus three
commits), and MLX core0.32.4. Compatible dependencies include Swift Crypto5.1.1
and SwiftSyntax604.0.0.

The Apple additions include empty-result error recovery, a JACCL package trait
and Swift Testing migration. The integration retains fork compiler ownership,
locking, invalid-assignment behavior and dedicated allocator lifetime tests.
Both Cmlx trait builds pass. The selected dependency tests and their original
failures are recorded in the
[integration review](https://github.com/tijs/mei/tree/codex/dependency-refresh-20261106/artifacts/upstream-refresh-20261106/apple-main-integration).

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

The production candidate selects an existing attention kernel on M1 Max only
for single-query BF16 GQA in the measured 24,576–32,768-token band and matching
KV layouts. The existing threadgroup-limit fallback and final safety check
remain in place. An optimized diagnostic showed 7.2% faster decode and 0.7%
shorter total requests on Qwen at 30,000 tokens, with identical API-observable
output. That diagnostic had one measured loaded request per fresh process and
does not qualify the production candidate or establish a prefill improvement.
The production build is `282cbf9a202d85191ffe8f5b2df95a0109934592`, executable
`85ae551e76435dc3bf21f01c6b1c1f0699d803edefe6c80f36ba1d1023fd1756`, with the
same source-built Metal library. The refreshed offline suite passed 495 XCTest
cases (two actual-model skips, no failures) plus four Swift Testing cases. Qwen,
Groot and Ornith each passed all 13 native probes, including 65,536-token
admission, and all 42 model files rehashed unchanged after the runs. The six
untraced timing and API cells use this executable and remain separate from
the preceding artifact's results.

### Apple-main artifact before the selector

The preceding Apple-main artifact uses source `651d7c93f003aab10581a935c139ea717fa97335`,
executable `12fa6435558ce679b698804137e6c36ac9e031332408143096474405b33c1bd9`
and the unchanged source-built Metal hash below. All three models passed 13/13
native checks, including cache reuse and 65,536-token admission, with all 42 files
unchanged. Qwen/Ornith each record one pressure warning below the sustained stop
threshold. The fresh twelve-cell comparison passed all 168 observable checks,
108 API cases and 168 post-run model file checks. It retains all 120 measured
requests, using equal weight for the two fresh processes on each version.

| Model | Short decode | 30,000-token decode | Short request time | 30,000-token request time |
| --- | ---: | ---: | ---: | ---: |
| Qwen | +0.4% | −3.6% | −1.9% | +1.8% |
| Ornith | +20.7% | +11.8% | −17.5% | −7.1% |
| Groot | +1.6% | −0.6% | −2.8% | −0.3% |

Changes are relative to Mei 0.10.0 on a 32 GB M1 Max. Positive decode numbers
mean faster generation; negative request times mean shorter waits. Qwen's
loaded backend prefill took 2.0% longer. These are runtime measurements on
fixed requests; scored workflow tasks are a separate comparison. The
[final-artifact reviews](https://github.com/tijs/local-model-bench/tree/main/artifacts/mei-upstream-qualification-20261106)
preserve raw repeats and separate the preceding artifacts and diagnostics.

### Preceding batched artifact

The preceding frozen runtime, before the additional Apple-main merge, is source `588413434e9c5cc417e3a6da1bf9cef90598ba32`,
executable SHA-256
`195e0d43b09c25fa485c0955d9beda5a4ac40a04961791a97cc45ef399e9622e`,
with source-built Metal SHA-256
`5be6b742d5017c31a8db5b87cdf90265c8219867039eb871b42a737908aebc61`.
That artifact’s release build and full offline495-case suite (two skips, no failures)
plus four Swift Testing cases pass. Selected synthetic parser/cache/width/NoPE
tests pass62 XCTest cases (one skip) plus132 Swift Testing cases. A five-input
negative control reproduces malformed NoPE declarations before validation.

Fresh native checks on Groot/Qwen/Ornith pass13/13 each, including65536-token
context; all42 files rehash unchanged. Groot64/64 passive status GETs succeed
during active prefill (maximum observed24.72ms). An untraced twelve-cell ABBA
speed, observable-parity and live-API comparison is complete. Each version has two
fresh processes and five measured requests per short/30,000-token context per
process; cold/warmups are separate. Old timings are not substituted. Profiles
retain weights, precision, budgets, cache and resource guards. Storage migration
is part of this aggregate runtime comparison, which does not isolate every patch.

The completed Qwen group improves decode speed, but long requests average1.48%
slower overall, with about2% slower prefill. A predeclared same-binary row-sharing
diagnostic did not support shared projection views as the cause: retaining
original projections adds about416MiB at nearly identical request times. Keep
row sharing enabled and retain the small latency difference in reporting. This
diagnostic is one process per arm and does not establish statistical equivalence.

Ornith's complete group passes; Groot has improved short decode, slightly lower
loaded decode, and faster whole-request means in both contexts on these runs.
The mixed results remain explicit. All168 observable rows,108 live API cases
and168 post-run model-file checks pass, with all120 measured requests retained.
These observations qualify only the frozen artifact above, not the additional
Apple source merge or a later executable.

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
[contribution index and proof scopes](https://github.com/tijs/mei/blob/codex/dependency-refresh-20261106/artifacts/upstream-refresh-20261106/README.md).
Additional issue-first proposals cover the scoped batching retune (draft corePR54,
vMLX issue763 because core issues are disabled) and malformed NoPE declarations
(vMLX issue764/PR765). Two test-only proposals cover width policies (issue766/PR767)
and tool/prompt-retention boundaries (issue768/PR769), for14 total. These are
proposed contributions, not upstream acceptance.

## Remaining release work

Finish native, API and performance qualification on the new executable,
publish the final source/artifact identity, then verify GitHub assets
and Homebrew delivery. After release, rerun Qwen, Ornith and Groot on the released
Mei artifact plus the pinned Logics-SWE llama.cpp control, seeds 17 and 29, using
the frozen household/Python benchmark. Publish all reviewed attempts and current
weighted-average comparisons before declaring the broader upgrade complete.
