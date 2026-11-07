# Mei 0.11.0 — release preparation

**Unreleased candidate. Stable remains 0.10.0.** Dependency integration and
selected regression tests, Mei's offline suite and optimized build pass. All
three full native gates pass. Final-artifact performance/API qualification,
publication and fresh four-model results remain open.

Mei 0.11 integrates the refreshed vMLX/MLX dependency chain while preserving
Mei's cache budgets, raw completions and strict structured-output behavior.
The candidate pins vMLX `1f90918ac0872b134ac40c31a1f4a7b68b4982f5`, integrating
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

The final Apple-main artifact uses source `651d7c93f003aab10581a935c139ea717fa97335`,
executable `12fa6435558ce679b698804137e6c36ac9e031332408143096474405b33c1bd9`
and the unchanged source-built Metal hash below. All three models pass13/13
native checks, including cache reuse and65,536-token admission, with all42 files
unchanged. Qwen/Ornith each record one pressure warning below the sustained stop
threshold; complete resource observations remain in the evidence. A fresh12-cell
ABBA/API campaign is running; performance qualification is still pending.
Qwen and Ornith's full four-cell audits pass. Qwen's loaded-context slowdown
remains an explicit finding before promotion; Ornith's candidate process means
are faster for decode and whole-request time at both contexts. Groot's full
group is still pending. These observations retain all measured repeats and equal
process weights; they are runtime qualification, not scored tasks. The
[current final-artifact reviews](https://github.com/tijs/local-model-bench/tree/main/artifacts/mei-upstream-qualification-20261106)
separate them from preceding artifacts. Prepared operator diagnostics have not
been built or executed.

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

Finish and review all balanced qualification cells, resolve any reproduced
regression, publish the final source/artifact identity, then verify GitHub assets
and Homebrew delivery. After release, rerun Qwen, Ornith and Groot on the released
Mei artifact plus the pinned Logics-SWE llama.cpp control, seeds 17 and 29, using
the frozen household/Python benchmark. Publish all reviewed attempts and current
weighted-average comparisons before declaring the broader upgrade complete.
