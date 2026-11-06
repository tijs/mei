# Pending Apple-main execution

These commands are prepared, **not executed**. Finish and review the live
qualification campaign first. Confirm its exact process is terminal, no Mei
server owns ports8017/8024, and the normal memory/disk gates permit build work.
Keep every current5884134/71074155 observation under its original identity.

Use the existing integration worktree/build directory to avoid a second full
build tree. Recheck that it is clean, then fast-forward its work branch to the
reviewed243cb4f2 source. Preserve original Products/proofs before pruning any
owned intermediates for disk capacity. Do not touch model files or unrelated
research. Execute builds sequentially with two compiler jobs.

From `/Users/tijs/projects/vmlx-mei-upstream-sync-20261106`:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build --target Cmlx --jobs 2 --disable-default-traits
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build --target Cmlx --jobs 2 --traits JACCL
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --jobs 2 --no-parallel \
  --filter 'UpstreamEmptyResultIntegrationTests|CompiledFunctionLifetimeTests|MLXArrayTests|ModuleTests|QuantizationTests|TransformTests'
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --skip-build --no-parallel --disable-swift-testing \
  --filter MLXArraySiblingLifetimeTests
```

Run the allocator-lifetime class in that dedicated filtered process, with no
other tests/model processes concurrently. Require exactly its two XCTest cases
and no Swift Testing cases in that isolated process; otherwise correct the
invocation before treating its allocation measurements as evidence.
The new compiler-bypass test requires
the unsafe-compile opt-ins unset or zero (record only those non-secret settings).
Retain complete command/status/test-discovery logs, not a grep-based verdict.
Use the previous integration's focused cache/processor/width/NoPE/ownership
selection as an additional regression gate. Confirm the sparse-module tests and
the two allocator tests actually execute; retain both documented RNG exclusions.
Inspect compiler defines for both trait settings and Xcode's default define.

Only after these gates pass, pin the exact tested vMLX revision in Mei, run Mei's
full offline suite and source-staging regression, build/freeze the new executable
and adjacent Metal resource, and qualify the affected runtime behavior. Compare
Metal/source identities explicitly. Current binary/native/performance evidence
does not automatically transfer to this additional source. Stable publication,
installed-artifact verification and the eight fresh task trajectories follow.
