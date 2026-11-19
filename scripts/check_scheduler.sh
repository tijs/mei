#!/usr/bin/env bash
# Compile the actual scheduler and its tests without loading/linking MLX.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/mei-scheduler.XXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/Sources/MeiCore" "$work/Tests/MeiTests"
ln -s "$repo/Sources/MeiCore/Scheduling.swift" "$work/Sources/MeiCore/Scheduling.swift"
for name in AsyncTestSupport RequestSchedulerTests SchedulerTraceReplayTests; do
  ln -s "$repo/Tests/MeiTests/$name.swift" "$work/Tests/MeiTests/$name.swift"
done
cat > "$work/Package.swift" <<'SWIFT'
// swift-tools-version: 6.3
import PackageDescription
let package = Package(name: "MeiSchedulerChecks", platforms: [.macOS(.v15)], targets: [
    .target(name: "MeiCore"), .testTarget(name: "MeiTests", dependencies: ["MeiCore"])
])
SWIFT
DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer} \
  MEI_SCHEDULER_FIXTURE_ROOT="$repo" swift test --build-system native --package-path "$work" --jobs 2 "$@"
