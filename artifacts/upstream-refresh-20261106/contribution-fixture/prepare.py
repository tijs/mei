"""Prepare a small consumer of one clean upstream contribution branch.

This is a test-only path dependency, never the Mei release manifest. Production
library sources come unchanged from the named branch. XCTest/Swift Testing
fixtures are copied with upstream locking, and all copied inputs are hashed.
The Cmlx test helper's explicit metallib override is omitted: the build's exact
Metal bundle must instead be provisioned beside the resulting test executable.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PROJECTS = Path("/Users/tijs/projects")
BRANCHES = {
    "pressure": (
        "vmlx-upstream-store-pressure-20261106",
        ["CacheStorePressureTests.swift"],
    ),
    "budget": (
        "vmlx-upstream-retained-budget-20261106",
        ["SharedPrefixCacheBudgetTests.swift"],
    ),
    "replay": (
        "vmlx-upstream-rotating-replay-20261106",
        ["RotatingPrefixReplayTests.swift"],
    ),
    "processor": (
        "vmlx-upstream-request-processor-20261106",
        ["StructuredOutputGenerationTests.swift"],
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=BRANCHES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    branch, tests = BRANCHES[args.kind]
    source = PROJECTS / branch
    if args.output.exists():
        raise ValueError("refuse existing fixture")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    if status.strip():
        raise ValueError(
            "commit contribution before preparing its qualification fixture"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    root = args.output.resolve()
    root.mkdir(parents=True)
    inputs = []

    def copy(relative, target, transform=None):
        raw = (source / relative).read_bytes()
        text = raw.decode()
        if transform:
            text = transform(text)
        output = root / "Tests" / target / Path(relative).name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        inputs.append(
            {
                "source": relative,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "fixture": str(output.relative_to(root)),
                "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "adapted": transform is not None,
            }
        )

    copy("Tests/MLXLMTests/MLXMetalTestLock.swift", "NativeTests")

    def serialization_without_explicit_cmlx_override(text):
        text = text.replace("import Cmlx\n", "")
        start = text.index("private let explicitMLXTestMetalLibrary: Void = {")
        end = text.index("let mlxTestSerializationQueue", start)
        return text[:start] + "func prepareMLXTestMetalLibrary() {}\n\n" + text[end:]

    copy(
        "Tests/MLXLMTests/MLXTestSerialization.swift",
        "NativeTests",
        serialization_without_explicit_cmlx_override,
    )
    for test in tests:
        copy("Tests/MLXLMTests/" + test, "NativeTests")
    target_names = ["NativeTests"]
    if args.kind == "processor":
        copy(
            "Tests/MLXLMCommonFocusedTests/TokenIteratorAdditionalProcessorTests.swift",
            "ProcessorTests",
        )
        copy(
            "Tests/MLXLMCommonFocusedTests/FocusedMLXTestSupport.swift",
            "ProcessorTests",
        )
        target_names.append("ProcessorTests")
    deps = ", ".join(
        f'.product(name: "{name}", package: "{branch}")'
        for name in ["MLX", "MLXNN", "MLXLMCommon"]
    )
    targets = ",\n".join(
        f'        .testTarget(name: "{target}", dependencies: [{deps}])'
        for target in target_names
    )
    (root / "Package.swift").write_text(f'''// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "ContributionQualification",
    platforms: [.macOS(.v14)],
    dependencies: [.package(path: "{source}")],
    targets: [
{targets}
    ]
)
''')
    (root / "fixture.json").write_text(
        json.dumps(
            {
                "kind": args.kind,
                "upstream_base": "effcd981851786b2ef993b780fabec13f6207e50",
                "source": str(source),
                "revision": commit,
                "inputs": inputs,
                "scope": "Clean-base library source through a test-only local SwiftPM consumer; no release dependency substitution",
            },
            indent=2,
        )
        + "\n"
    )
    print(root)


if __name__ == "__main__":
    main()
