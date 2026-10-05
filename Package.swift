// swift-tools-version: 6.1
// Mei — a narrow, native Swift/MLX OpenAI-compatible inference server.
//
// vmlx-swift is pinned to the pushed `main` of the Mei-maintained fork
// (633fe166), which keeps upstream as its parent and carries the Mei
// cache/generation commits, the request logit-processor seam, and the
// associated focused tests. Each Mei change is a separate cherry-pickable
// commit suitable for a later upstream PR.
import PackageDescription

let package = Package(
    name: "Mei",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MeiCore", targets: ["MeiCore"]),
        .executable(name: "mei", targets: ["Mei"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/tijs/vmlx-swift.git",
            revision: "633fe166630ef04310aea7d5a1795555ab32970d"
        ),
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.65.0"),
    ],
    targets: [
        .target(
            name: "MeiCore",
            dependencies: [
                .product(name: "VMLX", package: "vmlx-swift"),
                .product(name: "MLX", package: "vmlx-swift"),
                .product(name: "MLXLMCommon", package: "vmlx-swift"),
                .product(name: "MLXLLM", package: "vmlx-swift"),
                .product(name: "MLXHuggingFace", package: "vmlx-swift"),
                .product(name: "VMLXTokenizers", package: "vmlx-swift"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
            ]
        ),
        .executableTarget(
            name: "Mei",
            dependencies: ["MeiCore"]
        ),
        .testTarget(
            name: "MeiTests",
            dependencies: [
                "MeiCore",
                .product(name: "MLX", package: "vmlx-swift"),
                .product(name: "MLXLMCommon", package: "vmlx-swift"),
                // The structured-output model matrix tests load the staged
                // checkpoints' real tokenizer.json through the same
                // `#huggingFaceTokenizerLoader()` bridge the Engine uses.
                .product(name: "MLXHuggingFace", package: "vmlx-swift"),
                .product(name: "VMLXTokenizers", package: "vmlx-swift"),
            ]
        ),
    ]
)