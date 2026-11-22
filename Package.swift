// swift-tools-version: 6.3
// Mei — a narrow, native Swift/MLX OpenAI-compatible inference server.
//
import PackageDescription

let package = Package(
    name: "Mei",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "MeiCore", targets: ["MeiCore"]),
        .executable(name: "mei", targets: ["Mei"]),
    ],
    dependencies: [
        .package(url: "https://github.com/tijs/vmlx-swift.git",
                 revision: "06f811fac11c8127e36c45146b46ea68f879e81f"),
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.65.0"),
    ],
    targets: [
        .target(
            name: "MeiCore",
            dependencies: [
                .product(name: "VMLX", package: "vmlx-swift"),
                .product(name: "MLX", package: "vmlx-swift"),
                .product(name: "MLXFast", package: "vmlx-swift"),
                .product(name: "MLXNN", package: "vmlx-swift"),
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
                .product(name: "MLXNN", package: "vmlx-swift"),
                .product(name: "MLXLLM", package: "vmlx-swift"),
                .product(name: "MLXLMCommon", package: "vmlx-swift"),
                // The structured-output model matrix tests load the staged
                // checkpoints' real tokenizer.json through the same
                // `#huggingFaceTokenizerLoader()` bridge the Engine uses.
                .product(name: "MLXHuggingFace", package: "vmlx-swift"),
                .product(name: "VMLXTokenizers", package: "vmlx-swift"),
                .product(name: "MLXVLM", package: "vmlx-swift"),
                .product(name: "NIOEmbedded", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            exclude: ["Fixtures/JANGH2", "Fixtures/AutomaticModelSettings"]
        ),
    ]
)
