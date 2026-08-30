// swift-tools-version: 6.1
// Mei — a narrow, native Swift/MLX OpenAI-compatible inference server.
//
// vmlx-swift is pinned to the exact revision the osaurus-ai/osaurus core
// package builds against (aeb5e21c, "Emit .info before cache persistence...",
// 2026-08-29-era upstream). Treat re-pinning as a deliberate, tested decision:
// every re-pin must re-run the acceptance suite and the long-context
// regression probes.
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
            url: "https://github.com/osaurus-ai/vmlx-swift.git",
            revision: "aeb5e21c195d8519609488ef75a25ce7e48d8f88"
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
            ]
        ),
    ]
)