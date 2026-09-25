// swift-tools-version: 6.1
// Mei — a narrow, native Swift/MLX OpenAI-compatible inference server.
//
// vmlx-swift is pinned to the pushed `main` of the Mei-maintained fork
// (fef563a5), which keeps upstream as its parent and carries the Mei
// cache/generation commits plus 18 more fork-side commits since the
// 0.4.2-era pin: the Bonsai 2 Prism-Hadamard model/runtime work
// (default-off gated), the quantization_config alias, and two upstream
// syncs (MLX C++ 0.32.2). Each Mei change is a separate cherry-pickable
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
            revision: "fef563a55b4f22d4530b3439c1edb233cfc44a8f"
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