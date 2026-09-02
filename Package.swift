// swift-tools-version: 6.1
// Mei — a narrow, native Swift/MLX OpenAI-compatible inference server.
//
// vmlx-swift is pinned to the Mei-maintained fork revision that contains
// the five reviewed cache/generation commits originally developed in Mei.
// The fork keeps upstream as its parent and each Mei change is a separate
// cherry-pickable commit suitable for a later upstream PR.
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
            revision: "91fed8be21319f92ce5220622c6dcde0b851bdae"
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