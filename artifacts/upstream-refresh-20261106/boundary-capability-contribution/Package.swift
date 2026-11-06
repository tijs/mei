// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "ContributionQualification",
    platforms: [.macOS(.v14)],
    dependencies: [.package(path: "/Users/tijs/projects/vmlx-upstream-boundary-capability-20261106")],
    targets: [
        .testTarget(name: "NativeTests", dependencies: [.product(name: "MLX", package: "vmlx-upstream-boundary-capability-20261106"), .product(name: "MLXNN", package: "vmlx-upstream-boundary-capability-20261106"), .product(name: "MLXLMCommon", package: "vmlx-upstream-boundary-capability-20261106")])
    ]
)
