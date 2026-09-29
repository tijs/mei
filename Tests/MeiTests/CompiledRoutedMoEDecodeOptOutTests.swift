import Foundation
import XCTest
@testable import MeiCore

/// Model-free unit tests for the F1/C2 compiled routed-MoE decode opt-out
/// (`ModelOptimizationProfile`). No model, no GPU, no server: every case is a
/// temporary directory with a config.json (and optionally sidecars).
///
/// Evidence behind the policy: the compiled routed-MoE decode region measured
/// control 67.11 vs 60.62 tok/s short decode (-9.7%) and -4.9% at 30k on the
/// text-only qwen3_5_moe topology, token-for-token correct (Kiem note
/// 21c84df0), and upstream #455 turned it on by default for that path. The
/// metadata shape below is taken from the real bundles: Ornith 1.5 and the
/// stripped Qwen3.6 text-only both declare root `model_type` qwen3_5_moe and
/// nested `qwen3_5_moe_text` with NO `vision_config`; the Qwen3.6 vision
/// bundle adds `vision_config` plus processor sidecars.
final class CompiledRoutedMoEDecodeOptOutTests: XCTestCase {

    // MARK: - Fixtures

    /// Shape shared by Ornith 1.5 35B and the stripped Qwen3.6 35B text-only
    /// bundle. The image/video token ids are present in BOTH real text-only
    /// bundles, so they must not count as vision signals. Computed, not
    /// static: `[String: Any]` is not Sendable and Swift 6 rejects shared
    /// static storage of it.
    private var textOnlyMoeMetadata: [String: Any] {
        [
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "hidden_size": 2048,
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "quantization": ["group_size": 64, "bits": 4, "mode": "affine"],
            "text_config": [
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
            ],
        ]
    }

    /// What the vision bundle adds: `vision_config` (on top of the same text
    /// metadata) and processor sidecars.
    private var visionMetadata: [String: Any] {
        var metadata = textOnlyMoeMetadata
        metadata["vision_config"] = [
            "model_type": "qwen3_5_vit",
            "hidden_size": 1152,
            "depth": 27,
        ]
        return metadata
    }

    private func makeModelDirectory(
        config: Any?,                       // [String: Any], a raw JSON string, or nil (no config.json)
        sidecars: [String] = [],
        name: String = "mei-c2-optout"
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        if let config {
            let data: Data
            if let dictionary = config as? [String: Any] {
                data = try JSONSerialization.data(withJSONObject: dictionary)
            } else {
                data = Data((config as? String ?? "").utf8)
            }
            try data.write(to: directory.appendingPathComponent("config.json"))
        }
        for sidecar in sidecars {
            try Data("{}".utf8).write(to: directory.appendingPathComponent(sidecar))
        }
        return directory
    }

    // MARK: - Target identification (text-only qwen3_5_moe)

    func testOrnithShapedTextOnlyBundleIsATarget() throws {
        let directory = try makeModelDirectory(
            config: textOnlyMoeMetadata,
            name: "Ornith-1.5-35B-A3B-MLX-4bit-aligned")
        XCTAssertTrue(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: directory.path))
    }

    func testStrippedQwen36TextOnlyBundleIsATarget() throws {
        // Same metadata signature as Ornith: qwen3_5_moe + qwen3_5_moe_text,
        // image/video token ids retained, and no vision metadata. The name is
        // deliberately the vision bundle's name — the decision must come from
        // metadata, never from the path.
        let directory = try makeModelDirectory(
            config: textOnlyMoeMetadata,
            name: "Qwen3.6-35B-A3B-4bit")
        XCTAssertTrue(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: directory.path))
    }

    func testRootLevelQwen35MoeTextModelTypeIsATarget() throws {
        let directory = try makeModelDirectory(
            config: ["model_type": "qwen3_5_moe_text", "hidden_size": 2048])
        XCTAssertTrue(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: directory.path))
    }

    func testVisionBundleWithVisionConfigIsExcluded() throws {
        let directory = try makeModelDirectory(
            config: visionMetadata,
            name: "Qwen3.6-35B-A3B-4bit-vision")
        XCTAssertFalse(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: directory.path))
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutDecision(
                modelDirectory: directory.path, operatorEnvironment: [:]),
            .notApplicable)
    }

    func testVisionBundleWithProcessorSidecarsIsExcluded() throws {
        // Without `vision_config` the sidecars alone must still exclude it:
        // the real vision repo carries processor_config.json,
        // preprocessor_config.json and video_preprocessor_config.json.
        for sidecar in [
            "processor_config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "image_processor_config.json",
        ] {
            let directory = try makeModelDirectory(
                config: textOnlyMoeMetadata, sidecars: [sidecar])
            XCTAssertFalse(
                ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
                    modelDirectory: directory.path),
                "\(sidecar) must exclude the bundle from the opt-out")
        }
    }

    func testDeclaredButNullVisionConfigIsNotProvenTextOnly() throws {
        // Fail closed: a declared vision slot is not positive proof of a
        // text-only bundle, so the opt-out does not fire.
        let directory = try makeModelDirectory(
            config: ["model_type": "qwen3_5_moe", "vision_config": NSNull()])
        XCTAssertFalse(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: directory.path))
    }

    // MARK: - Fail-closed: malformed / unknown metadata

    func testMissingConfigFailsClosed() throws {
        let directory = try makeModelDirectory(config: nil)
        XCTAssertFalse(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: directory.path))
        // A path that does not exist at all proves nothing either.
        XCTAssertFalse(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: "/tmp/mei-c2-optout-does-not-exist-\(UUID().uuidString)"))
    }

    func testOrdinarySidecarsDoNotExcludeATarget() throws {
        // The real stripped Qwen3.6 text-only bundle carries configuration.json
        // and generation_config.json; only vision/processor sidecars are
        // vision signals.
        let directory = try makeModelDirectory(
            config: textOnlyMoeMetadata,
            sidecars: ["configuration.json", "generation_config.json",
                       "tokenizer_config.json", "chat_template.jinja"])
        XCTAssertTrue(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: directory.path))
    }

    func testMalformedConfigFailsClosed() throws {
        let notJSON = try makeModelDirectory(config: "not-json")
        XCTAssertFalse(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: notJSON.path))
        // Valid JSON that is not an object must also prove nothing.
        let notAnObject = try makeModelDirectory(config: "[1, 2, 3]")
        XCTAssertFalse(ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
            modelDirectory: notAnObject.path))
    }

    func testOrdinaryAndDenseModelTypesAreNotTargets() throws {
        for modelType in ["llama", "qwen3_5", "qwen3_5_text", "gemma4", "qwen3_8"] {
            let directory = try makeModelDirectory(config: ["model_type": modelType])
            XCTAssertFalse(
                ModelOptimizationProfile.needsCompiledRoutedMoEDecodeOptOut(
                    modelDirectory: directory.path),
                "\(modelType) is not a text-only qwen3_5_moe target")
            XCTAssertEqual(
                ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutDecision(
                    modelDirectory: directory.path, operatorEnvironment: [:]),
                .notApplicable,
                "\(modelType) must keep its environment untouched")
        }
    }

    // MARK: - Environment application: defaults and explicit operator override

    func testDefaultEnvironmentAppliesTheKiemPlanSwitches() throws {
        let directory = try makeModelDirectory(config: textOnlyMoeMetadata)
        let decision = ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutDecision(
            modelDirectory: directory.path, operatorEnvironment: [:])
        XCTAssertEqual(decision, .optOut)
        // Exactly the Kiem plan pair (notes c85a5754 / 5fbb3daa), nothing else.
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutSwitches,
            [
                "VMLX_QWEN35_COMPILE_DECODE_REGIONS": "0",
                "VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE": "0",
            ])
    }

    func testExplicitOperatorOverrideOfEitherSwitchSuppressesTheWholeOptOut() throws {
        let directory = try makeModelDirectory(config: textOnlyMoeMetadata)

        // Force-on for an A/B leg: the operator keeps control of the pair.
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutDecision(
                modelDirectory: directory.path,
                operatorEnvironment: ["VMLX_QWEN35_COMPILE_DECODE_REGIONS": "1"]),
            .operatorOverride(switches: ["VMLX_QWEN35_COMPILE_DECODE_REGIONS"]))

        // The surgical region kill alone is also an explicit operator state.
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutDecision(
                modelDirectory: directory.path,
                operatorEnvironment: ["VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE": "0"]),
            .operatorOverride(switches: ["VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE"]))

        // Both supplied: reported in sorted order.
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutDecision(
                modelDirectory: directory.path,
                operatorEnvironment: [
                    "VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE": "1",
                    "VMLX_QWEN35_COMPILE_DECODE_REGIONS": "1",
                ]),
            .operatorOverride(switches: [
                "VMLX_QWEN35_COMPILE_DECODE_REGIONS",
                "VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE",
            ]))

        // Even an empty value counts as supplied: vmlx treats any non-"0"
        // value as enable, so the operator's intent is not overridden.
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutDecision(
                modelDirectory: directory.path,
                operatorEnvironment: ["VMLX_QWEN35_COMPILE_DECODE_REGIONS": ""]),
            .operatorOverride(switches: ["VMLX_QWEN35_COMPILE_DECODE_REGIONS"]))
    }

    func testUnrelatedOperatorEnvironmentDoesNotSuppressTheOptOut() throws {
        let directory = try makeModelDirectory(config: textOnlyMoeMetadata)
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutDecision(
                modelDirectory: directory.path,
                operatorEnvironment: [
                    "VMLX_ENABLE_UNSAFE_COMPILE": "0",
                    "PATH": "/usr/bin",
                    "VMLX_QWEN35_COMPILE_DECODE_REGIONS_OTHER": "1",
                ]),
            .optOut)
    }

    func testNonTargetBundleNeverTouchesTheEnvironmentEvenWithOperatorValues() throws {
        let directory = try makeModelDirectory(config: visionMetadata)
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutDecision(
                modelDirectory: directory.path,
                operatorEnvironment: ["VMLX_QWEN35_COMPILE_DECODE_REGIONS": "1"]),
            .notApplicable,
            "a vision bundle must stay untouched regardless of operator values")
    }

    // MARK: - Startup/reporting contract

    func testReportingContractNamesTheAppliedSwitches() {
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutMessage(for: .optOut),
            "mei: compiled routed-MoE decode opt-out for text-only qwen3_5_moe: "
                + "VMLX_QWEN35_COMPILE_DECODE_REGIONS=0 VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE=0")
    }

    func testReportingContractNamesTheSkippedOverride() {
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutMessage(
                for: .operatorOverride(switches: ["VMLX_QWEN35_COMPILE_DECODE_REGIONS"])),
            "mei: compiled routed-MoE decode opt-out skipped for text-only qwen3_5_moe: "
                + "operator environment sets VMLX_QWEN35_COMPILE_DECODE_REGIONS")
        XCTAssertEqual(
            ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutMessage(
                for: .operatorOverride(switches: [
                    "VMLX_QWEN35_COMPILE_DECODE_REGIONS",
                    "VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE",
                ])),
            "mei: compiled routed-MoE decode opt-out skipped for text-only qwen3_5_moe: "
                + "operator environment sets VMLX_QWEN35_COMPILE_DECODE_REGIONS, "
                + "VMLX_QWEN4_EXP_COMPILE_ROUTED_MOE")
    }

    func testOrdinaryModelsProduceNoReporting() {
        XCTAssertNil(ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutMessage(
            for: .notApplicable))
    }

    func testDecodeOptOutSwitchesAreAllReported() {
        // The startup switch log must cover every switch this policy can set,
        // or the benchmark cannot verify the effective state from server.log.
        XCTAssertTrue(
            Set(ModelOptimizationProfile.loggedDecodePolicySwitchNames).isSuperset(
                of: ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutSwitches.keys),
            "logged switches: \(ModelOptimizationProfile.loggedDecodePolicySwitchNames)")
    }
}
