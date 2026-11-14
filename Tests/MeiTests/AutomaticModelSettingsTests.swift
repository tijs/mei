import CryptoKit
import Foundation
import XCTest
@testable import MeiCore

final class AutomaticModelSettingsTests: XCTestCase {
    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/AutomaticModelSettings")
    }

    private func parse(_ profile: String, extra: [String] = []) throws -> ServerConfig {
        try ServerConfig.parse(arguments: [
            "--model-dir", fixtures.appendingPathComponent(profile).path,
            "--served-model-id", "test/model", "--cache-mode", "off",
        ] + extra, recommendedWorkingSetBytes: 28_000_000_000, environment: [:])
    }

    private func temporaryCopy(_ profile: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-model-identity-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent(profile), to: root)
        return root
    }

    func testEveryCuratedArtifactAutomaticallySelectsItsOwnMeasuredSettings() throws {
        for tuning in ModelTuningRegistry.all {
            let config = try parse(tuning.name)
            XCTAssertEqual(config.modelTuning, tuning)
            XCTAssertTrue(config.modelTuningWasDetected)
            XCTAssertEqual(config.prefillStepSize, tuning.prefillStepSize)
            XCTAssertEqual(config.ssmAnchorBoundaryCount, tuning.ssmAnchorBoundaries)
            XCTAssertEqual(config.maxTokensDefault, tuning.maxTokens)
        }
        // Shared architecture/tokenizer settings do not conflate these models.
        XCTAssertEqual(try parse("ornith-1.5-35b-a3b").ssmAnchorBoundaryCount, 0)
        XCTAssertEqual(try parse("qwen3.6-35b-a3b-text").ssmAnchorBoundaryCount, 2)
        XCTAssertEqual(try parse("groot-19b-a3b-text").prefillStepSize, 256)
    }

    func testRecognitionDoesNotDependOnDirectoryNameAndRejectsChangedMetadata() throws {
        let directory = try temporaryCopy("groot-19b-a3b-text")
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertEqual(ModelTuningIdentity.detect(modelDirectory: directory.path)?.name,
                       "groot-19b-a3b-text")
        let file = directory.appendingPathComponent("generation_config.json")
        var data = try Data(contentsOf: file)
        data.append(contentsOf: "\n".utf8)
        try data.write(to: file)
        XCTAssertNil(ModelTuningIdentity.detect(modelDirectory: directory.path),
                     "even a valid re-export is not silently treated as the pinned artifact")
    }

    func testAmbiguousAndStaleRegistryFingerprintsFailClosed() throws {
        let profile = "groot-19b-a3b-text"
        let fingerprint = try XCTUnwrap(ModelTuningIdentity.fingerprints.first { $0.profile == profile })
        let directory = fixtures.appendingPathComponent(profile).path
        XCTAssertNil(ModelTuningIdentity.detect(modelDirectory: directory,
                                                candidates: [fingerprint, fingerprint]))
        let stale = ModelTuningIdentity.Fingerprint(profile: profile,
            revision: String(repeating: "0", count: 40), hashes: fingerprint.hashes)
        XCTAssertNil(ModelTuningIdentity.detect(modelDirectory: directory, candidates: [stale]))
    }

    func testMissingMalformedAndOversizedMetadataFailClosed() throws {
        let directory = try temporaryCopy("groot-19b-a3b-text")
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("config.json")
        let malformed = Data("[1,2]".utf8)
        try malformed.write(to: file)
        var hashes = try XCTUnwrap(ModelTuningIdentity.fingerprints.last).hashes
        hashes["config.json"] = SHA256.hash(data: malformed).map { String(format: "%02x", $0) }.joined()
        let candidate = ModelTuningIdentity.Fingerprint(profile: "groot-19b-a3b-text",
            revision: "53f52e0eb8d67d2941010688b2a6b8807d9a3bea", hashes: hashes)
        XCTAssertNil(ModelTuningIdentity.detect(modelDirectory: directory.path, candidates: [candidate]))
        try Data(repeating: 32, count: 1_048_577).write(to: file)
        XCTAssertNil(ModelTuningIdentity.detect(modelDirectory: directory.path))
        try FileManager.default.removeItem(at: file)
        XCTAssertNil(ModelTuningIdentity.detect(modelDirectory: directory.path))
        XCTAssertNil(ModelTuningIdentity.detect(modelDirectory: "/missing/mei/model"))
    }

    func testExplicitModelAndIndividualFlagsBeatAutomaticSelection() throws {
        let explicit = try parse("groot-19b-a3b-text", extra: [
            "--model-profile", "ornith-1.5-35b-a3b",
        ])
        XCTAssertEqual(explicit.modelTuning?.name, "ornith-1.5-35b-a3b")
        XCTAssertFalse(explicit.modelTuningWasDetected)
        let configured = try parse("groot-19b-a3b-text", extra: [
            "--prefill-step-size", "128", "--ssm-anchor-boundaries", "1",
            "--max-tokens", "512", "--compiled-decode", "true",
        ])
        XCTAssertEqual(configured.modelTuning?.name, "groot-19b-a3b-text")
        XCTAssertEqual(configured.prefillStepSize, 128)
        XCTAssertEqual(configured.ssmAnchorBoundaryCount, 1)
        XCTAssertEqual(configured.maxTokensDefault, 512)
        XCTAssertTrue(configured.enableCompiledDecode)
    }

    func testExplicitModelNoneOptsOutOfAutomaticSettings() throws {
        let none = try parse("groot-19b-a3b-text", extra: ["--model-profile", "none"])
        XCTAssertNil(none.modelTuning)
        XCTAssertFalse(none.modelTuningWasDetected)
        XCTAssertEqual(none.optimizationProfile, .qwen35MoE)
    }

    func testOnlyKnownGrootAndQualifiedChunksEnableCompleteKernelGroup() throws {
        let expected = Dictionary(uniqueKeysWithValues:
            ModelTuning.retainedGrootKernelSwitches.map { ($0, "1") })
        for tuning in ModelTuningRegistry.all {
            for chunk in [256, 4096] {
                XCTAssertEqual(tuning.retainedKernelEnvironment(prefillStepSize: chunk,
                    compiledDecode: false, environment: [:]),
                    tuning.name == "groot-19b-a3b-text" ? expected : [:])
            }
        }
        let groot = try XCTUnwrap(ModelTuningRegistry.named("groot-19b-a3b-text"))
        XCTAssertTrue(groot.retainedKernelEnvironment(prefillStepSize: 1024,
            compiledDecode: false, environment: [:]).isEmpty)
        XCTAssertTrue(groot.retainedKernelEnvironment(prefillStepSize: 256,
            compiledDecode: true, environment: [:]).isEmpty)
    }

    func testExplicitKernelOrRegionOverrideStandsDownWholeAutomaticGroup() throws {
        let groot = try XCTUnwrap(ModelTuningRegistry.named("groot-19b-a3b-text"))
        for key in ModelTuning.retainedGrootKernelSwitches {
            for value in ["0", "1", "invalid"] {
                XCTAssertTrue(groot.retainedKernelEnvironment(prefillStepSize: 256,
                    compiledDecode: false, environment: [key: value]).isEmpty)
            }
        }
        for key in ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutSwitches.keys {
            XCTAssertTrue(groot.retainedKernelEnvironment(prefillStepSize: 256,
                compiledDecode: false, environment: [key: "1"]).isEmpty)
        }
        // The automatic group never changes the existing unsafe/micro-fusion
        // policy, whether it was explicitly enabled or disabled.
        for value in ["0", "1"] {
            let changes = groot.retainedKernelEnvironment(prefillStepSize: 256,
                compiledDecode: false, environment: ["VMLX_ENABLE_UNSAFE_COMPILE": value])
            XCTAssertEqual(changes.count, 3)
            XCTAssertNil(changes["VMLX_ENABLE_UNSAFE_COMPILE"])
        }
    }
}
