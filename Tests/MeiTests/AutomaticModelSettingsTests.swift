import CryptoKit
import Foundation
import Testing
@testable import MeiCore

@Suite("Automatic model settings")
struct AutomaticModelSettingsTests {
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

    @Test func everyCuratedArtifactAutomaticallySelectsItsOwnMeasuredSettings() throws {
        for tuning in ModelTuningRegistry.all {
            let config = try parse(tuning.name)
            #expect(config.modelTuning == tuning)
            #expect(config.modelTuningWasDetected)
            #expect(config.prefillStepSize == tuning.prefillStepSize)
            #expect(config.ssmAnchorBoundaryCount == tuning.ssmAnchorBoundaries)
            #expect(config.maxTokensDefault == tuning.maxTokens)
        }
        // Shared architecture/tokenizer settings do not conflate these models.
        #expect(try parse("ornith-1.5-35b-a3b").ssmAnchorBoundaryCount == 0)
        #expect(try parse("qwen3.6-35b-a3b-text").ssmAnchorBoundaryCount == 2)
        #expect(try parse("groot-19b-a3b-text").prefillStepSize == 256)
    }

    @Test func recognitionDoesNotDependOnDirectoryNameAndRejectsChangedMetadata() throws {
        let directory = try temporaryCopy("groot-19b-a3b-text")
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(ModelTuningIdentity.detect(modelDirectory: directory.path)?.name == "groot-19b-a3b-text")
        let file = directory.appendingPathComponent("generation_config.json")
        var data = try Data(contentsOf: file)
        data.append(contentsOf: "\n".utf8)
        try data.write(to: file)
        #expect(ModelTuningIdentity.detect(modelDirectory: directory.path) == nil, "even a valid re-export is not silently treated as the pinned artifact")
    }

    @Test func ambiguousAndStaleRegistryFingerprintsFailClosed() throws {
        let profile = "groot-19b-a3b-text"
        let fingerprint = try #require(ModelTuningIdentity.fingerprints.first { $0.profile == profile })
        let directory = fixtures.appendingPathComponent(profile).path
        #expect(ModelTuningIdentity.detect(modelDirectory: directory,
                                                candidates: [fingerprint, fingerprint]) == nil)
        let stale = ModelTuningIdentity.Fingerprint(profile: profile,
            revision: String(repeating: "0", count: 40), hashes: fingerprint.hashes)
        #expect(ModelTuningIdentity.detect(modelDirectory: directory, candidates: [stale]) == nil)
    }

    @Test func missingMalformedAndOversizedMetadataFailClosed() throws {
        let directory = try temporaryCopy("groot-19b-a3b-text")
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("config.json")
        let malformed = Data("[1,2]".utf8)
        try malformed.write(to: file)
        var hashes = try #require(ModelTuningIdentity.fingerprints.last).hashes
        hashes["config.json"] = SHA256.hash(data: malformed).map { String(format: "%02x", $0) }.joined()
        let candidate = ModelTuningIdentity.Fingerprint(profile: "groot-19b-a3b-text",
            revision: "53f52e0eb8d67d2941010688b2a6b8807d9a3bea", hashes: hashes)
        #expect(ModelTuningIdentity.detect(modelDirectory: directory.path, candidates: [candidate]) == nil)
        try Data(repeating: 32, count: 1_048_577).write(to: file)
        #expect(ModelTuningIdentity.detect(modelDirectory: directory.path) == nil)
        try FileManager.default.removeItem(at: file)
        #expect(ModelTuningIdentity.detect(modelDirectory: directory.path) == nil)
        #expect(ModelTuningIdentity.detect(modelDirectory: "/missing/mei/model") == nil)
    }

    @Test func explicitModelAndIndividualFlagsBeatAutomaticSelection() throws {
        let explicit = try parse("groot-19b-a3b-text", extra: [
            "--model-profile", "ornith-1.5-35b-a3b",
        ])
        #expect(explicit.modelTuning?.name == "ornith-1.5-35b-a3b")
        #expect(!explicit.modelTuningWasDetected)
        let configured = try parse("groot-19b-a3b-text", extra: [
            "--prefill-step-size", "128", "--ssm-anchor-boundaries", "1",
            "--max-tokens", "512", "--compiled-decode", "true",
        ])
        #expect(configured.modelTuning?.name == "groot-19b-a3b-text")
        #expect(configured.prefillStepSize == 128)
        #expect(configured.ssmAnchorBoundaryCount == 1)
        #expect(configured.maxTokensDefault == 512)
        #expect(configured.enableCompiledDecode)
    }

    @Test func explicitModelNoneOptsOutOfAutomaticSettings() throws {
        let none = try parse("groot-19b-a3b-text", extra: ["--model-profile", "none"])
        #expect(none.modelTuning == nil)
        #expect(!none.modelTuningWasDetected)
        #expect(none.optimizationProfile == .qwen35MoE)
    }

    @Test func onlyKnownGrootAndQualifiedChunksSelectBlockedPrefillWithStockDecode() throws {
        let expected = ["MEI_EXPERIMENTAL_ROUTER_TAIL": "0",
                        "MEI_EXPERIMENTAL_GDN": "0",
                        "MEI_EXPERIMENTAL_BLOCKED_GDN": "1"]
        for tuning in ModelTuningRegistry.all {
            for chunk in [256, 4096] {
                #expect(tuning.retainedKernelEnvironment(prefillStepSize: chunk,
                    compiledDecode: false, environment: [:]) == (tuning.name == "groot-19b-a3b-text" ? expected : [:]))
            }
        }
        let groot = try #require(ModelTuningRegistry.named("groot-19b-a3b-text"))
        #expect(groot.retainedKernelEnvironment(prefillStepSize: 1024,
            compiledDecode: false, environment: [:]).isEmpty)
        #expect(groot.retainedKernelEnvironment(prefillStepSize: 256,
            compiledDecode: true, environment: [:]).isEmpty)
    }

    @Test func explicitKernelOrRegionOverrideStandsDownWholeAutomaticGroup() throws {
        let groot = try #require(ModelTuningRegistry.named("groot-19b-a3b-text"))
        for key in ModelTuning.retainedGrootKernelSwitches {
            for value in ["0", "1", "invalid"] {
                #expect(groot.retainedKernelEnvironment(prefillStepSize: 256,
                    compiledDecode: false, environment: [key: value]).isEmpty)
            }
        }
        for key in ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutSwitches.keys {
            #expect(groot.retainedKernelEnvironment(prefillStepSize: 256,
                compiledDecode: false, environment: [key: "1"]).isEmpty)
        }
        // The automatic group never changes the existing unsafe/micro-fusion
        // policy, whether it was explicitly enabled or disabled.
        for value in ["0", "1"] {
            let changes = groot.retainedKernelEnvironment(prefillStepSize: 256,
                compiledDecode: false, environment: ["VMLX_ENABLE_UNSAFE_COMPILE": value])
            #expect(changes.count == 3)
            #expect(changes["VMLX_ENABLE_UNSAFE_COMPILE"] == nil)
        }
    }
}
