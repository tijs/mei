import Foundation
import XCTest
@testable import MeiCore

final class InklingSupportTests: XCTestCase {
    private func config() throws -> ServerConfig {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model_type":"inkling2_loop"}"#.utf8).write(to: root.appendingPathComponent("config.json"))
        try Data(#"{"temperature":1.0}"#.utf8)
            .write(to: root.appendingPathComponent("generation_config.json"))
        var config = ServerConfig(modelDirectory: root.path, servedModelID: "test")
        config.enableInklingMini = true; config.contextCap = 4096; config.cacheMode = .off
        return config
    }
    func testGreedyCompletionPrefillEligibility() throws {
        func step(_ profile: InklingSupport.Profile = .vocabularyQ4, _ cpu: String = "Apple M1 Max",
                  _ memory: UInt64 = 32 << 30, _ cap: Int = 26_624) -> Int? {
            InklingSupport.greedyCompletionStep(profile: profile, cpuBrand: cpu, memory: memory, cap: cap)
        }
        XCTAssertEqual(step(), 256)
        XCTAssertNil(step(.original))
        XCTAssertNil(step(.vocabularyQ4, "Apple M5"))
        XCTAssertNil(step(.vocabularyQ4, "Apple M1 Max", 16 << 30))
        XCTAssertNil(step(.vocabularyQ4, "Apple M1 Max", 32 << 30, 26_625))
        var fixture = try config()
        fixture.prefillStepSize = 128
        fixture.greedyCompletionPrefillStepSize = step()
        XCTAssertEqual(fixture.completionPrefillStep(temperature: 0), 256)
        XCTAssertEqual(fixture.completionPrefillStep(temperature: 0.1), 128)
    }

    func testCLISelectsInklingDefaultsWithoutInitializingMetal() throws {
        let fixture = try config()
        let directory = URL(fileURLWithPath: fixture.modelDirectory)
        func workingSet() -> Int? {
            XCTFail("Inkling profile resolution must run before Metal initialization")
            return nil
        }
        let args = ["--model-dir", fixture.modelDirectory]
        let c = try ServerConfig.parse(arguments: args, recommendedWorkingSetBytes: workingSet(), environment: [:])
        XCTAssertTrue(c.isInklingMini)
        XCTAssertEqual(c.contextCap, ProcessInfo.processInfo.physicalMemory >= 32 << 30 ? 26_624 : 8192)
        XCTAssertEqual(c.prefillStepSize, 128)
        XCTAssertEqual(c.maxTokensDefault, 4096)
        XCTAssertEqual(c.cacheLimitBytes, 536_870_912)
        XCTAssertTrue(c.cacheLimitExplicit)
        XCTAssertEqual(c.effectiveCacheMode, .off)
        XCTAssertEqual(c.temperature, 1)
        XCTAssertEqual(c.topP, 1)
        XCTAssertEqual(c.topK, 0)
        let cache = directory.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
        let namedDirectory = cache.appendingPathComponent("inkling-2-mini")
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: cache) }
        try FileManager.default.copyItem(at: directory, to: namedDirectory)
        let named = try ServerConfig.parse(
            arguments: ["--model", "inkling-2-mini"], recommendedWorkingSetBytes: workingSet(),
            environment: [:], modelCache: cache)
        XCTAssertEqual(named.modelDirectory, namedDirectory.path)
        XCTAssertEqual(named.servedModelID, "inkling-2-mini")
        XCTAssertTrue(named.isInklingMini)
        XCTAssertEqual(named.contextCap, c.contextCap)
        XCTAssertEqual(named.prefillStepSize, c.prefillStepSize)
        XCTAssertEqual(named.maxTokensDefault, c.maxTokensDefault)
        XCTAssertEqual(named.effectiveCacheMode, c.effectiveCacheMode)

        let limited = try ServerConfig.parse(
            arguments: args + ["--context-cap", "4096", "--temperature", "0", "--prefill-step-size", "64"],
            recommendedWorkingSetBytes: workingSet(), environment: [:])
        XCTAssertEqual(limited.contextCap, 4096)
        XCTAssertEqual(limited.temperature, 0)
        XCTAssertEqual(limited.prefillStepSize, 64)
        XCTAssertThrowsError(try ServerConfig.parse(
            arguments: args + ["--cache-mode", "disk"],
            recommendedWorkingSetBytes: workingSet(), environment: [:]))
    }

    func testVocabularyQuantSelectsMeasuredContextAndPermitsExplicitOverride() throws {
        let fixture = try config()
        let url = URL(fileURLWithPath: fixture.modelDirectory).appendingPathComponent("config.json")
        let q: [String: Any] = ["mode": "affine", "bits": 4, "group_size": 64]
        let header: [String: Any] = [
            "model_type": "inkling2_loop",
            "quantization": ["language_model.model.embed_tokens": q, "language_model.lm_head": q],
        ]
        try JSONSerialization.data(withJSONObject: header).write(to: url)
        let profile = try XCTUnwrap(InklingSupport.profile(directory: fixture.modelDirectory))
        XCTAssertEqual(profile, .vocabularyQ4)
        XCTAssertEqual(profile.defaultContext(physicalMemory: 32 << 30), 26_624)
        XCTAssertEqual(profile.defaultContext(physicalMemory: 24 << 30), 8192)
        XCTAssertEqual(InklingSupport.Profile.original.defaultContext(physicalMemory: 32 << 30), 26_624)
        XCTAssertEqual(InklingSupport.Profile.original.defaultContext(physicalMemory: 24 << 30), 8192)
        XCTAssertTrue(profile.usesBoundedLocalCache(contextCap: 8192))
        XCTAssertFalse(InklingSupport.Profile.original.usesBoundedLocalCache(contextCap: 8192))
        XCTAssertTrue(InklingSupport.Profile.original.usesBoundedLocalCache(contextCap: 32_768))
        let args = ["--model-dir", fixture.modelDirectory]
        let parsed = try ServerConfig.parse(arguments: args, recommendedWorkingSetBytes: nil)
        XCTAssertEqual(parsed.contextCap, profile.defaultContext(physicalMemory: ProcessInfo.processInfo.physicalMemory))
        let override = try ServerConfig.parse(arguments: args + ["--context-cap", "200000"],
                                               recommendedWorkingSetBytes: nil)
        let explicit = try ServerConfig.parse(arguments: args + ["--prefill-step-size", "64"],
                                              recommendedWorkingSetBytes: nil)
        XCTAssertNil(explicit.greedyCompletionPrefillStepSize)
        XCTAssertEqual(explicit.completionPrefillStep(temperature: 0), 64)
        XCTAssertNil(override.greedyCompletionPrefillStepSize)
        XCTAssertEqual(override.contextCap, 200_000)
        XCTAssertTrue(try InklingSupport.validateLoad(override))

        var tiered = header
        tiered["mei_affine_vocabulary_range"] = ["lower": 24_576, "upper": 199_936]
        try JSONSerialization.data(withJSONObject: tiered).write(to: url)
        XCTAssertEqual(InklingSupport.profile(directory: fixture.modelDirectory), .original)
        try Data(#"{"model_type":"inkling2_loop","quantization":false}"#.utf8).write(to: url)
        XCTAssertEqual(InklingSupport.profile(directory: fixture.modelDirectory), .original)
    }

    func testAdmissionAndUnsupportedModes() throws {
        let base = try config()
        XCTAssertTrue(try InklingSupport.validateLoad(base))
        for index in 0..<8 {
            var c = base
            switch index {
            case 0: c.enableInklingMini = false
            case 1: c.contextCap = InklingSupport.maximumContext + 1
            case 2: c.kvBits = 8
            case 3: c.enableCompiledDecode = true
            case 4: c.maxActiveGenerations = 2
            case 5: c.cacheMode = .disk
            case 6: c.cacheMode = .memory
            default: c.maxKVWindowSize = 512
            }
            XCTAssertThrowsError(try InklingSupport.validateLoad(c))
        }
    }
    func testCapacityAndHTTPRejection() throws {
        var c = try config(); c.isInklingMini = true
        XCTAssertNoThrow(try InklingSupport.validateCapacity(promptTokens: 4095, config: c))
        XCTAssertThrowsError(try InklingSupport.validateCapacity(promptTokens: 4096, config: c))
        let text = try ChatRequest(json: Data(#"{"model":"test","messages":[{"role":"user","content":"hello"}]}"#.utf8))
        XCTAssertNoThrow(try InklingSupport.validateInput(text, enabled: true))
        let format = try ChatRequest(json: Data(#"{"model":"test","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_object"}}"#.utf8))
        XCTAssertThrowsError(try InklingSupport.validateInput(format, enabled: true))
    }
    func testJSON5CannotBypassExperimentalAdmission() throws {
        var c = try config()
        let path = URL(fileURLWithPath: c.modelDirectory).appendingPathComponent("config.json")
        try Data("// authored JSON5\n{model_type:'inkling2_loop'}".utf8).write(to: path)
        XCTAssertTrue(InklingSupport.isModel(directory: c.modelDirectory))
        c.enableInklingMini = false
        XCTAssertThrowsError(try InklingSupport.validateLoad(c))
    }

    func testEightKAdmissionKeepsOneOutputTokenReserved() throws {
        var c = try config()
        c.contextCap = 8192; c.isInklingMini = true
        XCTAssertTrue(try InklingSupport.validateLoad(c))
        XCTAssertNoThrow(try InklingSupport.validateCapacity(promptTokens: 4179, config: c))
        XCTAssertNoThrow(try InklingSupport.validateCapacity(promptTokens: 8191, config: c))
        XCTAssertThrowsError(try InklingSupport.validateCapacity(promptTokens: 8192, config: c))
        XCTAssertThrowsError(try InklingSupport.validateCapacity(promptTokens: 8193, config: c))
        let larger = try ServerConfig.parse(
            arguments: ["--model-dir", c.modelDirectory, "--context-cap", "200000"],
            recommendedWorkingSetBytes: nil, environment: [:])
        XCTAssertEqual(larger.contextCap, 200_000)
        XCTAssertTrue(try InklingSupport.validateLoad(larger))
        XCTAssertNoThrow(try InklingSupport.validateCapacity(promptTokens: 199_999, config: larger))
        XCTAssertThrowsError(try InklingSupport.validateCapacity(promptTokens: 200_000, config: larger))
    }

}
