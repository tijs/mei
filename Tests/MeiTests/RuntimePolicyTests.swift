import Foundation
import XCTest
import MLX
import MLXLMCommon
@testable import MeiCore

final class RuntimePolicyTests: XCTestCase {
    private func parse(_ flags: [String] = [], env: [String: String] = [:], directory: String = "/nonexistent/model") throws -> ServerConfig {
        try ServerConfig.parse(arguments: ["--model-dir", directory, "--served-model-id", "org/model"] + flags,
                               recommendedWorkingSetBytes: nil, environment: env)
    }

    func testMLXPressDefaultAndPrecedenceAreExplicit() throws {
        XCTAssertEqual(try parse().modelLoadConfiguration.jangPress, .disabled)
        let old = try parse(env: ["JANGPRESS": "70"])
        XCTAssertEqual(old.modelLoadConfiguration.jangPress, .enabled(coldFraction: 0.70))
        XCTAssertEqual(old.mlxPressPolicySource, "environment:JANGPRESS")
        XCTAssertEqual(try parse(env: ["MLXPRESS": "off", "JANGPRESS": "70"]).mlxPressPolicy, .off)
        let explicit = try parse(["--mlxpress", "auto"], env: ["MLXPRESS": "garbage"])
        XCTAssertEqual(explicit.modelLoadConfiguration.jangPress, .auto(envFallback: false))
        XCTAssertEqual(explicit.mlxPressPolicySource, "cli")
        XCTAssertEqual(try parse(["--mlxpress", "off"], env: ["MLXPRESS": "70"]).mlxPressPolicy, .off)
        for raw in ["garbage", "-1", "96", "1.5", ""] {
            XCTAssertThrowsError(try parse(env: ["MLXPRESS": raw]))
            XCTAssertThrowsError(try parse(["--mlxpress", raw]))
        }
    }

    func testExplicitZeroSurvivesLoadPolicyAndAllocatorApplication() throws {
        let config = try parse(["--cache-limit-bytes", "0"])
        XCTAssertEqual(config.modelLoadConfiguration.maxResidentBytes, .absolute(0))
        XCTAssertEqual(config.runtimePolicies().allocatorCacheLimitRequestedBytes, 0)
        XCTAssertNil(try parse().runtimePolicies().allocatorCacheLimitRequestedBytes)
        let prior = Memory.cacheLimit
        defer { Memory.cacheLimit = prior }
        Memory.cacheLimit = 1048576
        Engine.applyExplicitMemoryLimits(config)
        XCTAssertEqual(Memory.cacheLimit, 0)
        let bounded = try parse(["--memory-limit-bytes", "123456", "--cache-limit-bytes", "45678"])
        XCTAssertEqual(bounded.modelLoadConfiguration.memoryLimit, .absolute(123456))
        XCTAssertEqual(bounded.modelLoadConfiguration.maxResidentBytes, .absolute(45678))
        for flag in ["--memory-limit-bytes", "--cache-limit-bytes"] {
            XCTAssertThrowsError(try parse([flag, "-1"]))
        }
    }

    func testModesAndLegacyConflictsAreOrderIndependent() throws {
        XCTAssertEqual(try parse().effectiveCacheMode, .memory)
        XCTAssertEqual(try parse(["--cache-mode", "off"]).effectiveCacheMode, .off)
        XCTAssertEqual(try parse(["--cache-mode", "memory"]).effectiveCacheMode, .memory)
        XCTAssertEqual(try parse(["--cache-reuse", "false"]).effectiveCacheMode, .off)
        XCTAssertEqual(try parse(["--kv-cache-dir", "/chosen"]).effectiveCacheMode, .disk)
        for flags in [
            ["--cache-mode", "disk", "--cache-reuse", "false"],
            ["--cache-reuse", "false", "--cache-mode", "disk"],
            ["--cache-mode", "off", "--cache-reuse", "true"],
            ["--cache-mode", "memory", "--kv-cache-dir", "/chosen"],
            ["--cache-mode", "off", "--kv-cache-dir", "/chosen"],
            ["--cache-mode", "disk", "--kv-cache-dir", ""],
            ["--cache-mode", "invalid"]
        ] { XCTAssertThrowsError(try parse(flags), "\(flags)") }
    }

    func testDiskDefaultsAreIsolatedStableAndDoNotCreateDirectories() throws {
        var first = try parse(["--cache-mode", "disk"])
        let second = try parse(["--cache-mode", "disk"])
        let directory = first.kvCacheDir
        XCTAssertNotEqual(directory, second.kvCacheDir)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory))
        try first.resolveRuntimePolicies()
        XCTAssertEqual(directory, first.kvCacheDir)
        XCTAssertTrue(first.disposableKVCache)
        let explicit = try parse(["--cache-mode", "disk", "--kv-cache-dir", "/chosen"])
        XCTAssertEqual(explicit.kvCacheDir, "/chosen")
        XCTAssertFalse(explicit.disposableKVCache)
    }

    func testRequiredDiskPolicyAlsoAppliesToLibraryCallers() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_type":"qwen3_5_moe"}"#.utf8).write(to: dir.appendingPathComponent("config.json"))
        var direct = ServerConfig(modelDirectory: dir.path, servedModelID: "org/model")
        try direct.resolveRuntimePolicies()
        XCTAssertEqual(direct.effectiveCacheMode, .disk)
        XCTAssertEqual(direct.cacheDecision, "model requires disk-backed reuse")
        XCTAssertThrowsError(try parse(["--cache-mode", "memory"], directory: dir.path))
        XCTAssertThrowsError(try parse(["--kv-cache-dir", ""], directory: dir.path))
        XCTAssertEqual(try parse(["--cache-mode", "off"], directory: dir.path).effectiveCacheMode, .off)
        XCTAssertEqual(try parse(directory: dir.path).effectiveCacheMode, direct.effectiveCacheMode)
    }

    func testPolicyStatusSeparatesRequestsAndActualActivation() throws {
        let config = try parse(["--cache-mode", "disk", "--mlxpress", "auto", "--cache-limit-bytes", "0"])
        let status = config.runtimePolicies(mlxPressEnabled: false)
        let data = try JSONEncoder().encode(status)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["cacheModeRequested"] as? String, "disk")
        XCTAssertEqual(json["cacheModeEffective"] as? String, "disk")
        XCTAssertEqual(json["mlxPressRequested"] as? String, "auto")
        XCTAssertEqual(json["mlxPressEnabled"] as? Bool, false)
        XCTAssertEqual(json["allocatorCacheLimitRequestedBytes"] as? Int, 0)
        XCTAssertEqual(json["kvCacheDirectory"] as? String, config.kvCacheDir)
    }
}
