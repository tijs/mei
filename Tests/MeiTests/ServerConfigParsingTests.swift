import Foundation
import XCTest
@testable import MeiCore

/// Non-Metal unit tests for the Mei CLI surface — including the fork
/// patches' flags (0001 kv-bits / 0002 nothing new / 0003
/// compiled-decode-threshold / 0004 max-kv-window) — so flag plumbing is
/// verified by `swift test` without touching the GPU (the full matrix's
/// Metal-touching cells stay gated on an uncontended window).
final class ServerConfigParsingTests: XCTestCase {
    let base = ["--model-dir", "/tmp/model", "--served-model-id", "mei/model"]

    private func parse(_ extra: [String]) throws -> ServerConfig {
        try ServerConfig.parse(arguments: base + extra)
    }

    func testForkFlagDefaultsMatchRollbackConfiguration() throws {
        let config = try parse([])
        XCTAssertNil(config.kvBits)
        XCTAssertEqual(config.kvGroupSize, 64)
        XCTAssertEqual(config.quantizedKVStart, 0)
        XCTAssertFalse(config.enableCompiledDecode)
        XCTAssertNil(config.compiledDecodeMaxPromptOffset)
        XCTAssertEqual(config.maxKVWindowSize, 0)
        XCTAssertEqual(config.ssmAnchorBoundaryCount, 0)
        XCTAssertTrue(config.enableSSMReDerive)
        XCTAssertEqual(config.prefillStepSize, 64)
        XCTAssertEqual(config.requestedOptimizationProfile, .auto)
        XCTAssertEqual(config.optimizationProfile, .generic)
        XCTAssertEqual(config.contextCap, 65_536)
        XCTAssertTrue(config.useMmapSafetensors)
    }

    func testKVBitsParses4And8() throws {
        XCTAssertEqual(try parse(["--kv-bits", "4"]).kvBits, 4)
        XCTAssertEqual(try parse(["--kv-bits", "8"]).kvBits, 8)
        XCTAssertEqual(try parse(["--kv-bits", "8", "--kv-group-size", "128"]).kvGroupSize, 128)
        XCTAssertEqual(try parse(["--kv-bits", "8", "--quantized-kv-start", "4"]).quantizedKVStart, 4)
    }

    func testCompiledDecodeThresholdParses() throws {
        let config = try parse(["--compiled-decode", "true", "--compiled-decode-threshold", "16384"])
        XCTAssertTrue(config.enableCompiledDecode)
        XCTAssertEqual(config.compiledDecodeMaxPromptOffset, 16_384)
        // 0 = never compile (explicit opt-out).
        let never = try parse(["--compiled-decode", "true", "--compiled-decode-threshold", "0"])
        XCTAssertEqual(never.compiledDecodeMaxPromptOffset, 0)
    }

    func testMaxKVWindowParses() throws {
        XCTAssertEqual(try parse(["--max-kv-window", "16384"]).maxKVWindowSize, 16_384)
        XCTAssertEqual(try parse(["--max-kv-window", "0"]).maxKVWindowSize, 0)
    }

    func testSSMAnchorBoundariesParses() throws {
        XCTAssertEqual(try parse(["--ssm-anchor-boundaries", "8"]).ssmAnchorBoundaryCount, 8)
        XCTAssertEqual(try parse(["--ssm-anchor-boundaries", "0"]).ssmAnchorBoundaryCount, 0)
        XCTAssertThrowsError(try parse(["--ssm-anchor-boundaries", "abc"])) { error in
            XCTAssertTrue(error is ConfigError)
        }
    }

    func testSSMReDeriveAndCacheFlags() throws {
        XCTAssertFalse(try parse(["--ssm-rederive", "false"]).enableSSMReDerive)
        let config = try parse(["--ssm-rederive", "true", "--kv-cache-dir", "/tmp/kv"])
        XCTAssertTrue(config.enableSSMReDerive)
        XCTAssertEqual(config.kvCacheDir, "/tmp/kv")
        XCTAssertTrue(try parse(["--cache-reuse", "false"]).cacheReuse == false)
    }

    func testMemoryLimitsParse() throws {
        let config = try parse(["--memory-limit-bytes", "26843545600", "--cache-limit-bytes", "8589934592"])
        XCTAssertEqual(config.memoryLimitBytes, 26_843_545_600)
        XCTAssertEqual(config.cacheLimitBytes, 8_589_934_592)
    }

    func testPrefillStepAndContextValidation() throws {
        XCTAssertEqual(try parse(["--prefill-step-size", "2048"]).prefillStepSize, 2048)
        XCTAssertThrowsError(try parse(["--prefill-step-size", "0"])) { error in
            XCTAssertTrue(error is ConfigError)
        }
        XCTAssertThrowsError(try parse(["--context-cap", "0"])) { error in
            XCTAssertTrue(error is ConfigError)
        }
    }

    func testAutoDetectsNestedOrnithModelAndUses512Prefill() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-profile-ornith-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let metadata = "{\"model_type\":\"qwen3_5_moe\",\"text_config\":{\"model_type\":\"qwen3_5_moe_text\"}}"
        try Data(metadata.utf8).write(to: directory.appendingPathComponent("config.json"))

        let config = try ServerConfig.parse(arguments: [
            "--model-dir", directory.path, "--served-model-id", "ornith/test"
        ])
        XCTAssertEqual(config.optimizationProfile, .ornith)
        XCTAssertEqual(config.prefillStepSize, 512)
    }

    func testMalformedOrUnknownMetadataFallsBackToGeneric() throws {
        let malformed = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-profile-malformed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: malformed, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: malformed) }
        try Data("not-json".utf8).write(to: malformed.appendingPathComponent("config.json"))
        XCTAssertEqual(ModelOptimizationProfile.detect(modelDirectory: malformed.path), .generic)

        let unknown = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-profile-unknown-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: unknown, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: unknown) }
        try Data("{\"model_type\":\"gemma4\"}".utf8)
            .write(to: unknown.appendingPathComponent("config.json"))
        XCTAssertEqual(ModelOptimizationProfile.detect(modelDirectory: unknown.path), .generic)
    }

    func testExplicitProfilesOverrideAutoDetection() throws {
        let ornith = try parse(["--optimization-profile", "ornith"])
        XCTAssertEqual(ornith.requestedOptimizationProfile, .ornith)
        XCTAssertEqual(ornith.optimizationProfile, .ornith)
        XCTAssertEqual(ornith.prefillStepSize, 512)

        let generic = try parse([
            "--optimization-profile", "generic", "--prefill-step-size", "128"
        ])
        XCTAssertEqual(generic.requestedOptimizationProfile, .generic)
        XCTAssertEqual(generic.optimizationProfile, .generic)
        XCTAssertEqual(generic.prefillStepSize, 128)
    }

    func testExplicitPrefillWinsOverOrnithProfile() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-profile-explicit-step-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{\"model_type\":\"qwen3_5_moe\"}".utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", directory.path, "--served-model-id", "ornith/test",
            "--prefill-step-size", "256"
        ])
        XCTAssertEqual(config.optimizationProfile, .ornith)
        XCTAssertEqual(config.prefillStepSize, 256)
    }

    func testUnknownOptimizationProfileRejected() {
        XCTAssertThrowsError(try parse(["--optimization-profile", "qwen"])) { error in
            guard case ConfigError.invalidValue(let message) = error else {
                return XCTFail("expected invalidValue, got \(error)")
            }
            XCTAssertTrue(message.contains("optimization-profile"))
        }
    }

    func testUnknownOptionRejected() {
        XCTAssertThrowsError(try parse(["--not-a-real-flag", "1"])) { error in
            guard case ConfigError.invalidValue(let message) = error else {
                return XCTFail("expected invalidValue, got \(error)")
            }
            XCTAssertTrue(message.contains("not-a-real-flag"))
        }
    }

    func testMissingRequiredFlagsRejected() {
        XCTAssertThrowsError(try ServerConfig.parse(arguments: ["--port", "8024"])) { error in
            guard case ConfigError.missingRequired(let field) = error else {
                return XCTFail("expected missingRequired, got \(error)")
            }
            XCTAssertEqual(field, "--model-dir")
        }
    }

    func testInvalidIntegerRejected() {
        XCTAssertThrowsError(try parse(["--kv-bits", "many"])) { error in
            guard case ConfigError.invalidValue(let message) = error else {
                return XCTFail("expected invalidValue, got \(error)")
            }
            XCTAssertTrue(message.contains("kv-bits"))
        }
    }

    // MARK: - Dense Qwen3.5/Qwen3.8 disk-KV safety default (0.1.0 release
    // fix for the in-memory-only paged KV SmallVector crash; trigger
    // isolated 2026-09-02 by the 2x2 evidence: cells A/C (in-memory) crash
    // at vmlx array.cpp:335, cells B/D (disk) pass — the disk tier is the
    // load-bearing element for dense qwen3_5, so cache-reuse on without an
    // explicit --kv-cache-dir must default to a disposable on-disk cache).
    //
    // Fan-out: qwen3_5 / qwen3_5_text are the dense Qwen3.5/Qwen3.8 MLX
    // model_type values (verified on Qwen3.8-27B-4bit and the Heretic
    // variant); the MoE/hybrid qwen3_5_moe family keeps operator-controlled
    // cache configuration (Ornith behavior preserved).

    private func makeModelDir(modelType: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-profile-kv-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["model_type": modelType])
            .write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    func testDenseQwen35DefaultsDisposableDiskKVWhenReuseOnAndNoExplicitDir() throws {
        let dir = try makeModelDir(modelType: "qwen3_5")
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", dir.path,
            "--served-model-id", "mlx-community/Qwen3.8-27B-4bit",
        ])
        XCTAssertEqual(config.optimizationProfile, .generic)
        XCTAssertTrue(config.cacheReuse)
        XCTAssertFalse(config.kvCacheDirExplicit)
        XCTAssertEqual(
            config.kvCacheDir,
            ServerConfig.defaultDisposableKVCacheDir(servedModelID: "mlx-community/Qwen3.8-27B-4bit"))
        XCTAssertTrue(config.kvCacheDir.contains("Qwen3.8-27B-4bit"))
    }

    func testExplicitGenericProfileWithDenseQwen35StillGetsDiskDefault() throws {
        let dir = try makeModelDir(modelType: "qwen3_5_text")
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", dir.path,
            "--served-model-id", "mlx-community/Qwen3.8-27B-4bit",
            "--optimization-profile", "generic",
        ])
        XCTAssertEqual(config.optimizationProfile, .generic)
        XCTAssertEqual(
            config.kvCacheDir,
            ServerConfig.defaultDisposableKVCacheDir(servedModelID: "mlx-community/Qwen3.8-27B-4bit"))
    }

    func testExplicitKVCacheDirWinsOverDenseQwen35Default() throws {
        let dir = try makeModelDir(modelType: "qwen3_5")
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", dir.path,
            "--served-model-id", "mlx-community/Qwen3.8-27B-4bit",
            "--kv-cache-dir", "/custom/kv",
        ])
        XCTAssertEqual(config.optimizationProfile, .generic)
        XCTAssertTrue(config.kvCacheDirExplicit)
        XCTAssertEqual(config.kvCacheDir, "/custom/kv")
    }

    func testCacheReuseFalseKeepsKVCacheDirEmptyForDenseQwen35() throws {
        let dir = try makeModelDir(modelType: "qwen3_5")
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", dir.path,
            "--served-model-id", "mlx-community/Qwen3.8-27B-4bit",
            "--cache-reuse", "false",
        ])
        XCTAssertFalse(config.cacheReuse)
        XCTAssertEqual(config.kvCacheDir, "")
        XCTAssertFalse(config.kvCacheDirExplicit)
    }

    func testOrnithMoeModelKeepsKVCacheDirEmptyDefault() throws {
        let dir = try makeModelDir(modelType: "qwen3_5_moe")
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", dir.path,
            "--served-model-id", "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        ])
        XCTAssertEqual(config.optimizationProfile, .ornith)
        XCTAssertEqual(config.prefillStepSize, 512)
        // Ornith behavior preserved: no implicit disk-KV default.
        XCTAssertEqual(config.kvCacheDir, "")
    }

    func testNonQwenGenericModelKeepsKVCacheDirEmptyDefault() throws {
        let dir = try makeModelDir(modelType: "gemma4")
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", dir.path,
            "--served-model-id", "mlx-community/gemma-4-26b-a4b-it-4bit",
        ])
        XCTAssertEqual(config.optimizationProfile, .generic)
        XCTAssertEqual(config.kvCacheDir, "")
    }

    func testMissingModelDirNeverDefaulted() throws {
        let config = try parse([])
        XCTAssertEqual(config.modelDirectory, "/tmp/model")
        XCTAssertFalse(ModelOptimizationProfile.denseQwen35NeedsDiskKVTier(modelDirectory: "/tmp/model"))
        XCTAssertEqual(config.kvCacheDir, "")
    }
}