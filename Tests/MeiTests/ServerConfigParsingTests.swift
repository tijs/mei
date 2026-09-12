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

    func testEveryProfilePinsARevisionAndNamesItsRepo() {
        for tuning in ModelTuningRegistry.all {
            XCTAssertFalse(tuning.repo.isEmpty, "\(tuning.name) has no repo")
            // 40-char commit, never a branch: settings are calibrated to one
            // artifact, and an upstream re-export must not silently apply.
            XCTAssertEqual(tuning.revision.count, 40,
                           "\(tuning.name) revision must be a full commit sha")
            XCTAssertNotEqual(tuning.revision, "main")
        }
    }

    func testModelProfileSelectsTheMeasuredSettingsForThatModel() throws {
        let ornith = try parse(["--model-profile", "ornith-1.5-35b-a3b"])
        XCTAssertEqual(ornith.optimizationProfile, .ornith)
        // Anchors OFF on Ornith: they cost hermes_ops-multi-step-chain 0/3.
        XCTAssertEqual(ornith.ssmAnchorBoundaryCount, 0)
        XCTAssertEqual(ornith.maxTokensDefault, 8192)

        let text = try parse(["--model-profile", "qwen3.6-35b-a3b-text"])
        XCTAssertEqual(text.optimizationProfile, .ornith)
        // Anchors ON here: -52% prefill, zero task cost across three pairs.
        XCTAssertEqual(text.ssmAnchorBoundaryCount, 2)
        XCTAssertEqual(text.prefillStepSize, 1024)
    }

    func testModelProfileIsCaseInsensitiveAndNamesAreStable() throws {
        XCTAssertEqual(try parse(["--model-profile", "Ornith-1.5-35B-A3B"])
                        .modelTuning?.name, "ornith-1.5-35b-a3b")
        // These names are a documented lookup key in the README; renaming one
        // silently breaks every operator command line that uses it.
        XCTAssertEqual(ModelTuningRegistry.names,
                       ["ornith-1.5-35b-a3b", "qwen3.6-35b-a3b-text",
                        "qwen3.6-35b-a3b"])
    }

    func testUnknownModelProfileIsRejectedAndListsTheKnownOnes() {
        XCTAssertThrowsError(try parse(["--model-profile", "llama-3"])) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains("ornith-1.5-35b-a3b"),
                          "the error must list what IS supported, got: \(text)")
        }
    }

    func testExplicitFlagsStillBeatTheProfile() throws {
        // Naming a model supplies measured defaults; it must not override an
        // operator who asked for something specific.
        let c = try parse(["--model-profile", "qwen3.6-35b-a3b-text",
                           "--ssm-anchor-boundaries", "0",
                           "--prefill-step-size", "256"])
        XCTAssertEqual(c.ssmAnchorBoundaryCount, 0)
        XCTAssertEqual(c.prefillStepSize, 256)
    }

    func testNoProfileLeavesPerModelTuningUnapplied() throws {
        let c = try parse([])
        XCTAssertNil(c.modelTuning)
        XCTAssertEqual(c.ssmAnchorBoundaryCount, 0)
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
        try Data("{\"model_type\":\"llama\"}".utf8)
            .write(to: unknown.appendingPathComponent("config.json"))
        XCTAssertEqual(ModelOptimizationProfile.detect(modelDirectory: unknown.path), .generic)
    }

    func testNamedProfileOverridesArchitectureDetection() throws {
        // The base model dir carries generic metadata, so detection alone would
        // say .generic. Naming a model must win — that is the point of selecting
        // by name rather than inferring.
        let ornith = try parse(["--model-profile", "ornith-1.5-35b-a3b"])
        XCTAssertEqual(ornith.requestedOptimizationProfile, .ornith)
        XCTAssertEqual(ornith.optimizationProfile, .ornith)
        XCTAssertEqual(ornith.prefillStepSize, 512)

        // An explicit flag still beats the profile's own prefill.
        let pinned = try parse([
            "--model-profile", "ornith-1.5-35b-a3b", "--prefill-step-size", "128"
        ])
        XCTAssertEqual(pinned.prefillStepSize, 128)
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



    func testDenseQwen35ProfileStaysAt64Prefill() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-profile-qwen35-step-\\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"model_type":"qwen3_5"}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", directory.path, "--served-model-id", "mlx/qwen35"
        ])
        XCTAssertEqual(config.optimizationProfile, .generic)
        XCTAssertEqual(config.prefillStepSize, 64)
    }

    func testUnknownModelProfileRejected() {
        XCTAssertThrowsError(try parse(["--model-profile", "qwen"])) { error in
            guard case ConfigError.invalidValue(let message) = error else {
                return XCTFail("expected invalidValue, got \(error)")
            }
            XCTAssertTrue(message.contains("model-profile"))
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

    // MARK: - Disk-KV safety default (0.1.0 release fix)
    // Dense qwen3_5/qwen3_8-style checkpoints crash the in-memory-only paged
    // KV tier (vmlx array.cpp:335 SmallVector crash; trigger isolated
    // 2026-09-02 by the 2x2 evidence: cells A/C (in-memory) crash, cells
    // prefixes on it (cached=0; disk tier restores 6173/6174, 2026-09-03
    // evidence) — so cache-reuse on without an explicit --kv-cache-dir must
    // default both families to a disposable on-disk cache.
    //
    // Fan-out: qwen3_5 / qwen3_5_text are the dense Qwen3.5/Qwen3.8 MLX
    // model_type values (verified on Qwen3.8-27B-4bit and the Heretic
    // (root + nested text_config of mlx-community/some-unknown-model).
    // The MoE/hybrid qwen3_5_moe family keeps operator-controlled cache
    // configuration (Ornith behavior preserved).

    private func makeModelDir(modelType: String) throws -> URL {
        try makeModelDir(config: ["model_type": modelType])
    }

    /// Variant for nested metadata (model_type under text_config/...).
    private func makeModelDir(config: [String: Any]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-profile-kv-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    private func assertDisposableDiskKVDefault(
        modelDir: URL, servedModelID: String, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", modelDir.path,
            "--served-model-id", servedModelID,
        ])
        XCTAssertEqual(config.optimizationProfile, .generic, file: file, line: line)
        XCTAssertTrue(config.cacheReuse, file: file, line: line)
        XCTAssertFalse(config.kvCacheDirExplicit, file: file, line: line)
        XCTAssertEqual(
            config.kvCacheDir,
            ServerConfig.defaultDisposableKVCacheDir(servedModelID: servedModelID),
            file: file, line: line)
        XCTAssertTrue(config.kvCacheDir.contains(servedModelID.replacingOccurrences(of: "/", with: "-")), file: file, line: line)
    }

    func testDenseQwen35DefaultsDisposableDiskKVWhenReuseOnAndNoExplicitDir() throws {
        let dir = try makeModelDir(modelType: "qwen3_5")
        defer { try? FileManager.default.removeItem(at: dir) }
        try assertDisposableDiskKVDefault(
            modelDir: dir, servedModelID: "mlx-community/Qwen3.8-27B-4bit")
    }

    func testDenseQwen35GetsDiskDefaultWithoutAnyNamedProfile() throws {
        // Architecture detection still protects a model nobody named: this
        // topology cannot restore from the paged tier, so it needs a disk KV
        // dir whether or not the operator knew to ask.
        let dir = try makeModelDir(modelType: "qwen3_5_text")
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", dir.path,
            "--served-model-id", "mlx-community/Qwen3.8-27B-4bit",
        ])
        XCTAssertNil(config.modelTuning, "no profile was named")
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

    func testOrnithMoeModelGetsDisposableDiskKVDefault() throws {
        let dir = try makeModelDir(modelType: "qwen3_5_moe")
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", dir.path,
            "--served-model-id", "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        ])
        XCTAssertEqual(config.optimizationProfile, .ornith)
        XCTAssertEqual(config.prefillStepSize, 512)
        // 0.4.2 (f1ba4af) deliberately reversed the old "no implicit disk-KV
        // default" contract for this family: the qwen3_5_moe hybrid cannot
        // restore from the paged in-memory tier, so without a disk tier every
        // turn of a bare `mei --model-dir X` cold-prefills. Measured bare over
        // four turns: 246 s -> 66 s. The default must therefore be a real
        // disposable directory, not "".
        XCTAssertFalse(config.kvCacheDir.isEmpty,
                       "qwen3_5_moe needs a disk KV tier for ordinary reuse")
        XCTAssertTrue(config.kvCacheDir.contains("mei-kv-cache"),
                      "expected a disposable cache dir, got \(config.kvCacheDir)")
    }

    func testUnrelatedGenericModelKeepsKVCacheDirEmptyDefault() throws {
        let dir = try makeModelDir(modelType: "llama")
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try ServerConfig.parse(arguments: [
            "--model-dir", dir.path,
            "--served-model-id", "mlx-community/Llama-3.3-70B-4bit",
        ])
        XCTAssertEqual(config.optimizationProfile, .generic)
        XCTAssertEqual(config.kvCacheDir, "")
    }

    func testMissingModelDirNeverDefaulted() throws {
        let config = try parse([])
        XCTAssertEqual(config.modelDirectory, "/tmp/model")
        XCTAssertFalse(ModelOptimizationProfile.needsDiskKVTier(modelDirectory: "/tmp/model"))
        XCTAssertEqual(config.kvCacheDir, "")
    }
}
