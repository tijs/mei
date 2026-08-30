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
        XCTAssertTrue(config.enableSSMReDerive)
        XCTAssertEqual(config.prefillStepSize, 512)
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
}