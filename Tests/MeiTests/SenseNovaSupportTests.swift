import Foundation
import XCTest
@testable import MeiCore

final class SenseNovaSupportTests: XCTestCase {
    private func bundle(_ changes: [String: Any] = [:]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        var root: [String: Any] = [
            "model_type": "sensenova_loop", "source_revision": SenseNovaSupport.releaseRevision,
            "mei_text_only_export": true, "activation_dtype": "float32",
            "quantization": ["bits": 8, "group_size": 64],
            "text_config": ["model_type": "qwen3_5_text"],
        ]
        root.merge(changes) { _, new in new }
        try JSONSerialization.data(withJSONObject: root).write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    private func parse(_ directory: URL, _ flags: [String] = []) throws -> ServerConfig {
        try ServerConfig.parse(arguments: ["--model-dir", directory.path,
            "--enable-sensenova-loop", "true"] + flags,
            recommendedWorkingSetBytes: nil, environment: [:])
    }

    func testExperimentalProfileHasBoundedDefaultsAndExplicitOptIn() throws {
        let directory = try bundle()
        XCTAssertThrowsError(try ServerConfig.parse(arguments: ["--model-dir", directory.path],
            recommendedWorkingSetBytes: nil, environment: [:]))
        let config = try parse(directory)
        XCTAssertEqual(config.optimizationProfile, .senseNovaLoop)
        XCTAssertTrue(config.isSenseNovaLoop)
        XCTAssertEqual(config.contextCap, 4096)
        XCTAssertEqual(config.maxKVSize, 4096)
        XCTAssertEqual(config.maxTokensDefault, 256)
        XCTAssertEqual(config.prefillStepSize, 64)
        XCTAssertNil(config.modelTuning)
        XCTAssertEqual(config.effectiveCacheMode, .disk)
        XCTAssertTrue(config.disposableKVCache)
    }

    func testUnsupportedPrecisionRevisionAndResourceOverridesFailBeforeLoad() throws {
        for change: [String: Any] in [
            ["activation_dtype": "checkpoint"], ["source_revision": "preview"],
            ["mei_text_only_export": false], ["quantization": ["bits": 4, "group_size": 64]],
        ] {
            XCTAssertThrowsError(try parse(bundle(change)))
        }
        let directory = try bundle()
        for flags in [["--context-cap", "8192"], ["--prefill-step-size", "128"],
            ["--max-active-generations", "2"], ["--compiled-decode", "true"],
            ["--max-kv-window", "512"], ["--kv-bits", "4"], ["--kv-bits", "8"]] {
            XCTAssertThrowsError(try parse(directory, flags))
        }
        let smaller = try parse(directory, ["--context-cap", "2048", "--cache-reuse", "false"])
        XCTAssertEqual(smaller.maxKVSize, 2048)
        XCTAssertNoThrow(try SenseNovaSupport.validateCapacity(promptTokens: 2047, config: smaller))
        XCTAssertThrowsError(try SenseNovaSupport.validateCapacity(promptTokens: 2048, config: smaller))
    }

    func testJSON5ExportsUseTheSameAdmissionAndCachePolicy() throws {
        let directory = try bundle()
        let url = directory.appendingPathComponent("config.json")
        let json = try String(contentsOf: url, encoding: .utf8)
        try ("// A JSON5 export\n" + json).write(to: url, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(try ServerConfig.parse(arguments: ["--model-dir", directory.path],
            recommendedWorkingSetBytes: nil, environment: [:]))
        let config = try parse(directory)
        XCTAssertTrue(config.isSenseNovaLoop)
        XCTAssertEqual(config.maxKVSize, 4096)
        XCTAssertEqual(config.effectiveCacheMode, .disk)
        XCTAssertThrowsError(try parse(directory, ["--cache-mode", "memory"]))
    }

    func testOrdinaryQwenBundleKeepsItsDefaults() throws {
        let directory = try bundle(["model_type": "qwen3_5_text"])
        let config = try ServerConfig.parse(arguments: ["--model-dir", directory.path],
            recommendedWorkingSetBytes: nil, environment: [:])
        let defaults = ServerConfig(modelDirectory: directory.path, servedModelID: "test")
        XCTAssertFalse(config.isSenseNovaLoop)
        XCTAssertEqual(config.optimizationProfile, .generic)
        XCTAssertEqual(config.contextCap, defaults.contextCap)
        XCTAssertEqual(config.maxKVSize, defaults.maxKVSize)
        XCTAssertEqual(config.maxTokensDefault, defaults.maxTokensDefault)
        XCTAssertThrowsError(try parse(directory))
    }

    func testMediaAndUnknownContentAreRejectedBeforeGeneration() throws {
        for content: Any in [
            [["type": "image_url", "image_url": ["url": "https://invalid.example/image"]]],
            [["type": "video", "video": "synthetic"]],
            [["type": "audio", "text": "do not silently retain this"]],
            [["type": 1, "text": "invalid type"]],
            ["type": "image_url"],
        ] {
            let json = try JSONSerialization.data(withJSONObject: ["model": "test",
                "messages": [["role": "user", "content": content]]])
            let request = try ChatRequest(json: json)
            XCTAssertTrue(request.containsUnsupportedContent)
            XCTAssertThrowsError(try SenseNovaSupport.validateInput(request, textOnly: true))
            XCTAssertNoThrow(try SenseNovaSupport.validateInput(request, textOnly: false))
        }
        for content: Any in ["Hello", [["type": "text", "text": "Hello"]], [["text": "Hello"]]] {
            let json = try JSONSerialization.data(withJSONObject: ["model": "test",
                "messages": [["role": "user", "content": content]]])
            let request = try ChatRequest(json: json)
            XCTAssertFalse(request.containsUnsupportedContent)
            XCTAssertEqual(request.messages.first?.content, "Hello")
            XCTAssertNoThrow(try SenseNovaSupport.validateInput(request, textOnly: true))
        }
    }
}
