import Foundation
import XCTest
@testable import MeiCore

final class InklingSupportTests: XCTestCase {
    private func config() throws -> ServerConfig {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model_type":"inkling2_loop"}"#.utf8).write(to: root.appendingPathComponent("config.json"))
        var config = ServerConfig(modelDirectory: root.path, servedModelID: "test")
        config.enableInklingMini = true; config.contextCap = 4096; config.cacheMode = .off
        return config
    }
    func testCLISelectsInklingDefaultsWithoutInitializingMetal() throws {
        let fixture = try config()
        let directory = URL(fileURLWithPath: fixture.modelDirectory)
        try Data(#"{"temperature":1.0}"#.utf8)
            .write(to: directory.appendingPathComponent("generation_config.json"))
        func workingSet() -> Int? {
            XCTFail("Inkling profile resolution must run before Metal initialization")
            return nil
        }
        let args = ["--model-dir", fixture.modelDirectory]
        let c = try ServerConfig.parse(arguments: args, recommendedWorkingSetBytes: workingSet(), environment: [:])
        XCTAssertTrue(c.isInklingMini)
        XCTAssertEqual(c.contextCap, 8192)
        XCTAssertEqual(c.prefillStepSize, 128)
        XCTAssertEqual(c.maxTokensDefault, 4096)
        XCTAssertEqual(c.cacheLimitBytes, 536_870_912)
        XCTAssertTrue(c.cacheLimitExplicit)
        XCTAssertEqual(c.effectiveCacheMode, .off)
        XCTAssertEqual(c.temperature, 1)
        XCTAssertEqual(c.topP, 1)
        XCTAssertEqual(c.topK, 0)
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

    func testAdmissionAndUnsupportedModes() throws {
        let base = try config()
        XCTAssertTrue(try InklingSupport.validateLoad(base))
        for index in 0..<8 {
            var c = base
            switch index {
            case 0: c.enableInklingMini = false
            case 1: c.contextCap = 8193
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
        c.contextCap = 8193
        XCTAssertThrowsError(try InklingSupport.validateLoad(c))
    }

}
