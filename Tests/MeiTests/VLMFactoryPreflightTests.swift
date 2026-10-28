import Foundation
import XCTest
import MLXLMCommon
import MLXVLM

/// Regression at the actual factory entry point: metadata refusal must happen
/// before a creator can allocate a model or read any weight shards.
final class VLMFactoryPreflightTests: XCTestCase {
    private enum Probe: Error { case modelCreationReached, tokenizerReached, processorReached }
    private struct NoTokenizer: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { throw Probe.tokenizerReached }
    }

    private func attempt(config: String, processor: String? = nil) async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(config.utf8).write(to: dir.appendingPathComponent("config.json"))
        if let processor {
            try Data(processor.utf8).write(to: dir.appendingPathComponent("preprocessor_config.json"))
        }
        let types = ModelTypeRegistry(creators: [
            "qwen3_5_moe": { _, _ in throw Probe.modelCreationReached },
            "fixture-vlm": { _, _ in throw Probe.modelCreationReached }
        ])
        let processors = ProcessorTypeRegistry(creators: [
            "FixtureProcessor": { _, _ in throw Probe.processorReached }
        ])
        let factory = VLMModelFactory(typeRegistry: types, processorRegistry: processors,
                                      modelRegistry: VLMRegistry.shared)
        let resolved = ModelConfiguration(directory: dir).resolved(modelDirectory: dir, tokenizerDirectory: dir)
        _ = try await factory._load(configuration: resolved, tokenizerLoader: NoTokenizer())
    }

    func testTextOnlyQwenRefusesVisionFactoryBeforeModelCreation() async throws {
        do {
            try await attempt(config: #"{"model_type":"qwen3_5_moe","text_config":{}}"#)
            XCTFail("Expected text-only factory refusal")
        } catch ModelFactoryError.unsupportedModelType { }
        catch { XCTFail("Expected metadata refusal before allocation, got \(error)") }
    }

    func testNullVisionDeclarationIsTextOnly() async throws {
        do {
            try await attempt(config: #"{"model_type":"qwen3_5_moe","vision_config":null}"#)
            XCTFail("Expected text-only factory refusal")
        } catch ModelFactoryError.unsupportedModelType { }
        catch { XCTFail("Null tower must refuse before allocation, got \(error)") }
    }

    func testVisionQwenStillReachesModelCreation() async throws {
        do {
            try await attempt(config: #"{"model_type":"qwen3_5_moe","vision_config":{}}"#,
                              processor: #"{"processor_class":"FixtureProcessor"}"#)
            XCTFail("Expected creation probe")
        } catch Probe.modelCreationReached { }
        catch { XCTFail("Declared vision must remain eligible, got \(error)") }
    }

    func testUnknownModelPreservesUnsupportedTypeError() async throws {
        do {
            try await attempt(config: #"{"model_type":"not-a-vlm"}"#)
            XCTFail("Expected unsupported model type")
        } catch ModelFactoryError.unsupportedModelType { }
        catch { XCTFail("Unknown model type must not become a processor error, got \(error)") }
    }

    func testMissingProcessorRefusesBeforeModelCreation() async throws {
        do {
            try await attempt(config: #"{"model_type":"fixture-vlm"}"#)
            XCTFail("Expected missing processor refusal")
        } catch ModelFactoryError.configurationFileError(let filename, _, _) {
            XCTAssertEqual(filename, "preprocessor_config.json")
        } catch { XCTFail("Expected processor error before allocation, got \(error)") }
    }

    func testMalformedProcessorRefusesBeforeModelCreation() async throws {
        do {
            try await attempt(config: #"{"model_type":"fixture-vlm"}"#, processor: "{")
            XCTFail("Expected malformed processor refusal")
        } catch ModelFactoryError.configurationDecodingError(let filename, _, _) {
            XCTAssertEqual(filename, "preprocessor_config.json")
        } catch { XCTFail("Expected processor decode error before allocation, got \(error)") }
    }

    func testUnknownProcessorRefusesBeforeModelCreation() async throws {
        do {
            try await attempt(config: #"{"model_type":"fixture-vlm"}"#,
                              processor: #"{"processor_class":"UnknownProcessor"}"#)
            XCTFail("Expected unsupported processor refusal")
        } catch ModelFactoryError.unsupportedProcessorType { }
        catch { XCTFail("Expected processor registration refusal before allocation, got \(error)") }
    }

    func testValidProcessorMetadataReachesModelCreation() async throws {
        do {
            try await attempt(config: #"{"model_type":"fixture-vlm"}"#,
                              processor: #"{"processor_class":"FixtureProcessor"}"#)
            XCTFail("Expected creation probe")
        } catch Probe.modelCreationReached { }
        catch { XCTFail("Valid processor metadata must reach creation, got \(error)") }
    }
}
