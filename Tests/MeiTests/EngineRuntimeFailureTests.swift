import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MeiCore

private struct ProbeTokenizer: Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { _ in String(repeating: "a", count: 300) }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "answer " }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1] }
}

private struct ProbeProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput { LMInput(tokens: MLXArray([Int32(1)])) }
}

private final class ProbeModel: Module, LanguageModel {
    let invalidAt: Int?
    var calls = 0
    init(invalidAt: Int?) { self.invalidAt = invalidAt }
    var kvHeads: [Int] { [] }
    var vocabularySize: Int { 3 }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let invalid = calls == invalidAt
        calls += 1
        if invalid {
            // Invalid shapes exercise vMLX’s actual decode error boundary.
            _ = MLX.matmul(MLXArray.zeros([1, 2]), MLXArray.zeros([3, 1]))
        }
        return MLXArray([Float(0), 1, -1]).reshaped(1, 1, 3)
    }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

final class EngineRuntimeFailureTests: XCTestCase {
    private func engine(invalidAt: Int?) -> Engine {
        let model = ProbeModel(invalidAt: invalidAt)
        let context = ModelContext(
            configuration: ModelConfiguration(id: "test/fake"), model: model,
            processor: ProbeProcessor(), tokenizer: ProbeTokenizer())
        var config = ServerConfig(modelDirectory: "/nonexistent", servedModelID: "test/fake")
        config.temperature = 0
        config.cacheReuse = false
        return Engine(container: ModelContainer(context: context), config: config)
    }

    private func chatRequest() throws -> ChatRequest {
        try ChatRequest(
            json: Data(
                #"{"model":"test/fake","messages":[{"role":"user","content":"hello"}],"max_tokens":4,"temperature":0}"#
                    .utf8))
    }

    func testBufferedChatAndCompletionSurfaceDecodeFailure() async throws {
        let chat = engine(invalidAt: 1)
        do {
            _ = try await chat.chatRun(request: chatRequest())
            XCTFail("expected generation failure")
        } catch {
            XCTAssertTrue(
                error.localizedDescription.contains("MLX error during decode"), "\(error)")
        }
        let recovered = try await chat.chatRun(request: chatRequest())
        XCTAssertEqual(recovered.finishReason, "length")
        XCTAssertFalse(recovered.text.isEmpty)

        let completion = engine(invalidAt: 1)
        let request = try CompletionRequest(
            json: Data(
                #"{"model":"test/fake","prompt":"hello","max_tokens":4,"temperature":0}"#.utf8))
        do {
            _ = try await completion.completionRun(request: request)
            XCTFail("expected generation failure")
        } catch {
            XCTAssertTrue(
                error.localizedDescription.contains("MLX error during decode"), "\(error)")
        }
    }

    func testStreamingFailureDoesNotEmitFinish() async throws {
        for failurePosition in [1, 3] {
            let engine = engine(invalidAt: failurePosition)
            let stream = await engine.chatRunStreaming(request: try chatRequest())
            var finished = false
            var text = ""
            do {
                for try await event in stream {
                    if case .finish = event { finished = true }
                    if case .chunk(let chunk) = event { text += chunk }
                }
                XCTFail("expected generation failure")
            } catch {
                XCTAssertTrue(
                    error.localizedDescription.contains("MLX error during decode"), "\(error)")
            }
            XCTAssertFalse(finished)
            if failurePosition == 1 {
                XCTAssertTrue(text.isEmpty)
            } else {
                XCTAssertFalse(text.isEmpty, "fixture must exercise failure after visible output")
            }
        }
    }

    func testHTTPStreamingErrorBeforeAndAfterVisibleOutput() async throws {
        for failurePosition in [1, 3] {
            let engine = engine(invalidAt: failurePosition)
            var config = ServerConfig(modelDirectory: "/nonexistent", servedModelID: "test/fake")
            config.port = 0
            let server = try MeiHTTPServer(
                router: Router(engine: engine, config: config), config: config)
            defer { server.shutdown() }
            let port = try XCTUnwrap(server.boundPort)
            var request = URLRequest(
                url: try XCTUnwrap(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")))
            request.httpMethod = "POST"
            request.timeoutInterval = 20
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data(
                #"{"model":"test/fake","messages":[{"role":"user","content":"hello"}],"max_tokens":4,"temperature":0,"stream":true}"#
                    .utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            let status = try XCTUnwrap(response as? HTTPURLResponse).statusCode
            let body = try XCTUnwrap(String(data: data, encoding: .utf8))
            if failurePosition == 1 {
                XCTAssertEqual(status, 500, body)
                XCTAssertTrue(body.contains("MLX error during decode"), body)
            } else {
                XCTAssertEqual(status, 200, body)
                XCTAssertTrue(body.contains("stream_error"), body)
                XCTAssertTrue(body.contains(String(repeating: "a", count: 20)), body)
                XCTAssertFalse(body.contains("[DONE]"), body)
                XCTAssertFalse(body.contains("\"finish_reason\":\"stop\""), body)
            }
        }
    }

}
