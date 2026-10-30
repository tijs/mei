import Foundation
import MLX
import MLXLMCommon
import MLXNN
import NIOEmbedded
import NIOHTTP1
import XCTest
@testable import MeiCore

/// No model execution is needed: this exercises the writer's event-to-wire
/// boundary with an injected stream, including events that serialize to "".
final class HTTPStreamingProgressTests: XCTestCase {
    private struct NoopTokenizer: Tokenizer {
        let bosToken: String? = nil
        let eosToken: String? = nil
        let unknownToken: String? = nil
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
    }

    private struct NoopProcessor: UserInputProcessor {
        func prepare(input: UserInput) throws -> LMInput { fatalError("not used") }
    }

    private final class NoopModel: Module, LanguageModel {
        var kvHeads: [Int] { [] }
        var vocabularySize: Int { 0 }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult { fatalError("not used") }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { fatalError("not used") }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    }

    func testInvisibleProgressBeforeErrorDoesNotCommitSuccessStatus() async throws {
        let config = ServerConfig(modelDirectory: "/nonexistent", servedModelID: "test/fake")
        let context = ModelContext(configuration: ModelConfiguration(id: "test/fake"), model: NoopModel(),
                                   processor: NoopProcessor(), tokenizer: NoopTokenizer())
        let engine = Engine(container: ModelContainer(context: context), config: config)
        let channel = EmbeddedChannel()
        let writer = ResponseWriter(channel: channel, router: Router(engine: engine, config: config))
        let request = try ChatRequest(json: Data(#"{"model":"test/fake","messages":[{"role":"user","content":"hello"}],"stream":true}"#.utf8))
        let stream = AsyncThrowingStream<StreamEvent, Error> { continuation in
            continuation.yield(.prefill(completed: 1, total: 2))
            continuation.finish(throwing: EngineError.generationFailed("fixture failure"))
        }
        await writer.streamSSE(stream: stream, request: request, config: config)
        channel.embeddedEventLoop.run()
        guard case .head(let head)? = try channel.readOutbound(as: HTTPServerResponsePart.self) else {
            return XCTFail("expected an HTTP response head")
        }
        XCTAssertEqual(head.status, .internalServerError)
    }
}
