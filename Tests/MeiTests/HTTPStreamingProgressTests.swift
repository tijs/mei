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

    func testStatusRespondsWhileInferenceActorIsBlocked() async throws {
        let config = ServerConfig(modelDirectory: "/nonexistent", servedModelID: "test/fake")
        let context = ModelContext(configuration: ModelConfiguration(id: "test/fake"), model: NoopModel(),
                                   processor: NoopProcessor(), tokenizer: NoopTokenizer())
        let engine = Engine(container: ModelContainer(context: context), config: config)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let blocker = Task.detached { await engine.blockForStatusTest(entered: entered, release: release) }
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                entered.wait()
                continuation.resume()
            }
        }
        // The watchdog makes an accidental actor hop fail in bounded time.
        // On the responsive path, release immediately after the real route.
        let watchdog = DispatchWorkItem { release.signal() }
        DispatchQueue.global().asyncAfter(deadline: .now() + 2, execute: watchdog)
        let start = ContinuousClock.now
        let result = await Router(engine: engine, config: config).route(
            method: .GET, uri: "/v1/mei/status", body: Data())
        let elapsed = start.duration(to: .now)
        watchdog.cancel()
        release.signal()
        await blocker.value
        XCTAssertLessThan(elapsed, .milliseconds(500), "status waited for the inference actor")
        guard case .plain(let status, _, let body) = result else {
            return XCTFail("expected status JSON")
        }
        XCTAssertEqual(status, .ok)
        let response = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])
        XCTAssertEqual(response["model"] as? String, "test/fake")
        XCTAssertNotNil(response["scheduler"])
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

private extension Engine {
    func blockForStatusTest(entered: DispatchSemaphore, release: DispatchSemaphore) {
        entered.signal()
        release.wait()
    }
}
