import XCTest
@testable import MeiCore

/// Deterministic coverage of request-over-server precedence for sampling,
/// penalties, seed, stop, and max-token arithmetic, plus the reasoning-mode
/// override rules. All pure — no model or container involved.
final class GenerationControlSelectionTests: XCTestCase {

    private func config(
        temperature: Float = 0.6,
        topP: Float = 0.95,
        topK: Int = 20,
        minP: Float = 0,
        repetitionPenalty: Float? = nil,
        presencePenalty: Float? = nil,
        frequencyPenalty: Float? = nil,
        contextCap: Int = 65_536,
        maxTokensDefault: Int = 32_768
    ) -> ServerConfig {
        var config = ServerConfig(modelDirectory: "/tmp/model", servedModelID: "test-model")
        config.temperature = temperature
        config.topP = topP
        config.topK = topK
        config.minP = minP
        config.repetitionPenalty = repetitionPenalty
        config.presencePenalty = presencePenalty
        config.frequencyPenalty = frequencyPenalty
        config.contextCap = contextCap
        config.maxTokensDefault = maxTokensDefault
        return config
    }

    private func chatRequest(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        maxTokens: Int? = nil,
        stop: [String]? = nil,
        seed: UInt64? = nil,
        repetitionPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil
    ) -> ChatRequest {
        ChatRequest(
            model: "test-model",
            messages: [.init(role: "user", content: "hi")],
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            maxTokens: maxTokens,
            stream: false,
            stop: stop,
            tools: nil,
            toolChoice: nil,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            frequencyPenalty: frequencyPenalty,
            seed: seed,
            includeUsage: false,
            reasoningEffort: nil)
    }

    // MARK: - Request-over-server precedence

    func testRequestValuesOverrideServerDefaults() {
        let config = config(temperature: 0.6, topP: 0.95, topK: 20, minP: 0)
        let request = chatRequest(
            temperature: 0.1, topP: 0.3, topK: 5, minP: 0.02,
            stop: ["END"], seed: 99,
            repetitionPenalty: 1.1, presencePenalty: 0.5, frequencyPenalty: -0.5)
        let controls = GenerationControlSelection.resolve(
            request: request, config: config, promptTokenCount: 10)
        XCTAssertEqual(controls.temperature, 0.1)
        XCTAssertEqual(controls.topP, 0.3)
        XCTAssertEqual(controls.topK, 5)
        XCTAssertEqual(controls.minP, 0.02)
        XCTAssertEqual(controls.repetitionPenalty, 1.1)
        XCTAssertEqual(controls.presencePenalty, 0.5)
        XCTAssertEqual(controls.frequencyPenalty, -0.5)
        XCTAssertEqual(controls.randomSeed, 99)
        XCTAssertEqual(controls.extraStopStrings, ["END"])
    }

    func testServerDefaultsApplyWhenRequestOmits() {
        let config = config(temperature: 0.7, topP: 0.8, topK: 30, minP: 0.05,
                            repetitionPenalty: 1.2, presencePenalty: 0.25, frequencyPenalty: 0.75)
        let controls = GenerationControlSelection.resolve(
            request: chatRequest(), config: config, promptTokenCount: 10)
        XCTAssertEqual(controls.temperature, 0.7)
        XCTAssertEqual(controls.topP, 0.8)
        XCTAssertEqual(controls.topK, 30)
        XCTAssertEqual(controls.minP, 0.05)
        XCTAssertEqual(controls.repetitionPenalty, 1.2)
        XCTAssertEqual(controls.presencePenalty, 0.25)
        XCTAssertEqual(controls.frequencyPenalty, 0.75)
        XCTAssertNil(controls.randomSeed, "no server seed exists; seed must stay request-only")
        XCTAssertNil(controls.extraStopStrings)
    }

    func testEmptyStopArrayMeansNoStop() {
        let config = config()
        let controls = GenerationControlSelection.resolve(
            request: chatRequest(stop: []), config: config, promptTokenCount: 10)
        XCTAssertNil(controls.extraStopStrings)
        let withStop = GenerationControlSelection.resolve(
            request: chatRequest(stop: ["x"]), config: config, promptTokenCount: 10)
        XCTAssertEqual(withStop.extraStopStrings, ["x"])
    }

    func testRequestOverridesDoNotLeakAcrossRequests() {
        // Resolution is pure and stateless: an override request must not
        // bleed into the next request, and re-resolving the same override
        // must be order-independent (the seam behind byte-stable greedy
        // runs at temperature 0 + fixed seed).
        let config = config(temperature: 0.7)
        let overridden = chatRequest(temperature: 0.0, seed: 7)
        let first = GenerationControlSelection.resolve(
            request: overridden, config: config, promptTokenCount: 10)
        let plain = GenerationControlSelection.resolve(
            request: chatRequest(), config: config, promptTokenCount: 10)
        XCTAssertEqual(plain.temperature, 0.7, "the previous request's temperature must not leak")
        XCTAssertNil(plain.randomSeed, "the previous request's seed must not leak")
        XCTAssertNil(plain.extraStopStrings)
        let again = GenerationControlSelection.resolve(
            request: overridden, config: config, promptTokenCount: 10)
        XCTAssertEqual(first, again)
        XCTAssertEqual(first.temperature, 0.0)
        XCTAssertEqual(first.randomSeed, 7)
    }

    // MARK: - Max token arithmetic

    func testMaxTokensClampedToKVCapacity() {
        let config = config(contextCap: 100)
        let controls = GenerationControlSelection.resolve(
            request: chatRequest(maxTokens: 500), config: config, promptTokenCount: 40)
        // kvCapacity = contextCap + 4096 for the generation headroom window.
        XCTAssertEqual(controls.maxTokens, min(500, max(1, 100 + 4096 - 40)))
    }

    func testMaxTokensNeverBelowOneOnFullContext() {
        let config = config(contextCap: 10)
        let controls = GenerationControlSelection.resolve(
            request: chatRequest(maxTokens: 100), config: config, promptTokenCount: 10_000)
        XCTAssertEqual(controls.maxTokens, 1, "a full context must still allow a 1-token generation")
    }

    func testServerDefaultMaxTokensUsedWhenRequestOmits() {
        let config = config(maxTokensDefault: 8192)
        let controls = GenerationControlSelection.resolve(
            request: chatRequest(), config: config, promptTokenCount: 100)
        XCTAssertEqual(controls.maxTokens, 8192)
    }

    func testResolveMaxTokensArithmetic() {
        XCTAssertEqual(GenerationControlSelection.resolveMaxTokens(requested: 50, promptTokenCount: 10, kvCapacity: 100), 50)
        XCTAssertEqual(GenerationControlSelection.resolveMaxTokens(requested: 200, promptTokenCount: 10, kvCapacity: 100), 90)
        XCTAssertEqual(GenerationControlSelection.resolveMaxTokens(requested: 1, promptTokenCount: 100, kvCapacity: 100), 1)
        XCTAssertEqual(GenerationControlSelection.resolveMaxTokens(requested: 1, promptTokenCount: 500, kvCapacity: 100), 1)
    }

    // MARK: - Completion requests

    func testCompletionRequestResolution() {
        let config = config(temperature: 0.6)
        let request = CompletionRequest(
            model: "m", prompt: "hello", temperature: 0.0, topP: nil, topK: nil,
            minP: nil, maxTokens: 16, stream: false, stop: nil,
            repetitionPenalty: nil, presencePenalty: nil, frequencyPenalty: nil,
            seed: 7, includeUsage: false)
        let controls = GenerationControlSelection.resolve(
            request: request, config: config, promptTokenCount: 3)
        XCTAssertEqual(controls.temperature, 0.0)
        XCTAssertEqual(controls.topP, config.topP)
        XCTAssertEqual(controls.maxTokens, 16)
        XCTAssertEqual(controls.randomSeed, 7)
    }

    // MARK: - Reasoning mode precedence

    func testReasoningOverrideRules() {
        // effort with unset server flag drives enable_thinking.
        XCTAssertEqual(
            Engine.resolveEnableThinking(
                request: chatRequest(reasoningEffort: "none"), configEnableThinking: nil),
            false)
        XCTAssertEqual(
            Engine.resolveEnableThinking(
                request: chatRequest(reasoningEffort: "high"), configEnableThinking: nil),
            true)
        XCTAssertEqual(
            Engine.resolveEnableThinking(
                request: chatRequest(reasoningEffort: "low"), configEnableThinking: nil),
            true)
        // no effort, no flag: template default.
        XCTAssertNil(
            Engine.resolveEnableThinking(
                request: chatRequest(), configEnableThinking: nil))
        // explicit server flag always wins over a request effort.
        XCTAssertEqual(
            Engine.resolveEnableThinking(
                request: chatRequest(reasoningEffort: "none"), configEnableThinking: true),
            true)
        XCTAssertEqual(
            Engine.resolveEnableThinking(
                request: chatRequest(reasoningEffort: "high"), configEnableThinking: false),
            false)
    }

    private func chatRequest(reasoningEffort: String?) -> ChatRequest {
        var request = chatRequest()
        request.reasoningEffort = reasoningEffort
        return request
    }

    // MARK: - Stop reason mapping

    func testStopReasonMapping() {
        XCTAssertEqual(Engine.mapStopReason(.stop, toolCallCount: 0), "stop")
        XCTAssertEqual(Engine.mapStopReason(.length, toolCallCount: 0), "length")
        XCTAssertEqual(Engine.mapStopReason(.cancelled, toolCallCount: 0), "stop")
        XCTAssertEqual(Engine.mapStopReason(.stop, toolCallCount: 2), "tool_calls")
        XCTAssertEqual(Engine.mapStopReason(.length, toolCallCount: 1), "tool_calls")
    }
}