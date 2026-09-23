import XCTest
import NIOHTTP1
@testable import MeiCore

/// Deterministic response-shape coverage: error envelope, /v1/models identity,
/// the minimal legacy /v1/completions shape, usage arithmetic on the wire,
/// and the error-status mapping for engine failures.
final class OpenAIResponseShapeTests: XCTestCase {

    private func decodeObject(_ encoded: String) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(encoded.utf8)) as? [String: Any])
    }

    private func usageRun(
        prompt: Int = 10, completion: Int = 5, cached: Int = 8
    ) -> GenerationRun {
        var run = GenerationRun()
        run.text = "Ready."
        run.promptTokenCount = prompt
        run.completionTokenCount = completion
        run.cachedTokenCount = cached
        run.decodeTokensPerSecond = 47.5
        run.finishReason = "stop"
        return run
    }

    // MARK: - Error envelope

    func testErrorEnvelopeShape() throws {
        let payload = ResponseSerializer().errorPayload("boom")
        let object = try decodeObject(payload)
        let error = try XCTUnwrap(object["error"] as? [String: Any])
        XCTAssertEqual(error["message"] as? String, "boom")
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertNil(error["code"], "code is omitted when absent (OpenAI includes it as null; Mei omits)")
        XCTAssertEqual(object.count, 1, "envelope must contain only the error key")
    }

    func testErrorEnvelopeCarriesCode() throws {
        let payload = ResponseSerializer().errorPayload("ctx", type: "engine_error", code: "engine_error")
        let error = try XCTUnwrap(decodeObject(payload)["error"] as? [String: Any])
        XCTAssertEqual(error["code"] as? String, "engine_error")
        XCTAssertEqual(error["type"] as? String, "engine_error")
    }

    func testEngineErrorMessagesAndStatusMapping() {
        let cap = EngineError.overContextCap(promptTokens: 65_537, cap: 65_536)
        XCTAssertTrue(cap.errorDescription!.contains("65537"))
        XCTAssertTrue(cap.errorDescription!.contains("65536"))
        XCTAssertTrue(cap.errorDescription!.contains("context cap"))

        XCTAssertEqual(Router.errorStatus(.overContextCap(promptTokens: 1, cap: 1)), .badRequest)
        XCTAssertEqual(Router.errorStatus(.emptyPrompt), .badRequest)
        XCTAssertEqual(Router.errorStatus(.modelNotLoaded), .internalServerError)
        XCTAssertEqual(Router.errorStatus(.generationFailed("x")), .internalServerError)
        XCTAssertEqual(Router.errorStatus(.modelDirectoryMissing("/x")), .internalServerError)
    }

    // MARK: - /v1/models

    func testModelsResponseShape() throws {
        let response = ModelsResponse(data: [
            .init(id: "org/Model-4bit", created: 1_700_000_000)
        ])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["object"] as? String, "list")
        let entries = try XCTUnwrap(object["data"] as? [[String: Any]])
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries[0]["id"] as? String, "org/Model-4bit")
        XCTAssertEqual(entries[0]["object"] as? String, "model")
        XCTAssertEqual(entries[0]["owned_by"] as? String, "mei")
        XCTAssertEqual(entries[0]["created"] as? NSNumber, 1_700_000_000)
    }

    // MARK: - Usage arithmetic (integer fields)

    func testUsageArithmeticIsIntegralJSON() throws {
        let response = Router.completionResponse(run: usageRun(), model: "m", emitReasoning: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? NSNumber, 10)
        XCTAssertEqual(usage["completion_tokens"] as? NSNumber, 5)
        XCTAssertEqual(usage["total_tokens"] as? NSNumber, 15)
        let details = try XCTUnwrap(usage["prompt_tokens_details"] as? [String: Any])
        XCTAssertEqual(details["cached_tokens"] as? NSNumber, 8)
        // NaN-free perf fields ride as numbers, not strings.
        XCTAssertEqual(usage["tokens_per_second"] as? NSNumber, 47.5)
    }

    func testChatResponseShape() throws {
        let response = Router.completionResponse(run: usageRun(), model: "m", emitReasoning: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["object"] as? String, "chat.completion")
        XCTAssertEqual(object["model"] as? String, "m")
        XCTAssertNotNil(object["id"] as? String)
        XCTAssertNotNil(object["created"] as? NSNumber)
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        XCTAssertEqual(choices.count, 1)
        XCTAssertEqual(choices[0]["index"] as? NSNumber, 0)
        XCTAssertEqual(choices[0]["finish_reason"] as? String, "stop")
        XCTAssertNil(choices[0]["logprobs"])
        XCTAssertEqual((choices[0]["message"] as? [String: Any])?["role"] as? String, "assistant")
    }

    func testToolCallResponseShape() throws {
        var run = GenerationRun()
        run.toolCalls = [.init(id: "call_1", name: "add_numbers", argumentsJSON: #"{"a":15,"b":27}"#)]
        run.finishReason = "tool_calls"
        run.text = ""
        run.promptTokenCount = 3
        run.completionTokenCount = 12
        let response = Router.completionResponse(run: run, model: "m", emitReasoning: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(response)) as? [String: Any])
        let message = try XCTUnwrap((object["choices"] as? [[String: Any]])?[0]["message"] as? [String: Any])
        XCTAssertNil(message["content"], "tool-only runs must not fabricate empty content")
        let calls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]])
        XCTAssertEqual(calls[0]["id"] as? String, "call_1")
        XCTAssertEqual(calls[0]["type"] as? String, "function")
        XCTAssertEqual((calls[0]["function"] as? [String: Any])?["name"] as? String, "add_numbers")
        XCTAssertEqual((calls[0]["function"] as? [String: Any])?["arguments"] as? String, #"{"a":15,"b":27}"#)
        XCTAssertEqual((object["choices"] as? [[String: Any]])?[0]["finish_reason"] as? String, "tool_calls")
    }

    func testReasoningFieldRespectsEmitFlag() throws {
        var run = GenerationRun()
        run.text = "answer"
        run.reasoning = "secret thought"
        run.promptTokenCount = 1
        run.completionTokenCount = 1
        let with = Router.completionResponse(run: run, model: "m", emitReasoning: true)
        let without = Router.completionResponse(run: run, model: "m", emitReasoning: false)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let choicesWith = try XCTUnwrap(
            (JSONSerialization.jsonObject(with: encoder.encode(with)) as? [String: Any])?["choices"] as? [[String: Any]])
        let choicesWithout = try XCTUnwrap(
            (JSONSerialization.jsonObject(with: encoder.encode(without)) as? [String: Any])?["choices"] as? [[String: Any]])
        let messageWith = try XCTUnwrap(choicesWith[0]["message"] as? [String: Any])
        let messageWithout = try XCTUnwrap(choicesWithout[0]["message"] as? [String: Any])
        XCTAssertEqual(messageWith["reasoning_content"] as? String, "secret thought")
        XCTAssertNil(messageWithout["reasoning_content"])
    }

    // MARK: - Legacy /v1/completions shape

    func testLegacyCompletionResponseShape() throws {
        var run = usageRun()
        run.text = "Hello there"
        let response = Router.legacyCompletionResponse(run: run, model: "m")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(response)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["object"] as? String, "text_completion")
        XCTAssertEqual(object["model"] as? String, "m")
        XCTAssertTrue((object["id"] as? String)?.hasPrefix("cmpl-") == true)
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        XCTAssertEqual(choices.count, 1)
        XCTAssertEqual(choices[0]["text"] as? String, "Hello there")
        XCTAssertEqual(choices[0]["index"] as? NSNumber, 0)
        XCTAssertEqual(choices[0]["finish_reason"] as? String, "stop")
        XCTAssertNil(choices[0]["logprobs"])
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? NSNumber, 10)
        XCTAssertEqual(usage["completion_tokens"] as? NSNumber, 5)
        XCTAssertEqual(usage["total_tokens"] as? NSNumber, 15)
    }

    func testStreamingSSEErrorFrameShape() throws {
        // The mid-stream error frame a streaming run emits when generation
        // itself fails after the 200 headers are out.
        let payload = ResponseSerializer().errorPayload("generation failed: boom", type: "invalid_request_error", code: "stream_error")
        let object = try decodeObject(payload)
        let error = try XCTUnwrap(object["error"] as? [String: Any])
        XCTAssertEqual(error["code"] as? String, "stream_error")
    }
}