import XCTest
@testable import MeiCore

final class OpenAITypesTests: XCTestCase {
    func testChatRequestDecodingFull() throws {
        let json = """
        {
          "model": "ornith-test",
          "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 15 + 27?"}
          ],
          "temperature": 0,
          "top_p": 0.9,
          "top_k": 20,
          "max_tokens": 128,
          "stream": false,
          "stop": ["END"],
          "seed": 42,
          "tool_choice": {"type": "function", "function": {"name": "add_numbers"}}
        }
        """
        let request = try ChatRequest(json: Data(json.utf8))
        XCTAssertEqual(request.model, "ornith-test")
        XCTAssertEqual(request.messages.count, 2)
        XCTAssertEqual(request.messages[0].role, "system")
        XCTAssertEqual(request.messages[0].content, "You are helpful.")
        XCTAssertEqual(request.messages[1].content, "What is 15 + 27?")
        XCTAssertEqual(request.temperature, 0)
        XCTAssertEqual(request.topP, 0.9)
        XCTAssertEqual(request.topK, 20)
        XCTAssertEqual(request.maxTokens, 128)
        XCTAssertFalse(request.stream)
        XCTAssertEqual(request.stop, ["END"])
        XCTAssertEqual(request.seed, 42)
        XCTAssertNotNil(request.toolChoice)
    }

    func testChatRequestContentArray() throws {
        let json = """
        {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
        """
        let request = try ChatRequest(json: Data(json.utf8))
        XCTAssertEqual(request.messages[0].content, "hello")
    }

    func testChatRequestStreamOptions() throws {
        let json = """
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": true,
         "stream_options": {"include_usage": true}}
        """
        let request = try ChatRequest(json: Data(json.utf8))
        XCTAssertTrue(request.stream)
        XCTAssertTrue(request.includeUsage)
    }

    func testChatRequestToolMessages() throws {
        let json = """
        {
          "model": "m",
          "messages": [
            {"role": "user", "content": "add"},
            {"role": "assistant", "content": null,
             "tool_calls": [{"id": "call_1", "type": "function",
               "function": {"name": "add_numbers", "arguments": "{\\"a\\": 15, \\"b\\": 27}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"}
          ]
        }
        """
        let request = try ChatRequest(json: Data(json.utf8))
        XCTAssertEqual(request.messages.count, 3)
        let assistant = request.messages[1]
        XCTAssertEqual(assistant.role, "assistant")
        XCTAssertEqual(assistant.toolCalls?.count, 1)
        XCTAssertEqual(assistant.toolCalls?[0].name, "add_numbers")
        XCTAssertEqual(assistant.toolCalls?[0].argumentsJSON, #"{"a": 15, "b": 27}"#)
        XCTAssertEqual(request.messages[2].role, "tool")
        XCTAssertEqual(request.messages[2].toolCallID, "call_1")
        XCTAssertEqual(request.messages[2].content, "42")
    }

    func testCompletionRequest() throws {
        let json = """
        {"model": "m", "prompt": " hello hello", "max_tokens": 1, "temperature": 0}
        """
        let request = try CompletionRequest(json: Data(json.utf8))
        XCTAssertEqual(request.prompt, " hello hello")
        XCTAssertEqual(request.maxTokens, 1)
        XCTAssertEqual(request.temperature, 0)
    }

    func testToolChoiceObjectJsonPath() throws {
        let json = """
        {"model": "m", "messages": [{"role": "user", "content": "x"}],
         "tool_choice": {"type": "function", "function": {"name": "add_numbers"}}}
        """
        let request = try ChatRequest(json: Data(json.utf8))
        guard case .object(let object) = request.toolChoice else {
            return XCTFail("expected object tool_choice")
        }
        guard case .object(let function) = object["function"] else {
            return XCTFail("expected function object")
        }
        guard case .string(let name) = function["name"] else {
            return XCTFail("expected function name")
        }
        XCTAssertEqual(name, "add_numbers")
    }

    func testResponseEncoding() throws {
        var run = GenerationRun()
        run.text = "Ready."
        run.promptTokenCount = 10
        run.completionTokenCount = 3
        run.finishReason = "stop"
        let response = Router.completionResponse(run: run, model: "m", emitReasoning: true)
        let data = try JSONEncoder().encode(response)
        let dict = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(dict["object"] as? String, "chat.completion")
        let choices = try XCTUnwrap(dict["choices"] as? [[String: Any]])
        XCTAssertEqual(choices[0]["finish_reason"] as? String, "stop")
        let message = try XCTUnwrap(choices[0]["message"] as? [String: Any])
        XCTAssertEqual(message["content"] as? String, "Ready.")
        let usage = try XCTUnwrap(dict["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? Int, 10)
        XCTAssertEqual(usage["completion_tokens"] as? Int, 3)
    }

    func testToolCallArgumentsRoundTrip() throws {
        let arguments = ["a": MeiJSONValue.number(15), "b": MeiJSONValue.number(27)]
        let jsonString = try MeiJSONValue.object(arguments).jsonString()
        XCTAssertEqual(jsonString, #"{"a":15,"b":27}"#)
        let parsed = MeiJSONValue.parseObject(from: jsonString)
        XCTAssertEqual(parsed, .object(arguments))
    }

    // MARK: - Usage contract

    /// A run with every usage-relevant counter populated.
    private func usageRun() -> GenerationRun {
        var run = GenerationRun()
        run.promptTokenCount = 10
        run.completionTokenCount = 5
        run.cachedTokenCount = 8
        run.decodeTokensPerSecond = 47.5
        run.promptTokensPerSecond = 300.1
        // Leave perf fields intentionally; the contract keeps absent-perf nil.
        return run
    }

    func testUsageContractFieldTypesAndArithmetic() throws {
        let usage = Router.usage(run: usageRun())
        XCTAssertEqual(usage.promptTokens, 10)
        XCTAssertEqual(usage.completionTokens, 5)
        XCTAssertEqual(usage.totalTokens, 15, "total_tokens must be prompt + completion")
        XCTAssertEqual(usage.promptTokensDetails?.cachedTokens, 8)
        XCTAssertEqual(usage.tokensPerSecond, 47.5)
        XCTAssertEqual(usage.promptTokensPerSecond, 300.1)

        let data = try JSONEncoder().encode(usage)
        let dict = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        // All count fields must be integral JSON numbers, not strings.
        XCTAssertEqual(dict["prompt_tokens"] as? NSNumber, 10)
        XCTAssertEqual(dict["completion_tokens"] as? NSNumber, 5)
        XCTAssertEqual(dict["total_tokens"] as? NSNumber, 15)
        let details = try XCTUnwrap(dict["prompt_tokens_details"] as? [String: Any])
        XCTAssertEqual(details["cached_tokens"] as? NSNumber, 8)
    }

    func testNonStreamingResponsesAlwaysIncludeUsage() throws {
        // /v1/chat/completions and /v1/completions both serialize through
        // Router.completionResponse, so this single assertion pins the shared
        // shape for both paths.
        let response = Router.completionResponse(run: usageRun(), model: "m", emitReasoning: true)
        let data = try JSONEncoder().encode(response)
        let dict = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let usage = try XCTUnwrap(dict["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? NSNumber, 10)
        XCTAssertEqual(usage["completion_tokens"] as? NSNumber, 5)
        XCTAssertEqual(usage["total_tokens"] as? NSNumber, 15)
        XCTAssertEqual(
            (usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? NSNumber, 8)
    }

    func testStreamingFinishUsageCountParityWithNonStreaming() throws {
        let run = usageRun()
        // Non-streaming response usage is the contract reference.
        let nonStreaming = Router.completionResponse(run: run, model: "m", emitReasoning: true)
        let nonStreamingData = try JSONEncoder().encode(nonStreaming)
        let nonStreamingDict = try XCTUnwrap(
            JSONSerialization.jsonObject(with: nonStreamingData) as? [String: Any])
        let nonStreamingUsage = try XCTUnwrap(nonStreamingDict["usage"] as? [String: Any])

        // The same run through the streaming finish (include_usage: true) must
        // emit a usage chunk whose counts are byte-identical in value.
        let frames = Router.finishSSEData(
            id: "chatcmpl-test", run: run, model: "m", created: 1_700_000_000,
            includeUsage: true)
        let usageFrame = try XCTUnwrap(
            frames.first { frame in frame.contains("\"usage\":") },
            "include_usage=true must emit a usage chunk")
        let usageData = Data(usageFrame.utf8)
        let usageDict = try XCTUnwrap(JSONSerialization.jsonObject(with: usageData) as? [String: Any])
        let usage = try XCTUnwrap(usageDict["usage"] as? [String: Any])
        XCTAssertEqual(usage["prompt_tokens"] as? NSNumber, nonStreamingUsage["prompt_tokens"] as? NSNumber)
        XCTAssertEqual(usage["completion_tokens"] as? NSNumber, nonStreamingUsage["completion_tokens"] as? NSNumber)
        XCTAssertEqual(usage["total_tokens"] as? NSNumber, nonStreamingUsage["total_tokens"] as? NSNumber)
        XCTAssertEqual(
            (usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? NSNumber,
            (nonStreamingUsage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? NSNumber)
        XCTAssertEqual(usage["tokens_per_second"] as? NSNumber, nonStreamingUsage["tokens_per_second"] as? NSNumber)
    }

    func testStreamingUsageAbsentWhenIncludeUsageFalse() throws {
        let frames = Router.finishSSEData(
            id: "chatcmpl-test", run: usageRun(), model: "m", created: 1_700_000_000,
            includeUsage: false)
        XCTAssertFalse(
            frames.contains { $0.contains("\"usage\":") },
            "streaming without stream_options.include_usage must NOT emit a usage chunk")
        XCTAssertEqual(frames.last, "[DONE]")
    }
}