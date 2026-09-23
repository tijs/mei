import XCTest
@testable import MeiCore

/// Deterministic SSE coverage over the exact wire format the streaming
/// handler ships: content/reasoning deltas, role chunks, finish frames,
/// [DONE], usage inclusion/omission, tool-call fragmentation and multiple
/// tool indexes, malformed/incomplete frames, and streaming/non-streaming
/// usage parity. Everything here is pure — no engine, no model.
final class RouterSSEFrameTests: XCTestCase {

    private var router: Router {
        Router(engine: nil, config: .init(modelDirectory: "/tmp/m", servedModelID: "mei-model"))
    }

    private func usageRun() -> GenerationRun {
        var run = GenerationRun()
        run.promptTokenCount = 10
        run.completionTokenCount = 5
        run.cachedTokenCount = 8
        run.decodeTokensPerSecond = 47.5
        run.text = "Hello."
        run.finishReason = "stop"
        return run
    }

    private func parse(_ data: String) -> SSEAssemblyResult {
        SSEFrameParser.parse(Data(data.utf8))
    }

    // MARK: - Content and reasoning deltas

    func testContentDeltaFrameShape() {
        let frame = router.sseFrame(
            id: "chatcmpl-x", event: .chunk("Hel"), model: "mei-model",
            emitReasoning: true, includeUsage: false)
        XCTAssertTrue(frame.hasPrefix("data: "), frame)
        XCTAssertTrue(frame.hasSuffix("\n\n"), frame)
        let result = parse(frame)
        XCTAssertEqual(result.content, "Hel")
        XCTAssertEqual(result.role, "assistant", "content deltas carry the assistant role")
    }

    func testContentDeltasAssembleInOrder() {
        var body = ""
        for fragment in ["Hello, ", "world", "!"] {
            body += router.sseFrame(
                id: "chatcmpl-x", event: .chunk(fragment), model: "mei-model",
                emitReasoning: true, includeUsage: false)
        }
        let result = parse(body)
        XCTAssertEqual(result.content, "Hello, world!")
        XCTAssertEqual(result.role, "assistant")
    }

    func testReasoningDeltaEmittedOnlyWhenEnabled() {
        let emitted = router.sseFrame(
            id: "chatcmpl-x", event: .reasoning("think think"), model: "mei-model",
            emitReasoning: true, includeUsage: false)
        XCTAssertEqual(parse(emitted).reasoning, "think think")
        let suppressed = router.sseFrame(
            id: "chatcmpl-x", event: .reasoning("think think"), model: "mei-model",
            emitReasoning: false, includeUsage: false)
        XCTAssertEqual(suppressed, "", "reasoning deltas must be suppressed when emit_reasoning is false")
    }

    func testPrefillEventsProduceNoFrames() {
        let frame = router.sseFrame(
            id: "chatcmpl-x", event: .prefill(completed: 1, total: 10), model: "mei-model",
            emitReasoning: true, includeUsage: false)
        XCTAssertEqual(frame, "")
    }

    // MARK: - Finish frames and [DONE]

    func testFinishFrameCarriesFinishReason() {
        let frames = Router.finishSSEData(
            id: "chatcmpl-x", run: usageRun(), model: "mei-model",
            created: 1_700_000_000, includeUsage: false)
        XCTAssertEqual(frames.count, 2, "finish chunk + [DONE]")
        XCTAssertEqual(frames.last, "[DONE]")
        let finish = try? JSONSerialization.jsonObject(with: Data(frames[0].utf8)) as? [String: Any]
        let choice = (finish?["choices"] as? [[String: Any]])?.first
        XCTAssertEqual(choice?["finish_reason"] as? String, "stop")
        XCTAssertNil((choice?["delta"] as? [String: Any])?["content"] as? String)
    }

    func testSSEFinishSequenceOrderThroughWireFormat() {
        // The complete terminal sequence a stream sends: finish chunk, then
        // usage chunk (include_usage), then [DONE].
        var body = ""
        body += router.sseFrame(
            id: "chatcmpl-x", event: .chunk("done"), model: "mei-model",
            emitReasoning: true, includeUsage: true)
        body += router.sseFrame(
            id: "chatcmpl-x", event: .finish(usageRun()), model: "mei-model",
            emitReasoning: true, includeUsage: true)
        let result = parse(body)
        XCTAssertEqual(result.sawDone, true)
        XCTAssertEqual(result.finishReason, "stop")
        XCTAssertEqual(result.usageFrameCount, 1)
        XCTAssertEqual(result.usage?.promptTokens, 10)
        XCTAssertEqual(result.usage?.cachedTokens, 8)
        XCTAssertEqual(result.malformedFrames, [])
    }

    func testUsageOmittedWhenIncludeUsageFalse() {
        var body = ""
        body += router.sseFrame(
            id: "chatcmpl-x", event: .chunk("done"), model: "mei-model",
            emitReasoning: true, includeUsage: false)
        body += router.sseFrame(
            id: "chatcmpl-x", event: .finish(usageRun()), model: "mei-model",
            emitReasoning: true, includeUsage: false)
        let result = parse(body)
        XCTAssertEqual(result.usageFrameCount, 0)
        XCTAssertNil(result.usage)
        XCTAssertTrue(result.sawDone)
        XCTAssertEqual(result.finishReason, "stop")
    }

    // MARK: - Usage parity with non-streaming

    func testStreamingUsageParityWithNonStreaming() throws {
        let run = usageRun()
        // Non-streaming contract reference.
        let nonStreaming = Router.completionResponse(run: run, model: "mei-model", emitReasoning: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let nonStreamingDict = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoder.encode(nonStreaming)) as? [String: Any])
        let reference = try XCTUnwrap(nonStreamingDict["usage"] as? [String: Any])

        // Same run through the wire-level SSE finish with include_usage.
        var body = ""
        body += router.sseFrame(
            id: "chatcmpl-x", event: .finish(run), model: "mei-model",
            emitReasoning: true, includeUsage: true)
        let streamed = parse(body)
        let usage = try XCTUnwrap(streamed.usage)
        XCTAssertEqual(usage.promptTokens, (reference["prompt_tokens"] as? NSNumber)?.intValue)
        XCTAssertEqual(usage.completionTokens, (reference["completion_tokens"] as? NSNumber)?.intValue)
        XCTAssertEqual(usage.totalTokens, (reference["total_tokens"] as? NSNumber)?.intValue)
        XCTAssertEqual(
            usage.cachedTokens,
            ((reference["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? NSNumber)?.intValue)
        XCTAssertEqual(
            usage.tokensPerSecond,
            (reference["tokens_per_second"] as? NSNumber)?.doubleValue)
    }

    func testUsageChunkHasEmptyChoices() throws {
        let frames = Router.finishSSEData(
            id: "chatcmpl-x", run: usageRun(), model: "mei-model",
            created: 1_700_000_000, includeUsage: true)
        let usageFrame = try XCTUnwrap(frames.first { $0.contains("\"usage\":") })
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(usageFrame.utf8)) as? [String: Any])
        XCTAssertEqual(object["object"] as? String, "chat.completion.chunk")
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        XCTAssertTrue(choices.isEmpty, "the usage chunk must not carry choices (OpenAI contract)")
        XCTAssertNotNil(object["usage"])
    }

    // MARK: - Tool-call deltas: fragmentation and indexes

    func testFragmentedToolCallReassembles() {
        // OpenAI-style fragmentation: id on the first fragment, name and
        // arguments split across frames of the same index.
        let f1 = Router.toolCallSSEData(
            call: .init(id: "call_1", name: "", argumentsJSON: ""), index: 0,
            id: "chatcmpl-x", model: "mei-model", created: 1)
        let f2 = Router.toolCallSSEData(
            call: .init(id: "call_1", name: "add_", argumentsJSON: #"{"a":15,"b":"#), index: 0,
            id: "chatcmpl-x", model: "mei-model", created: 1)
        let f3 = Router.toolCallSSEData(
            call: .init(id: "call_1", name: "numbers", argumentsJSON: "27}"), index: 0,
            id: "chatcmpl-x", model: "mei-model", created: 1)
        let result = parse(f1 + f2 + f3)
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].name, "add_numbers")
        XCTAssertEqual(result.toolCalls[0].id, "call_1")
        XCTAssertEqual(result.toolCalls[0].arguments, #"{"a":15,"b":27}"#)
        XCTAssertEqual(result.incompleteToolCalls, [])
    }

    func testMultipleToolIndexesStayDistinct() {
        var body = ""
        body += router.sseFrame(
            id: "chatcmpl-x",
            event: .toolCall(index: 0, call: .init(id: "call_a", name: "search_files", argumentsJSON: #"{"q":"a"}"#)),
            model: "mei-model", emitReasoning: true, includeUsage: false)
        body += router.sseFrame(
            id: "chatcmpl-x",
            event: .toolCall(index: 1, call: .init(id: "call_b", name: "read_file", argumentsJSON: #"{"p":"b"}"#)),
            model: "mei-model", emitReasoning: true, includeUsage: false)
        let result = parse(body)
        XCTAssertEqual(result.toolCalls.map(\.index), [0, 1])
        XCTAssertEqual(result.toolCalls.map(\.name), ["search_files", "read_file"])
        XCTAssertEqual(result.toolCalls.map(\.id), ["call_a", "call_b"])
        XCTAssertEqual(result.toolCalls[0].arguments, #"{"q":"a"}"#)
    }

    func testFragmentedFramesAcrossIndexesDoNotCrossMerge() {
        // Two calls whose fragments interleave must still merge cleanly per
        // index (a regression for the old single-index serializer).
        let a1 = Router.toolCallSSEData(
            call: .init(id: "call_a", name: "search_", argumentsJSON: "{\"q\":\""), index: 0,
            id: "chatcmpl-x", model: "mei-model", created: 1)
        let b1 = Router.toolCallSSEData(
            call: .init(id: "call_b", name: "read_", argumentsJSON: "{\"p\":\""), index: 1,
            id: "chatcmpl-x", model: "mei-model", created: 1)
        let a2 = Router.toolCallSSEData(
            call: .init(id: "call_a", name: "files", argumentsJSON: "mei\"}"), index: 0,
            id: "chatcmpl-x", model: "mei-model", created: 1)
        let b2 = Router.toolCallSSEData(
            call: .init(id: "call_b", name: "file", argumentsJSON: "/tmp/x\"}"), index: 1,
            id: "chatcmpl-x", model: "mei-model", created: 1)
        let result = parse(a1 + b1 + a2 + b2)
        XCTAssertEqual(result.toolCalls.count, 2)
        XCTAssertEqual(result.toolCalls[0].name, "search_files")
        XCTAssertEqual(result.toolCalls[0].arguments, #"{"q":"mei"}"#)
        XCTAssertEqual(result.toolCalls[1].name, "read_file")
        XCTAssertEqual(result.toolCalls[1].arguments, #"{"p":"/tmp/x"}"#)
    }

    func testInterleavedToolCallAndContentFrames() {
        var body = ""
        body += router.sseFrame(id: "chatcmpl-x", event: .chunk("I will "), model: "mei-model", emitReasoning: true, includeUsage: false)
        body += router.sseFrame(id: "chatcmpl-x", event: .toolCall(index: 0, call: .init(id: "c1", name: "f", argumentsJSON: "{}")), model: "mei-model", emitReasoning: true, includeUsage: false)
        body += router.sseFrame(id: "chatcmpl-x", event: .chunk("help."), model: "mei-model", emitReasoning: true, includeUsage: false)
        let result = parse(body)
        XCTAssertEqual(result.content, "I will help.")
        XCTAssertEqual(result.toolCalls.count, 1)
        XCTAssertEqual(result.toolCalls[0].name, "f")
    }

    // MARK: - Malformed and incomplete frames

    func testMalformedFrameRecordedAndSkipped() {
        let good = router.sseFrame(id: "chatcmpl-x", event: .chunk("ok"), model: "mei-model", emitReasoning: true, includeUsage: false)
        let body = good + "data: {this is not json\n\n" + good
        let result = parse(body)
        XCTAssertEqual(result.content, "okok", "well-formed frames around a malformed frame still assemble")
        XCTAssertEqual(result.malformedFrames.count, 1)
        XCTAssertTrue(result.malformedFrames[0].contains("not json"))
    }

    func testNonDataLinesIgnored() {
        let good = router.sseFrame(id: "chatcmpl-x", event: .chunk("ok"), model: "mei-model", emitReasoning: true, includeUsage: false)
        let result = parse(": comment\n\n" + good + "event: message\n\n")
        XCTAssertEqual(result.content, "ok")
        XCTAssertEqual(result.malformedFrames, [])
    }

    func testIncompleteToolArgumentsFlagged() {
        var body = ""
        body += router.sseFrame(id: "chatcmpl-x", event: .toolCall(index: 0, call: .init(id: "c1", name: "f", argumentsJSON: #"{"a":1"#)), model: "mei-model", emitReasoning: true, includeUsage: false)
        body += router.sseFrame(id: "chatcmpl-x", event: .finish(usageRun()), model: "mei-model", emitReasoning: true, includeUsage: false)
        let result = parse(body)
        XCTAssertEqual(result.incompleteToolCalls, [0], "arguments that never close are flagged so a caller can resend")
    }

    func testMissingTerminalDoneCanStillAssemble() {
        let body = router.sseFrame(id: "chatcmpl-x", event: .chunk("partial"), model: "mei-model", emitReasoning: true, includeUsage: false)
        let result = parse(body)
        XCTAssertEqual(result.content, "partial")
        XCTAssertFalse(result.sawDone)
    }

    // MARK: - Error stream contract

    func testErrorStreamFrameShape() throws {
        let frame = router.errorSSEFrame(message: "generation exploded")
        XCTAssertTrue(frame.hasPrefix("data: "), frame)
        XCTAssertTrue(frame.hasSuffix("\n\n"), frame)
        let payload = String(frame.dropFirst("data: ".count).dropLast(2))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
        let error = try XCTUnwrap(object["error"] as? [String: Any])
        XCTAssertEqual(error["message"] as? String, "generation exploded")
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertEqual(error["code"] as? String, "stream_error")
        XCTAssertNil(object["choices"], "the error frame carries no choices")
        XCTAssertNil(object["usage"], "the error frame carries no usage")
    }

    func testErrorStreamTerminalFrameIsNotDone() {
        // An errored stream ends AT the error frame: no finish frame, no
        // usage chunk, no [DONE]. [DONE] is the success terminator only —
        // requiring it after an error frame would falsely signal success.
        let frame = router.errorSSEFrame(message: "generation exploded")
        XCTAssertFalse(frame.contains("[DONE]"))
        XCTAssertFalse(frame.contains("finish_reason"))
        XCTAssertFalse(frame.contains("\"usage\""))
    }

    func testErrorFrameAndSuccessTerminationAreDistinct() {
        // Success always terminates with [DONE] (finishSSEData, with or
        // without include_usage); failure terminates with the error frame and
        // never with [DONE]. The two terminal shapes must not overlap.
        let success = Router.finishSSEData(
            id: "chatcmpl-x", run: usageRun(), model: "mei-model",
            created: 1_700_000_000, includeUsage: false)
        XCTAssertEqual(success.last, "[DONE]")
        XCTAssertTrue(success.dropLast().allSatisfy { $0.contains("finish_reason") })
        let failure = router.errorSSEFrame(message: "generation exploded")
        XCTAssertFalse(failure.contains("[DONE]"))
    }
}