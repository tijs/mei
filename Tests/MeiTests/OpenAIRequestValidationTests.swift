import XCTest
@testable import MeiCore

/// Deterministic coverage of the P0 request boundary: decoding, required
/// fields, role/content-part rules, max_tokens vs max_completion_tokens
/// policy, stream_options, tools/tool_choice validation, sampling ranges, and
/// loud rejection of deferred platform fields.
final class OpenAIRequestValidationTests: XCTestCase {

    private func chat(_ json: String) throws -> ChatRequest {
        try ChatRequest(json: Data(json.utf8))
    }

    private func completion(_ json: String) throws -> CompletionRequest {
        try CompletionRequest(json: Data(json.utf8))
    }

    private enum ErrKind {
        case invalidBody, invalidField, conflict, deferred
    }

    private static func kind(of error: APIRequestError) -> ErrKind {
        switch error {
        case .invalidBody: return .invalidBody
        case .invalidField: return .invalidField
        case .conflict: return .conflict
        case .deferred: return .deferred
        }
    }

    private func assertThrows<T>(
        _ expression: @autoclosure () throws -> T,
        kind expected: ErrKind,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            _ = try expression()
            XCTFail("expected APIRequestError.\(expected) but request decoded", file: file, line: line)
        } catch let error as APIRequestError {
            if Self.kind(of: error) != expected {
                XCTFail("wrong error kind: \(error)", file: file, line: line)
            }
        } catch {
            XCTFail("wrong error type: \(error)", file: file, line: line)
        }
    }

    private func message(_ role: String, _ content: String) -> String {
        #"{"role": "\#(role)", "content": "\#(content)"}"#
    }

    // MARK: - Required fields

    func testModelIsRequired() {
        assertThrows(try chat(#"{"messages": [{"role": "user", "content": "hi"}]}"#), kind: .invalidField)
        assertThrows(try chat(#"{"model": "", "messages": [{"role": "user", "content": "hi"}]}"#), kind: .invalidField)
    }

    func testMessagesRequiredAndNotEmpty() {
        assertThrows(try chat(#"{"model": "m"}"#), kind: .invalidField)
        assertThrows(try chat(#"{"model": "m", "messages": []}"#), kind: .invalidField)
    }

    func testMalformedJSONBody() {
        assertThrows(try chat("not json"), kind: .invalidBody)
        assertThrows(try chat(#"{"model": "m", "messages": [{"role": "user", "content": 42}]}"#), kind: .invalidField)
        assertThrows(try completion(#"{"model": "m", "prompt": 42}"#), kind: .invalidBody)
    }

    // MARK: - Roles and content parts

    func testSupportedRolesDecode() throws {
        let json = #"""
        {"model": "m", "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"}
        ]}
        """#
        let request = try chat(json)
        XCTAssertEqual(request.messages.map(\.role), ["system", "user", "assistant", "user", "assistant", "tool"])
    }

    func testDeveloperRoleIsDeferred() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "developer", "content": "x"}]}"#),
            kind: .deferred)
    }

    func testUnknownRoleRejected() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "goblin", "content": "x"}]}"#),
            kind: .invalidField)
    }

    func testContentRequiredForSystemUserAndTool() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "system", "content": null}]}"#),
            kind: .invalidField)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user"}]}"#),
            kind: .invalidField)
        assertThrows(
            try chat(#"{"model": "m", "messages": [                {"role": "assistant", "content": null, "tool_calls": [                    {"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},                {"role": "tool", "tool_call_id": "c1", "content": null}]}"#),
            kind: .invalidField)
    }

    func testAssistantNeedsContentOrToolCalls() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "assistant", "content": null}]}"#),
            kind: .invalidField)
    }

    func testToolRoleRequiresToolCallID() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "tool", "content": "42"}]}"#),
            kind: .invalidField)
    }

    func testToolCallsOnlyOnAssistant() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [                {"role": "user", "content": "x", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]}]}"#),
            kind: .invalidField)
    }

    func testToolCallWithoutFunctionNameRejected() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1"}]}]}"#),
            kind: .invalidField)
    }

    func testContentPartArrayJoinsText() throws {
        let request = try chat(#"{"model": "m", "messages": [            {"role": "user", "content": [                {"type": "text", "text": "one"},                {"type": "text", "text": "two"}]}]}"#)
        XCTAssertEqual(request.messages[0].content, "one\ntwo")
    }

    func testNonTextContentPartIsDeferred() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [                {"role": "user", "content": [                    {"type": "text", "text": "ok"},                    {"type": "image_url", "image_url": {"url": "x"}}]}]}"#),
            kind: .deferred)
        assertThrows(
            try chat(#"{"model": "m", "messages": [                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}}]}]}"#),
            kind: .deferred)
    }

    func testTextPartWithoutTextRejected() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": [{"type": "text"}]}]}"#),
            kind: .invalidField)
    }

    // MARK: - max_tokens / max_completion_tokens policy

    func testMaxCompletionTokensAliasAlone() throws {
        let request = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 128}"#)
        XCTAssertEqual(request.maxTokens, 128)
    }

    func testMaxTokensAloneStillWorks() throws {
        let request = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}"#)
        XCTAssertEqual(request.maxTokens, 64)
    }

    func testEqualMaxTokensFormsAccepted() throws {
        let request = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64, "max_completion_tokens": 64}"#)
        XCTAssertEqual(request.maxTokens, 64)
    }

    func testConflictingMaxTokensFormsRejected() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64, "max_completion_tokens": 128}"#),
            kind: .conflict)
    }

    func testZeroOrNegativeMaxTokensRejected() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}"#),
            kind: .invalidField)
        assertThrows(
            try completion(#"{"model": "m", "prompt": "hi", "max_completion_tokens": -1}"#),
            kind: .invalidField)
    }

    // MARK: - Sampling range validation

    func testTemperatureRange() {
        assertThrows(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": -0.5}"#), kind: .invalidField)
        assertThrows(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 2.5}"#), kind: .invalidField)
        _ = try? chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 2}"#)
    }

    func testTopPRange() {
        assertThrows(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "top_p": 1.5}"#), kind: .invalidField)
        _ = try? chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "top_p": 0.9}"#)
    }

    func testMinPRange() {
        assertThrows(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "min_p": -0.1}"#), kind: .invalidField)
        _ = try? chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "min_p": 0.05}"#)
    }

    func testTopKPositive() {
        assertThrows(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "top_k": 0}"#), kind: .invalidField)
        _ = try? chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "top_k": 20}"#)
    }

    func testPenaltyRanges() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "presence_penalty": 3}"#),
            kind: .invalidField)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "frequency_penalty": -3}"#),
            kind: .invalidField)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "repetition_penalty": -1}"#),
            kind: .invalidField)
        _ = try? chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "presence_penalty": -2, "frequency_penalty": 2, "repetition_penalty": 1.1}"#)
    }

    // MARK: - stop, seed, reasoning_effort

    func testStopStringOrArray() throws {
        XCTAssertEqual(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "stop": "END"}"#).stop, ["END"])
        XCTAssertEqual(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "stop": ["a", "b"]}"#).stop, ["a", "b"])
        XCTAssertNil(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}]}"#).stop)
    }

    func testSeedDecodes() throws {
        XCTAssertEqual(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "seed": 42}"#).seed, 42)
        XCTAssertNil(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}]}"#).seed)
    }

    func testReasoningEffortValues() throws {
        XCTAssertEqual(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "reasoning_effort": "low"}"#).reasoningEffort, "low")
        XCTAssertEqual(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "reasoning_effort": "none"}"#).reasoningEffort, "none")
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "reasoning_effort": "ultra"}"#),
            kind: .invalidField)
    }

    // MARK: - stream_options

    func testStreamOptionsIncludeUsage() throws {
        let on = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": true, "stream_options": {"include_usage": true}}"#)
        XCTAssertTrue(on.includeUsage)
        let off = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": true, "stream_options": {"include_usage": false}}"#)
        XCTAssertFalse(off.includeUsage)
        XCTAssertFalse(try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": true}"#).includeUsage)
    }

    func testStreamOptionsWithoutStreamRejected() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "stream_options": {"include_usage": true}}"#),
            kind: .invalidField)
    }

    func testStreamOptionsUnknownKeyRejected() {
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": true, "stream_options": {"continuous_usage_stats": true}}"#),
            kind: .invalidField)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": true, "stream_options": {"include_usage": "yes"}}"#),
            kind: .invalidField)
    }

    // MARK: - tools / tool_choice

    func testToolsValidation() throws {
        let good = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": [            {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}"#)
        XCTAssertEqual(good.tools?.count, 1)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": [{"type": "code_interpreter"}]}"#),
            kind: .deferred)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {}}]}"#),
            kind: .invalidField)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": ["not-an-object"]}"#),
            kind: .invalidField)
    }

    func testToolChoiceValidation() throws {
        for value in ["auto", "none", "required"] {
            let request = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": "\#(value)"}"#)
            XCTAssertEqual(request.toolChoice, .string(value))
        }
        // A bare function name is the legacy forced-tool form (documented).
        _ = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": "search_files"}"#)
        // The OpenAI object form.
        _ = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": {"type": "function", "function": {"name": "add_numbers"}}}"#)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": ""}"#),
            kind: .invalidField)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": {"type": "function"}}"#),
            kind: .invalidField)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": {"type": "function", "function": {"name": ""}}}"#),
            kind: .invalidField)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": [1, 2]}"#),
            kind: .invalidField)
    }

    // MARK: - Deferred platform fields

    func testDeferredChatFieldsRejected() {
        // (field name, raw JSON value to attach to it)
        let payloads: [(String, String)] = [
            ("response_format", #"{"type": "json_object"}"#),
            ("logprobs", "true"),
            ("n", "2"),
            ("prediction", #"{"type": "content", "content": "x"}"#),
            ("store", "true"),
            ("metadata", #"{"k": "v"}"#),
            ("service_tier", #""auto""#),
            ("modalities", #"["text"]"#),
            ("audio", #"{"voice": "alloy", "format": "wav"}"#),
            ("parallel_tool_calls", "false"),
            ("functions", #"[{"name": "f"}]"#),
            ("function_call", #""f""#),
        ]
        for (fieldName, value) in payloads {
            do {
                _ = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "\#(fieldName)": \#(value)}"#)
                XCTFail("\(fieldName) must be rejected")
            } catch let error as APIRequestError {
                guard case .deferred(let message) = error else {
                    return XCTFail("\(fieldName): expected deferred, got \(error)")
                }
                XCTAssertTrue(message.contains(fieldName), "deferred message should name \(fieldName): \(message)")
            } catch {
                XCTFail("\(fieldName): unexpected error \(error)")
            }
        }
    }

    func testNAllowedOnlyWhenOne() throws {
        _ = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 1}"#)
        assertThrows(
            try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 0}"#),
            kind: .deferred)
    }

    func testParallelToolCallsTrueAllowed() throws {
        _ = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "parallel_tool_calls": true}"#)
    }

    func testUserFieldIsIgnored() throws {
        // `user` is a platform tracking id with no inference semantics; the
        // contract documents it as accepted-and-ignored, so it must not throw.
        _ = try chat(#"{"model": "m", "messages": [{"role": "user", "content": "x"}], "user": "abc-123"}"#)
    }

    // MARK: - Legacy /v1/completions

    func testCompletionRequiredFields() {
        assertThrows(try completion(#"{"prompt": "hi"}"#), kind: .invalidField)
        assertThrows(try completion(#"{"model": "m"}"#), kind: .invalidField)
        _ = try? completion(#"{"model": "m", "prompt": "hi"}"#)
    }

    func testCompletionDeferredFieldsRejected() {
        for field in ["echo", "suffix", "best_of", "logprobs", "top_logprobs", "stream_options"] {
            do {
                _ = try completion(#"{"model": "m", "prompt": "hi", "\#(field)": true}"#)
                XCTFail("\(field) must be rejected")
            } catch let error as APIRequestError {
                guard case .deferred = error else {
                    return XCTFail("\(field): expected deferred, got \(error)")
                }
            } catch {
                XCTFail("\(field): unexpected error \(error)")
            }
        }
    }

    func testCompletionStreamRejectedAtRouteLevelOnly() throws {
        // Decoding accepts the field (legacy OpenAI allows stream=true); the
        // ROUTER rejects it with a 400 before any generation. Decode must not
        // throw here — the route-level guard is covered by RouterSSEFrameTests
        // and the live probe.
        let request = try completion(#"{"model": "m", "prompt": "hi", "stream": true}"#)
        XCTAssertTrue(request.stream)
    }

    func testCompletionMaxCompletionTokensAlias() throws {
        XCTAssertEqual(try completion(#"{"model": "m", "prompt": "hi", "max_completion_tokens": 32}"#).maxTokens, 32)
        assertThrows(
            try completion(#"{"model": "m", "prompt": "hi", "max_tokens": 16, "max_completion_tokens": 32}"#),
            kind: .conflict)
    }

    func testReasoningContentExtensionParity() throws {
        // Mei's reasoning_content extension on assistant messages must round
        // trip through decoding.
        let request = try chat(#"{"model": "m", "messages": [            {"role": "assistant", "content": "answer", "reasoning_content": "thought"}]}"#)
        XCTAssertEqual(request.messages[0].reasoningContent, "thought")
    }
}