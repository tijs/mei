import XCTest
@testable import MeiCore

/// Deterministic coverage of the chat-template mapping seam
/// (`MessageMapping`): message roles, assistant tool_calls (dual view),
/// tool messages, reasoning pass-through, and the shipped `tool_choice`
/// degradation — a bare-name string or an object with a TOP-LEVEL `name` maps
/// to `tool_choice:"required"` + `tool_choice_name`; the official nested
/// `{"type":"function","function":{"name":X}}` form degrades to
/// `tool_choice:"required"` WITHOUT extracting the nested name (§9.2 of the
/// compatibility contract). Pure — no engine, no model.
final class MessageMappingTests: XCTestCase {

    private func mapping(_ message: APIMessage) -> [String: any Sendable] {
        MessageMapping.templateDictionary(from: message)
    }

    private func context(
        enableThinking: Bool? = nil,
        reasoningEffort: String? = nil,
        toolChoice: MeiJSONValue? = nil
    ) -> [String: any Sendable]? {
        MessageMapping.additionalContext(
            enableThinking: enableThinking, reasoningEffort: reasoningEffort, toolChoice: toolChoice)
    }

    // MARK: - tool_choice mapping

    func testKeywordToolChoicesPassThrough() {
        for keyword in ["auto", "none", "required"] {
            let context = context(toolChoice: .string(keyword))
            XCTAssertEqual(context?["tool_choice"] as? String, keyword, keyword)
            XCTAssertNil(context?["tool_choice_name"], "keyword forms must not name a tool")
        }
    }

    func testBareFunctionNameForcesThatTool() {
        let context = context(toolChoice: .string("search_files"))
        XCTAssertEqual(context?["tool_choice"] as? String, "required")
        XCTAssertEqual(context?["tool_choice_name"] as? String, "search_files")
    }

    func testObjectFormWithTopLevelNameExtractsName() {
        // The legacy bare-name object shape used by some clients.
        let context = context(toolChoice: .object(["name": .string("add_numbers")]))
        XCTAssertEqual(context?["tool_choice"] as? String, "required")
        XCTAssertEqual(context?["tool_choice_name"] as? String, "add_numbers")
    }

    func testNestedObjectFormDegradesToRequiredWithoutExtractingName() {
        // The OFFICIAL OpenAI forced-tool shape. Shipped behavior: the nested
        // function.name is not extracted, so the request degrades to
        // tool_choice="required" (any tool). Pinned so the degradation cannot
        // silently change under refactors.
        let nested = MeiJSONValue.object([
            "type": .string("function"),
            "function": .object(["name": .string("add_numbers")]),
        ])
        let context = context(toolChoice: nested)
        XCTAssertEqual(context?["tool_choice"] as? String, "required")
        XCTAssertNil(
            context?["tool_choice_name"],
            "the nested function.name must NOT be extracted (documented degradation)")
    }

    func testNoOverridesProduceNoAdditionalContext() {
        XCTAssertNil(context())
        let withReasoning = context(reasoningEffort: "low")
        XCTAssertEqual(withReasoning?["reasoning_effort"] as? String, "low")
        let withThinking = context(enableThinking: false)
        XCTAssertEqual(withThinking?["enable_thinking"] as? Bool, false)
    }

    // MARK: - Message mapping

    func testToolMessageCarriesToolCallID() {
        let dict = mapping(APIMessage(role: "tool", content: "42", toolCallID: "call_1", toolCalls: nil, reasoningContent: nil))
        XCTAssertEqual(dict["role"] as? String, "tool")
        XCTAssertEqual(dict["content"] as? String, "42")
        XCTAssertEqual(dict["tool_call_id"] as? String, "call_1")
        XCTAssertNil(dict["tool_calls"])
    }

    func testAssistantToolCallsCarryBothViews() throws {
        let dict = mapping(APIMessage(
            role: "assistant", content: nil,
            toolCallID: nil,
            toolCalls: [
                .init(id: "call_1", name: "add_numbers", argumentsJSON: #"{"a":15,"b":27}"#),
            ],
            reasoningContent: "thought"))
        XCTAssertEqual(dict["role"] as? String, "assistant")
        XCTAssertNil(dict["content"], "tool-only assistant messages must not fabricate content")
        XCTAssertEqual(dict["reasoning_content"] as? String, "thought")

        let calls = try XCTUnwrap(dict["tool_calls"] as? [[String: any Sendable]])
        XCTAssertEqual(calls.count, 1)
        let call = calls[0]
        XCTAssertEqual(call["id"] as? String, "call_1")
        XCTAssertEqual(call["type"] as? String, "function")
        XCTAssertEqual(call["name"] as? String, "add_numbers")
        // The argument object must be the parsed JSON, not its string form.
        let arguments = try XCTUnwrap(call["arguments"] as? [String: any Sendable])
        XCTAssertEqual(arguments["a"] as? Double, 15)
        XCTAssertEqual(arguments["b"] as? Double, 27)
        // Templates read either convention; both must be present and identical.
        let functionView = try XCTUnwrap(call["function"] as? [String: any Sendable])
        XCTAssertEqual(functionView["name"] as? String, "add_numbers")
        let functionArguments = try XCTUnwrap(functionView["arguments"] as? [String: any Sendable])
        XCTAssertEqual(functionArguments["a"] as? Double, 15)
    }

    func testSystemMessageMapsPlainly() {
        let dict = mapping(APIMessage(role: "system", content: "sys", toolCallID: nil, toolCalls: nil, reasoningContent: nil))
        XCTAssertEqual(dict["role"] as? String, "system")
        XCTAssertEqual(dict["content"] as? String, "sys")
        XCTAssertNil(dict["tool_calls"])
        XCTAssertNil(dict["tool_call_id"])
    }
}