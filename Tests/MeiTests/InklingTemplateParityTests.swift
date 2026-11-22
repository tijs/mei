import Foundation
import MLXHuggingFace
import MLXLMCommon
import VMLXTokenizers
import XCTest
@testable import MeiCore

final class InklingTemplateParityTests: XCTestCase {
    func testAuthoredTemplateTokenIDsForTextReasoningAndToolContinuation() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let model = env["INKLING_MODEL_DIR"], let fixture = env["INKLING_TEMPLATE_PROTOCOL"] else {
            throw XCTSkip("Requires independently rendered pinned Inkling tokenizer/template cases")
        }
        let value = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: fixture))) as! [String: Any]
        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: URL(fileURLWithPath: model))
        for item in value["cases"] as! [[String: Any]] {
            var messages = item["messages"] as! [[String: Any]]
            // OpenAI transports arguments as JSON text. Mei must restore the
            // parsed object required by the model's authored native template.
            for index in messages.indices {
                if var calls = messages[index]["tool_calls"] as? [[String: Any]] {
                    for j in calls.indices {
                        var function = calls[j]["function"] as! [String: Any]
                        if let arguments = function["arguments"] as? [String: Any] {
                            function["arguments"] = String(decoding: try JSONSerialization.data(withJSONObject: arguments), as: UTF8.self)
                        }
                        calls[j]["function"] = function
                    }
                    messages[index]["tool_calls"] = calls
                }
            }
            var payload: [String: Any] = ["model": "inkling-template-test", "messages": messages]
            if let tools = item["tools"] as? [[String: Any]] { payload["tools"] = tools }
            let request = try ChatRequest(json: JSONSerialization.data(withJSONObject: payload))
            let actual = try tokenizer.applyChatTemplate(
                messages: request.messages.map { MessageMapping.templateDictionary(from: $0) },
                tools: MessageMapping.templateTools(request.tools), additionalContext: ["reasoning_effort": "none"])
            let expected = (item["ids"] as! [NSNumber]).map { $0.intValue }
            XCTAssertEqual(actual, expected, item["name"] as! String)
        }
    }
}
