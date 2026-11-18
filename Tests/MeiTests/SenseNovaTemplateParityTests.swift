import Foundation
import MLXHuggingFace
import MLXLMCommon
import VMLXTokenizers
import XCTest
@testable import MeiCore

final class SenseNovaTemplateParityTests: XCTestCase {
    func testPinnedSwiftRenderingMatchesIndependentAuthoredTokenIDs() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["SENSENOVA_MODEL_DIR"],
              let protocolPath = environment["SENSENOVA_TEMPLATE_PROTOCOL"] else {
            throw XCTSkip("Requires the independently frozen release template protocol")
        }
        struct ProtocolValue: Decodable {
            struct Case: Decodable {
                let name: String
                let messages: [[String: String]]
                let ids: [Int]
            }
            let cases: [Case]
        }
        let value = try JSONDecoder().decode(ProtocolValue.self,
            from: Data(contentsOf: URL(fileURLWithPath: protocolPath)))
        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: URL(fileURLWithPath: modelPath))
        for item in value.cases {
            let payload = try JSONSerialization.data(withJSONObject: ["model": "synthetic", "messages": item.messages])
            let request = try ChatRequest(json: payload)
            let messages = request.messages.map { MessageMapping.templateDictionary(from: $0) }
            let actual = try tokenizer.applyChatTemplate(messages: messages, tools: nil,
                additionalContext: ["enable_thinking": false])
            XCTAssertEqual(actual, item.ids, item.name)
        }
    }
}
