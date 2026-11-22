import Foundation
import XCTest
@testable import MeiCore

final class InklingToolArgumentStreamTests: XCTestCase {
    private func tools(type: String = "string") -> [MeiJSONValue] {
        [.object(["type": .string("function"), "function": .object([
            "name": .string("write_script"), "parameters": .object([
                "type": .string("object"), "additionalProperties": .bool(false),
                "properties": .object(["source": .object(["type": .string(type)])])
            ])
        ])])]
    }
    private func raw(_ args: String) -> String {
        "<|message_model|>write_script<|content_invoke_tool_json|>{\"name\":\"write_script\",\"args\":"
            + args + "}<|end_message|>"
    }
    func testLongCodeStreamsActualArgumentsBeforeEnvelopeCloses() throws {
        let source = String(repeating: "print(\"荷😺 <|end_message|> }{ \\\"\")\n", count: 100)
        let args = try MeiJSONValue.object(["source": .string(source)]).jsonString()
        var stream = InklingToolArgumentStream(tools: tools())
        var deltas = [SSEChatChunk.DeltaToolCall]()
        let text = raw(args)
        for character in text.dropLast("<|end_message|>".count) {
            if let delta = stream.feed(String(character), index: 0) { deltas.append(delta) }
        }
        XCTAssertGreaterThan(deltas.count, 100)
        XCTAssertEqual(deltas.compactMap(\.id).count, 1)
        XCTAssertEqual(deltas.compactMap { $0.function?.name }, ["write_script"])
        XCTAssertEqual(deltas.compactMap { $0.function?.arguments }.joined(), args)
        XCTAssertNil(stream.feed("<|end_message|>", index: 0))
        XCTAssertEqual(try stream.finish(name: "write_script", argumentsJSON: args), deltas.first?.id)
    }
    func testEveryChunkBoundaryReassemblesThroughRealSSEFrames() throws {
        let args = try MeiJSONValue.object(["source": .string("a = '\"}荷'\nprint(a)\\")]).jsonString()
        let text = raw(args)
        for width in 1...text.count {
            var stream = InklingToolArgumentStream(tools: tools())
            var cursor = text.startIndex, assembled = "", name = "", ids = [String]()
            while cursor < text.endIndex {
                let end = text.index(cursor, offsetBy: width, limitedBy: text.endIndex) ?? text.endIndex
                if let delta = stream.feed(String(text[cursor..<end]), index: 2) {
                    let frame = Router.toolCallDeltaSSEData(delta: delta, id: "test", model: "inkling", created: 1)
                    let data = Data(frame.dropFirst(6).trimmingCharacters(in: .whitespacesAndNewlines).utf8)
                    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
                    let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
                    let wireDelta = try XCTUnwrap(choices[0]["delta"] as? [String: Any])
                    let calls = try XCTUnwrap(wireDelta["tool_calls"] as? [[String: Any]])
                    XCTAssertEqual(calls[0]["index"] as? Int, 2)
                    let function = try XCTUnwrap(calls[0]["function"] as? [String: Any])
                    assembled += function["arguments"] as? String ?? ""
                    name += function["name"] as? String ?? ""
                    if let id = calls[0]["id"] as? String { ids.append(id) }
                }
                cursor = end
            }
            XCTAssertEqual(assembled, args)
            XCTAssertEqual(name, "write_script")
            XCTAssertEqual(ids.count, 1)
            XCTAssertEqual(try stream.finish(name: name, argumentsJSON: args), ids.first)
        }
    }
    func testDistinctCallsResetIDsAndIndexes() throws {
        var stream = InklingToolArgumentStream(tools: tools())
        let a = try XCTUnwrap(stream.feed(raw("{\"source\":\"first\"}"), index: 0))
        XCTAssertEqual(try stream.finish(name: "write_script", argumentsJSON: "{\"source\":\"first\"}"), a.id)
        let b = try XCTUnwrap(stream.feed(raw("{\"source\":\"second\"}"), index: 1))
        XCTAssertNotEqual(a.id, b.id)
        XCTAssertEqual(a.index, 0); XCTAssertEqual(b.index, 1)
        XCTAssertEqual(try stream.finish(name: "write_script", argumentsJSON: "{\"source\":\"second\"}"), b.id)
    }
    func testNumericSchemasAndUnknownToolsKeepBufferedPath() throws {
        var numeric = InklingToolArgumentStream(tools: tools(type: "integer"))
        XCTAssertNil(numeric.feed(raw("{\"source\":\"42\"}"), index: 0))
        XCTAssertNil(try numeric.finish(name: "write_script", argumentsJSON: "{\"source\":42}"))
        var unavailable = InklingToolArgumentStream(tools: [])
        XCTAssertNil(unavailable.feed(raw("{\"source\":\"x\"}"), index: 0))
        XCTAssertNil(try unavailable.finish(name: "write_script", argumentsJSON: "{\"source\":\"x\"}"))
    }
    func testIncompleteAndMismatchedArgumentsFailCompletion() throws {
        var stream = InklingToolArgumentStream(tools: tools())
        XCTAssertNotNil(stream.feed("<|message_model|>write_script<|content_invoke_tool_json|>{\"name\":\"write_script\",\"args\":{\"source\":\"unfinished", index: 0))
        XCTAssertThrowsError(try stream.finish(name: "write_script", argumentsJSON: "{\"source\":\"unfinished\"}"))
        XCTAssertNotNil(stream.feed(raw("{\"source\":\"original\"}"), index: 0))
        XCTAssertThrowsError(try stream.finish(name: "write_script", argumentsJSON: "{\"source\":\"different\"}"))
    }
    func testArgsBeforeNameAndLiteralMarkerRemainJSONData() throws {
        var stream = InklingToolArgumentStream(tools: tools())
        let args = "{\"source\":\"literal <|end_message|> and { braces }\"}"
        let input = "<|message_model|>write_script<|content_invoke_tool_json|>{\"args\":" + args + ",\"name\":\"write_script\"}<|end_message|>"
        let delta = try XCTUnwrap(stream.feed(input, index: 0))
        XCTAssertEqual(delta.function?.arguments, args)
        XCTAssertEqual(try stream.finish(name: "write_script", argumentsJSON: args), delta.id)
    }
}
