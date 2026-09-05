import XCTest

@testable import MeiCore

/// Regression guard for the streaming multi-tool-call corruption: Mei's SSE
/// mapper previously hard-coded every tool-call delta at `index: 0`. OpenAI
/// streaming clients merge `tool_calls` deltas keyed by that index, so a
/// generation that returned two distinct tool calls produced one corrupted
/// call whose name and arguments were concatenated (observed in the wild as
/// `search_filessearch_files`, `search_filessearch_filessearch_files`,
/// `search_filesterminal`, and `read_fileread_file`).
///
/// The seam under test is the real `Router` tool-call → SSE-frame serializer
/// (the exact JSON that goes over the wire), driven the same way the
/// streaming HTTP handler drives it per emitted `.toolCall` event.
final class RouterSSEToolCallIndexingTests: XCTestCase {

    /// Strip the `data: ` SSE prefix and decode the frame's JSON body.
    private func decodeFrame(_ sseFrame: String) throws -> [String: Any] {
        let line = sseFrame.split(separator: "\n").first.map(String.init) ?? ""
        var encoded = line.hasPrefix("data:") ? String(line.dropFirst(5)) : line
        encoded = encoded.trimmingCharacters(in: .whitespacesAndNewlines)
        let object = try JSONSerialization.jsonObject(with: Data(encoded.utf8)) as? [String: Any]
        return try XCTUnwrap(object, "frame did not decode to JSON: \(sseFrame)")
    }

    /// Collect `tool_calls` deltas across SSE frames, merging by OpenAI's
    /// `index` key exactly like a streaming client would.
    private func reassemble(
        _ frames: [String]
    ) throws -> [Int: (id: String?, name: String, arguments: String)] {
        var calls: [Int: (id: String?, name: String, arguments: String)] = [:]
        for frame in frames {
            let json = try decodeFrame(frame)
            for choice in (json["choices"] as? [[String: Any]]) ?? [] {
                let delta = (choice["delta"] as? [String: Any]) ?? [:]
                for part in (delta["tool_calls"] as? [[String: Any]]) ?? [] {
                    let index = (part["index"] as? Int) ?? -1
                    let function = (part["function"] as? [String: Any]) ?? [:]
                    let name = (function["name"] as? String) ?? ""
                    let arguments = (function["arguments"] as? String) ?? ""
                    if let existing = calls[index] {
                        calls[index] = (
                            existing.id,
                            existing.name + name,
                            existing.arguments + arguments)
                    } else {
                        calls[index] = ((part["id"] as? String), name, arguments)
                    }
                }
            }
        }
        return calls
    }

    /// A generation that emits two distinct tool calls must serialize those
    /// calls under distinct OpenAI streaming indexes, so a client merging by
    /// index recovers two separate calls instead of one concatenated blob.
    func testTwoToolCallEventsGetDistinctStreamingIndexes() throws {
        let search = GenerationRun.ToolCallEmitting(
            id: "call_search", name: "search_files", argumentsJSON: #"{"query":"mei"}"#)
        let read = GenerationRun.ToolCallEmitting(
            id: "call_read", name: "read_file", argumentsJSON: #"{"path":"/tmp/x"}"#)

        // The streaming handler emits one `.toolCall` event per call; each is
        // serialized through the same Router seam with its own index.
        let frameSearch = Router.toolCallSSEData(
            call: search, index: 0, id: "chatcmpl-test", model: "mei-model", created: 1)
        let frameRead = Router.toolCallSSEData(
            call: read, index: 1, id: "chatcmpl-test", model: "mei-model", created: 1)

        let calls = try reassemble([frameSearch, frameRead])

        XCTAssertEqual(
            calls.count, 2,
            "two tool calls must occupy two distinct indexes (was: both merged at 0) — got \(calls)")
        XCTAssertEqual(calls[0]?.name, "search_files")
        XCTAssertEqual(calls[0]?.id, "call_search")
        XCTAssertEqual(calls[1]?.name, "read_file")
        XCTAssertEqual(calls[1]?.id, "call_read")
    }

    /// Single-call streaming must keep index 0 with id + arguments intact.
    func testSingleToolCallKeepsIndexZeroWithIdAndArguments() throws {
        let call = GenerationRun.ToolCallEmitting(
            id: "call_add", name: "add_numbers", argumentsJSON: #"{"a":15,"b":27}"#)
        let frame = Router.toolCallSSEData(
            call: call, index: 0, id: "chatcmpl-test", model: "mei-model", created: 1)

        let calls = try reassemble([frame])
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0]?.name, "add_numbers")
        XCTAssertEqual(calls[0]?.id, "call_add")
        XCTAssertEqual(calls[0]?.arguments, #"{"a":15,"b":27}"#)
    }
}