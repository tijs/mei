import Foundation

/// Streams authored argument bytes for text-only tools. Numeric schemas keep
/// the completed-call path because their arguments may require schema coercion.
/// A partial call is never a completed call: finish validates the whole parsed
/// invocation before the engine can emit its successful terminal frame.
struct InklingToolArgumentStream {
    private let eligibleNames: Set<String>
    private var buffer = ""
    private var emittedBytes = 0
    private var streamedID: String?
    private var recipient: String?
    private var argumentJSON = ""

    init(tools: [MeiJSONValue]?) {
        eligibleNames = Set((tools ?? []).compactMap { tool in
            guard case .object(let object) = tool,
                case .object(let function)? = object["function"],
                case .string(let name)? = function["name"],
                case .object(let schema)? = function["parameters"],
                case .bool(false)? = schema["additionalProperties"],
                case .object(let properties)? = schema["properties"],
                properties.values.allSatisfy({ property in
                    guard case .object(let value) = property else { return false }
                    return value["type"] == .string("string")
                })
            else { return nil }
            return name
        })
    }

    mutating func feed(_ text: String, index: Int) -> SSEChatChunk.DeltaToolCall? {
        let marker = "<|message_model|>"
        if buffer.isEmpty && !text.hasPrefix(marker) && !marker.hasPrefix(text) { return nil }
        buffer += text
        guard buffer.hasPrefix(marker) || marker.hasPrefix(buffer) else {
            buffer = ""
            return nil
        }
        guard let header = buffer.range(of: "<|content_invoke_tool_json|>"),
            buffer.hasPrefix(marker)
        else { return nil }
        let name = String(buffer[buffer.index(buffer.startIndex, offsetBy: marker.count)..<header.lowerBound])
        guard eligibleNames.contains(name) else { return nil }
        let body = Array(buffer[header.upperBound...].utf8)
        guard let range = Self.argumentsRange(body) else { return nil }
        let bytes = Array(body[range])
        guard bytes.count > emittedBytes else { return nil }
        let first = streamedID == nil
        if first { streamedID = "call_" + UUID().uuidString.lowercased() }
        recipient = name
        argumentJSON = String(decoding: bytes, as: UTF8.self)
        let delta = String(decoding: bytes.dropFirst(emittedBytes), as: UTF8.self)
        emittedBytes = bytes.count
        return .init(index: index, id: first ? streamedID : nil,
                     type: first ? "function" : nil,
                     function: .init(name: first ? name : nil, arguments: delta))
    }

    enum StreamError: Error { case completedCallMismatch }

    /// Returns the ID already sent to the client, or nil for a buffered call.
    mutating func finish(name: String, argumentsJSON: String) throws -> String? {
        defer {
            buffer = ""; emittedBytes = 0; streamedID = nil
            recipient = nil; argumentJSON = ""
        }
        guard let streamedID else { return nil }
        guard recipient == name,
            let streamed = MeiJSONValue.parseObject(from: argumentJSON),
            let completed = MeiJSONValue.parseObject(from: argumentsJSON),
            streamed == completed
        else { throw StreamError.completedCallMismatch }
        return streamedID
    }

    /// Locate only the outer envelope's args object, including a partial body.
    /// JSON strings are scanned lexically so braces, escaped quotes and native
    /// marker spellings inside source code never become transport boundaries.
    private static func argumentsRange(_ bytes: [UInt8]) -> Range<Int>? {
        var i = 0
        func whitespace() {
            while i < bytes.count && [9, 10, 13, 32].contains(bytes[i]) { i += 1 }
        }
        func stringEnd(_ start: Int) -> Int? {
            var j = start + 1
            while j < bytes.count {
                if bytes[j] == 92 { j += 2 }
                else if bytes[j] == 34 { return j + 1 }
                else { j += 1 }
            }
            return nil
        }
        whitespace()
        guard i < bytes.count, bytes[i] == 123 else { return nil }
        i += 1
        while i < bytes.count {
            whitespace()
            guard i < bytes.count, bytes[i] == 34, let end = stringEnd(i),
                let key = try? JSONDecoder().decode(String.self, from: Data(bytes[i..<end]))
            else { return nil }
            i = end; whitespace()
            guard i < bytes.count, bytes[i] == 58 else { return nil }
            i += 1; whitespace()
            guard i < bytes.count else { return nil }
            if key == "args" {
                guard bytes[i] == 123 else { return nil }
                let start = i
                var depth = 0
                while i < bytes.count {
                    if bytes[i] == 34 {
                        guard let end = stringEnd(i) else { return start..<bytes.count }
                        i = end; continue
                    }
                    if bytes[i] == 123 { depth += 1 }
                    if bytes[i] == 125 {
                        depth -= 1
                        if depth == 0 { return start..<(i + 1) }
                    }
                    i += 1
                }
                return start..<bytes.count
            }
            // Authored envelopes have a string name before args. Unsupported
            // extra/compound fields fall back to completed-call serialization.
            guard bytes[i] == 34, let end = stringEnd(i) else { return nil }
            i = end; whitespace()
            guard i < bytes.count, bytes[i] == 44 else { return nil }
            i += 1
        }
        return nil
    }
}
