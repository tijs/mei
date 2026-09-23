import Foundation

/// A streaming usage block decoded from an SSE chunk, mirroring the fields
/// the benchmark and tests compare between streamed and non-streamed runs.
/// `cached_tokens` rides inside `prompt_tokens_details`, like the
/// non-streaming usage contract.
public struct SSEUsageReport: Decodable, Equatable, Sendable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int
    public var cachedTokens: Int?
    public var tokensPerSecond: Double?
    public var promptTokensPerSecond: Double?

    private struct Details: Decodable {
        var cachedTokens: Int?
        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }
    }

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case details = "prompt_tokens_details"
        case tokensPerSecond = "tokens_per_second"
        case promptTokensPerSecond = "prompt_tokens_per_second"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        promptTokens = try container.decode(Int.self, forKey: .promptTokens)
        completionTokens = try container.decode(Int.self, forKey: .completionTokens)
        totalTokens = try container.decode(Int.self, forKey: .totalTokens)
        cachedTokens = try container.decodeIfPresent(Details.self, forKey: .details)?.cachedTokens
        tokensPerSecond = try container.decodeIfPresent(Double.self, forKey: .tokensPerSecond)
        promptTokensPerSecond = try container.decodeIfPresent(Double.self, forKey: .promptTokensPerSecond)
    }
}

/// One reassembled streaming tool call: the fragments of all deltas that
/// shared an `index` are concatenated exactly like a streaming client merges
/// them.
public struct SSEDeltaToolCall: Equatable, Sendable {
    public var index: Int
    public var id: String?
    public var name: String
    public var arguments: String
}

/// The deterministic reassembly of one SSE body. Semantics, pinned by tests:
///
/// - only `data:` lines are inspected; any other line is ignored (SSE allows
///   comments and event metadata);
/// - a `data:` line whose payload is not valid JSON is recorded in
///   `malformedFrames` and skipped — the rest of the stream still assembles;
/// - `[DONE]` sets `sawDone` and stops nothing (trailing frames are still
///   parsed, matching a client that already closed its parser);
/// - tool-call fragments are merged by `index`; when the merge completes, any
///   call whose concatenated `arguments` do not parse as JSON (including an
///   empty string — a call that never received its argument fragment) is
///   listed in `incompleteToolCalls`.
public struct SSEAssemblyResult: Equatable, Sendable {
    public var content = ""
    public var reasoning = ""
    public var role: String?
    public var toolCalls: [SSEDeltaToolCall] = []
    public var finishReason: String?
    public var usage: SSEUsageReport?
    public var usageFrameCount = 0
    public var sawDone = false
    public var malformedFrames: [String] = []
    public var incompleteToolCalls: [Int] = []
}

public enum SSEFrameParser {
    public static func parse(_ data: Data) -> SSEAssemblyResult {
        let text = String(data: data, encoding: .utf8) ?? ""
        var result = SSEAssemblyResult()
        var merged: [Int: (id: String?, name: String, arguments: String)] = [:]

        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.hasPrefix("data:") else { continue }
            let encoded = line.dropFirst(5).trimmingCharacters(in: .whitespacesAndNewlines)
            if encoded == "[DONE]" {
                result.sawDone = true
                continue
            }
            guard let event = try? JSONSerialization.jsonObject(with: Data(encoded.utf8)) as? [String: Any] else {
                result.malformedFrames.append(line)
                continue
            }
            if let usage = event["usage"] as? [String: Any],
                let data = try? JSONSerialization.data(withJSONObject: usage),
                let report = try? JSONDecoder().decode(SSEUsageReport.self, from: data)
            {
                result.usage = report
                result.usageFrameCount += 1
            }
            for choice in event["choices"] as? [[String: Any]] ?? [] {
                let delta = choice["delta"] as? [String: Any] ?? [:]
                if let role = delta["role"] as? String { result.role = role }
                if let content = delta["content"] as? String { result.content += content }
                if let reasoning = delta["reasoning_content"] as? String { result.reasoning += reasoning }
                for part in delta["tool_calls"] as? [[String: Any]] ?? [] {
                    let index = (part["index"] as? Int) ?? -1
                    let function = part["function"] as? [String: Any] ?? [:]
                    let name = (function["name"] as? String) ?? ""
                    let arguments = (function["arguments"] as? String) ?? ""
                    if var existing = merged[index] {
                        existing.name += name
                        existing.arguments += arguments
                        merged[index] = existing
                    } else {
                        merged[index] = ((part["id"] as? String), name, arguments)
                    }
                }
                if let finishReason = choice["finish_reason"] as? String {
                    result.finishReason = finishReason
                }
            }
        }

        for (index, call) in merged.sorted(by: { $0.key < $1.key }) {
            let complete = !call.arguments.isEmpty
                && (try? JSONSerialization.jsonObject(with: Data(call.arguments.utf8))) != nil
            if !complete {
                result.incompleteToolCalls.append(index)
            }
            result.toolCalls.append(SSEDeltaToolCall(
                index: index, id: call.id, name: call.name, arguments: call.arguments))
        }
        return result
    }
}