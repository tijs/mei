import Foundation

// MARK: - Flexible JSON value (tool_choice, tool arguments)

public enum MeiJSONValue: Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: MeiJSONValue])
    case array([MeiJSONValue])
    case null

    public var anyValue: Any {
        switch self {
        case .string(let s): return s
        case .number(let n): return n
        case .bool(let b): return b
        case .object(let o): return o.mapValues { $0.anyValue }
        case .array(let a): return a.map { $0.anyValue }
        case .null: return NSNull()
        }
    }

    /// Compact JSON string of this value (used for tool arguments output).
    public func jsonString() throws -> String {
        let data = try JSONSerialization.data(withJSONObject: anyValue)
        guard let string = String(data: data, encoding: .utf8) else {
            throw MeiJSONValueError.notSerializable
        }
        return string
    }

    public enum MeiJSONValueError: Error {
        case notSerializable
    }

    /// Decode a JSON object string into this enum (tool arguments, etc.).
    public static func parseObject(from string: String) -> MeiJSONValue? {
        guard let data = string.data(using: .utf8),
            let raw = try? JSONSerialization.jsonObject(with: data),
            let dict = raw as? [String: Any]
        else { return nil }
        return MeiJSONValue.from(dict)
    }

    public static func from(_ value: Any) -> MeiJSONValue {
        switch value {
        case let string as String: return .string(string)
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return .bool(number.boolValue)
            }
            return .number(number.doubleValue)
        case let bool as Bool: return .bool(bool)
        case let dict as [String: Any]: return .object(dict.mapValues { from($0) })
        case let array as [Any]: return .array(array.map { from($0) })
        case is NSNull: return .null
        default: return .null
        }
    }

    /// Convert a nested MeiJSONValue tree into `[String: any Sendable]` for
    /// chat-template consumption (applyChatTemplate wants Sendable dicts).
    public static func templateSendable(_ value: MeiJSONValue) -> any Sendable {
        switch value {
        case .string(let s): return s
        case .number(let n): return n
        case .bool(let b): return b
        case .null: return NSNull()
        case .object(let o):
            return o.mapValues { templateSendable($0) as any Sendable }
        case .array(let a):
            return a.map { templateSendable($0) as any Sendable }
        }
    }
}

extension MeiJSONValue: Decodable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let bool = try? container.decode(Bool.self) {
            self = .bool(bool)
        } else if let number = try? container.decode(Double.self) {
            self = .number(number)
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let array = try? container.decode([MeiJSONValue].self) {
            self = .array(array)
        } else if let object = try? container.decode([String: MeiJSONValue].self) {
            self = .object(object)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "invalid JSON value")
        }
    }
}

// MARK: - Request DTOs

public struct ChatRequest: Sendable {
    public var model: String
    public var messages: [APIMessage]
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var maxTokens: Int?
    public var stream: Bool
    public var stop: [String]?
    public var tools: [MeiJSONValue]?
    public var toolChoice: MeiJSONValue?
    public var repetitionPenalty: Double?
    public var presencePenalty: Double?
    public var frequencyPenalty: Double?
    public var seed: UInt64?
    public var includeUsage: Bool
    public var reasoningEffort: String?
}

public struct APIMessage: Sendable {
    public var role: String
    public var content: String?
    public var toolCallID: String?
    /// Parsed tool_calls entries: id, name, arguments (JSON object string).
    public var toolCalls: [APIToolCall]?
    public var reasoningContent: String?

    public struct APIToolCall: Sendable {
        public var id: String?
        public var name: String
        public var argumentsJSON: String
    }
}

public struct CompletionRequest: Sendable {
    public var model: String
    public var prompt: String
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var maxTokens: Int?
    public var stream: Bool
    public var stop: [String]?
    public var repetitionPenalty: Double?
    public var presencePenalty: Double?
    public var frequencyPenalty: Double?
    public var seed: UInt64?
    public var includeUsage: Bool
}

// MARK: - Decoding

private struct FlexibleString: Decodable {
    let value: String?
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([ContentPart].self) {
            value = array.compactMap { $0.text }.joined(separator: "\n")
        } else {
            value = nil
        }
    }
}

private struct ContentPart: Decodable {
    let text: String?
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text = try? container.decode(String.self, forKey: .text)
    }
    private enum CodingKeys: String, CodingKey {
        case text
    }
}

private struct FlexibleInt: Decodable {
    let value: Int?
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = Int(double)
        } else {
            value = nil
        }
    }
}

private struct FlexibleStop: Decodable {
    let strings: [String]?
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let single = try? container.decode(String.self) {
            strings = [single]
        } else if let array = try? container.decode([String].self) {
            strings = array
        } else {
            strings = nil
        }
    }
}

private extension KeyedDecodingContainer {
    func optional<T: Decodable>(_ key: KeyedDecodingContainer.Key) -> T? {
        (try? decodeIfPresent(T.self, forKey: key)) ?? nil
    }
}

extension ChatRequest {
    public init(json: Data) throws {
        let decoder = JSONDecoder()
        struct Raw: Decodable {
            let model: String
            let messages: [RawMessage]
            let temperature: Double?
            let topP: Double?
            let topK: FlexibleInt?
            let minP: Double?
            let maxTokens: FlexibleInt?
            let stream: Bool?
            let stop: FlexibleStop?
            let tools: [MeiJSONValue]?
            let toolChoice: MeiJSONValue?
            let repetitionPenalty: Double?
            let presencePenalty: Double?
            let frequencyPenalty: Double?
            let seed: UInt64?
            let reasoningEffort: String?

            enum CodingKeys: String, CodingKey {
                case model, messages, temperature, stop, tools, seed, stream
                case topP = "top_p"
                case topK = "top_k"
                case minP = "min_p"
                case maxTokens = "max_tokens"
                case toolChoice = "tool_choice"
                case repetitionPenalty = "repetition_penalty"
                case presencePenalty = "presence_penalty"
                case frequencyPenalty = "frequency_penalty"
                case reasoningEffort = "reasoning_effort"
            }
        }
        struct RawMessage: Decodable {
            let role: String
            let content: FlexibleString?
            let toolCallID: String?
            let toolCalls: [RawToolCall]?
            let reasoningContent: String?

            enum CodingKeys: String, CodingKey {
                case role, content
                case toolCallID = "tool_call_id"
                case toolCalls = "tool_calls"
                case reasoningContent = "reasoning_content"
            }
        }
        struct RawToolCall: Decodable {
            let id: String?
            let function: RawFunction?

            struct RawFunction: Decodable {
                let name: String?
                let arguments: String?
            }
        }

        let raw = try decoder.decode(Raw.self, from: json)
        self.model = raw.model
        self.messages = raw.messages.map { message in
            let toolCalls = message.toolCalls?.compactMap { call -> APIMessage.APIToolCall? in
                guard let name = call.function?.name else { return nil }
                return APIMessage.APIToolCall(
                    id: call.id,
                    name: name,
                    argumentsJSON: call.function?.arguments ?? "{}"
                )
            }
            return APIMessage(
                role: message.role,
                content: message.content?.value,
                toolCallID: message.toolCallID,
                toolCalls: (toolCalls?.isEmpty == false) ? toolCalls : nil,
                reasoningContent: message.reasoningContent
            )
        }
        self.temperature = raw.temperature
        self.topP = raw.topP
        self.topK = raw.topK?.value
        self.minP = raw.minP
        self.maxTokens = raw.maxTokens?.value
        self.stream = raw.stream ?? false
        self.stop = raw.stop?.strings
        self.tools = raw.tools
        self.toolChoice = raw.toolChoice
        self.repetitionPenalty = raw.repetitionPenalty
        self.presencePenalty = raw.presencePenalty
        self.frequencyPenalty = raw.frequencyPenalty
        self.seed = raw.seed
        self.reasoningEffort = raw.reasoningEffort
        // stream_options.include_usage lives at the top level of the payload,
        // not inside messages; pull it via a raw JSON probe.
        let root = try JSONSerialization.jsonObject(with: json) as? [String: Any]
        let options = root?["stream_options"] as? [String: Any]
        self.includeUsage = (options?["include_usage"] as? Bool) ?? false
    }
}

extension CompletionRequest {
    public init(json: Data) throws {
        let decoder = JSONDecoder()
        struct Raw: Decodable {
            let model: String
            let prompt: String
            let temperature: Double?
            let topP: Double?
            let topK: FlexibleInt?
            let minP: Double?
            let maxTokens: FlexibleInt?
            let stream: Bool?
            let stop: FlexibleStop?
            let repetitionPenalty: Double?
            let presencePenalty: Double?
            let frequencyPenalty: Double?
            let seed: UInt64?

            enum CodingKeys: String, CodingKey {
                case model, prompt, temperature, stream, stop, seed
                case topP = "top_p"
                case topK = "top_k"
                case minP = "min_p"
                case maxTokens = "max_tokens"
                case repetitionPenalty = "repetition_penalty"
                case presencePenalty = "presence_penalty"
                case frequencyPenalty = "frequency_penalty"
            }
        }
        let raw = try decoder.decode(Raw.self, from: json)
        self.model = raw.model
        self.prompt = raw.prompt
        self.temperature = raw.temperature
        self.topP = raw.topP
        self.topK = raw.topK?.value
        self.minP = raw.minP
        self.maxTokens = raw.maxTokens?.value
        self.stream = raw.stream ?? false
        self.stop = raw.stop?.strings
        self.repetitionPenalty = raw.repetitionPenalty
        self.presencePenalty = raw.presencePenalty
        self.frequencyPenalty = raw.frequencyPenalty
        self.seed = raw.seed
        let root = try JSONSerialization.jsonObject(with: json) as? [String: Any]
        let options = root?["stream_options"] as? [String: Any]
        self.includeUsage = (options?["include_usage"] as? Bool) ?? false
    }
}

// MARK: - Response DTOs

public struct ChatCompletionResponse: Encodable, Sendable {
    public var id: String
    public var object = "chat.completion"
    public var created: Int
    public var model: String
    public var choices: [Choice]
    public var usage: Usage?

    public struct Choice: Encodable, Sendable {
        public var index = 0
        public var message: ResponseMessage
        public var finishReason: String?
        public var logprobs: Int? = nil

        public enum CodingKeys: String, CodingKey {
            case index, message, logprobs
            case finishReason = "finish_reason"
        }
    }

    public struct ResponseMessage: Encodable, Sendable {
        public var role = "assistant"
        public var content: String?
        public var toolCalls: [ResponseToolCall]?
        public var reasoningContent: String?

        public enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
            case reasoningContent = "reasoning_content"
        }
    }

    public struct ResponseToolCall: Encodable, Sendable {
        public var id: String
        public var type = "function"
        public var function: ResponseFunction

        public struct ResponseFunction: Encodable, Sendable {
            public var name: String
            public var arguments: String
        }
    }

    public struct Usage: Encodable, Sendable {
        public var promptTokens: Int
        public var completionTokens: Int
        public var totalTokens: Int
        public var promptTokensDetails: PromptTokensDetails?

        public struct PromptTokensDetails: Encodable, Sendable {
            public var cachedTokens: Int
        }

        public enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
            case promptTokensDetails = "prompt_tokens_details"
        }
    }

    public enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
    }
}

public struct SSEChatChunk: Encodable, Sendable {
    public var id: String
    public var object = "chat.completion.chunk"
    public var created: Int
    public var model: String
    public var choices: [Choice]
    public var usage: ChatCompletionResponse.Usage?

    public struct Choice: Encodable, Sendable {
        public var index = 0
        public var delta: Delta
        public var finishReason: String?

        public enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    public struct Delta: Encodable, Sendable {
        public var role: String?
        public var content: String?
        public var reasoningContent: String?
        public var toolCalls: [DeltaToolCall]?

        public enum CodingKeys: String, CodingKey {
            case role, content
            case reasoningContent = "reasoning_content"
            case toolCalls = "tool_calls"
        }
    }

    public struct DeltaToolCall: Encodable, Sendable {
        public var index: Int
        public var id: String?
        public var type: String?
        public var function: Function?

        public struct Function: Encodable, Sendable {
            public var name: String?
            public var arguments: String?
        }
    }

    public enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
    }
}

public struct ModelsResponse: Encodable, Sendable {
    public var object = "list"
    public var data: [ModelEntry]

    public struct ModelEntry: Encodable, Sendable {
        public var id: String
        public var object = "model"
        public var created: Int
        public var ownedBy = "mei"

        public enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }
    }
}

public struct APIErrorEnvelope: Encodable, Sendable {
    public var error: APIErrorDetail

    public struct APIErrorDetail: Encodable, Sendable {
        public var message: String
        public var type: String
        public var code: String?
    }
}

/// The assembled result of one generation run, shared by the streaming and
/// non-streaming paths so both report identical content/usage.
public struct GenerationRun: Sendable {
    public var text = ""
    public var reasoning = ""
    public var toolCalls: [ToolCallEmitting] = []
    public var finishReason: String = "stop"
    public var promptTokenCount = 0
    public var completionTokenCount = 0
    public var cachedTokenCount = 0
    public var decodeTokensPerSecond: Double = 0
    public var promptTokensPerSecond: Double = 0
    public var prefillMilliseconds: Double = 0
    public var generateMilliseconds: Double = 0
    public var wallMilliseconds: Double = 0
    public var cacheHit = false

    public struct ToolCallEmitting: Sendable {
        public var id: String?
        public var name: String
        public var argumentsJSON: String
    }
}