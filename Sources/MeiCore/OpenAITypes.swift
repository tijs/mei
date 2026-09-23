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
    /// `.sortedKeys` keeps tool-call arguments byte-deterministic across runs
    /// (same request → identical arguments JSON, regardless of dictionary
    /// hashing).
    public func jsonString() throws -> String {
        let data = try JSONSerialization.data(withJSONObject: anyValue, options: [.sortedKeys])
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

/// `content` in its two P0 shapes: a plain string, or an array of text parts.
/// Decoding is strict by contract: a part that is not `type: "text"` is a
/// deferred multimodal feature and is rejected loudly instead of being
/// silently dropped from the prompt (silent dropping changed the request's
/// semantics while the field looked accepted).
private struct FlexibleString: Decodable {
    let value: String?
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            // content: null is legal (assistant messages carrying tool_calls)
            value = nil
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([ContentPart].self) {
            for part in array {
                guard let type = part.type, type == "text" else {
                    if let type = part.type {
                        throw APIRequestError.deferred(
                            "content part type '\(type)' is deferred in the Mei P0 "
                                + "contract — only text parts are supported (multimodal input "
                                + "is not implemented)")
                    }
                    throw APIRequestError.invalidField(
                        "content part must declare \"type\": \"text\" in the Mei P0 contract")
                }
                guard part.text != nil else {
                    throw APIRequestError.invalidField(
                        "content part of type 'text' must carry a text value")
                }
            }
            value = array.compactMap { $0.text }.joined(separator: "\n")
        } else {
            throw APIRequestError.invalidField(
                "message content must be a string or an array of text parts")
        }
    }
}

private struct ContentPart: Decodable {
    let type: String?
    let text: String?
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = try? container.decode(String.self, forKey: .type)
        text = try? container.decode(String.self, forKey: .text)
    }
    private enum CodingKeys: String, CodingKey {
        case type, text
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
            let model: String?
            let messages: [RawMessage]?
            let temperature: Double?
            let topP: Double?
            let topK: FlexibleInt?
            let minP: Double?
            let maxTokens: FlexibleInt?
            let maxCompletionTokens: FlexibleInt?
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
                case maxCompletionTokens = "max_completion_tokens"
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

        let raw: Raw
        let root: [String: Any]
        do {
            root = (try JSONSerialization.jsonObject(with: json) as? [String: Any]) ?? [:]
            raw = try decoder.decode(Raw.self, from: json)
        } catch let error as APIRequestError {
            throw error
        } catch {
            throw APIRequestError.invalidBody(
                "request body is not a valid /v1/chat/completions payload: \(error.localizedDescription)")
        }

        guard let model = raw.model, !model.isEmpty else {
            throw APIRequestError.invalidField("model is required")
        }
        guard let messages = raw.messages, !messages.isEmpty else {
            throw APIRequestError.invalidField("messages is required and must not be empty")
        }
        let endpoint = "/v1/chat/completions"
        // Map first (tolerantly: a missing function name/arguments becomes an
        // empty string), then validate the mapped messages — validation sees
        // exactly the shape that reaches the template.
        let mappedMessages = messages.map { message in
            let toolCalls = message.toolCalls?.map { call -> APIMessage.APIToolCall in
                APIMessage.APIToolCall(
                    id: call.id,
                    name: call.function?.name ?? "",
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
        try APIValidation.validateMessageArray(mappedMessages)
        try APIValidation.validateSampling(
            temperature: raw.temperature,
            topP: raw.topP,
            topK: raw.topK?.value,
            minP: raw.minP,
            repetitionPenalty: raw.repetitionPenalty,
            presencePenalty: raw.presencePenalty,
            frequencyPenalty: raw.frequencyPenalty)
        if let effort = raw.reasoningEffort {
            guard APIValidation.reasoningEffortValues.contains(effort) else {
                throw APIRequestError.invalidField(
                    "reasoning_effort: '\(effort)' is not a supported value "
                        + "(low, medium, high, none) in the Mei P0 contract")
            }
        }
        // Deferred platform fields reject loudly (see APIValidation):
        for (field, reason) in APIValidation.deferredChatFields where root[field] != nil {
            throw APIRequestError.deferred(field + ": " + reason)
        }
        if root["n"] != nil {
            guard let n = root["n"] as? NSNumber, n.doubleValue == 1 else {
                throw APIRequestError.deferred(
                    "n: only n=1 is supported in the Mei P0 contract (multiple choices are deferred)")
            }
        }
        if let parallel = root["parallel_tool_calls"] as? Bool, !parallel {
            throw APIRequestError.deferred(
                "parallel_tool_calls=false is not supported in the Mei P0 contract: Mei always "
                    + "allows multiple tool calls per turn (the platform default). Send the "
                    + "field only with value true, or omit it.")
        }
        // stream_options: the only recognized option is include_usage, and it
        // requires a stream. Unknown option keys reject (a future platform
        // option must never be accepted and then dropped).
        let stream = raw.stream ?? false
        var includeUsage = false
        if let rawOptions = root["stream_options"] {
            guard let options = rawOptions as? [String: Any] else {
                throw APIRequestError.invalidField("stream_options must be an object")
            }
            let unknown = Set(options.keys).subtracting(["include_usage"])
            if !unknown.isEmpty {
                throw APIRequestError.invalidField(
                    "stream_options: unsupported option(s) \(unknown.sorted().joined(separator: ", ")) "
                        + "in the Mei P0 contract (only include_usage is supported)")
            }
            guard stream else {
                throw APIRequestError.invalidField(
                    "stream_options requires stream=true in the Mei P0 contract")
            }
            if let include = options["include_usage"] {
                guard let include = include as? Bool else {
                    throw APIRequestError.invalidField(
                        "stream_options.include_usage must be a boolean")
                }
                includeUsage = include
            }
        }
        let maxTokens = try APIValidation.resolveMaxTokens(
            maxTokens: raw.maxTokens?.value,
            maxCompletionTokens: raw.maxCompletionTokens?.value,
            endpoint: endpoint)
        let toolChoice = raw.toolChoice
        try APIValidation.validateTools(raw.tools, endpoint: endpoint)
        try APIValidation.validateToolChoice(toolChoice)

        self.model = model
        self.messages = mappedMessages
        self.temperature = raw.temperature
        self.topP = raw.topP
        self.topK = raw.topK?.value
        self.minP = raw.minP
        self.maxTokens = maxTokens
        self.stream = stream
        self.stop = raw.stop?.strings
        self.tools = raw.tools
        self.toolChoice = toolChoice
        self.repetitionPenalty = raw.repetitionPenalty
        self.presencePenalty = raw.presencePenalty
        self.frequencyPenalty = raw.frequencyPenalty
        self.seed = raw.seed
        self.reasoningEffort = raw.reasoningEffort
        self.includeUsage = includeUsage
    }
}

extension CompletionRequest {
    public init(json: Data) throws {
        let decoder = JSONDecoder()
        struct Raw: Decodable {
            let model: String?
            let prompt: String?
            let temperature: Double?
            let topP: Double?
            let topK: FlexibleInt?
            let minP: Double?
            let maxTokens: FlexibleInt?
            let maxCompletionTokens: FlexibleInt?
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
                case maxCompletionTokens = "max_completion_tokens"
                case repetitionPenalty = "repetition_penalty"
                case presencePenalty = "presence_penalty"
                case frequencyPenalty = "frequency_penalty"
            }
        }

        let raw: Raw
        let root: [String: Any]
        do {
            root = (try JSONSerialization.jsonObject(with: json) as? [String: Any]) ?? [:]
            raw = try decoder.decode(Raw.self, from: json)
        } catch let error as APIRequestError {
            throw error
        } catch {
            throw APIRequestError.invalidBody(
                "request body is not a valid /v1/completions payload: \(error.localizedDescription)")
        }

        let endpoint = "/v1/completions"
        guard let model = raw.model, !model.isEmpty else {
            throw APIRequestError.invalidField("model is required")
        }
        guard let prompt = raw.prompt else {
            throw APIRequestError.invalidField("prompt is required")
        }
        for (field, reason) in APIValidation.deferredCompletionFields where root[field] != nil {
            throw APIRequestError.deferred(field + ": " + reason)
        }
        if root["n"] != nil {
            guard let n = root["n"] as? NSNumber, n.doubleValue == 1 else {
                throw APIRequestError.deferred(
                    "n: only n=1 is supported in the Mei P0 contract (multiple choices are deferred)")
            }
        }
        let stream = raw.stream ?? false
        let maxTokens = try APIValidation.resolveMaxTokens(
            maxTokens: raw.maxTokens?.value,
            maxCompletionTokens: raw.maxCompletionTokens?.value,
            endpoint: endpoint)
        try APIValidation.validateSampling(
            temperature: raw.temperature,
            topP: raw.topP,
            topK: raw.topK?.value,
            minP: raw.minP,
            repetitionPenalty: raw.repetitionPenalty,
            presencePenalty: raw.presencePenalty,
            frequencyPenalty: raw.frequencyPenalty)

        self.model = model
        self.prompt = prompt
        self.temperature = raw.temperature
        self.topP = raw.topP
        self.topK = raw.topK?.value
        self.minP = raw.minP
        self.maxTokens = maxTokens
        self.stream = stream
        self.stop = raw.stop?.strings
        self.repetitionPenalty = raw.repetitionPenalty
        self.presencePenalty = raw.presencePenalty
        self.frequencyPenalty = raw.frequencyPenalty
        self.seed = raw.seed
        self.includeUsage = false
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
        // Mei engine extensions (optional; absent for streaming chunks when
        // not finalized). These are how the benchmark harness reads the
        // engine's own decode/prompt tok/s and allocator footprint — no
        // proxy-side timing guesses.
        public var tokensPerSecond: Double?
        public var promptTokensPerSecond: Double?
        public var prefillMilliseconds: Double?
        public var generateMilliseconds: Double?
        public var memoryActiveBytes: Int?
        public var memoryCacheBytes: Int?
        public var memoryPeakBytes: Int?

        public struct PromptTokensDetails: Encodable, Sendable {
            public var cachedTokens: Int

            public enum CodingKeys: String, CodingKey {
                case cachedTokens = "cached_tokens"
            }
        }

        public enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
            case promptTokensDetails = "prompt_tokens_details"
            case tokensPerSecond = "tokens_per_second"
            case promptTokensPerSecond = "prompt_tokens_per_second"
            case prefillMilliseconds = "prefill_ms"
            case generateMilliseconds = "generate_ms"
            case memoryActiveBytes = "mei_memory_active_bytes"
            case memoryCacheBytes = "mei_memory_cache_bytes"
            case memoryPeakBytes = "mei_memory_peak_bytes"
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

/// Minimal legacy `/v1/completions` response in the OpenAI `text_completion`
/// shape: one choice carrying `text` instead of a chat `message`. The usage
/// block is byte-identical to the chat usage block (same `Router.usage`), so
/// the benchmark's context/admission probes read either endpoint the same way.
public struct TextCompletionResponse: Encodable, Sendable {
    public var id: String
    public var object = "text_completion"
    public var created: Int
    public var model: String
    public var choices: [Choice]
    public var usage: ChatCompletionResponse.Usage?

    public struct Choice: Encodable, Sendable {
        public var text: String
        public var index = 0
        public var logprobs: Int? = nil
        public var finishReason: String?

        public enum CodingKeys: String, CodingKey {
            case text, index, logprobs
            case finishReason = "finish_reason"
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
    /// MLX allocator snapshot captured when the run finished: active (live
    /// arrays), cache (recyclable buffer pool), and program peak so far.
    /// All bytes; 0 until the first run (the engine patches these before the
    /// run leaves the actor).
    public var memoryActiveBytes = 0
    public var memoryCacheBytes = 0
    public var memoryPeakBytes = 0

    public struct ToolCallEmitting: Sendable {
        public var id: String?
        public var name: String
        public var argumentsJSON: String
    }
}

/// Runtime memory + device report for /v1/mei/status. Replacement for the
/// nonexistent `get_physical_memory` Cmlx API: MLX's allocator exposes
/// active/cache/peak via `Memory.snapshot()` and the Metal device carries
/// the working-set budget.
public struct MeiMemoryReport: Sendable, Codable {
    public var activeBytes: Int
    public var cacheBytes: Int
    public var peakBytes: Int
    public var memoryLimitBytes: Int
    public var cacheLimitBytes: Int
    public var recommendedWorkingSetBytes: Int?

    /// Snapshot of GPU/device facts (architecture, physical memory).
    public struct Device: Sendable, Codable {
        public var architecture: String
        public var memoryBytes: Int
    }

    public var device: Device?
}

/// Shape of GET /v1/mei/status — convenience status surface for the
/// benchmark harness (probe records it; OpenAI-compat surface stays clean).
public struct MeiStatusResponse: Sendable, Codable {
    public var status: String
    public var model: String
    public var contextCap: Int
    public var prefillStepSize: Int
    public var maxTokens: Int
    public var kvBits: Int?
    public var cacheReuse: Bool
    public var uptimeSeconds: Int
    public var memory: MeiMemoryReport

    /// Live prefix-cache counters when the coordinator is enabled (nil when
    /// --cache-reuse false or before the first request).
    public var cache: MeiCacheStatus?
}

/// Prefix-cache counters surfaced by /v1/mei/status (translated from the
/// coordinator's stats snapshot so the JSON shape stays stable across
/// vmlx-swift versions).
public struct MeiCacheStatus: Sendable, Codable {
    public var pagedEnabled: Bool
    public var pagedHits: Int
    public var pagedMisses: Int
    public var pagedEvictions: Int
    public var ssmHits: Int
    public var ssmMisses: Int
    public var isHybrid: Bool
}