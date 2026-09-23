import Foundation

/// Errors raised while validating an OpenAI-format request body. Every case
/// maps to HTTP 400 with the OpenAI error envelope (`type:
/// invalid_request_error`). The message names the offending field so a client
/// can act without guessing.
public enum APIRequestError: LocalizedError, Equatable, Sendable {
    /// The body is not a well-formed request for this endpoint.
    case invalidBody(String)
    /// A present field violates the P0 contract (missing required value,
    /// out-of-range number, unknown role, ...).
    case invalidField(String)
    /// Two fields disagree and both are present (max_tokens vs
    /// max_completion_tokens).
    case conflict(String)
    /// A recognized OpenAI-platform feature that Mei deliberately does not
    /// implement. Rejected loudly instead of being silently ignored, so the
    /// field can never look supported while its semantics are dropped.
    case deferred(String)

    public var errorDescription: String? {
        switch self {
        case .invalidBody(let message): message
        case .invalidField(let message): message
        case .conflict(let message): message
        case .deferred(let message): message
        }
    }
}

/// The P0 request-validation rules, shared by `/v1/chat/completions` and
/// `/v1/completions`. Pure and deterministic so the exact rejection behavior
/// is unit-tested without a model.
public enum APIValidation {
    /// P0 message roles. `developer` is deferred (see the compatibility
    /// contract): it is rejected with a dedicated message rather than folded
    /// into `system`, because folding would silently change template
    /// semantics for clients that distinguish the two.
    public static let supportedRoles: Set<String> = ["system", "user", "assistant", "tool"]

    /// Accepted `reasoning_effort` values. Unknown values are rejected at the
    /// request boundary; an unsupported value would otherwise ride through to
    /// the template, which has no defined behavior for it.
    public static let reasoningEffortValues: Set<String> = ["low", "medium", "high", "none"]

    /// Deferred top-level `/v1/chat/completions` fields that must reject
    /// loudly. Each carries request semantics Mei does not implement; silently
    /// accepting them would make the field look supported while the request
    /// behavior diverged from the OpenAI platform (the exact failure mode the
    /// P0 contract forbids). Fields absent from this list (e.g. `user`) are
    /// ignored like OpenAI ignores them; the contract documents that rule.
    public static let deferredChatFields: [String: String] = [
        "response_format": "structured outputs (response_format) are deferred in the Mei P0 contract",
        "logprobs": "logprobs are deferred in the Mei P0 contract",
        "top_logprobs": "logprobs are deferred in the Mei P0 contract",
        "prediction": "prediction is deferred in the Mei P0 contract",
        "store": "store is deferred in the Mei P0 contract",
        "metadata": "metadata is deferred in the Mei P0 contract",
        "service_tier": "service tiers are deferred in the Mei P0 contract",
        "modalities": "multimodal output modalities are deferred in the Mei P0 contract",
        "audio": "audio output is deferred in the Mei P0 contract",
        "functions": "the deprecated functions parameter is not supported; use tools",
        "function_call": "the deprecated function_call parameter is not supported; use tool_choice",
    ]

    /// Deferred fields of the legacy `/v1/completions` request.
    public static let deferredCompletionFields: [String: String] = [
        "echo": "echo is deferred in the Mei P0 contract",
        "suffix": "suffix is deferred in the Mei P0 contract",
        "best_of": "best_of (n > 1) is deferred in the Mei P0 contract",
        "logprobs": "logprobs are deferred in the Mei P0 contract",
        "top_logprobs": "logprobs are deferred in the Mei P0 contract",
        "stream_options": "stream_options does not apply to /v1/completions in the Mei P0 contract",
    ]

    /// Validate the sampling/penalty ranges OpenAI defines. Out-of-range
    /// values are rejected instead of drifting into sampler code that has no
    /// defined behavior for them. `nil` fields are untouched (validation is
    /// only about present values).
    public static func validateSampling(
        temperature: Double?,
        topP: Double?,
        topK: Int?,
        minP: Double?,
        repetitionPenalty: Double?,
        presencePenalty: Double?,
        frequencyPenalty: Double?
    ) throws {
        if let temperature {
            guard (0.0...2.0).contains(temperature) else {
                throw APIRequestError.invalidField(
                    "temperature: \(temperature) is outside the supported range 0...2")
            }
        }
        if let topP {
            guard (0.0...1.0).contains(topP) else {
                throw APIRequestError.invalidField(
                    "top_p: \(topP) is outside the supported range 0...1")
            }
        }
        if let topK {
            guard topK >= 1 else {
                throw APIRequestError.invalidField(
                    "top_k: \(topK) must be >= 1")
            }
        }
        if let minP {
            guard (0.0...1.0).contains(minP) else {
                throw APIRequestError.invalidField(
                    "min_p: \(minP) is outside the supported range 0...1")
            }
        }
        if let repetitionPenalty {
            guard repetitionPenalty >= 0 else {
                throw APIRequestError.invalidField(
                    "repetition_penalty: \(repetitionPenalty) must be >= 0")
            }
        }
        if let presencePenalty {
            guard (-2.0...2.0).contains(presencePenalty) else {
                throw APIRequestError.invalidField(
                    "presence_penalty: \(presencePenalty) is outside the supported range -2...2")
            }
        }
        if let frequencyPenalty {
            guard (-2.0...2.0).contains(frequencyPenalty) else {
                throw APIRequestError.invalidField(
                    "frequency_penalty: \(frequencyPenalty) is outside the supported range -2...2")
            }
        }
    }

    /// Resolve `max_tokens` vs `max_completion_tokens` per the frozen policy:
    /// `max_completion_tokens` is an alias for `max_tokens`; either form may
    /// be used alone; both present with equal values are accepted; both
    /// present with different values are a hard conflict (the request is
    /// rejected even though one of them alone would have been honored —
    /// silently keeping one would hide the incompatibility from the client).
    /// Returns nil when neither form is present (server default applies).
    public static func resolveMaxTokens(
        maxTokens: Int?,
        maxCompletionTokens: Int?,
        endpoint: String
    ) throws -> Int? {
        guard let maxTokens else {
            return try maxCompletionTokens.map { try Self.validated($0, endpoint: endpoint) }
        }
        guard let maxCompletionTokens else { return try Self.validated(maxTokens, endpoint: endpoint) }
        guard maxTokens == maxCompletionTokens else {
            throw APIRequestError.conflict(
                "max_tokens (\(maxTokens)) and max_completion_tokens (\(maxCompletionTokens)) "
                    + "conflict: the Mei P0 contract accepts either form alone, or both with the "
                    + "same value")
        }
        return try Self.validated(maxTokens, endpoint: endpoint)
    }

    private static func validated(_ value: Int, endpoint: String) throws -> Int {
        guard value >= 1 else {
            throw APIRequestError.invalidField(
                "max_tokens: \(value) must be >= 1 (\(endpoint))")
        }
        return value
    }

    /// Validate a chat `messages` array (already mapped to APIMessage) against
    /// the P0 role/content rules. Runs before anything reaches the template,
    /// so every malformed branch is a clean 400, not a template error.
    public static func validateMessageArray(_ messages: [APIMessage]) throws {
        for (index, message) in messages.enumerated() {
            let prefix = "messages[\(index)]"
            guard supportedRoles.contains(message.role) else {
                if message.role == "developer" {
                    throw APIRequestError.deferred(
                        "\(prefix).role 'developer' is deferred in the Mei P0 contract; use a system message")
                }
                throw APIRequestError.invalidField(
                    "\(prefix).role '\(message.role)' is not supported in the Mei P0 contract — "
                        + "supported roles: system, user, assistant, tool")
            }
            switch message.role {
            case "assistant":
                let hasToolCalls = message.toolCalls?.isEmpty == false
                if message.content == nil && !hasToolCalls {
                    throw APIRequestError.invalidField(
                        "\(prefix): assistant messages need content or tool_calls")
                }
                for (callIndex, call) in (message.toolCalls ?? []).enumerated() {
                    guard !call.name.isEmpty else {
                        throw APIRequestError.invalidField(
                            "\(prefix).tool_calls[\(callIndex)].function.name is required")
                    }
                    guard !call.argumentsJSON.isEmpty else {
                        throw APIRequestError.invalidField(
                            "\(prefix).tool_calls[\(callIndex)].function.arguments must not be empty")
                    }
                }
            case "tool":
                guard let toolCallID = message.toolCallID, !toolCallID.isEmpty else {
                    throw APIRequestError.invalidField(
                        "\(prefix).tool_call_id is required for role 'tool'")
                }
                guard message.content != nil else {
                    throw APIRequestError.invalidField(
                        "\(prefix).content is required for role 'tool'")
                }
                if message.toolCalls?.isEmpty == false {
                    throw APIRequestError.invalidField(
                        "\(prefix): tool_calls are only allowed on assistant messages")
                }
            case "system", "user":
                guard message.content != nil else {
                    throw APIRequestError.invalidField(
                        "\(prefix).content is required for role '\(message.role)'")
                }
                if message.toolCalls?.isEmpty == false {
                    throw APIRequestError.invalidField(
                        "\(prefix): tool_calls are only allowed on assistant messages")
                }
            default:
                // Unreachable after the role guard above; kept for exhaustiveness.
                break
            }
        }
    }

    /// Validate the `tools` array: every entry must be a function tool with a
    /// name. Other tool types (code_interpreter, file_search) are platform
    /// features without Mei semantics and reject loudly.
    public static func validateTools(_ tools: [MeiJSONValue]?, endpoint: String) throws {
        guard let tools else { return }
        for (index, tool) in tools.enumerated() {
            guard case .object(let object) = tool else {
                throw APIRequestError.invalidField(
                    "tools[\(index)]: each tool must be an object")
            }
            guard case .string(let type)? = object["type"], type == "function" else {
                if let type = object["type"] {
                    throw APIRequestError.deferred(
                        "tools[\(index)].type '\(type)' is deferred in the Mei P0 contract "
                            + "— only function tools are supported")
                }
                throw APIRequestError.invalidField(
                    "tools[\(index)].type is required (only 'function' is supported in the Mei P0 contract)")
            }
            guard case .object(let function)? = object["function"] else {
                throw APIRequestError.invalidField(
                    "tools[\(index)].function is required")
            }
            guard case .string(let name)? = function["name"], !name.isEmpty else {
                throw APIRequestError.invalidField(
                    "tools[\(index)].function.name is required")
            }
        }
    }

    /// Validate `tool_choice`: the three keywords, a bare function name
    /// (legacy forced-tool form, documented in the contract), or the OpenAI
    /// object form. An object that names no function means "required, any
    /// tool" (documented); an object without the `function` key at all is a
    /// malformed request.
    public static func validateToolChoice(_ choice: MeiJSONValue?) throws {
        guard let choice else { return }
        switch choice {
        case .string(let name):
            guard !name.isEmpty else {
                throw APIRequestError.invalidField("tool_choice: a tool name must not be empty")
            }
        case .object(let object):
            if let function = object["function"] {
                guard case .object(let fn) = function else {
                    throw APIRequestError.invalidField(
                        "tool_choice.function must be an object")
                }
                if let name = fn["name"] {
                    guard case .string(let value) = name, !value.isEmpty else {
                        throw APIRequestError.invalidField(
                            "tool_choice.function.name must be a non-empty string when present")
                    }
                }
                if let type = object["type"] {
                    guard case .string(let value) = type, value == "function" else {
                        throw APIRequestError.invalidField(
                            "tool_choice.type must be 'function' when present")
                    }
                }
            } else {
                throw APIRequestError.invalidField(
                    "tool_choice: object form requires a function member "
                        + "({type: function, function: {name: ...}}) in the Mei P0 contract")
            }
        default:
            throw APIRequestError.invalidField(
                "tool_choice: must be 'auto', 'none', 'required', a function name, "
                    + "or {type: function, function: {name: ...}} in the Mei P0 contract")
        }
    }
}