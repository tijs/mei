import Foundation

// MARK: - Chat Completions `response_format` (request contract)

/// The decoded `response_format` field of `POST /v1/chat/completions`.
///
/// Scope: chat completions only. The legacy `/v1/completions` request DTO
/// deliberately has no `response_format` field, so the field stays inert on
/// that route — it is out of the frozen P0 contract
/// (docs/OPENAI-COMPATIBILITY.md §7).
///
/// Decoding is pure request→value mapping. Whether the format can actually be
/// honored is a separate question answered by the model-free compiler
/// (`JSONSchemaCompiler`, pre-generation via `Router.validateStructuredRequest`)
/// and by token-level constrained decoding in the engine
/// (`StructuredGeneration` / `JSONGrammarLogitProcessor`).
public enum ResponseFormat: Sendable, Equatable {
    /// Ordinary text generation. Both an absent field and the explicit
    /// `{"type": "text"}` form decode here and keep the ordinary generation
    /// path byte-compatible.
    case text

    /// `{"type": "json_object"}`: output must be a syntactically valid JSON
    /// value, enforced by token-level constrained decoding.
    case jsonObject

    /// `{"type": "json_schema", "json_schema": {...}}`: strict schema form.
    case jsonSchema(JSONSchemaFormat)
}

/// A strict `json_schema` format as supplied in the request. The schema is
/// preserved verbatim as JSON values; validating it against the supported
/// subset is the model-free compiler's job (`JSONSchemaCompiler`), kept
/// deliberately separate from HTTP decoding so it can be fuzzed and reused
/// by buffered and streamed generation.
public struct JSONSchemaFormat: Sendable, Equatable {
    public var name: String
    /// Always `true` after decoding: the first milestone accepts only
    /// explicitly strict schemas. `strict: false` or an omitted `strict` is
    /// an explicit 400 rather than a silent downgrade.
    public var strict: Bool
    public var schema: MeiJSONValue

    public init(name: String, strict: Bool, schema: MeiJSONValue) {
        self.name = name
        self.strict = strict
        self.schema = schema
    }
}

/// Typed decode/validation errors for the `response_format` field.
///
/// All of them map to HTTP 400 with `param: "response_format"` (see
/// `Router.errorResult`): malformed or unsupported formats are rejected
/// explicitly instead of being ignored.
public enum ResponseFormatError: Error, Sendable, Equatable {
    /// `response_format` is present but not a JSON object.
    case notObject
    /// `response_format.type` is absent (or null).
    case missingType
    /// `response_format.type` is present but not a string.
    case invalidType
    /// `response_format.type` names an unsupported format.
    case unsupportedType(String)
    /// `response_format.json_schema` is absent or not an object.
    case missingJSONSchema
    /// `response_format.json_schema.name` is absent, empty, or not a string.
    case missingName
    /// `response_format.json_schema.schema` is absent (or null).
    case missingSchema
    /// `response_format.json_schema.schema` is present but not an object.
    case invalidSchema
    /// `response_format.json_schema.strict` is absent, not a boolean, or not
    /// `true` — only explicitly strict schemas are supported.
    case invalidStrict
    /// The request combines a structured format with `tools`. Tool calls and
    /// token-level JSON constraints are not proven compatible yet, so the
    /// combination is rejected explicitly instead of silently answering
    /// without one of the two guarantees.
    case structuredToolsUnsupported
}

extension ResponseFormatError {
    /// The OpenAI-style `error.param` naming the offending request field.
    public var param: String { "response_format" }

    /// Stable machine-readable `error.code`.
    public var code: String {
        switch self {
        case .structuredToolsUnsupported: return "response_format_unsupported"
        default: return "invalid_response_format"
        }
    }

    /// Human-readable message naming the precise offending sub-field.
    public var message: String {
        switch self {
        case .notObject:
            return "response_format must be a JSON object"
        case .missingType:
            return "response_format.type is required"
        case .invalidType:
            return "response_format.type must be a string"
        case .unsupportedType(let type):
            return
                "response_format.type '\(type)' is not supported; supported types: 'text', 'json_object', 'json_schema'"
        case .missingJSONSchema:
            return "response_format.json_schema is required and must be a JSON object"
        case .missingName:
            return "response_format.json_schema.name is required and must be a non-empty string"
        case .missingSchema:
            return "response_format.json_schema.schema is required"
        case .invalidSchema:
            return "response_format.json_schema.schema must be a JSON object"
        case .invalidStrict:
            return "response_format.json_schema.strict must be true: only strict schemas are supported"
        case .structuredToolsUnsupported:
            return
                "response_format structured output cannot be combined with 'tools' yet: tool calls and token-level JSON constraints are not proven compatible; remove one of the two fields"
        }
    }
}

extension ResponseFormatError: LocalizedError {
    public var errorDescription: String? { message }
}

extension ResponseFormat {
    /// Decode and validate the raw `response_format` JSON value.
    ///
    /// An explicit `null` is treated as absent (`.text`): several JSON
    /// encoders emit null for an unset optional field, and rejecting it would
    /// break such clients without protecting anything.
    public static func decode(from value: MeiJSONValue) throws -> ResponseFormat {
        let object: [String: MeiJSONValue]
        switch value {
        case .object(let dictionary):
            object = dictionary
        case .null:
            return .text
        default:
            throw ResponseFormatError.notObject
        }

        switch object["type"] {
        case .string(let type):
            switch type {
            case "text":
                return .text
            case "json_object":
                return .jsonObject
            case "json_schema":
                return .jsonSchema(try decodeJSONSchema(object["json_schema"]))
            default:
                throw ResponseFormatError.unsupportedType(type)
            }
        case nil, .null?:
            throw ResponseFormatError.missingType
        default:
            throw ResponseFormatError.invalidType
        }
    }

    private static func decodeJSONSchema(_ value: MeiJSONValue?) throws -> JSONSchemaFormat {
        guard case .object(let wrapper)? = value else {
            throw ResponseFormatError.missingJSONSchema
        }
        guard case .string(let name)? = wrapper["name"], !name.isEmpty else {
            throw ResponseFormatError.missingName
        }
        guard case .bool(let strict)? = wrapper["strict"], strict else {
            throw ResponseFormatError.invalidStrict
        }
        let schema: MeiJSONValue
        switch wrapper["schema"] {
        case .object(let object)?:
            schema = .object(object)
        case nil, .null?:
            throw ResponseFormatError.missingSchema
        default:
            throw ResponseFormatError.invalidSchema
        }
        return JSONSchemaFormat(name: name, strict: strict, schema: schema)
    }
}
