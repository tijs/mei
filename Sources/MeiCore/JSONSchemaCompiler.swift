import CryptoKit
import Foundation

// MARK: - Model-free schema/format compiler boundary

/// The tokenizer- and engine-independent half of structured generation.
///
/// It validates the supported strict JSON-Schema subset — `strict: true`,
/// root object, `properties`, `required`, `additionalProperties: false`,
/// scalar fields, and string `enum` — and compiles it into a deterministic
/// constraint value plus cache key. Unsupported keywords and ambiguous
/// constructs are rejected explicitly (fail closed) instead of being ignored
/// or downgraded.
///
/// Deliberately free of HTTP, engine, and tokenizer types so it can be fuzzed
/// in isolation and reused by buffered and streamed generation. The instance
/// validator (`CompiledJSONSchema.validate`) is a diagnostic building block
/// only: the real guarantee must come from token-level constrained decoding,
/// not from validating output afterward.
public enum JSONSchemaCompiler {

    /// Compile a decoded `response_format` into its model-free constraint.
    public static func compile(_ format: ResponseFormat) throws -> CompiledResponseFormat {
        switch format {
        case .text:
            return .text
        case .jsonObject:
            return .jsonObject
        case .jsonSchema(let schemaFormat):
            return .jsonSchema(try compile(schemaFormat))
        }
    }

    /// Compile a strict `json_schema` format against the supported subset.
    public static func compile(_ format: JSONSchemaFormat) throws -> CompiledJSONSchema {
        guard format.strict else { throw JSONSchemaCompileError.strictNotSupported }
        guard !format.name.isEmpty else { throw JSONSchemaCompileError.invalidName }
        guard isJSONRepresentable(format.name) else {
            throw JSONSchemaCompileError.invalidJSONString(format.name)
        }

        guard case .object(let root) = format.schema else {
            throw JSONSchemaCompileError.rootNotObject
        }
        guard case .string("object")? = root["type"] else {
            throw JSONSchemaCompileError.rootTypeNotObject
        }
        let rootKeys: Set<String> = ["type", "properties", "required", "additionalProperties"]
        for key in root.keys.sorted() where !rootKeys.contains(key) {
            throw JSONSchemaCompileError.rootKeywordUnsupported(key)
        }
        guard case .bool(false)? = root["additionalProperties"] else {
            throw JSONSchemaCompileError.additionalPropertiesMustBeFalse
        }
        guard case .object(let rawProperties)? = root["properties"] else {
            throw JSONSchemaCompileError.propertiesInvalid
        }
        guard case .array(let requiredValues)? = root["required"] else {
            throw JSONSchemaCompileError.requiredInvalid
        }
        var requiredNames: [String] = []
        var seenRequired = Set<String>()
        for value in requiredValues {
            guard case .string(let name) = value else { throw JSONSchemaCompileError.requiredInvalid }
            guard isJSONRepresentable(name) else { throw JSONSchemaCompileError.invalidJSONString(name) }
            guard seenRequired.insert(name).inserted else {
                throw JSONSchemaCompileError.requiredNameDuplicated(name)
            }
            requiredNames.append(name)
        }

        var properties: [CompiledJSONSchema.Property] = []
        for name in rawProperties.keys.sorted() {
            guard !name.isEmpty else { throw JSONSchemaCompileError.propertyNameEmpty }
            guard isJSONRepresentable(name) else { throw JSONSchemaCompileError.invalidJSONString(name) }
            properties.append(try compileProperty(name: name, value: rawProperties[name]!))
        }

        let definedNames = Set(properties.map(\.name))
        for name in requiredNames.sorted() where !definedNames.contains(name) {
            throw JSONSchemaCompileError.requiredNameNotDefined(name)
        }

        let sortedRequired = requiredNames.sorted()
        let canonical = try canonicalJSONString(name: format.name, properties: properties, required: sortedRequired)
        return CompiledJSONSchema(
            name: format.name,
            properties: properties,
            required: sortedRequired,
            constraintKey: "json_schema:v1:" + sha256Hex(canonical))
    }

    // MARK: - Property compilation

    private static func compileProperty(name: String, value: MeiJSONValue) throws -> CompiledJSONSchema.Property {
        guard case .object(let property) = value else {
            throw JSONSchemaCompileError.propertyNotObject(name)
        }
        let scalarType: CompiledJSONSchema.ScalarType
        switch property["type"] {
        case .string(let typeName)?:
            guard let parsed = CompiledJSONSchema.ScalarType(rawValue: typeName) else {
                throw JSONSchemaCompileError.propertyTypeUnsupported(name, typeName)
            }
            scalarType = parsed
        case nil, .null?:
            throw JSONSchemaCompileError.propertyTypeMissing(name)
        default:
            throw JSONSchemaCompileError.propertyTypeUnsupported(name, jsonKindName(property["type"]!))
        }

        var allowedValues: [String]? = nil
        if let enumValue = property["enum"] {
            guard scalarType == .string else {
                throw JSONSchemaCompileError.propertyEnumRequiresStringType(name)
            }
            guard case .array(let values) = enumValue else {
                throw JSONSchemaCompileError.propertyEnumInvalid(name)
            }
            var strings: [String] = []
            var seen = Set<String>()
            for entry in values {
                guard case .string(let string) = entry else {
                    throw JSONSchemaCompileError.propertyEnumInvalid(name)
                }
                guard isJSONRepresentable(string) else {
                    throw JSONSchemaCompileError.invalidJSONString(string)
                }
                guard seen.insert(string).inserted else {
                    throw JSONSchemaCompileError.propertyEnumDuplicated(name, string)
                }
                strings.append(string)
            }
            guard !strings.isEmpty else { throw JSONSchemaCompileError.propertyEnumInvalid(name) }
            allowedValues = strings.sorted()
        }

        let allowedKeys: Set<String> = ["type", "enum"]
        for key in property.keys.sorted() where !allowedKeys.contains(key) {
            throw JSONSchemaCompileError.propertyKeywordUnsupported(name, key)
        }

        return CompiledJSONSchema.Property(name: name, type: scalarType, allowedValues: allowedValues)
    }

    // MARK: - Canonicalization

    /// Deterministic canonical JSON for the compiled format: property order,
    /// `required` order, and enum order are normalized so equivalent schemas
    /// produce byte-identical canonical text (and therefore the same key).
    private static func canonicalJSONString(
        name: String,
        properties: [CompiledJSONSchema.Property],
        required: [String]
    ) throws -> String {
        var propertyTree: [String: Any] = [:]
        for property in properties {
            var field: [String: Any] = ["type": property.type.rawValue]
            if let allowedValues = property.allowedValues {
                field["enum"] = allowedValues
            }
            propertyTree[property.name] = field
        }
        let schemaTree: [String: Any] = [
            "type": "object",
            "properties": propertyTree,
            "required": required,
            "additionalProperties": false,
        ]
        let formatTree: [String: Any] = [
            "type": "json_schema",
            "json_schema": [
                "name": name,
                "strict": true,
                "schema": schemaTree,
            ] as [String: Any],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: formatTree, options: [.sortedKeys]),
            let string = String(data: data, encoding: .utf8)
        else {
            throw JSONSchemaCompileError.schemaNotSerializable
        }
        return string
    }

    private static func sha256Hex(_ string: String) -> String {
        SHA256.hash(data: Data(string.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Compiled constraint values

/// A compiled `response_format`: the model-free constraint the generation
/// slices will consume.
public enum CompiledResponseFormat: Equatable, Sendable {
    case text
    case jsonObject
    case jsonSchema(CompiledJSONSchema)

    /// Deterministic cache key identifying the compiled constraint.
    public var constraintKey: String {
        switch self {
        case .text: return "text"
        case .jsonObject: return "json_object"
        case .jsonSchema(let schema): return schema.constraintKey
        }
    }
}

/// Compiled strict object schema: scalar fields with optional string enums,
/// required names, and `additionalProperties: false`. Canonical (name-sorted)
/// order makes the value and its `constraintKey` deterministic.
public struct CompiledJSONSchema: Equatable, Sendable {
    public let name: String
    /// Field constraints in canonical (name-sorted) order.
    public let properties: [Property]
    /// Required field names in canonical (name-sorted) order.
    public let required: [String]
    /// SHA-256 over the canonical JSON serialization of the format.
    public let constraintKey: String

    public struct Property: Equatable, Sendable {
        public let name: String
        public let type: ScalarType
        /// Non-nil iff the schema declared a string enum (canonical order).
        public let allowedValues: [String]?
    }

    public enum ScalarType: String, Equatable, Sendable {
        case string
        case number
        case integer
        case boolean
    }

    /// Pure check of one decoded JSON value against the compiled constraint.
    ///
    /// Diagnostic building block only — it must not be used as the structured
    /// guarantee; that must come from token-level constrained decoding.
    public func validate(_ value: MeiJSONValue) -> Bool {
        guard case .object(let object) = value else { return false }
        var constraints: [String: Property] = [:]
        constraints.reserveCapacity(properties.count)
        for property in properties {
            constraints[property.name] = property
        }
        for key in object.keys where constraints[key] == nil {
            return false  // additionalProperties: false
        }
        for name in required where object[name] == nil {
            return false
        }
        for (key, fieldValue) in object {
            guard let property = constraints[key], property.accepts(fieldValue) else { return false }
        }
        return true
    }
}

extension CompiledJSONSchema.Property {
    func accepts(_ value: MeiJSONValue) -> Bool {
        switch type {
        case .string:
            guard case .string(let string) = value, isJSONRepresentable(string) else { return false }
            if let allowedValues, !allowedValues.contains(string) { return false }
            return true
        case .number:
            guard case .number(let number) = value else { return false }
            return number.isFinite
        case .integer:
            guard case .number(let number) = value else { return false }
            return number.isFinite && number == number.rounded()
        case .boolean:
            guard case .bool = value else { return false }
            return true
        }
    }
}

// MARK: - Compile errors

/// Typed schema-subset rejections. Kept separate from `ResponseFormatError`
/// (HTTP decoding) on purpose: the compiler boundary is model-free and the
/// HTTP mapping layer can translate these when the constraint is wired in.
public enum JSONSchemaCompileError: Error, Sendable, Equatable {
    case rootNotObject
    case rootKeywordUnsupported(String)
    case rootTypeNotObject
    case additionalPropertiesMustBeFalse
    case propertiesInvalid
    case requiredInvalid
    case requiredNameDuplicated(String)
    case requiredNameNotDefined(String)
    case propertyNotObject(String)
    case propertyNameEmpty
    case propertyTypeMissing(String)
    case propertyTypeUnsupported(String, String)
    case propertyEnumRequiresStringType(String)
    case propertyEnumInvalid(String)
    case propertyEnumDuplicated(String, String)
    case propertyKeywordUnsupported(String, String)
    case invalidJSONString(String)
    case schemaNotSerializable
    case invalidName
    case strictNotSupported

    public var message: String {
        switch self {
        case .rootNotObject:
            return "schema must be a JSON object"
        case .rootKeywordUnsupported(let keyword):
            return
                "unsupported JSON Schema keyword '\(keyword)' at the schema root; supported root keywords: type, properties, required, additionalProperties"
        case .rootTypeNotObject:
            return "schema root must declare \"type\": \"object\""
        case .additionalPropertiesMustBeFalse:
            return "schema \"additionalProperties\" must be exactly false"
        case .propertiesInvalid:
            return "schema \"properties\" is required and must be an object"
        case .requiredInvalid:
            return "schema \"required\" is required and must be an array of strings"
        case .requiredNameDuplicated(let name):
            return "duplicate name '\(name)' in \"required\""
        case .requiredNameNotDefined(let name):
            return "\"required\" names '\(name)' but it is not defined in \"properties\""
        case .propertyNotObject(let name):
            return "property '\(name)' must be a JSON object"
        case .propertyNameEmpty:
            return "property names must be non-empty strings"
        case .propertyTypeMissing(let name):
            return "property '\(name)' must declare \"type\" (one of: string, number, integer, boolean)"
        case .propertyTypeUnsupported(let name, let type):
            return
                "property '\(name)' declares unsupported type '\(type)'; supported types: string, number, integer, boolean (nested objects/arrays are not supported in this subset)"
        case .propertyEnumRequiresStringType(let name):
            return "property '\(name)' declares \"enum\" but its type is not \"string\"; only string enums are supported"
        case .propertyEnumInvalid(let name):
            return "property '\(name)' has an invalid \"enum\": it must be a non-empty array of unique strings"
        case .propertyEnumDuplicated(let name, let value):
            return "property '\(name)' repeats enum value '\(value)'"
        case .propertyKeywordUnsupported(let name, let keyword):
            return "property '\(name)' uses unsupported JSON Schema keyword '\(keyword)'; supported: type, enum"
        case .invalidJSONString(let string):
            return "string '\(string)' cannot be represented in JSON"
        case .schemaNotSerializable:
            return "schema could not be canonicalized to JSON"
        case .invalidName:
            return "json_schema name must be a non-empty string"
        case .strictNotSupported:
            return "only strict schemas (\"strict\": true) are supported"
        }
    }
}

extension JSONSchemaCompileError: LocalizedError {
    public var errorDescription: String? { message }
}

// MARK: - Helpers

/// A string is JSON-representable iff it survives strict UTF-8/JSON encoding.
/// Swift's String views repair ill-formed UTF-16, so the strict serialization
/// path is the honest check for lone surrogates and similar damage.
private func isJSONRepresentable(_ string: String) -> Bool {
    (try? JSONSerialization.data(withJSONObject: [string])) != nil
}

private func jsonKindName(_ value: MeiJSONValue) -> String {
    switch value {
    case .string: return "string"
    case .number: return "number"
    case .bool: return "boolean"
    case .object: return "object"
    case .array: return "array"
    case .null: return "null"
    }
}
