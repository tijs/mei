import CryptoKit
import Foundation

// MARK: - Model-free schema/format compiler boundary

/// The tokenizer- and engine-independent half of structured generation.
///
/// It validates the supported strict JSON-Schema subset — `strict: true`, a
/// root object, recursively nested strict objects (`properties`, `required`,
/// `additionalProperties: false` at every level), arrays with `items` and
/// optional `minItems`/`maxItems`, scalar fields (`string`/`number`/
/// `integer`/`boolean`), nullable scalar unions (`type: [scalar, "null"]`),
/// enums on every scalar type (with an optional null member on nullable
/// fields), and numeric constraints on `number`/`integer` (`minimum`,
/// `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`) — and
/// compiles it into a deterministic constraint value plus cache key.
/// Unsupported keywords and ambiguous constructs are rejected explicitly
/// (fail closed) instead of being ignored or downgraded.
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
        let rootSchema = try compileObjectSchema(root, path: [])
        let canonical = try canonicalJSONString(name: format.name, root: rootSchema)
        return CompiledJSONSchema(
            name: format.name,
            root: rootSchema,
            constraintKey: "json_schema:v1:" + sha256Hex(canonical))
    }

    // MARK: - Value compilation

    /// Compile one object schema node (root or nested). Every strict object
    /// shares the same shape: `type: "object"`, `properties`, `required`,
    /// `additionalProperties: false`. Errors at nested nodes are located by
    /// their path from the root.
    private static func compileObjectSchema(
        _ raw: [String: MeiJSONValue],
        path: [String]
    ) throws -> CompiledJSONSchema.ObjectSchema {
        let allowedKeys: Set<String> = ["type", "properties", "required", "additionalProperties"]
        for key in raw.keys.sorted() where !allowedKeys.contains(key) {
            if path.isEmpty { throw JSONSchemaCompileError.rootKeywordUnsupported(key) }
            throw JSONSchemaCompileError.objectKeywordUnsupported(pathString(path), key)
        }
        guard case .bool(false)? = raw["additionalProperties"] else {
            throw locateObjectShapeError(path, .additionalPropertiesMustBeFalse)
        }
        guard case .object(let rawProperties)? = raw["properties"] else {
            throw locateObjectShapeError(path, .propertiesInvalid)
        }
        guard case .array(let requiredValues)? = raw["required"] else {
            throw locateObjectShapeError(path, .requiredInvalid)
        }
        var requiredNames: [String] = []
        var seenRequired = Set<String>()
        for value in requiredValues {
            guard case .string(let name) = value else {
                throw locateObjectShapeError(path, .requiredInvalid)
            }
            guard isJSONRepresentable(name) else { throw JSONSchemaCompileError.invalidJSONString(name) }
            guard seenRequired.insert(name).inserted else {
                throw locateObjectShapeError(path, .requiredNameDuplicated(name))
            }
            requiredNames.append(name)
        }

        var properties: [CompiledJSONSchema.Property] = []
        for name in rawProperties.keys.sorted() {
            guard !name.isEmpty else { throw locateObjectShapeError(path, .propertyNameEmpty) }
            guard isJSONRepresentable(name) else { throw JSONSchemaCompileError.invalidJSONString(name) }
            properties.append(
                CompiledJSONSchema.Property(
                    name: name,
                    value: try compileValueSchema(rawProperties[name]!, path: path + [name])))
        }

        let definedNames = Set(properties.map(\.name))
        for name in requiredNames.sorted() where !definedNames.contains(name) {
            throw locateObjectShapeError(path, .requiredNameNotDefined(name))
        }
        return CompiledJSONSchema.ObjectSchema(properties: properties, required: requiredNames.sorted())
    }

    /// Compile one value schema: a scalar, a nullable scalar union, a nested
    /// object, or an array of any supported value (recursive). Errors name the
    /// schema node's path, so deep rejections stay traceable.
    private static func compileValueSchema(
        _ value: MeiJSONValue,
        path: [String]
    ) throws -> CompiledJSONSchema.Value {
        guard case .object(let raw) = value else {
            throw JSONSchemaCompileError.propertyNotObject(pathString(path))
        }
        switch raw["type"] {
        case .string(let typeName)?:
            switch typeName {
            case "object":
                return .object(try compileObjectSchema(raw, path: path))
            case "array":
                return .array(try compileArrayItems(raw, path: path))
            default:
                guard let scalarType = CompiledJSONSchema.ScalarType(rawValue: typeName) else {
                    throw JSONSchemaCompileError.propertyTypeUnsupported(pathString(path), typeName)
                }
                return .scalar(try compileScalar(raw, baseType: scalarType, nullable: false, path: path))
            }
        case .array(let entries)?:
            let scalarType = try nullableUnionScalarType(entries, path: path)
            return .scalar(try compileScalar(raw, baseType: scalarType, nullable: true, path: path))
        case nil, .null?:
            throw JSONSchemaCompileError.propertyTypeMissing(pathString(path))
        default:
            throw JSONSchemaCompileError.propertyTypeUnsupported(
                pathString(path), jsonKindName(raw["type"]!))
        }
    }

    /// Compile an array node: only `type`, `items`, `minItems`, and
    /// `maxItems` are allowed; `items` is required and is itself a supported
    /// value schema; the count keywords are non-negative integers and must
    /// form a non-empty range.
    private static func compileArrayItems(
        _ raw: [String: MeiJSONValue],
        path: [String]
    ) throws -> CompiledJSONSchema.ArraySchema {
        let allowedKeys: Set<String> = ["type", "items", "minItems", "maxItems"]
        for key in raw.keys.sorted() where !allowedKeys.contains(key) {
            throw JSONSchemaCompileError.arrayKeywordUnsupported(pathString(path), key)
        }
        guard case .object(let items)? = raw["items"] else {
            throw JSONSchemaCompileError.arrayItemsInvalid(pathString(path))
        }
        let itemValue = try compileValueSchema(.object(items), path: path + ["items"])
        let minItems = try compileArrayCount(raw, keyword: "minItems", path: path)
        let maxItems = try compileArrayCount(raw, keyword: "maxItems", path: path)
        if let minItems, let maxItems, minItems > maxItems {
            throw JSONSchemaCompileError.arrayCountRangeInvalid(pathString(path))
        }
        return CompiledJSONSchema.ArraySchema(items: itemValue, minItems: minItems, maxItems: maxItems)
    }

    /// Compile `minItems`/`maxItems`: a non-negative integer item count.
    private static func compileArrayCount(
        _ raw: [String: MeiJSONValue],
        keyword: String,
        path: [String]
    ) throws -> Int? {
        guard let value = raw[keyword] else { return nil }
        guard case .number(let number) = value, number.isFinite,
            number >= 0, number.rounded() == number,
            let count = Int(exactly: number)
        else {
            throw JSONSchemaCompileError.arrayCountConstraintInvalid(pathString(path), keyword)
        }
        return count
    }

    /// Parse the only supported `type` union: exactly one supported scalar
    /// plus `"null"`, in either order. Anything else (two scalars, a non-scalar
    /// member, duplicates, other lengths) is rejected.
    private static func nullableUnionScalarType(
        _ entries: [MeiJSONValue],
        path: [String]
    ) throws -> CompiledJSONSchema.ScalarType {
        guard entries.count == 2,
            case .string(let first) = entries[0],
            case .string(let second) = entries[1]
        else {
            throw JSONSchemaCompileError.propertyTypeUnionUnsupported(pathString(path))
        }
        let scalarName: String
        if first == "null" {
            scalarName = second
        } else if second == "null" {
            scalarName = first
        } else {
            throw JSONSchemaCompileError.propertyTypeUnionUnsupported(pathString(path))
        }
        guard scalarName != "null", let scalarType = CompiledJSONSchema.ScalarType(rawValue: scalarName) else {
            throw JSONSchemaCompileError.propertyTypeUnionUnsupported(pathString(path))
        }
        return scalarType
    }

    /// Compile a scalar constraint (optionally nullable) with an optional
    /// enum and, for `number`/`integer`, optional numeric constraints.
    ///
    /// - `enum` values must match the declared scalar type; string enums are
    ///   as before, number/integer enums compare by exact decimal value, and
    ///   boolean enums list `true`/`false`.
    /// - A null enum member requires the nullable union; when a declared enum
    ///   omits null, null is not an accepted value (JSON-Schema intersection).
    /// - Numeric keywords are accepted only on `number`/`integer`, must be
    ///   finite JSON numbers, and `multipleOf` must be strictly positive.
    /// - The declared constraints must be satisfiable for the declared type;
    ///   a schema that can never generate a value is rejected here instead of
    ///   failing closed mid-generation.
    private static func compileScalar(
        _ raw: [String: MeiJSONValue],
        baseType: CompiledJSONSchema.ScalarType,
        nullable: Bool,
        path: [String]
    ) throws -> CompiledJSONSchema.Scalar {
        let numericKeywords: Set<String> = [
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        ]
        var allowedKeys: Set<String> = ["type", "enum"]
        if baseType == .number || baseType == .integer { allowedKeys.formUnion(numericKeywords) }
        for key in raw.keys.sorted() where !allowedKeys.contains(key) {
            throw JSONSchemaCompileError.propertyKeywordUnsupported(pathString(path), key)
        }

        var enumValues: CompiledJSONSchema.ScalarEnumValues? = nil
        var enumAllowsNull = false
        if let enumValue = raw["enum"] {
            guard case .array(let values) = enumValue, !values.isEmpty else {
                throw JSONSchemaCompileError.propertyEnumInvalid(pathString(path))
            }
            var strings: [String] = []
            var numbers: [DecimalLiteral] = []
            var booleans: [Bool] = []
            var seenStrings = Set<String>()
            var seenNumbers = Set<DecimalLiteral>()
            var seenBooleans = Set<Bool>()
            for entry in values {
                switch entry {
                case .string(let string):
                    guard baseType == .string else {
                        throw JSONSchemaCompileError.propertyEnumValueTypeMismatch(pathString(path))
                    }
                    guard isJSONRepresentable(string) else {
                        throw JSONSchemaCompileError.invalidJSONString(string)
                    }
                    guard seenStrings.insert(string).inserted else {
                        throw JSONSchemaCompileError.propertyEnumDuplicated(pathString(path), string)
                    }
                    strings.append(string)
                case .number(let number):
                    guard baseType == .number || baseType == .integer, number.isFinite,
                        let decimal = DecimalLiteral(double: number)
                    else {
                        throw JSONSchemaCompileError.propertyEnumValueTypeMismatch(pathString(path))
                    }
                    if baseType == .integer, !decimal.isInteger {
                        throw JSONSchemaCompileError.propertyEnumValueTypeMismatch(pathString(path))
                    }
                    guard seenNumbers.insert(decimal).inserted else {
                        throw JSONSchemaCompileError.propertyEnumDuplicated(pathString(path), decimal.canonicalText)
                    }
                    numbers.append(decimal)
                case .bool(let bool):
                    guard baseType == .boolean else {
                        throw JSONSchemaCompileError.propertyEnumValueTypeMismatch(pathString(path))
                    }
                    guard seenBooleans.insert(bool).inserted else {
                        throw JSONSchemaCompileError.propertyEnumDuplicated(
                            pathString(path), bool ? "true" : "false")
                    }
                    booleans.append(bool)
                case .null:
                    guard nullable else {
                        throw JSONSchemaCompileError.propertyEnumInvalid(pathString(path))
                    }
                    guard !enumAllowsNull else {
                        throw JSONSchemaCompileError.propertyEnumDuplicated(pathString(path), "null")
                    }
                    enumAllowsNull = true
                case .object, .array:
                    throw JSONSchemaCompileError.propertyEnumValueTypeMismatch(pathString(path))
                }
            }
            switch baseType {
            case .string: enumValues = .strings(strings.sorted())
            case .number, .integer: enumValues = .numbers(numbers.sorted())
            case .boolean: enumValues = .booleans(booleans.sorted { !$0 && $1 })
            }
        }

        var numeric: NumericConstraints? = nil
        if baseType == .number || baseType == .integer {
            func constraint(_ keyword: String) throws -> DecimalLiteral? {
                guard let value = raw[keyword] else { return nil }
                guard case .number(let number) = value, number.isFinite,
                    let decimal = DecimalLiteral(double: number)
                else {
                    throw JSONSchemaCompileError.propertyNumericConstraintInvalid(pathString(path), keyword)
                }
                return decimal
            }
            let minimum = try constraint("minimum")
            let maximum = try constraint("maximum")
            let exclusiveMinimum = try constraint("exclusiveMinimum")
            let exclusiveMaximum = try constraint("exclusiveMaximum")
            var multipleOf = try constraint("multipleOf")
            if let value = multipleOf {
                guard value > .zero else {
                    throw JSONSchemaCompileError.propertyNumericConstraintInvalid(pathString(path), "multipleOf")
                }
                multipleOf = value
            }
            if minimum != nil || maximum != nil || exclusiveMinimum != nil || exclusiveMaximum != nil
                || multipleOf != nil {
                numeric = NumericConstraints(
                    minimum: minimum, maximum: maximum,
                    exclusiveMinimum: exclusiveMinimum, exclusiveMaximum: exclusiveMaximum,
                    multipleOf: multipleOf)
            }
        }

        let scalar = CompiledJSONSchema.Scalar(
            type: baseType, enumValues: enumValues, enumAllowsNull: enumAllowsNull, nullable: nullable,
            numeric: numeric)

        // Fail closed on a schema that can never generate a value.
        if baseType == .number || baseType == .integer {
            let constraintSet =
                scalar.numericConstraintSet ?? NumericConstraintSet(bounds: nil, allowedValues: nil)
            if !scalar.acceptsNull, !constraintSet.isSatisfiable(kind: baseType) {
                throw JSONSchemaCompileError.propertyConstraintsUnsatisfiable(pathString(path))
            }
        }
        return scalar
    }

    /// Dot-joined path of a schema node from the root (array items use the
    /// `items` segment, e.g. `rows.items.v`).
    private static func pathString(_ path: [String]) -> String {
        path.joined(separator: ".")
    }

    /// Locate an object-node shape error: the root keeps its bare error
    /// (there is only one root), any nested object is wrapped with its path.
    private static func locateObjectShapeError(
        _ path: [String],
        _ error: JSONSchemaCompileError
    ) -> JSONSchemaCompileError {
        path.isEmpty ? error : .nested(pathString(path), error)
    }

    // MARK: - Canonicalization

    /// Deterministic canonical JSON for the compiled format: property order,
    /// `required` order, enum order, and nullable type-union order are
    /// normalized so equivalent schemas produce byte-identical canonical text
    /// (and therefore the same key).
    private static func canonicalJSONString(
        name: String,
        root: CompiledJSONSchema.ObjectSchema
    ) throws -> String {
        let formatTree: [String: Any] = [
            "type": "json_schema",
            "json_schema": [
                "name": name,
                "strict": true,
                "schema": canonicalObjectTree(root),
            ] as [String: Any],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: formatTree, options: [.sortedKeys]),
            let string = String(data: data, encoding: .utf8)
        else {
            throw JSONSchemaCompileError.schemaNotSerializable
        }
        return string
    }

    /// Canonical JSON tree for one object node (recursive).
    private static func canonicalObjectTree(_ object: CompiledJSONSchema.ObjectSchema) -> Any {
        var propertyTree: [String: Any] = [:]
        for property in object.properties {
            propertyTree[property.name] = canonicalValueTree(property.value)
        }
        return [
            "type": "object",
            "properties": propertyTree,
            "required": object.required,
            "additionalProperties": false,
        ]
    }

    /// Canonical JSON tree for one value schema (recursive). A nullable type
    /// union always renders as `[scalar, "null"]` regardless of input order,
    /// and a null enum member always renders last.
    private static func canonicalValueTree(_ value: CompiledJSONSchema.Value) -> Any {
        switch value {
        case .scalar(let scalar):
            var field: [String: Any] = [:]
            field["type"] =
                scalar.nullable ? [scalar.type.rawValue, "null"] : scalar.type.rawValue
            if let enumValues = scalar.enumValues {
                field["enum"] = canonicalEnumTree(enumValues, allowsNull: scalar.enumAllowsNull)
            }
            if let numeric = scalar.numeric {
                if let minimum = numeric.minimum { field["minimum"] = minimum.canonicalText }
                if let maximum = numeric.maximum { field["maximum"] = maximum.canonicalText }
                if let exclusiveMinimum = numeric.exclusiveMinimum {
                    field["exclusiveMinimum"] = exclusiveMinimum.canonicalText
                }
                if let exclusiveMaximum = numeric.exclusiveMaximum {
                    field["exclusiveMaximum"] = exclusiveMaximum.canonicalText
                }
                if let multipleOf = numeric.multipleOf { field["multipleOf"] = multipleOf.canonicalText }
            }
            return field
        case .object(let object):
            return canonicalObjectTree(object)
        case .array(let schema):
            var field: [String: Any] = ["type": "array", "items": canonicalValueTree(schema.items)]
            if let minItems = schema.minItems { field["minItems"] = minItems }
            if let maxItems = schema.maxItems { field["maxItems"] = maxItems }
            return field
        }
    }

    /// Canonical enum tree: string and boolean members render as JSON values;
    /// number members render as `{"number": "<exact decimal>"}` so the tree
    /// stays injective and free of binary floating-point artifacts. A null
    /// member always renders last.
    private static func canonicalEnumTree(
        _ values: CompiledJSONSchema.ScalarEnumValues, allowsNull: Bool
    ) -> [Any] {
        var tree: [Any]
        switch values {
        case .strings(let strings): tree = strings
        case .numbers(let numbers): tree = numbers.map { ["number": $0.canonicalText] }
        case .booleans(let booleans): tree = booleans
        }
        if allowsNull { tree.append(NSNull()) }
        return tree
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

/// Compiled strict schema: a root object whose fields may be scalars,
/// nullable scalar unions, nested strict objects, or arrays of any supported
/// value — recursively. Canonical (name-sorted) order makes the value and its
/// `constraintKey` deterministic.
public struct CompiledJSONSchema: Equatable, Sendable {
    public let name: String
    /// The root object schema (the strict subset always roots in an object).
    public let root: ObjectSchema
    /// SHA-256 over the canonical JSON serialization of the format.
    public let constraintKey: String

    /// Root field constraints in canonical (name-sorted) order.
    public var properties: [Property] { root.properties }
    /// Root required field names in canonical (name-sorted) order.
    public var required: [String] { root.required }

    /// A strict object node: fields plus required names.
    public struct ObjectSchema: Equatable, Sendable {
        /// Field constraints in canonical (name-sorted) order.
        public let properties: [Property]
        /// Required field names in canonical (name-sorted) order.
        public let required: [String]
    }

    /// One named field of an object node.
    public struct Property: Equatable, Sendable {
        public let name: String
        public let value: Value
    }

    /// A supported value schema, recursively.
    public indirect enum Value: Equatable, Sendable {
        /// A scalar (or `[scalar, "null"]` union) with optional enum and
        /// numeric constraints.
        case scalar(Scalar)
        /// A strict nested object.
        case object(ObjectSchema)
        /// An array with a supported `items` schema and optional
        /// `minItems`/`maxItems`.
        case array(ArraySchema)
    }

    /// An array schema node: the item schema plus optional count constraints.
    public struct ArraySchema: Equatable, Sendable {
        public let items: Value
        public let minItems: Int?
        public let maxItems: Int?
    }

    /// A scalar constraint: one scalar type, optionally nullable, with an
    /// optional enum and optional numeric constraints (bounds and
    /// `multipleOf` for `number`/`integer`). `enumAllowsNull` records whether
    /// a declared enum listed null itself — a nullable type union whose enum
    /// omits null rejects null (the JSON-Schema intersection semantics).
    public struct Scalar: Equatable, Sendable {
        public let type: ScalarType
        /// Non-nil iff the schema declared an enum; the case matches `type`
        /// and the values are in canonical order.
        public let enumValues: ScalarEnumValues?
        /// True iff the declared enum listed null (only meaningful with an enum).
        public let enumAllowsNull: Bool
        /// True iff the `type` union included `"null"`.
        public let nullable: Bool
        /// Non-nil iff the schema declared numeric constraints
        /// (`number`/`integer` types only).
        public let numeric: NumericConstraints?

        /// String enum values in canonical order (accessor for the flat
        /// milestone's string-enum shape).
        public var allowedValues: [String]? {
            if case .strings(let values)? = enumValues { return values }
            return nil
        }

        /// Whether null is an accepted value: the type union must allow it and
        /// a declared enum must not exclude it.
        public var acceptsNull: Bool { nullable && (enumValues == nil || enumAllowsNull) }

        /// The number-grammar constraint set for `number`/`integer` scalars,
        /// or nil when the field is unconstrained (the byte-compatible fast
        /// path: no per-byte feasibility work happens).
        var numericConstraintSet: NumericConstraintSet? {
            guard type == .number || type == .integer else { return nil }
            var numbers: [DecimalLiteral]? = nil
            if case .numbers(let values)? = enumValues { numbers = values }
            guard numeric != nil || numbers != nil else { return nil }
            return NumericConstraintSet(bounds: numeric, allowedValues: numbers)
        }
    }

    /// Enum values for a scalar field; the case matches the field's type.
    public enum ScalarEnumValues: Equatable, Sendable {
        case strings([String])
        case numbers([DecimalLiteral])
        case booleans([Bool])
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
        root.accepts(value)
    }
}

extension CompiledJSONSchema.ObjectSchema {
    /// `additionalProperties: false` semantics: keys must be declared, every
    /// required name must be present, and every value must satisfy its field
    /// schema.
    func accepts(_ value: MeiJSONValue) -> Bool {
        guard case .object(let object) = value else { return false }
        var constraints: [String: CompiledJSONSchema.Property] = [:]
        constraints.reserveCapacity(properties.count)
        for property in properties {
            constraints[property.name] = property
        }
        for key in object.keys where constraints[key] == nil {
            return false
        }
        for name in required where object[name] == nil {
            return false
        }
        for (key, fieldValue) in object {
            guard let property = constraints[key], property.value.accepts(fieldValue) else { return false }
        }
        return true
    }
}

extension CompiledJSONSchema.Value {
    func accepts(_ value: MeiJSONValue) -> Bool {
        switch self {
        case .scalar(let scalar):
            return scalar.accepts(value)
        case .object(let object):
            return object.accepts(value)
        case .array(let schema):
            guard case .array(let array) = value else { return false }
            if let minItems = schema.minItems, array.count < minItems { return false }
            if let maxItems = schema.maxItems, array.count > maxItems { return false }
            return array.allSatisfy { schema.items.accepts($0) }
        }
    }
}

extension CompiledJSONSchema.Scalar {
    func accepts(_ value: MeiJSONValue) -> Bool {
        switch value {
        case .null:
            return acceptsNull
        case .string(let string):
            guard type == .string, isJSONRepresentable(string) else { return false }
            if case .strings(let allowed)? = enumValues, !allowed.contains(string) { return false }
            return true
        case .number(let number):
            guard (type == .number || type == .integer), number.isFinite,
                let decimal = DecimalLiteral(double: number)
            else { return false }
            if type == .integer, !decimal.isInteger { return false }
            if let constraintSet = numericConstraintSet, !constraintSet.accepts(decimal) { return false }
            return true
        case .bool(let bool):
            guard type == .boolean else { return false }
            if case .booleans(let allowed)? = enumValues, !allowed.contains(bool) { return false }
            return true
        case .object, .array:
            return false
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
    /// A `type` array outside the single-scalar-plus-`null` union form.
    case propertyTypeUnionUnsupported(String)
    /// An array property whose `items` is missing, not an object, or not a
    /// supported value schema (the payload is the array schema's path).
    case arrayItemsInvalid(String)
    /// An object schema node (not the root) with an unsupported keyword; the
    /// payload is the node's path and the keyword.
    case objectKeywordUnsupported(String, String)
    /// An array schema node with an unsupported keyword; the payload is the
    /// node's path and the keyword.
    case arrayKeywordUnsupported(String, String)
    /// An array node whose `minItems`/`maxItems` is not a non-negative
    /// integer; the payload is the node's path and the keyword.
    case arrayCountConstraintInvalid(String, String)
    /// An array node whose `minItems` exceeds its `maxItems`.
    case arrayCountRangeInvalid(String)
    /// An error inside a nested value schema, located by its path from the
    /// root (property names joined with `.`, array items as `items`). The
    /// underlying error is never itself a `.nested`.
    indirect case nested(String, JSONSchemaCompileError)
    /// An `enum` member whose JSON type does not match the declared scalar
    /// type (including fractional members of `integer` enums).
    case propertyEnumValueTypeMismatch(String)
    case propertyEnumInvalid(String)
    case propertyEnumDuplicated(String, String)
    case propertyKeywordUnsupported(String, String)
    /// A numeric constraint keyword whose value is not a finite JSON number
    /// (or `multipleOf` is not strictly positive); the payload is the
    /// property's path and the keyword.
    case propertyNumericConstraintInvalid(String, String)
    /// The declared constraints cannot be satisfied by any value of the
    /// declared type (and the field cannot be null).
    case propertyConstraintsUnsatisfiable(String)
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
            return
                "property '\(name)' must declare \"type\" (one of: string, number, integer, boolean, object, array, or exactly one of the scalars plus \"null\")"
        case .propertyTypeUnsupported(let name, let type):
            return
                "property '\(name)' declares unsupported type '\(type)'; supported: string, number, integer, boolean, object, array, or exactly one of the scalars plus \"null\""
        case .propertyTypeUnionUnsupported(let name):
            return
                "property '\(name)' declares an unsupported type union; supported unions are exactly one of \"string\", \"number\", \"integer\", \"boolean\" plus \"null\" (e.g. [\"string\", \"null\"])"
        case .arrayItemsInvalid(let path):
            return "array schema at '\(path)' must declare \"items\" as a supported schema object"
        case .objectKeywordUnsupported(let path, let keyword):
            return
                "object schema at '\(path)' uses unsupported JSON Schema keyword '\(keyword)'; supported: type, properties, required, additionalProperties"
        case .arrayKeywordUnsupported(let path, let keyword):
            return
                "array schema at '\(path)' uses unsupported JSON Schema keyword '\(keyword)'; supported: type, items, minItems, maxItems"
        case .arrayCountConstraintInvalid(let path, let keyword):
            return
                "array schema at '\(path)' declares an invalid \"\(keyword)\": it must be a non-negative integer"
        case .arrayCountRangeInvalid(let path):
            return "array schema at '\(path)' declares minItems > maxItems; no array can satisfy it"
        case .nested(let path, let underlying):
            return "at '\(path)': \(underlying.message)"
        case .propertyEnumValueTypeMismatch(let name):
            return
                "property '\(name)' has an \"enum\" member whose type does not match the declared scalar type (integer enums must be whole numbers)"
        case .propertyEnumInvalid(let name):
            return "property '\(name)' has an invalid \"enum\": it must be a non-empty array of unique values matching the declared type"
        case .propertyEnumDuplicated(let name, let value):
            return "property '\(name)' repeats enum value '\(value)'"
        case .propertyKeywordUnsupported(let name, let keyword):
            return
                "property '\(name)' uses unsupported JSON Schema keyword '\(keyword)'; supported: type, enum, and on number/integer minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf"
        case .propertyNumericConstraintInvalid(let name, let keyword):
            return
                "property '\(name)' declares an invalid \"\(keyword)\": it must be a finite JSON number (and multipleOf must be strictly positive)"
        case .propertyConstraintsUnsatisfiable(let name):
            return
                "property '\(name)' declares numeric constraints that no value of the declared type can satisfy"
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
