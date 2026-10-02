import XCTest
@testable import MeiCore

/// Model-free tests for the schema/format compiler boundary — slice 2 of the
/// structured-generation plan.
///
/// The compiler is deliberately tokenizer-, engine-, and HTTP-free: it
/// validates the supported strict JSON-Schema subset (`strict: true`, root
/// object, `properties`, `required`, `additionalProperties: false`, scalar
/// fields, string `enum`) and produces a deterministic compiled constraint
/// plus cache key. Unsupported keywords and ambiguous constructs are rejected
/// explicitly (fail closed) rather than ignored or downgraded.
///
/// The CoCore structured-output canary schema (`{"status": {"type": "string",
/// "enum": ["ok"]}}` under a strict flat object) is the first supported shape;
/// the post-generation validator here is a diagnostic building block only —
/// the real guarantee must come from token-level constrained decoding.
final class JSONSchemaCompilerTests: XCTestCase {

    // MARK: - Helpers

    private func mei(_ json: String) throws -> MeiJSONValue {
        try JSONDecoder().decode(MeiJSONValue.self, from: Data(json.utf8))
    }

    private func compileSchema(_ schemaJSON: String, name: String = "canary_status") throws -> CompiledJSONSchema {
        try JSONSchemaCompiler.compile(JSONSchemaFormat(name: name, strict: true, schema: try mei(schemaJSON)))
    }

    /// The exact schema CoCore's structured-output canary sends
    /// (docs/COCORE.md §2).
    private func canarySchema() throws -> CompiledJSONSchema {
        try compileSchema(
            #"{"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}}, "required": ["status"], "additionalProperties": false}"#
        )
    }

    /// A lone UTF-16 surrogate: not representable in JSON/UTF-8, only
    /// constructible programmatically (Swift literals refuse it and
    /// JSONDecoder rejects the escape).
    private var invalidJSONString: String { String(utf16CodeUnits: [0xD800], count: 1) }

    // MARK: - Supported subset compiles

    func testCoCoreFlatObjectSchemaCompiles() throws {
        let compiled = try canarySchema()
        XCTAssertEqual(compiled.name, "canary_status")
        XCTAssertEqual(compiled.required, ["status"])
        XCTAssertEqual(compiled.properties.count, 1)
        let status = try XCTUnwrap(compiled.properties.first)
        XCTAssertEqual(status.name, "status")
        XCTAssertEqual(status.type, .string)
        XCTAssertEqual(status.allowedValues, ["ok"])
        XCTAssertFalse(compiled.constraintKey.isEmpty)
        XCTAssertEqual(
            try canarySchema().constraintKey, compiled.constraintKey,
            "compiling the same schema twice must produce the same key")
    }

    func testEmptyObjectSchemaCompiles() throws {
        let compiled = try compileSchema(
            #"{"type": "object", "properties": {}, "required": [], "additionalProperties": false}"#)
        XCTAssertTrue(compiled.properties.isEmpty)
        XCTAssertTrue(compiled.required.isEmpty)
        XCTAssertTrue(compiled.validate(try mei("{}")))
        XCTAssertFalse(compiled.validate(try mei(#"{"x": 1}"#)))
    }

    func testTextAndJSONObjectFormatsCompile() throws {
        XCTAssertEqual(try JSONSchemaCompiler.compile(ResponseFormat.text), .text)
        XCTAssertEqual(try JSONSchemaCompiler.compile(ResponseFormat.jsonObject), .jsonObject)
        XCTAssertEqual(try JSONSchemaCompiler.compile(ResponseFormat.text).constraintKey, "text")
        XCTAssertEqual(try JSONSchemaCompiler.compile(ResponseFormat.jsonObject).constraintKey, "json_object")
    }

    /// The full request path: the decoded canary body compiles to the same
    /// constraint as the schema compiled directly.
    func testFullFormatCompilesFromDecodedRequest() throws {
        let json = #"""
        {"model": "m", "messages": [{"role": "user", "content": "hi"}],
         "response_format": {"type": "json_schema", "json_schema": {"name": "canary_status", "strict": true,
           "schema": {"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}},
                      "required": ["status"], "additionalProperties": false}}}}
        """#
        let request = try ChatRequest(json: Data(json.utf8))
        guard case .jsonSchema(let compiled) = try JSONSchemaCompiler.compile(request.responseFormat) else {
            return XCTFail("expected a compiled json_schema constraint")
        }
        XCTAssertEqual(compiled.required, ["status"])
        XCTAssertEqual(compiled.constraintKey, try canarySchema().constraintKey)
    }

    // MARK: - Instance validation (diagnostic building block)

    func testRequiredKeysAreEnforced() throws {
        let compiled = try canarySchema()
        XCTAssertTrue(compiled.validate(try mei(#"{"status": "ok"}"#)))
        XCTAssertFalse(compiled.validate(try mei("{}")), "a missing required key must fail closed")
    }

    func testExtraKeysFailClosed() throws {
        let compiled = try canarySchema()
        XCTAssertFalse(
            compiled.validate(try mei(#"{"status": "ok", "extra": true}"#)),
            "additionalProperties: false must reject extra keys")
    }

    func testEnumValuesAreEnforced() throws {
        let compiled = try canarySchema()
        XCTAssertTrue(compiled.validate(try mei(#"{"status": "ok"}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"status": "nope"}"#)), "value outside the enum must fail closed")
        XCTAssertFalse(compiled.validate(try mei(#"{"status": 1}"#)), "non-string enum value must fail closed")
    }

    func testWrongScalarTypesFailClosed() throws {
        let compiled = try compileSchema(
            #"{"type": "object", "properties": {"s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": "number"}, "b": {"type": "boolean"}}, "required": ["s", "i", "n", "b"], "additionalProperties": false}"#
        )
        XCTAssertTrue(compiled.validate(try mei(#"{"s": "x", "i": 3, "n": 1.5, "b": true}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"s": "x", "i": 4, "n": 2, "b": false}"#)), "integer accepts integral numbers")
        XCTAssertFalse(compiled.validate(try mei(#"{"s": 1, "i": 3, "n": 1.5, "b": true}"#)), "string field given a number")
        XCTAssertFalse(compiled.validate(try mei(#"{"s": "x", "i": 3.5, "n": 1.5, "b": true}"#)), "integer field given a fraction")
        XCTAssertFalse(compiled.validate(try mei(#"{"s": "x", "i": 3, "n": "1.5", "b": true}"#)), "number field given a string")
        XCTAssertFalse(compiled.validate(try mei(#"{"s": "x", "i": 3, "n": 1.5, "b": 1}"#)), "boolean field given a number")
        XCTAssertFalse(compiled.validate(try mei(#"{"s": null, "i": 3, "n": 1.5, "b": true}"#)), "null is not a scalar field value")
    }

    func testNonFiniteNumbersFailClosed() throws {
        let compiled = try compileSchema(
            #"{"type": "object", "properties": {"n": {"type": "number"}, "i": {"type": "integer"}}, "required": ["n", "i"], "additionalProperties": false}"#
        )
        XCTAssertFalse(
            compiled.validate(.object(["n": .number(.nan), "i": .number(1)])),
            "NaN is not a valid JSON number")
        XCTAssertFalse(
            compiled.validate(.object(["n": .number(.infinity), "i": .number(1)])),
            "infinity is not a valid JSON number")
        XCTAssertFalse(
            compiled.validate(.object(["n": .number(1), "i": .number(.nan)])),
            "NaN is not a valid integer")
        XCTAssertTrue(compiled.validate(.object(["n": .number(1.5), "i": .number(2)])))
    }

    func testInvalidJSONStringsFailClosed() throws {
        // Instance side: a string that cannot be represented in JSON/UTF-8.
        let compiled = try compileSchema(
            #"{"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"], "additionalProperties": false}"#
        )
        XCTAssertFalse(
            compiled.validate(.object(["s": .string(invalidJSONString)])),
            "a non-JSON-representable string must fail closed")

        // Schema side: property names, required names, and enum values must be
        // representable; otherwise the compiled constraint/key would not be
        // well-defined.
        let badPropertyName = MeiJSONValue.object([
            "type": .string("object"),
            "properties": .object([invalidJSONString: .object(["type": .string("string")])]),
            "required": .array([.string(invalidJSONString)]),
            "additionalProperties": .bool(false),
        ])
        XCTAssertThrowsError(
            try JSONSchemaCompiler.compile(JSONSchemaFormat(name: "x", strict: true, schema: badPropertyName))
        ) { error in
            guard case .invalidJSONString(let bad)? = error as? JSONSchemaCompileError else {
                return XCTFail("expected invalidJSONString, got \(error)")
            }
            XCTAssertEqual(bad, self.invalidJSONString)
        }

        let badEnumValue = MeiJSONValue.object([
            "type": .string("object"),
            "properties": .object([
                "status": .object(["type": .string("string"), "enum": .array([.string(invalidJSONString)])])
            ]),
            "required": .array([.string("status")]),
            "additionalProperties": .bool(false),
        ])
        XCTAssertThrowsError(
            try JSONSchemaCompiler.compile(JSONSchemaFormat(name: "x", strict: true, schema: badEnumValue))
        ) { error in
            guard case .invalidJSONString(let bad)? = error as? JSONSchemaCompileError else {
                return XCTFail("expected invalidJSONString, got \(error)")
            }
            XCTAssertEqual(bad, self.invalidJSONString)
        }

        // Name side.
        XCTAssertThrowsError(
            try JSONSchemaCompiler.compile(
                JSONSchemaFormat(
                    name: invalidJSONString, strict: true,
                    schema: try mei(#"{"type": "object", "properties": {}, "required": [], "additionalProperties": false}"#)))
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .invalidJSONString(self.invalidJSONString))
        }
    }

    // MARK: - Unsupported constructs fail closed

    func testUnsupportedNestedStructuresRejected() throws {
        let nestedObject =
            #"{"type": "object", "properties": {"o": {"type": "object", "properties": {"x": {"type": "string"}}}}, "required": ["o"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(nestedObject)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyTypeUnsupported("o", "object"))
        }
        let nestedArray =
            #"{"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string"}}}, "required": ["a"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(nestedArray)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyTypeUnsupported("a", "array"))
        }
        XCTAssertThrowsError(
            try compileSchema(#"{"type": "array", "items": {"type": "string"}}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .rootTypeNotObject)
        }
        XCTAssertThrowsError(
            try compileSchema(
                #"{"type": "object", "properties": {"n": {"type": "null"}}, "required": ["n"], "additionalProperties": false}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyTypeUnsupported("n", "null"))
        }
        // Nullable unions are a later feature; an array-valued `type` is not
        // silently interpreted.
        XCTAssertThrowsError(
            try compileSchema(
                #"{"type": "object", "properties": {"u": {"type": ["string", "null"]}}, "required": ["u"], "additionalProperties": false}"#)
        )
    }

    func testUnsupportedSchemaKeywordsRejected() throws {
        XCTAssertThrowsError(
            try compileSchema(
                #"{"type": "object", "properties": {}, "required": [], "additionalProperties": false, "minProperties": 1}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .rootKeywordUnsupported("minProperties"))
        }
        for (keyword, value) in [("pattern", #""^a$""#), ("minLength", "1"), ("format", #""date-time""#)] {
            let schema =
                #"{"type": "object", "properties": {"x": {"type": "string", "\#(keyword)": \#(value)}}, "required": ["x"], "additionalProperties": false}"#
            XCTAssertThrowsError(try compileSchema(schema), keyword) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .propertyKeywordUnsupported("x", keyword))
            }
        }
    }

    func testDuplicateRequiredNamesRejected() throws {
        XCTAssertThrowsError(
            try compileSchema(
                #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "a"], "additionalProperties": false}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .requiredNameDuplicated("a"))
        }
    }

    func testRequiredNameMustBeDefined() throws {
        XCTAssertThrowsError(
            try compileSchema(
                #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "b"], "additionalProperties": false}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .requiredNameNotDefined("b"))
        }
    }

    func testEmptyPropertyNameRejected() throws {
        XCTAssertThrowsError(
            try compileSchema(
                #"{"type": "object", "properties": {"": {"type": "string"}}, "required": [], "additionalProperties": false}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyNameEmpty)
        }
    }

    func testEnumShapeRejected() throws {
        let cases: [(String, JSONSchemaCompileError)] = [
            (
                #"{"type": "object", "properties": {"s": {"type": "string", "enum": []}}, "required": ["s"], "additionalProperties": false}"#,
                .propertyEnumInvalid("s")
            ),
            (
                #"{"type": "object", "properties": {"s": {"type": "string", "enum": [1]}}, "required": ["s"], "additionalProperties": false}"#,
                .propertyEnumInvalid("s")
            ),
            (
                #"{"type": "object", "properties": {"s": {"type": "string", "enum": "ok"}}, "required": ["s"], "additionalProperties": false}"#,
                .propertyEnumInvalid("s")
            ),
            (
                #"{"type": "object", "properties": {"s": {"type": "string", "enum": ["ok", "ok"]}}, "required": ["s"], "additionalProperties": false}"#,
                .propertyEnumDuplicated("s", "ok")
            ),
            (
                #"{"type": "object", "properties": {"n": {"type": "number", "enum": ["ok"]}}, "required": ["n"], "additionalProperties": false}"#,
                .propertyEnumRequiresStringType("n")
            ),
        ]
        for (schema, expected) in cases {
            XCTAssertThrowsError(try compileSchema(schema), "\(expected)") { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, expected)
            }
        }
    }

    func testAdditionalPropertiesMustBeFalse() throws {
        let schemas = [
            #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}"#,
            #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": true}"#,
            #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": {"type": "string"}}"#,
        ]
        for schema in schemas {
            XCTAssertThrowsError(try compileSchema(schema), schema) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .additionalPropertiesMustBeFalse)
            }
        }
    }

    func testPropertiesAndRequiredShapesRejected() throws {
        let propertiesCases = [
            #"{"type": "object", "required": [], "additionalProperties": false}"#,
            #"{"type": "object", "properties": "x", "required": [], "additionalProperties": false}"#,
        ]
        for schema in propertiesCases {
            XCTAssertThrowsError(try compileSchema(schema), schema) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .propertiesInvalid)
            }
        }
        let requiredCases = [
            #"{"type": "object", "properties": {}, "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": "a", "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": [1], "additionalProperties": false}"#,
        ]
        for schema in requiredCases {
            XCTAssertThrowsError(try compileSchema(schema), schema) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .requiredInvalid)
            }
        }
    }

    func testRootShapeRejected() throws {
        for schema in ["[]", #""x""#, "5"] {
            XCTAssertThrowsError(try compileSchema(schema), schema) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .rootNotObject)
            }
        }
        let nonObjectRoots = [
            #"{"properties": {}, "required": [], "additionalProperties": false}"#,
            #"{"type": "string", "properties": {}, "required": [], "additionalProperties": false}"#,
            #"{"type": 5, "properties": {}, "required": [], "additionalProperties": false}"#,
        ]
        for schema in nonObjectRoots {
            XCTAssertThrowsError(try compileSchema(schema), schema) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .rootTypeNotObject)
            }
        }
    }

    func testCompileRejectsNonStrictAndEmptyName() throws {
        let schema = try mei(#"{"type": "object", "properties": {}, "required": [], "additionalProperties": false}"#)
        XCTAssertThrowsError(
            try JSONSchemaCompiler.compile(JSONSchemaFormat(name: "x", strict: false, schema: schema))
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .strictNotSupported)
        }
        XCTAssertThrowsError(
            try JSONSchemaCompiler.compile(JSONSchemaFormat(name: "", strict: true, schema: schema))
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .invalidName)
        }
    }

    // MARK: - Canonical key

    func testEquivalentKeyOrderingProducesSameCanonicalConstraintKey() throws {
        let a = try compileSchema(
            #"{"type": "object", "properties": {"b": {"type": "boolean"}, "a": {"type": "string"}}, "required": ["a", "b"], "additionalProperties": false}"#
        )
        let b = try compileSchema(
            #"{"additionalProperties": false, "required": ["b", "a"], "properties": {"a": {"type": "string"}, "b": {"type": "boolean"}}, "type": "object"}"#
        )
        XCTAssertEqual(a, b, "equivalent schemas must compile to the same constraint")
        XCTAssertEqual(a.constraintKey, b.constraintKey, "key order must not change the canonical key")

        let enumA = try compileSchema(
            #"{"type": "object", "properties": {"s": {"type": "string", "enum": ["ok", "no"]}}, "required": ["s"], "additionalProperties": false}"#
        )
        let enumB = try compileSchema(
            #"{"type": "object", "properties": {"s": {"type": "string", "enum": ["no", "ok"]}}, "required": ["s"], "additionalProperties": false}"#
        )
        XCTAssertEqual(enumA, enumB, "enum order must not change the compiled constraint")
        XCTAssertEqual(enumA.constraintKey, enumB.constraintKey)

        let different = try compileSchema(
            #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false}"#
        )
        XCTAssertNotEqual(a.constraintKey, different.constraintKey, "different schemas must produce different keys")
    }
}
