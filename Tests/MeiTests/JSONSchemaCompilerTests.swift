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
        guard case .scalar(let scalar) = status.value else {
            return XCTFail("status must compile to a scalar, got \(status.value)")
        }
        XCTAssertEqual(scalar.type, .string)
        XCTAssertEqual(scalar.allowedValues, ["ok"])
        XCTAssertFalse(scalar.nullable)
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

    // MARK: - Recursive subset: nested objects, arrays/items, nullable unions

    func testNestedObjectSchemaCompilesAndValidates() throws {
        let compiled = try compileSchema(
            #"{"type": "object", "properties": {"meta": {"type": "object", "properties": {"id": {"type": "integer"}, "label": {"type": "string", "enum": ["a", "b"]}}, "required": ["id"], "additionalProperties": false}}, "required": ["meta"], "additionalProperties": false}"#
        )
        guard case .object(let meta) = try XCTUnwrap(compiled.properties.first).value else {
            return XCTFail("meta must compile to a nested object")
        }
        XCTAssertEqual(meta.required, ["id"])
        XCTAssertEqual(meta.properties.map(\.name), ["id", "label"])
        XCTAssertTrue(compiled.validate(try mei(#"{"meta": {"id": 3}}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"meta": {"id": 3, "label": "b"}}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"meta": {}}"#)), "nested required keys must be enforced")
        XCTAssertFalse(
            compiled.validate(try mei(#"{"meta": {"id": 3, "extra": 1}}"#)),
            "nested additionalProperties: false must be enforced")
        XCTAssertFalse(compiled.validate(try mei(#"{"meta": {"id": "3"}}"#)), "nested scalar types must be enforced")
        XCTAssertFalse(compiled.validate(try mei(#"{"meta": {"id": 3, "label": "c"}}"#)), "nested enums must be enforced")
        XCTAssertFalse(compiled.validate(try mei(#"{"meta": 3}"#)))
        XCTAssertFalse(compiled.validate(try mei("{}")))
    }

    func testArraySchemaCompilesAndValidates() throws {
        let compiled = try compileSchema(
            #"{"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}, "counts": {"type": "array", "items": {"type": "integer"}}}, "required": ["tags", "counts"], "additionalProperties": false}"#
        )
        XCTAssertTrue(compiled.validate(try mei(#"{"tags": [], "counts": []}"#)), "empty arrays are valid")
        XCTAssertTrue(compiled.validate(try mei(#"{"tags": ["a", "b"], "counts": [1, 2, -3]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"tags": [1], "counts": []}"#)), "array items must match")
        XCTAssertFalse(compiled.validate(try mei(#"{"tags": ["a", null], "counts": []}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"tags": "a", "counts": []}"#)), "a scalar is not an array")
        XCTAssertFalse(compiled.validate(try mei(#"{"tags": [], "counts": [1.5]}"#)), "integer items reject fractions")
    }

    func testRecursiveShapesCompileAndValidate() throws {
        // object -> array -> object -> array -> boolean, with a nullable leaf.
        let compiled = try compileSchema(
            #"{"type": "object", "properties": {"rows": {"type": "array", "items": {"type": "object", "properties": {"cells": {"type": "array", "items": {"type": "boolean"}}, "note": {"type": ["string", "null"]}}, "required": ["cells", "note"], "additionalProperties": false}}}, "required": ["rows"], "additionalProperties": false}"#
        )
        XCTAssertTrue(compiled.validate(try mei(#"{"rows": []}"#)))
        XCTAssertTrue(
            compiled.validate(try mei(#"{"rows": [{"cells": [true, false], "note": null}, {"cells": [], "note": "x"}]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"rows": [{"cells": [1], "note": null}]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"rows": [{"cells": [], "note": 1}]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"rows": [{"cells": []}]}"#)), "nested required keys are enforced")
        XCTAssertFalse(
            compiled.validate(try mei(#"{"rows": [{"cells": [], "note": null, "x": 1}]}"#)),
            "nested additionalProperties: false is enforced")
    }

    func testNullableUnionsCompileAndValidate() throws {
        let compiled = try compileSchema(
            #"{"type": "object", "properties": {"s": {"type": ["string", "null"]}, "n": {"type": ["number", "null"]}, "i": {"type": ["integer", "null"]}, "b": {"type": ["boolean", "null"]}}, "required": ["s", "n", "i", "b"], "additionalProperties": false}"#
        )
        XCTAssertTrue(compiled.validate(try mei(#"{"s": null, "n": null, "i": null, "b": null}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"s": "x", "n": 1.5, "i": 3, "b": false}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"s": 1, "n": 1.5, "i": 3, "b": false}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"s": "x", "n": true, "i": 3, "b": false}"#)))
        XCTAssertFalse(
            compiled.validate(try mei(#"{"s": "x", "n": 1.5, "i": 3.5, "b": false}"#)),
            "a nullable integer still rejects fractions")
        XCTAssertFalse(compiled.validate(try mei(#"{"s": "x", "n": 1.5, "i": 3, "b": 1}"#)))

        // The order of the union entries does not matter.
        let reversed = try compileSchema(
            #"{"type": "object", "properties": {"s": {"type": ["null", "string"]}}, "required": ["s"], "additionalProperties": false}"#)
        XCTAssertTrue(reversed.validate(try mei(#"{"s": null}"#)))
        XCTAssertTrue(reversed.validate(try mei(#"{"s": "x"}"#)))
        XCTAssertFalse(reversed.validate(try mei(#"{"s": 1}"#)))
    }

    func testNullableStringEnumSemantics() throws {
        let withNull = try compileSchema(
            #"{"type": "object", "properties": {"e": {"type": ["string", "null"], "enum": ["b", "a", null]}}, "required": ["e"], "additionalProperties": false}"#)
        XCTAssertTrue(withNull.validate(try mei(#"{"e": null}"#)))
        XCTAssertTrue(withNull.validate(try mei(#"{"e": "a"}"#)))
        XCTAssertFalse(withNull.validate(try mei(#"{"e": "c"}"#)))
        XCTAssertFalse(withNull.validate(try mei(#"{"e": 1}"#)))

        // When the enum omits null, null is not an accepted value even though
        // the type union allows it.
        let withoutNull = try compileSchema(
            #"{"type": "object", "properties": {"e": {"type": ["string", "null"], "enum": ["a"]}}, "required": ["e"], "additionalProperties": false}"#)
        XCTAssertFalse(withoutNull.validate(try mei(#"{"e": null}"#)))
        XCTAssertTrue(withoutNull.validate(try mei(#"{"e": "a"}"#)))

        let onlyNull = try compileSchema(
            #"{"type": "object", "properties": {"e": {"type": ["string", "null"], "enum": [null]}}, "required": ["e"], "additionalProperties": false}"#)
        XCTAssertTrue(onlyNull.validate(try mei(#"{"e": null}"#)))
        XCTAssertFalse(onlyNull.validate(try mei(#"{"e": "a"}"#)))

        // null in an enum requires the nullable union; other enum shapes stay rejected.
        XCTAssertThrowsError(
            try compileSchema(
                #"{"type": "object", "properties": {"e": {"type": "string", "enum": ["a", null]}}, "required": ["e"], "additionalProperties": false}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyEnumInvalid("e"))
        }
        XCTAssertThrowsError(
            try compileSchema(
                #"{"type": "object", "properties": {"e": {"type": ["string", "null"], "enum": ["a", "a", null]}}, "required": ["e"], "additionalProperties": false}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyEnumDuplicated("e", "a"))
        }
        XCTAssertThrowsError(
            try compileSchema(
                #"{"type": "object", "properties": {"e": {"type": ["string", "null"], "enum": [null, null]}}, "required": ["e"], "additionalProperties": false}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyEnumDuplicated("e", "null"))
        }
        // Integer enums are supported now; a nullable integer enum still
        // carries the nullable-union semantics.
        let nullableIntegerEnum = try compileSchema(
            #"{"type": "object", "properties": {"e": {"type": ["integer", "null"], "enum": [1, null]}}, "required": ["e"], "additionalProperties": false}"#)
        XCTAssertTrue(nullableIntegerEnum.validate(try mei(#"{"e": 1}"#)))
        XCTAssertTrue(nullableIntegerEnum.validate(try mei(#"{"e": null}"#)))
        XCTAssertFalse(nullableIntegerEnum.validate(try mei(#"{"e": 2}"#)))
    }

    // MARK: - Unsupported constructs fail closed

    func testUnsupportedNestedConstructsStillFailClosed() throws {
        // Nested objects and arrays are part of the recursive subset now;
        // their malformed variants must still be rejected explicitly.
        let nestedBadKeyword =
            #"{"type": "object", "properties": {"o": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "additionalProperties": false, "minProperties": 1}}, "required": ["o"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(nestedBadKeyword)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .objectKeywordUnsupported("o", "minProperties"))
        }
        let nestedAdditional =
            #"{"type": "object", "properties": {"o": {"type": "object", "properties": {}, "required": [], "additionalProperties": true}}, "required": ["o"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(nestedAdditional)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .nested("o", .additionalPropertiesMustBeFalse))
        }
        let nestedMissingRequired =
            #"{"type": "object", "properties": {"o": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["y"], "additionalProperties": false}}, "required": ["o"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(nestedMissingRequired)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .nested("o", .requiredNameNotDefined("y")))
        }
        // Deeper levels name their path.
        let deepBadProperty =
            #"{"type": "object", "properties": {"o": {"type": "object", "properties": {"x": {"type": "string", "minLength": 1}}, "required": ["x"], "additionalProperties": false}}, "required": ["o"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(deepBadProperty)) { error in
            XCTAssertEqual(
                error as? JSONSchemaCompileError,
                .propertyKeywordUnsupported("o.x", "minLength"))
        }
        let arrayMissingItems =
            #"{"type": "object", "properties": {"a": {"type": "array"}}, "required": ["a"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(arrayMissingItems)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .arrayItemsInvalid("a"))
        }
        let arrayItemsNotObject =
            #"{"type": "object", "properties": {"a": {"type": "array", "items": true}}, "required": ["a"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(arrayItemsNotObject)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .arrayItemsInvalid("a"))
        }
        let arrayBadKeyword =
            #"{"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string"}, "uniqueItems": true}}, "required": ["a"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(arrayBadKeyword)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .arrayKeywordUnsupported("a", "uniqueItems"))
        }
        let itemsBadType =
            #"{"type": "object", "properties": {"a": {"type": "array", "items": {"type": "null"}}}, "required": ["a"], "additionalProperties": false}"#
        XCTAssertThrowsError(try compileSchema(itemsBadType)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyTypeUnsupported("a.items", "null"))
        }
        // `$ref` and other combinators stay unsupported at every level.
        let nestedReference =
            ##"{"type": "object", "properties": {"o": {"type": "object", "properties": {}, "required": [], "additionalProperties": false, "$ref": "#/x"}}, "required": ["o"], "additionalProperties": false}"##
        XCTAssertThrowsError(try compileSchema(nestedReference)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .objectKeywordUnsupported("o", "$ref"))
        }

        // Root must stay an object; a bare null type stays unsupported.
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
    }

    func testUnsupportedTypeUnionsRejected() throws {
        let schemas = [
            #"{"type": "object", "properties": {"u": {"type": ["string", "integer"]}}, "required": ["u"], "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"u": {"type": ["object", "null"]}}, "required": ["u"], "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"u": {"type": ["array", "null"]}}, "required": ["u"], "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"u": {"type": ["null"]}}, "required": ["u"], "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"u": {"type": ["null", "null"]}}, "required": ["u"], "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"u": {"type": ["string", "string"]}}, "required": ["u"], "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"u": {"type": ["string", "null", "integer"]}}, "required": ["u"], "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"u": {"type": []}}, "required": ["u"], "additionalProperties": false}"#,
            #"{"type": "object", "properties": {"u": {"type": ["string", 5]}}, "required": ["u"], "additionalProperties": false}"#,
        ]
        for schema in schemas {
            XCTAssertThrowsError(try compileSchema(schema), schema) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .propertyTypeUnionUnsupported("u"))
            }
        }
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
                .propertyEnumValueTypeMismatch("s")
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
                .propertyEnumValueTypeMismatch("n")
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

    func testRecursiveCanonicalKeyIsOrderIndependent() throws {
        let a = try compileSchema(
            #"{"type": "object", "properties": {"o": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "additionalProperties": false}, "a": {"type": "array", "items": {"type": ["integer", "null"]}}}, "required": ["o", "a"], "additionalProperties": false}"#
        )
        let b = try compileSchema(
            #"{"additionalProperties": false, "required": ["a", "o"], "properties": {"a": {"items": {"type": ["null", "integer"]}, "type": "array"}, "o": {"required": ["x"], "additionalProperties": false, "properties": {"x": {"type": "string"}}, "type": "object"}}, "type": "object"}"#
        )
        XCTAssertEqual(a, b, "equivalent recursive schemas must compile to the same constraint")
        XCTAssertEqual(a.constraintKey, b.constraintKey, "key/union order must not change the canonical key")

        let different = try compileSchema(
            #"{"type": "object", "properties": {"o": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": false}}, "required": ["o"], "additionalProperties": false}"#
        )
        XCTAssertNotEqual(a.constraintKey, different.constraintKey, "different nested schemas must produce different keys")
    }

    func testFlatCanaryConstraintKeyIsFrozen() throws {
        // The canonical cache key of the flat CoCore canary subset is frozen:
        // the recursive-schema refactor must not silently change keys for
        // schemas that were already supported.
        XCTAssertEqual(
            try canarySchema().constraintKey,
            "json_schema:v1:af38e1cfae52d821a5f94f1e69cb53cb817bc32f7e181c69cb6d6e80388cde6d")
    }
}
