import XCTest
@testable import MeiCore

/// Model-free tests for the expanded strict-schema matrix (the "expand the
/// schema matrix deliberately" slice): numeric constraints on `number`/
/// `integer` (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`,
/// `multipleOf`), enums on every scalar type, and array `minItems`/`maxItems`
/// — compilation, canonical constraint keys, instance validation, and the
/// explicit rejection matrix for invalid or still-unsupported constructs.
///
/// The numeric semantics under test are exact decimal semantics: schema
/// numbers enter through their shortest round-trip decimal form, generated
/// literals through their exact digits, so `0.1 × 3 = 0.3` and
/// `0.3 / 0.1 = 3` hold exactly (no binary floating-point artifacts).
final class SchemaMatrixTests: XCTestCase {

    // MARK: - Helpers

    private func mei(_ json: String) throws -> MeiJSONValue {
        try JSONDecoder().decode(MeiJSONValue.self, from: Data(json.utf8))
    }

    private func compileSchema(_ schemaJSON: String, name: String = "matrix") throws -> CompiledJSONSchema {
        try JSONSchemaCompiler.compile(
            JSONSchemaFormat(name: name, strict: true, schema: try mei(schemaJSON)))
    }

    /// One object with a single constrained scalar field, for compact cases.
    private func scalarSchema(_ fieldSchemaJSON: String, field: String = "v") throws -> CompiledJSONSchema {
        try compileSchema(
            #"{"type": "object", "properties": {"\#(field)": \#(fieldSchemaJSON)}, "required": ["\#(field)"], "additionalProperties": false}"#
        )
    }

    // MARK: - Numeric constraints compile and validate

    func testIntegerBoundsAndMultipleOfCompileAndValidate() throws {
        let compiled = try scalarSchema(
            #"{"type": "integer", "minimum": 10, "maximum": 99, "multipleOf": 5}"#)
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 15}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 10}"#)), "inclusive minimum")
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 95}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 12}"#)), "not a multiple of 5")
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 9}"#)), "below the minimum")
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 100}"#)), "above the maximum")
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 15.5}"#)), "fraction under integer")
    }

    func testNumberExclusiveBoundsAndFractionalMultipleOf() throws {
        let compiled = try scalarSchema(
            #"{"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1, "multipleOf": 0.25}"#)
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 0.5}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 0.25}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 0}"#)), "the exclusive lower bound")
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 1}"#)), "the exclusive upper bound")
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 0.3}"#)), "not a multiple of 0.25")
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 1.5}"#)), "above the upper bound")
    }

    func testNumericComparisonsUseExactDecimalsNotBinaryFloats() throws {
        // 0.3 / 0.1 = 3 exactly here; the same check over binary doubles fails.
        let compiled = try scalarSchema(#"{"type": "number", "multipleOf": 0.1}"#)
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 0.3}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 0.1}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 0.35}"#)))

        // 1e-1 is exactly 0.1, so it matches a decimal bound of 0.1.
        let bounded = try scalarSchema(#"{"type": "number", "maximum": 0.1}"#)
        XCTAssertTrue(bounded.validate(try mei(#"{"v": 1e-1}"#)))
        XCTAssertFalse(bounded.validate(try mei(#"{"v": 0.1000001}"#)))
    }

    func testNumericConstraintsApplyToNullableUnions() throws {
        let compiled = try scalarSchema(
            #"{"type": ["number", "null"], "minimum": 0, "multipleOf": 0.5}"#)
        XCTAssertTrue(compiled.validate(try mei(#"{"v": null}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 1.5}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 1.25}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"v": -0.5}"#)))
    }

    // MARK: - Enum coverage on every scalar type

    func testEnumCoverageForEveryScalarType() throws {
        let compiled = try compileSchema(
            #"{"type": "object", "properties": {"s": {"type": "string", "enum": ["b", "a"]}, "i": {"type": "integer", "enum": [2, 1]}, "n": {"type": "number", "enum": [0.5, 1]}, "b": {"type": "boolean", "enum": [true]}}, "required": ["s", "i", "n", "b"], "additionalProperties": false}"#
        )
        XCTAssertTrue(compiled.validate(try mei(#"{"s": "a", "i": 1, "n": 0.5, "b": true}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"s": "b", "i": 2, "n": 1, "b": true}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"s": "c", "i": 1, "n": 0.5, "b": true}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"s": "a", "i": 3, "n": 0.5, "b": true}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"s": "a", "i": 1, "n": 0.75, "b": true}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"s": "a", "i": 1, "n": 0.5, "b": false}"#)))

        // Number enum values are canonicalized by exact decimal value.
        guard case .scalar(let numberField) = compiled.root.properties[2].value else {
            return XCTFail("expected the number field")
        }
        XCTAssertEqual(
            numberField.enumValues,
            CompiledJSONSchema.ScalarEnumValues.numbers([
                DecimalLiteral(negative: false, digits: [5], exponent: -1),
                DecimalLiteral(negative: false, digits: [1], exponent: 0),
            ]))
    }

    func testNullableIntegerEnumSemantics() throws {
        let withNull = try scalarSchema(#"{"type": ["integer", "null"], "enum": [1, null]}"#)
        XCTAssertTrue(withNull.validate(try mei(#"{"v": null}"#)))
        XCTAssertTrue(withNull.validate(try mei(#"{"v": 1}"#)))
        XCTAssertFalse(withNull.validate(try mei(#"{"v": 2}"#)))

        let withoutNull = try scalarSchema(#"{"type": ["integer", "null"], "enum": [1]}"#)
        XCTAssertFalse(withoutNull.validate(try mei(#"{"v": null}"#)), "enum omitting null rejects null")
        XCTAssertTrue(withoutNull.validate(try mei(#"{"v": 1}"#)))

        // An enum of only null keeps the null branch and rejects every number.
        let onlyNull = try scalarSchema(#"{"type": ["integer", "null"], "enum": [null]}"#)
        XCTAssertTrue(onlyNull.validate(try mei(#"{"v": null}"#)))
        XCTAssertFalse(onlyNull.validate(try mei(#"{"v": 1}"#)))
    }

    func testEnumValuesCombineWithNumericConstraints() throws {
        let compiled = try scalarSchema(#"{"type": "integer", "enum": [1, 2, 3, 4], "minimum": 3}"#)
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 3}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"v": 4}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 2}"#)), "below the minimum")
        XCTAssertFalse(compiled.validate(try mei(#"{"v": 5}"#)), "outside the enum")
    }

    // MARK: - Array count constraints

    func testArrayCountConstraintsCompileAndValidate() throws {
        let compiled = try scalarSchema(
            #"{"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2}"#)
        XCTAssertFalse(compiled.validate(try mei(#"{"v": []}"#)), "below minItems")
        XCTAssertTrue(compiled.validate(try mei(#"{"v": [1]}"#)))
        XCTAssertTrue(compiled.validate(try mei(#"{"v": [1, 2]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"v": [1, 2, 3]}"#)), "above maxItems")
        XCTAssertFalse(compiled.validate(try mei(#"{"v": ["x"]}"#)), "items still enforced")

        // minItems 0 is distinct from absent but both accept the empty array.
        let zeroMin = try scalarSchema(
            #"{"type": "array", "items": {"type": "integer"}, "minItems": 0}"#)
        XCTAssertTrue(zeroMin.validate(try mei(#"{"v": []}"#)))
        XCTAssertNotEqual(zeroMin.constraintKey, compiled.constraintKey)

        // Count constraints nest recursively.
        let nested = try compileSchema(
            #"{"type": "object", "properties": {"rows": {"type": "array", "items": {"type": "object", "properties": {"cells": {"type": "array", "items": {"type": "boolean"}, "minItems": 2}}, "required": ["cells"], "additionalProperties": false}, "minItems": 1, "maxItems": 1}}, "required": ["rows"], "additionalProperties": false}"#
        )
        XCTAssertTrue(nested.validate(try mei(#"{"rows": [{"cells": [true, false]}]}"#)))
        XCTAssertFalse(nested.validate(try mei(#"{"rows": []}"#)))
        XCTAssertFalse(nested.validate(try mei(#"{"rows": [{"cells": [true]}]}"#)))
        XCTAssertFalse(
            nested.validate(try mei(#"{"rows": [{"cells": [true, false]}, {"cells": [true, false]}]}"#)))
    }

    // MARK: - Canonical constraint keys

    func testMatrixCanonicalKeyIsOrderIndependentAndValueCanonical() throws {
        let a = try scalarSchema(
            #"{"type": "integer", "minimum": 10, "maximum": 99, "multipleOf": 5, "enum": [20, 15]}"#)
        let b = try scalarSchema(
            #"{"enum": [15, 20.0], "multipleOf": 5, "maximum": 99, "minimum": 10, "type": "integer"}"#)
        XCTAssertEqual(a, b, "key order, enum order, and 20 vs 20.0 must not change the constraint")
        XCTAssertEqual(a.constraintKey, b.constraintKey)

        let differentMinimum = try scalarSchema(
            #"{"type": "integer", "minimum": 11, "maximum": 99, "multipleOf": 5, "enum": [20, 15]}"#)
        XCTAssertNotEqual(a.constraintKey, differentMinimum.constraintKey)

        let differentMultiple = try scalarSchema(
            #"{"type": "integer", "minimum": 10, "maximum": 99, "multipleOf": 10, "enum": [20, 15]}"#)
        XCTAssertNotEqual(a.constraintKey, differentMultiple.constraintKey)

        let withCounts = try compileSchema(
            #"{"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}}, "required": ["a"], "additionalProperties": false}"#)
        let reordered = try compileSchema(
            #"{"type": "object", "properties": {"a": {"maxItems": 3, "items": {"type": "string"}, "type": "array", "minItems": 1}}, "required": ["a"], "additionalProperties": false}"#)
        XCTAssertEqual(withCounts.constraintKey, reordered.constraintKey)

        let booleanEnum = try scalarSchema(#"{"type": "boolean", "enum": [false, true]}"#)
        let booleanEnumReordered = try scalarSchema(#"{"type": "boolean", "enum": [true, false]}"#)
        XCTAssertEqual(booleanEnum.constraintKey, booleanEnumReordered.constraintKey)
    }

    func testFlatAndRecursiveConstraintKeysStayFrozen() throws {
        // The previously supported subsets must not silently change keys.
        let flat = try compileSchema(
            #"{"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}}, "required": ["status"], "additionalProperties": false}"#,
            name: "canary_status")
        XCTAssertEqual(
            flat.constraintKey,
            "json_schema:v1:af38e1cfae52d821a5f94f1e69cb53cb817bc32f7e181c69cb6d6e80388cde6d")
        let recursive = try compileSchema(
            #"{"type": "object", "properties": {"meta": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"], "additionalProperties": false}, "tags": {"type": "array", "items": {"type": "string"}}, "note": {"type": ["string", "null"]}}, "required": ["meta", "tags", "note"], "additionalProperties": false}"#,
            name: "nested_v1")
        XCTAssertEqual(
            recursive.constraintKey,
            "json_schema:v1:b4aa5e5933ec2dcddf8c920cdc34d5d3ad41d5ea1b72f03b17b0839bf175f97c",
            "the recursive slice key is frozen")
        XCTAssertEqual(
            recursive.constraintKey, try RecursiveSchemaFixture.schema().constraintKey,
            "the fixture compiles to the same frozen key")
    }

    // MARK: - Rejection matrix: invalid numeric constraints

    func testNumericKeywordsOnNonNumericTypesRejected() throws {
        for (fieldSchema, keyword) in [
            (#"{"type": "string", "minimum": 1}"#, "minimum"),
            (#"{"type": "boolean", "maximum": 1}"#, "maximum"),
            (#"{"type": "string", "multipleOf": 2}"#, "multipleOf"),
            (#"{"type": "string", "exclusiveMinimum": 1}"#, "exclusiveMinimum"),
        ] {
            XCTAssertThrowsError(try scalarSchema(fieldSchema), keyword) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .propertyKeywordUnsupported("v", keyword))
            }
        }
    }

    func testInvalidNumericConstraintValuesRejected() throws {
        let cases: [(String, String)] = [
            (#"{"type": "number", "minimum": "1"}"#, "minimum"),
            (#"{"type": "number", "maximum": true}"#, "maximum"),
            (#"{"type": "number", "exclusiveMinimum": "0"}"#, "exclusiveMinimum"),
            (#"{"type": "integer", "multipleOf": 0}"#, "multipleOf"),
            (#"{"type": "integer", "multipleOf": -1}"#, "multipleOf"),
            (#"{"type": "integer", "multipleOf": true}"#, "multipleOf"),
        ]
        for (fieldSchema, keyword) in cases {
            XCTAssertThrowsError(try scalarSchema(fieldSchema), keyword) { error in
                XCTAssertEqual(
                    error as? JSONSchemaCompileError, .propertyNumericConstraintInvalid("v", keyword))
            }
        }
    }

    func testUnsatisfiableConstraintsRejected() throws {
        let unsatisfiable = [
            #"{"type": "integer", "minimum": 5, "maximum": 3}"#,
            #"{"type": "integer", "minimum": 1.1, "maximum": 1.9}"#,
            #"{"type": "number", "exclusiveMinimum": 1, "maximum": 1}"#,
            #"{"type": "number", "minimum": 2, "exclusiveMaximum": 2}"#,
            #"{"type": "integer", "minimum": 1, "maximum": 1, "multipleOf": 2}"#,
            // Integer multiples of 1.5 are multiples of 3: [1, 2] and [4, 5]
            // contain none, so both schemas can never generate a value.
            #"{"type": "integer", "minimum": 1, "maximum": 2, "multipleOf": 1.5}"#,
            #"{"type": "integer", "minimum": 4, "maximum": 5, "multipleOf": 1.5}"#,
            #"{"type": "integer", "enum": [1], "minimum": 2}"#,
            #"{"type": "number", "enum": [0.5], "exclusiveMaximum": 0.5}"#,
        ]
        for fieldSchema in unsatisfiable {
            XCTAssertThrowsError(try scalarSchema(fieldSchema), fieldSchema) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .propertyConstraintsUnsatisfiable("v"))
            }
        }

        // Satisfiable boundary shapes must keep compiling.
        let satisfiable = [
            #"{"type": "integer", "minimum": 1, "maximum": 2, "multipleOf": 2}"#,
            // 3 is the integer multiple of 1.5 inside [1, 4].
            #"{"type": "integer", "minimum": 1, "maximum": 4, "multipleOf": 1.5}"#,
            // Every integer is a multiple of 0.25 and 0.1.
            #"{"type": "integer", "minimum": 1, "maximum": 2, "multipleOf": 0.25}"#,
            #"{"type": "integer", "minimum": 1, "maximum": 2, "multipleOf": 0.1}"#,
            #"{"type": "number", "exclusiveMinimum": 1, "maximum": 2}"#,
            #"{"type": "number", "minimum": 0.1, "maximum": 0.1}"#,
            #"{"type": "integer", "minimum": 1.5, "maximum": 2.5}"#,
        ]
        for fieldSchema in satisfiable {
            XCTAssertNoThrow(try scalarSchema(fieldSchema), fieldSchema)
        }

        // A nullable field keeps its null branch even when the numeric branch
        // is unsatisfiable; the null value stays generatable.
        let nullableImpossible = try scalarSchema(
            #"{"type": ["integer", "null"], "minimum": 5, "maximum": 3}"#)
        XCTAssertTrue(nullableImpossible.validate(try mei(#"{"v": null}"#)))
        XCTAssertFalse(nullableImpossible.validate(try mei(#"{"v": 4}"#)))
    }

    // MARK: - Rejection matrix: enum shapes

    func testEnumTypeMismatchesRejected() throws {
        let cases = [
            #"{"type": "number", "enum": ["ok"]}"#,
            #"{"type": "integer", "enum": [1.5]}"#,
            #"{"type": "integer", "enum": ["1"]}"#,
            #"{"type": "boolean", "enum": [1]}"#,
            #"{"type": "string", "enum": [true]}"#,
            #"{"type": "number", "enum": [{"a": 1}]}"#,
            #"{"type": "integer", "enum": [[1]]}"#,
        ]
        for fieldSchema in cases {
            XCTAssertThrowsError(try scalarSchema(fieldSchema), fieldSchema) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .propertyEnumValueTypeMismatch("v"))
            }
        }
    }

    func testDuplicateEnumValuesRejectedByExactValue() throws {
        XCTAssertThrowsError(try scalarSchema(#"{"type": "number", "enum": [1, 1.0]}"#)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyEnumDuplicated("v", "1e0"))
        }
        XCTAssertThrowsError(try scalarSchema(#"{"type": "integer", "enum": [7, 7]}"#)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyEnumDuplicated("v", "7e0"))
        }
        XCTAssertThrowsError(try scalarSchema(#"{"type": "boolean", "enum": [true, true]}"#)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyEnumDuplicated("v", "true"))
        }
        XCTAssertThrowsError(try scalarSchema(#"{"type": "boolean", "enum": [false, false]}"#)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyEnumDuplicated("v", "false"))
        }
    }

    // MARK: - Rejection matrix: array counts and still-unsupported keywords

    func testArrayCountShapeRejected() throws {
        let cases: [(String, JSONSchemaCompileError)] = [
            (
                #"{"type": "array", "items": {"type": "string"}, "minItems": -1}"#,
                .arrayCountConstraintInvalid("v", "minItems")
            ),
            (
                #"{"type": "array", "items": {"type": "string"}, "maxItems": 1.5}"#,
                .arrayCountConstraintInvalid("v", "maxItems")
            ),
            (
                #"{"type": "array", "items": {"type": "string"}, "minItems": "1"}"#,
                .arrayCountConstraintInvalid("v", "minItems")
            ),
            (
                #"{"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 1}"#,
                .arrayCountRangeInvalid("v")
            ),
        ]
        for (fieldSchema, expected) in cases {
            XCTAssertThrowsError(try scalarSchema(fieldSchema), fieldSchema) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, expected)
            }
        }
    }

    func testStillUnsupportedKeywordsRemainRejected() throws {
        // Type-specific keywords outside the OpenAI strict subset stay 400s.
        for (fieldSchema, keyword) in [
            (#"{"type": "string", "minLength": 1}"#, "minLength"),
            (#"{"type": "string", "maxLength": 3}"#, "maxLength"),
            (#"{"type": "string", "pattern": "^a$"}"#, "pattern"),
            (#"{"type": "string", "format": "date-time"}"#, "format"),
            (#"{"type": "string", "const": "a"}"#, "const"),
            (#"{"type": "string", "anyOf": [{"type": "string"}]}"#, "anyOf"),
            (#"{"type": "number", "minLength": 1}"#, "minLength"),
            // Annotations are not part of the accepted subset yet: they are
            // rejected explicitly rather than silently ignored.
            (#"{"type": "string", "description": "a name"}"#, "description"),
            (#"{"type": "string", "title": "Name"}"#, "title"),
        ] {
            XCTAssertThrowsError(try scalarSchema(fieldSchema), keyword) { error in
                XCTAssertEqual(error as? JSONSchemaCompileError, .propertyKeywordUnsupported("v", keyword))
            }
        }

        XCTAssertThrowsError(
            try scalarSchema(#"{"type": "array", "items": {"type": "string"}, "uniqueItems": true}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .arrayKeywordUnsupported("v", "uniqueItems"))
        }
        XCTAssertThrowsError(
            try scalarSchema(#"{"type": "object", "properties": {}, "required": [], "additionalProperties": false, "minProperties": 1}"#)
        ) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .objectKeywordUnsupported("v", "minProperties"))
        }
        XCTAssertThrowsError(try scalarSchema(##"{"$ref": "#/$defs/x"}"##)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyTypeMissing("v"))
        }
        XCTAssertThrowsError(try scalarSchema(#"{"anyOf": [{"type": "string"}]}"#)) { error in
            XCTAssertEqual(error as? JSONSchemaCompileError, .propertyTypeMissing("v"))
        }
    }

    // MARK: - Brute-force consistency against an independent oracle

    /// Test-local exact decimal for small literals, built on `Int` arithmetic
    /// only — deliberately independent of `DecimalLiteral`. Value is
    /// `units / 10^scale`.
    private struct SmallDecimal: Equatable {
        var units: Int64
        var scale: Int

        init?(parsing literal: String) {
            let chars = Array(literal.utf8)
            var index = 0
            var negative = false
            if index < chars.count, chars[index] == 45 {
                negative = true
                index += 1
            }
            var digits: [Int64] = []
            var scale = 0
            var sawDigit = false
            while index < chars.count, (48...57).contains(chars[index]) {
                digits.append(Int64(chars[index] - 48))
                sawDigit = true
                index += 1
            }
            if index < chars.count, chars[index] == 46 {
                index += 1
                while index < chars.count, (48...57).contains(chars[index]) {
                    digits.append(Int64(chars[index] - 48))
                    scale += 1
                    sawDigit = true
                    index += 1
                }
            }
            guard sawDigit else { return nil }
            if index < chars.count, chars[index] == 101 || chars[index] == 69 {
                index += 1
                var exponentNegative = false
                if index < chars.count, chars[index] == 43 || chars[index] == 45 {
                    exponentNegative = chars[index] == 45
                    index += 1
                }
                var exponent = 0
                var sawExponentDigit = false
                while index < chars.count, (48...57).contains(chars[index]) {
                    exponent = exponent * 10 + Int(chars[index] - 48)
                    sawExponentDigit = true
                    index += 1
                }
                guard sawExponentDigit else { return nil }
                scale -= exponentNegative ? -exponent : exponent
            }
            guard index == chars.count else { return nil }
            var units: Int64 = 0
            for digit in digits { units = units * 10 + digit }
            self.init(units: negative ? -units : units, scale: scale)
        }

        init(units: Int64, scale: Int) {
            var units = units
            var scale = scale
            while scale < 0 {
                units *= 10
                scale += 1
            }
            while scale > 0, units % 10 == 0 {
                units /= 10
                scale -= 1
            }
            self.units = units
            self.scale = scale
        }

        private static func pow10(_ exponent: Int) -> Int64 {
            var result: Int64 = 1
            for _ in 0..<exponent { result *= 10 }
            return result
        }

        static func compare(_ lhs: SmallDecimal, _ rhs: SmallDecimal) -> Int {
            let scale = Swift.max(lhs.scale, rhs.scale)
            let left = lhs.units * pow10(scale - lhs.scale)
            let right = rhs.units * pow10(scale - rhs.scale)
            if left == right { return 0 }
            return left < right ? -1 : 1
        }

        func isMultiple(of other: SmallDecimal) -> Bool {
            let scale = Swift.max(self.scale, other.scale)
            let numerator = units * Self.pow10(scale - self.scale)
            let denominator = other.units * Self.pow10(scale - other.scale)
            guard denominator != 0 else { return false }
            return numerator % denominator == 0
        }
    }

    /// Independent evaluation of the numeric constraint keywords for a small
    /// literal value. Returns nil when the literal is not a valid JSON number.
    private func oracleAccepts(
        literal: String,
        minimum: String? = nil,
        maximum: String? = nil,
        exclusiveMinimum: String? = nil,
        exclusiveMaximum: String? = nil,
        multipleOf: String? = nil,
        enumValues: [String]? = nil
    ) -> Bool? {
        guard let value = SmallDecimal(parsing: literal) else { return nil }
        if let minimum, let bound = SmallDecimal(parsing: minimum), SmallDecimal.compare(value, bound) < 0 {
            return false
        }
        if let maximum, let bound = SmallDecimal(parsing: maximum), SmallDecimal.compare(value, bound) > 0 {
            return false
        }
        if let exclusiveMinimum, let bound = SmallDecimal(parsing: exclusiveMinimum),
            SmallDecimal.compare(value, bound) <= 0 {
            return false
        }
        if let exclusiveMaximum, let bound = SmallDecimal(parsing: exclusiveMaximum),
            SmallDecimal.compare(value, bound) >= 0 {
            return false
        }
        if let multipleOf, let multiple = SmallDecimal(parsing: multipleOf), !value.isMultiple(of: multiple) {
            return false
        }
        if let enumValues {
            let allowed = enumValues.compactMap { SmallDecimal(parsing: $0) }
            if !allowed.contains(value) { return false }
        }
        return true
    }

    /// Feeds `{"v":<literal>}` into a copy of the prepared state.
    private func acceptsLiteral(_ literal: String, prepared: JSONGrammarState) -> Bool {
        var state = prepared
        for byte in (literal + "}").utf8 where !state.consume(byte: byte) { return false }
        return state.status == .complete
    }

    /// Builds a state already past `{"v":` for one field schema.
    private func preparedState(_ fieldSchemaJSON: String) throws -> JSONGrammarState {
        let schema = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "matrix", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"v": \#(fieldSchemaJSON)}, "required": ["v"], "additionalProperties": false}"#
                )))
        var state = JSONGrammarState(schema: schema)
        XCTAssertTrue(feed(#"{"v":"#, into: &state))
        return state
    }

    private func feed(_ text: String, into state: inout JSONGrammarState) -> Bool {
        for byte in text.utf8 where !state.consume(byte: byte) { return false }
        return true
    }

    func testIntegerGrammarMatchesIndependentOracleAcrossLiterals() throws {
        let cases: [(
            schema: String, minimum: String?, maximum: String?, exclusiveMinimum: String?,
            exclusiveMaximum: String?, multipleOf: String?, enumValues: [String]?
        )] = [
            (#"{"type": "integer", "minimum": 10, "maximum": 99, "multipleOf": 5}"#, "10", "99", nil, nil, "5", nil),
            (#"{"type": "integer", "multipleOf": 7}"#, nil, nil, nil, nil, "7", nil),
            (#"{"type": "integer", "minimum": -20, "maximum": 20, "multipleOf": 3}"#, "-20", "20", nil, nil, "3", nil),
            (#"{"type": "integer", "enum": [0, 1, 2]}"#, nil, nil, nil, nil, nil, ["0", "1", "2"]),
            (#"{"type": "integer", "minimum": 0, "enum": [-5, 5]}"#, "0", nil, nil, nil, nil, ["-5", "5"]),
            (
                #"{"type": "integer", "minimum": 5, "maximum": 25, "enum": [5, 10, 15, 20, 25]}"#,
                "5", "25", nil, nil, nil, ["5", "10", "15", "20", "25"]
            ),
            (#"{"type": "integer", "exclusiveMinimum": -3, "exclusiveMaximum": 3}"#, nil, nil, "-3", "3", nil, nil),
            (#"{"type": "integer", "minimum": 100, "maximum": 200, "multipleOf": 25}"#, "100", "200", nil, nil, "25", nil),
            (#"{"type": "integer", "maximum": -5, "multipleOf": 5}"#, nil, "-5", nil, nil, "5", nil),
            (#"{"type": "integer", "exclusiveMaximum": 0}"#, nil, nil, nil, "0", nil, nil),
            (#"{"type": "integer", "minimum": -99, "maximum": -10, "multipleOf": 11}"#, "-99", "-10", nil, nil, "11", nil),
            // Fractional multiples: the reachable integers are the multiples
            // of the least positive integer multiple (1.5 -> 3, 0.3 -> 3,
            // 2.5 -> 5).
            (#"{"type": "integer", "minimum": 1, "maximum": 4, "multipleOf": 1.5}"#, "1", "4", nil, nil, "1.5", nil),
            (#"{"type": "integer", "minimum": 1, "maximum": 10, "multipleOf": 0.3}"#, "1", "10", nil, nil, "0.3", nil),
            (#"{"type": "integer", "minimum": -9, "maximum": 9, "multipleOf": 2.5}"#, "-9", "9", nil, nil, "2.5", nil),
        ]
        for testCase in cases {
            let prepared = try preparedState(testCase.schema)
            var literals: [String] = ["-0"]
            for value in -999...999 { literals.append(String(value)) }
            for literal in literals {
                let accepted = acceptsLiteral(literal, prepared: prepared)
                let expected = try XCTUnwrap(
                    oracleAccepts(
                        literal: literal, minimum: testCase.minimum, maximum: testCase.maximum,
                        exclusiveMinimum: testCase.exclusiveMinimum, exclusiveMaximum: testCase.exclusiveMaximum,
                        multipleOf: testCase.multipleOf, enumValues: testCase.enumValues))
                XCTAssertEqual(accepted, expected, "literal \(literal) under \(testCase.schema)")
            }
        }
    }

    func testNumberGrammarMatchesIndependentOracleAcrossLiterals() throws {
        let cases: [(
            schema: String, minimum: String?, maximum: String?, exclusiveMinimum: String?,
            exclusiveMaximum: String?, multipleOf: String?, enumValues: [String]?
        )] = [
            (
                #"{"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1, "multipleOf": 0.25}"#,
                nil, nil, "0", "1", "0.25", nil
            ),
            (
                #"{"type": "number", "minimum": -1.5, "maximum": 2.5, "multipleOf": 0.5}"#,
                "-1.5", "2.5", nil, nil, "0.5", nil
            ),
            (
                #"{"type": "number", "enum": [0.1, 0.25, 1, -2]}"#,
                nil, nil, nil, nil, nil, ["0.1", "0.25", "1", "-2"]
            ),
            (
                #"{"type": "number", "multipleOf": 0.1}"#,
                nil, nil, nil, nil, "0.1", nil
            ),
            (
                #"{"type": "number", "maximum": 0}"#,
                nil, "0", nil, nil, nil, nil
            ),
            (
                #"{"type": "number", "exclusiveMinimum": -1, "exclusiveMaximum": 0}"#,
                nil, nil, "-1", "0", nil, nil
            ),
            (
                #"{"type": "number", "enum": [0.5, 1.5, 2], "minimum": 1}"#,
                "1", nil, nil, nil, nil, ["0.5", "1.5", "2"]
            ),
            (
                #"{"type": "number", "minimum": 0.1, "maximum": 0.2}"#,
                "0.1", "0.2", nil, nil, nil, nil
            ),
            (
                #"{"type": "number", "multipleOf": 0.05, "maximum": 1}"#,
                nil, "1", nil, nil, "0.05", nil
            ),
            (
                #"{"type": "number", "exclusiveMinimum": -0.5, "exclusiveMaximum": 0.5, "multipleOf": 0.25}"#,
                nil, nil, "-0.5", "0.5", "0.25", nil
            ),
        ]
        var literals: [String] = []
        let intParts = ["0", "1", "2", "3", "5", "9", "10", "15", "25", "99", "100"]
        let fractionParts = ["", ".0", ".1", ".2", ".25", ".3", ".5", ".75", ".9", ".99"]
        let exponentParts = ["", "e0", "e1", "e2", "e-1", "e-2", "e-3", "E1", "e+1"]
        for intPart in intParts {
            for fraction in fractionParts {
                for exponent in exponentParts {
                    literals.append(intPart + fraction + exponent)
                    literals.append("-" + intPart + fraction + exponent)
                }
            }
        }
        for testCase in cases {
            let prepared = try preparedState(testCase.schema)
            for literal in literals {
                let accepted = acceptsLiteral(literal, prepared: prepared)
                let expected = try XCTUnwrap(
                    oracleAccepts(
                        literal: literal, minimum: testCase.minimum, maximum: testCase.maximum,
                        exclusiveMinimum: testCase.exclusiveMinimum, exclusiveMaximum: testCase.exclusiveMaximum,
                        multipleOf: testCase.multipleOf, enumValues: testCase.enumValues))
                XCTAssertEqual(accepted, expected, "literal \(literal) under \(testCase.schema)")
            }
        }
    }

    // MARK: - The shared fixture schema compiles

    func testSchemaMatrixFixtureCompilesAndValidatesItsDocument() throws {
        let compiled = try SchemaMatrixFixture.schema()
        XCTAssertTrue(compiled.validate(try mei(SchemaMatrixFixture.document)))
        XCTAssertFalse(compiled.validate(try mei(#"{"level":12,"ratio":0.5,"mode":"fast","retries":null,"ok":true,"flags":[true]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"level":15,"ratio":0.3,"mode":"fast","retries":null,"ok":true,"flags":[true]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"level":15,"ratio":0.5,"mode":"nope","retries":null,"ok":true,"flags":[true]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"level":15,"ratio":0.5,"mode":"fast","retries":3,"ok":true,"flags":[true]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"level":15,"ratio":0.5,"mode":"fast","retries":null,"ok":false,"flags":[true]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"level":15,"ratio":0.5,"mode":"fast","retries":null,"ok":true,"flags":[]}"#)))
        XCTAssertFalse(compiled.validate(try mei(#"{"level":15,"ratio":0.5,"mode":"fast","retries":null,"ok":true,"flags":[true,false,true]}"#)))
    }
}
