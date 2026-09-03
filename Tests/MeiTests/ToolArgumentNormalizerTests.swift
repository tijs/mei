import XCTest

@testable import MeiCore

/// Schema-aware normalization of tool-call arguments: fields the tool's JSON
/// Schema declares as `number`/`integer` must serialize as JSON numbers even
/// when the model emitted them as numeric strings (Gemma-4 does this), while
/// `string`, `boolean`, and nested object/array values must pass through
/// unchanged so legitimate strings are never corrupted.
final class ToolArgumentNormalizerTests: XCTestCase {
  /// Build a `parameters` schema object for `add_numbers`.
  private func addSchema() -> MeiJSONValue {
    .object([
      "type": .string("object"),
      "properties": .object([
        "a": .object(["type": .string("number")]),
        "b": .object(["type": .string("integer")]),
        "label": .object(["type": .string("string")]),
        "verbose": .object(["type": .string("boolean")]),
      ]),
      "required": .array([.string("a"), .string("b"), .string("label")]),
    ])
  }

  func testNumericStringFieldsCoerceToJSONNumbers() throws {
    // Gemma emits the integer args quoted; the schema says number/integer.
    let arguments: MeiJSONValue = .object([
      "a": .string("15"),
      "b": .string("27"),
    ])
    let normalized = ToolArgumentNormalizer.normalize(
      arguments: arguments, parametersSchema: addSchema())
    XCTAssertEqual(
      try normalized.jsonString(), #"{"a":15,"b":27}"#,
      "numeric schema fields must serialize as JSON numbers")
    // The strings are now numbers in the typed tree, not just the text.
    guard case .object(let object) = normalized else { return XCTFail("expected object") }
    guard case .number(let a) = object["a"] else { return XCTFail("a should be number") }
    guard case .number(let b) = object["b"] else { return XCTFail("b should be number") }
    XCTAssertEqual(a, 15)
    XCTAssertEqual(b, 27)
  }

  func testReadymadeNumberFieldsStayNumbers() throws {
    // The GGUF/reference path already emits numbers; normalization is a no-op.
    let arguments: MeiJSONValue = .object([
      "a": .number(15),
      "b": .number(27),
    ])
    let normalized = ToolArgumentNormalizer.normalize(
      arguments: arguments, parametersSchema: addSchema())
    XCTAssertEqual(try normalized.jsonString(), #"{"a":15,"b":27}"#)
  }

  func testStringSchemaFieldsRemainStrings() throws {
    let arguments: MeiJSONValue = .object([
      "a": .string("15"),
      "b": .string("27"),
      "label": .string("42"),  // string-typed field that looks numeric must stay a string
    ])
    let normalized = ToolArgumentNormalizer.normalize(
      arguments: arguments, parametersSchema: addSchema())
    XCTAssertEqual(
      try normalized.jsonString(),
      #"{"a":15,"b":27,"label":"42"}"#,
      "string-schema fields must stay strings even when numeric-looking")
  }

  func testBooleanFieldsRemainBooleans() throws {
    let arguments: MeiJSONValue = .object([
      "a": .string("15"),
      "verbose": .bool(true),
    ])
    let normalized = ToolArgumentNormalizer.normalize(
      arguments: arguments, parametersSchema: addSchema())
    XCTAssertEqual(try normalized.jsonString(), #"{"a":15,"verbose":true}"#)
  }

  func testNonNumericStringsInNumberFieldsStayStrings() throws {
    // A field declared numeric but carrying a non-numeric string must not
    // be mangled; leave it verbatim rather than corrupt it.
    let arguments: MeiJSONValue = .object([
      "a": .string("not-a-number"),
      "b": .string("27"),
    ])
    let normalized = ToolArgumentNormalizer.normalize(
      arguments: arguments, parametersSchema: addSchema())
    guard case .object(let object) = normalized else { return XCTFail("expected object") }
    guard case .string(let a) = object["a"] else { return XCTFail("non-numeric stays string") }
    XCTAssertEqual(a, "not-a-number")
  }

  func testNestedObjectNumbersCoerceRecursively() throws {
    let schema: MeiJSONValue = .object([
      "type": .string("object"),
      "properties": .object([
        "target": .object([
          "type": .string("object"),
          "properties": .object([
            "mark": .object(["type": .string("integer")]),
            "name": .object(["type": .string("string")]),
          ]),
        ])
      ]),
    ])
    let arguments: MeiJSONValue = .object([
      "target": .object(["mark": .string("1"), "name": .string("tensor_0")])
    ])
    let normalized = ToolArgumentNormalizer.normalize(
      arguments: arguments, parametersSchema: schema)
    XCTAssertEqual(
      try normalized.jsonString(),
      #"{"target":{"mark":1,"name":"tensor_0"}}"#,
      "nested object numeric fields coerce while nested strings survive")
  }

  func testArrayOfObjectsNumbersCoerce() throws {
    let schema: MeiJSONValue = .object([
      "type": .string("object"),
      "properties": .object([
        "points": .object([
          "type": .string("array"),
          "items": .object([
            "type": .string("object"),
            "properties": .object([
              "x": .object(["type": .string("integer")]),
              "tag": .object(["type": .string("string")]),
            ]),
          ]),
        ])
      ]),
    ])
    let arguments: MeiJSONValue = .object([
      "points": .array([
        .object(["x": .string("1"), "tag": .string("10")]),
        .object(["x": .string("2"), "tag": .string("20")]),
      ])
    ])
    let normalized = ToolArgumentNormalizer.normalize(
      arguments: arguments, parametersSchema: schema)
    XCTAssertEqual(
      try normalized.jsonString(),
      #"{"points":[{"tag":"10","x":1},{"tag":"20","x":2}]}"#,
      "array-of-object numeric fields coerce while element strings survive")
  }

  func testNoSchemaLeavesArgumentsUntouched() throws {
    let arguments: MeiJSONValue = .object([
      "a": .string("15"),
      "b": .string("27"),
    ])
    let normalized = ToolArgumentNormalizer.normalize(arguments: arguments, parametersSchema: nil)
    XCTAssertEqual(normalized, arguments)
    guard case .object(let object) = normalized else { return XCTFail("expected object") }
    guard case .string(let a) = object["a"] else { return XCTFail("no schema -> verbatim string") }
    XCTAssertEqual(a, "15")
  }

  func testUntypedPropertyLeavesValueVerbatim() throws {
    // A property with no `type` is ambiguous; do not guess by coercing.
    let schema: MeiJSONValue = .object([
      "type": .string("object"),
      "properties": .object([
        "ambiguous": .object(["description": .string("no type")])
      ]),
    ])
    let arguments: MeiJSONValue = .object(["ambiguous": .string("27")])
    let normalized = ToolArgumentNormalizer.normalize(
      arguments: arguments, parametersSchema: schema)
    XCTAssertEqual(try normalized.jsonString(), #"{"ambiguous":"27"}"#)
  }
}
