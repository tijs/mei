import XCTest
@testable import MeiCore

/// Model-free tests for the byte-level JSON grammar automaton — the first half
/// of the constrained-decoding core (slice 3 of the structured-generation
/// plan; the token-level half is `JSONGrammarProcessorTests`).
///
/// The automaton is deliberately tokenizer-, engine-, and HTTP-free: it
/// consumes bytes, tracks the JSON grammar state, and reports completion or
/// failure. It is the single source of truth the token-level mask consults, so
/// it must pin, at minimum: legal JSON whitespace, quoted strings and escapes
/// (including surrogate pairs and strict UTF-8), numbers, booleans, null,
/// nested objects/arrays, complete-root acceptance, premature EOS rejection,
/// illegal prose, and the strict flat `json_schema` subset (any key order,
/// required keys, `additionalProperties: false`, string enums).
final class JSONGrammarStateTests: XCTestCase {

    // MARK: - Helpers

    private func mei(_ json: String) throws -> MeiJSONValue {
        try JSONDecoder().decode(MeiJSONValue.self, from: Data(json.utf8))
    }

    /// The exact schema CoCore's structured-output canary sends
    /// (docs/COCORE.md §2).
    private func canarySchema() throws -> CompiledJSONSchema {
        try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "canary_status", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}}, "required": ["status"], "additionalProperties": false}"#
                )))
    }

    private func scalarSchema() throws -> CompiledJSONSchema {
        try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "scalars", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": "number"}, "b": {"type": "boolean"}}, "required": ["s", "i", "n", "b"], "additionalProperties": false}"#
                )))
    }

    private func freeState(maxDepth: Int = JSONGrammarState.defaultMaximumNestingDepth) -> JSONGrammarState {
        JSONGrammarState(jsonObjectWithMaximumNestingDepth: maxDepth)
    }

    /// Feeds `text` byte by byte; returns false as soon as a byte is rejected.
    @discardableResult
    private func feed(_ text: String, into state: inout JSONGrammarState) -> Bool {
        for byte in text.utf8 where !state.consume(byte: byte) { return false }
        return true
    }

    private func assertAccepts(
        _ text: String, _ state: JSONGrammarState, file: StaticString = #filePath, line: UInt = #line
    ) {
        var state = state
        XCTAssertTrue(feed(text, into: &state), "\(text) must be fully consumable", file: file, line: line)
        XCTAssertEqual(state.status, .complete, "\(text) must complete the root value", file: file, line: line)
        XCTAssertTrue(state.canAcceptEndOfSequence, "\(text) must accept EOS", file: file, line: line)
    }

    private func assertRejected(
        _ text: String, _ state: JSONGrammarState, file: StaticString = #filePath, line: UInt = #line
    ) {
        var state = state
        XCTAssertFalse(feed(text, into: &state), "\(text) must be rejected", file: file, line: line)
        XCTAssertEqual(state.status, .failed, "\(text) must leave the automaton failed", file: file, line: line)
    }

    private func assertIncomplete(
        _ text: String, _ state: JSONGrammarState, file: StaticString = #filePath, line: UInt = #line
    ) {
        var state = state
        XCTAssertTrue(feed(text, into: &state), "\(text) must be a legal prefix", file: file, line: line)
        XCTAssertEqual(state.status, .inProgress, "\(text) must not complete the root value", file: file, line: line)
    }

    // MARK: - json_object: complete roots

    func testCompleteJSONValuesAreAccepted() {
        let state = freeState()
        assertAccepts("{}", state)
        assertAccepts("[]", state)
        assertAccepts(#"{"a": 1}"#, state)
        assertAccepts(#"["a", 1, true, false, null]"#, state)
        assertAccepts(#""hello""#, state)
        assertAccepts("42", state)
        assertAccepts("-1.5e+10", state)
        assertAccepts("true", state)
        assertAccepts("false", state)
        assertAccepts("null", state)
        assertAccepts(#"{"a": [1, {"b": null}], "c": "x"}"#, state)
        assertAccepts(#"[[]]"#, state)
    }

    func testWhitespaceIsLegalAroundTokens() {
        let state = freeState()
        assertAccepts(" \t\r\n{ \"a\" : 1 , \"b\" : [ ] } \n", state)
        assertAccepts("  \n42\n  ", state)
        assertAccepts("\n[ 1 , 2 ]\t", state)
    }

    func testWhitespaceIsNotLegalInsideTokens() {
        let state = freeState()
        assertRejected("t rue", state)
        assertRejected("1 2", state)
        assertRejected(#"{"a": 1 2}"#, state)
        assertRejected(#""a" "b""#, state)
        assertRejected("nul l", state)
    }

    func testIncompleteValuesAreNotAccepted() {
        let state = freeState()
        assertIncomplete("{", state)
        assertIncomplete(#"{"a": 1"#, state)
        assertIncomplete("[1, ", state)
        assertIncomplete("tru", state)
        assertIncomplete("-", state)
        assertIncomplete("1.", state)
        assertIncomplete("1e", state)
        assertIncomplete(#""abc"#, state)
        assertIncomplete(#""abc\"#, state)
    }

    func testPrematureEndOfSequenceIsRejected() {
        var state = freeState()
        XCTAssertTrue(feed("{", into: &state))
        XCTAssertFalse(state.canAcceptEndOfSequence)
        XCTAssertFalse(state.markEndOfSequence())
        XCTAssertEqual(state.status, .inProgress, "a rejected EOS must not poison the state")

        var number = freeState()
        XCTAssertTrue(feed("1.", into: &number))
        XCTAssertFalse(number.canAcceptEndOfSequence)
        XCTAssertFalse(number.markEndOfSequence())

        var done = freeState()
        XCTAssertTrue(feed(#"{"a": 1}"#, into: &done))
        XCTAssertTrue(done.canAcceptEndOfSequence)
        XCTAssertTrue(done.markEndOfSequence())
        XCTAssertEqual(done.status, .finished)
        XCTAssertFalse(done.consume(byte: 0x20), "nothing may advance after EOS")
        XCTAssertEqual(done.status, .finished, "post-EOS rejection must not degrade the finished status")
    }

    // MARK: - json_object: illegal prose

    func testIllegalProseIsRejected() {
        let state = freeState()
        assertRejected("Hello", state)
        assertRejected("'quoted'", state)
        assertRejected("nan", state)
        assertRejected("Infinity", state)
        assertRejected("+1", state)
        assertRejected(".5", state)
        assertRejected("0x10", state)
        assertRejected("TRUE", state)
        assertRejected("nulll", state)
        assertRejected(#"{"a": 1} trailing"#, state)
        assertRejected(#"{"a" 1}"#, state)
        assertRejected(#"{"a": 1,}"#, state)
        assertRejected(#"{"a": 1 "b": 2}"#, state)
        assertRejected(#"{,}"#, state)
        assertRejected(#"{"a": }"#, state)
        assertRejected(#"[1,]"#, state)
        assertRejected(#"{"a": 1}" x"#, state)
        assertRejected(#"{"a": 1},{"a": 1}"#, state)
    }

    // MARK: - json_object: strings and escapes

    func testStringEscapesAreAccepted() {
        let state = freeState()
        assertAccepts(#""a\"b\\c\/d\be\ff\ng\rh\ti""#, state)
        assertAccepts(#""\u0041\u00e9\uD83D\uDE00""#, state)
        assertAccepts(#""caf\u00e9""#, state)
        assertAccepts(#"{"k\u0041y": "v\u00e4lue"}"#, state)
    }

    func testInvalidStringEscapesAreRejected() {
        let state = freeState()
        assertRejected(#""\x""#, state)
        assertRejected(#""\u12""#, state)
        assertRejected(#""\u12G4""#, state)
        assertRejected(#""\uDC00""#, state)
        assertRejected(#""\uD800\u0041""#, state)
        assertRejected(#""\uD800x""#, state)
        assertRejected("\"raw\nnewline\"", state)
        assertRejected("\"ctrl\u{01}\"", state)
    }

    func testStrictUTF8IsEnforcedInsideStrings() {
        let state = freeState()
        assertAccepts("\"é\"", state)
        assertAccepts("\"𝄞\"", state)
        assertAccepts(#"{"é": "𝄞"}"#, state)

        // A multi-byte sequence split across fragments is a legal prefix.
        var partial = freeState()
        XCTAssertTrue(partial.consume(byte: 0x22))
        XCTAssertTrue(partial.consume(byte: 0xF0))
        XCTAssertEqual(partial.status, .inProgress)
        XCTAssertTrue(partial.consume(byte: 0x9D))
        XCTAssertTrue(partial.consume(byte: 0x84))
        XCTAssertTrue(partial.consume(byte: 0x9E))
        XCTAssertTrue(partial.consume(byte: 0x22))
        XCTAssertEqual(partial.status, .complete)

        // Invalid lead bytes, overlongs, and encoded surrogates fail closed.
        var invalidLead = freeState()
        XCTAssertTrue(invalidLead.consume(byte: 0x22))
        XCTAssertFalse(invalidLead.consume(byte: 0xFF))
        var overlongLead = freeState()
        XCTAssertTrue(overlongLead.consume(byte: 0x22))
        XCTAssertFalse(overlongLead.consume(byte: 0xC0), "0xC0 is never a valid UTF-8 lead byte (overlong)")
        var overlongContinuation = freeState()
        XCTAssertTrue(overlongContinuation.consume(byte: 0x22))
        XCTAssertTrue(overlongContinuation.consume(byte: 0xE0))
        XCTAssertFalse(overlongContinuation.consume(byte: 0x9F), "E0 9F would be an overlong encoding")
        var encodedSurrogate = freeState()
        XCTAssertTrue(encodedSurrogate.consume(byte: 0x22))
        XCTAssertTrue(encodedSurrogate.consume(byte: 0xED))
        XCTAssertFalse(encodedSurrogate.consume(byte: 0xA0))
        var bareContinuation = freeState()
        XCTAssertTrue(bareContinuation.consume(byte: 0x22))
        XCTAssertFalse(bareContinuation.consume(byte: 0x80))
    }

    // MARK: - json_object: numbers

    func testNumbers() {
        let state = freeState()
        assertAccepts("0", state)
        assertAccepts("-0", state)
        assertAccepts("1234567890", state)
        assertAccepts("-12.5", state)
        assertAccepts("1e10", state)
        assertAccepts("1E-10", state)
        assertAccepts("1.5e+3", state)
        assertAccepts("0.0", state)
        assertIncomplete("-", state)
        assertIncomplete("1.", state)
        assertIncomplete("1e", state)
        assertIncomplete("1e+", state)
        assertRejected("01", state)
        assertRejected("1..5", state)
        assertRejected("1e+5.5", state)
        assertRejected("--5", state)
        assertRejected("1-", state)
        assertRejected(#"[1 2]"#, state)
        assertRejected(#"{"a": 1.}"#, state)
    }

    // MARK: - json_object: nesting and depth cap

    func testNestingAndDepthCap() {
        let state = freeState(maxDepth: 2)
        assertAccepts("[[1]]", state)
        assertAccepts(#"{"a": {"b": 1}}"#, state)
        assertRejected("[[[1]]]", state)
        assertRejected(#"{"a": {"b": {"c": 1}}}"#, state)
        assertRejected(#"{"a": [{"b": 1}]}"#, state)
    }

    // MARK: - strict json_schema subset

    func testCanarySchemaAcceptsWhitespaceAndCompleteRoot() throws {
        let state = JSONGrammarState(schema: try canarySchema())
        assertAccepts(#"{"status":"ok"}"#, state)
        assertAccepts(" \n { \"status\" : \"ok\" } \t", state)
        assertAccepts("{\"status\": \"ok\"}", state)
    }

    func testSchemaRootMustBeObject() throws {
        let state = JSONGrammarState(schema: try canarySchema())
        assertRejected(#"["ok"]"#, state)
        assertRejected(#""ok""#, state)
        assertRejected("42", state)
        assertRejected("true", state)
        assertRejected("null", state)
        assertIncomplete("{", state)
    }

    func testRequiredKeysAreEnforcedAtClose() throws {
        let state = JSONGrammarState(schema: try canarySchema())
        assertRejected("{}", state)
        assertRejected(#"{"status":"ok",}"#, state)
        assertIncomplete(#"{"status":"ok""#, state)
        assertAccepts(#"{"status":"ok"}"#, state)
    }

    func testAdditionalPropertiesAreRejected() throws {
        let state = JSONGrammarState(schema: try canarySchema())
        assertRejected(#"{"extra":1}"#, state)
        assertRejected(#"{"status":"ok","extra":1}"#, state)
        assertRejected(#"{"status":"ok","status":"ok"}"#, state)
        assertRejected(#"{"status":"ok",""#, state)
    }

    func testSchemaKeysMayAppearInAnyOrder() throws {
        let schema = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "pair", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "boolean"}}, "required": ["a", "b"], "additionalProperties": false}"#
                )))
        let state = JSONGrammarState(schema: schema)
        assertAccepts(#"{"a":"x","b":true}"#, state)
        assertAccepts(#"{"b":false,"a":""}"#, state)
        assertIncomplete(#"{"a":"x""#, state)
        assertRejected(#"{"a":"x"}"#, state)
        assertRejected(#"{"a":"x","a":"y"}"#, state)
        assertRejected(#"{"b":true,"c":1}"#, state)
    }

    func testSchemaEnumValuesAreEnforced() throws {
        let state = JSONGrammarState(schema: try canarySchema())
        assertRejected(#"{"status":"nope"}"#, state)
        assertRejected(#"{"status":"okay"}"#, state)
        assertRejected(#"{"status":42}"#, state)
        assertRejected(#"{"status":true}"#, state)
        assertIncomplete(#"{"status":"o"#, state)
        assertAccepts(#"{"status":"ok"}"#, state)
    }

    func testSchemaEnumMatchesDecodedEscapes() throws {
        let state = JSONGrammarState(schema: try canarySchema())
        assertAccepts(#"{"status":"\u006f\u006b"}"#, state)
        assertRejected(#"{"status":"\u004F\u004B"}"#, state)
        assertRejected(#"{"status":"\uD83D\uDE00"}"#, state)
    }

    func testSchemaEnumMatchesNonASCIIAndEscapedContent() throws {
        let emoji = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "mood", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"mood": {"type": "string", "enum": ["𝄞"]}}, "required": ["mood"], "additionalProperties": false}"#
                )))
        let emojiState = JSONGrammarState(schema: emoji)
        assertAccepts(#"{"mood":"𝄞"}"#, emojiState)
        assertAccepts(#"{"mood":"\uD834\uDD1E"}"#, emojiState)
        assertRejected(#"{"mood":"\uD834\uDD1F"}"#, emojiState)
        assertRejected(#"{"mood":"𝄞x"}"#, emojiState)
        assertRejected(#"{"mood":"x𝄞"}"#, emojiState)

        let quoted = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "quoted", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"s": {"type": "string", "enum": ["a\"b"]}}, "required": ["s"], "additionalProperties": false}"#
                )))
        let quotedState = JSONGrammarState(schema: quoted)
        assertAccepts(#"{"s":"a\"b"}"#, quotedState)
        assertAccepts(#"{"s":"a\u0022b"}"#, quotedState)
        assertRejected(#"{"s":"ab"}"#, quotedState)

        // Non-ASCII property names decode like enum values do.
        let cafe = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "cafe", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"café": {"type": "string"}}, "required": ["café"], "additionalProperties": false}"#
                )))
        let cafeState = JSONGrammarState(schema: cafe)
        assertAccepts(#"{"café":"x"}"#, cafeState)
        assertAccepts(#"{"caf\u00e9":"x"}"#, cafeState)
        assertRejected(#"{"cafe":"x"}"#, cafeState)
    }

    func testSchemaScalarTypes() throws {
        let state = JSONGrammarState(schema: try scalarSchema())
        assertAccepts(#"{"s":"x","i":3,"n":1.5,"b":true}"#, state)
        assertAccepts(#"{"b":false,"n":-2.5e3,"i":-7,"s":""}"#, state)
        assertAccepts(#"{"s":"x","i":0,"n":0.5,"b":true}"#, state)
        assertRejected(#"{"s":1,"i":3,"n":1.5,"b":true}"#, state)
        assertRejected(#"{"s":"x","i":3.5,"n":1.5,"b":true}"#, state)
        assertRejected(#"{"s":"x","i":3e2,"n":1.5,"b":true}"#, state)
        assertRejected(#"{"s":"x","i":3,"n":"1.5","b":true}"#, state)
        assertRejected(#"{"s":"x","i":3,"n":1.5,"b":1}"#, state)
        assertRejected(#"{"s":"x","i":3,"n":1.5,"b":null}"#, state)
        assertRejected(#"{"s":"x","i":3,"n":1.5}"#, state)
        assertRejected(#"{"s":"x","i":3,"n":1.5,"b":true,"extra":1}"#, state)
        assertIncomplete(#"{"s":"x","i":3,"n":1.5,"b":true"#, state)
    }

    func testSchemaStringFieldIsAnyJSONString() throws {
        let state = JSONGrammarState(schema: try scalarSchema())
        assertAccepts(#"{"s":"he said \"hi\"\n","i":0,"n":0,"b":false}"#, state)
        assertAccepts(#"{"s":"é𝄞","i":0,"n":0,"b":false}"#, state)
        assertAccepts(#"{"s":"\uD83D\uDE00","i":0,"n":0,"b":false}"#, state)
    }

    func testSchemaRejectsProseBeforeAndAfter() throws {
        let state = JSONGrammarState(schema: try canarySchema())
        assertRejected(#"prose {"status":"ok"}"#, state)
        assertRejected(#"{"status":"ok"} prose"#, state)
        assertRejected(#"{"status":"ok"}{"status":"ok"}"#, state)
        assertRejected(#"{"status":"ok"},"#, state)
    }

    func testSchemaKeyMustMatchADefinedProperty() throws {
        let state = JSONGrammarState(schema: try canarySchema())
        assertRejected(#"{"statuz":"ok"}"#, state)
        assertRejected(#"{"statusx":"ok"}"#, state)
        assertRejected(#"{"":"ok"}"#, state)
        assertIncomplete(#"{"st"#, state)
        assertAccepts(#"{"status":"ok"}"#, state)
    }

    // MARK: - Recursive json_schema subset: nested objects, arrays, nullable unions

    func testNestedObjectGrammar() throws {
        let state = JSONGrammarState(schema: try RecursiveSchemaFixture.schema())
        assertAccepts(#"{"meta":{"id":3},"tags":[],"note":null}"#, state)
        assertAccepts(#"{"note":"x","tags":["a","b"],"meta":{"id":-7}}"#, state)
        assertAccepts(" { \"meta\" : { \"id\" : 3 } , \"tags\" : [ ] , \"note\" : null } \n", state)
        assertIncomplete(#"{"meta":{"id":1}"#, state)
        // Nested required keys are enforced at the nested close.
        assertRejected(#"{"meta":{}"#, state)
        // Nested additionalProperties: false.
        assertRejected(#"{"meta":{"id":1,"extra":2}"#, state)
        // The key matcher is scoped to the nested object's own keys.
        assertRejected(#"{"meta":{"tags":[]}"#, state)
        // Duplicate nested key.
        assertRejected(#"{"meta":{"id":1,"id":2}"#, state)
        // Nested value types are enforced.
        assertRejected(#"{"meta":{"id":"3"}"#, state)
        assertRejected(#"{"meta":[]"#, state)
        // Root-level keys are not silently reusable inside `meta`.
        assertRejected(#"{"meta":{"id":1},"meta":{"id":2}"#, state)
        // The root close still enforces the root required keys.
        assertRejected(#"{"meta":{"id":1}}"#, state)
    }

    func testArrayItemsGrammar() throws {
        let state = JSONGrammarState(schema: try RecursiveSchemaFixture.schema())
        assertAccepts(#"{"meta":{"id":1},"tags":["a","b",""],"note":null}"#, state)
        assertRejected(#"{"meta":{"id":1},"tags":["a",1],"note":null}"#, state)
        assertRejected(#"{"meta":{"id":1},"tags":["a",],"note":null}"#, state)
        assertRejected(#"{"meta":{"id":1},"tags":[},"note":null}"#, state)
        assertRejected(#"{"meta":{"id":1},"tags":{"a":1},"note":null}"#, state)
        assertIncomplete(#"{"meta":{"id":1},"tags":["a"]"#, state)
        assertIncomplete(#"{"meta":{"id":1},"tags":["a""#, state)
    }

    func testRecursiveArrayOfObjectsGrammar() throws {
        let schema = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "rows", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"rows": {"type": "array", "items": {"type": "object", "properties": {"v": {"type": ["number", "null"]}}, "required": ["v"], "additionalProperties": false}}}, "required": ["rows"], "additionalProperties": false}"#
                )))
        let state = JSONGrammarState(schema: schema)
        assertAccepts(#"{"rows":[]}"#, state)
        assertAccepts(#"{"rows":[{"v":1},{"v":null},{"v":-2.5}]}"#, state)
        assertRejected(#"{"rows":[{"v":"x"}]}"#, state)
        assertRejected(#"{"rows":[{}]}"#, state)
        assertRejected(#"{"rows":[{"v":1,"x":2}]}"#, state)
        assertRejected(#"{"rows":[{"v":1},]}"#, state)
        assertIncomplete(#"{"rows":[{"v":1}]"#, state)
    }

    func testNestedArrayOfArraysGrammar() throws {
        let schema = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "grid", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"grid": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}}, "required": ["grid"], "additionalProperties": false}"#
                )))
        let state = JSONGrammarState(schema: schema)
        assertAccepts(#"{"grid":[[1],[2,3],[]]}"#, state)
        assertRejected(#"{"grid":[[1],[2,"x"]]}"#, state)
        assertRejected(#"{"grid":[[1] 2]}"#, state)
        assertRejected(#"{"grid":[[1],2]}"#, state)
    }

    func testNullableUnionGrammar() throws {
        let state = JSONGrammarState(schema: try RecursiveSchemaFixture.schema())
        assertAccepts(#"{"meta":{"id":1},"tags":[],"note":null}"#, state)
        assertAccepts(#"{"meta":{"id":1},"tags":[],"note":"anything"}"#, state)
        assertRejected(#"{"meta":{"id":1},"tags":[],"note":1}"#, state)
        assertRejected(#"{"meta":{"id":1},"tags":[],"note":true}"#, state)
        assertRejected(#"{"meta":{"id":1},"tags":[],"note":[]}"#, state)
        assertIncomplete(#"{"meta":{"id":1},"tags":[],"note":nul"#, state)
        assertIncomplete(#"{"meta":{"id":1},"tags":[],"note":null"#, state)
    }

    func testNullableEnumGrammar() throws {
        let withNull = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "nullable_enum", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"e": {"type": ["string", "null"], "enum": ["ok", null]}}, "required": ["e"], "additionalProperties": false}"#
                )))
        let withNullState = JSONGrammarState(schema: withNull)
        assertAccepts(#"{"e":null}"#, withNullState)
        assertAccepts(#"{"e":"ok"}"#, withNullState)
        assertAccepts(#"{"e":"\u006f\u006b"}"#, withNullState)
        assertRejected(#"{"e":"nope"}"#, withNullState)
        assertRejected(#"{"e":1}"#, withNullState)
        assertRejected(#"{"e":nulll}"#, withNullState)

        // An enum without null rejects null even though the union allows it.
        let withoutNull = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "nullable_enum", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"e": {"type": ["string", "null"], "enum": ["ok"]}}, "required": ["e"], "additionalProperties": false}"#
                )))
        let withoutNullState = JSONGrammarState(schema: withoutNull)
        assertRejected(#"{"e":null}"#, withoutNullState)
        assertAccepts(#"{"e":"ok"}"#, withoutNullState)

        // An enum of only null accepts null and no string.
        let onlyNull = try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "nullable_enum", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"e": {"type": ["string", "null"], "enum": [null]}}, "required": ["e"], "additionalProperties": false}"#
                )))
        let onlyNullState = JSONGrammarState(schema: onlyNull)
        assertAccepts(#"{"e":null}"#, onlyNullState)
        assertRejected(#"{"e":"ok"}"#, onlyNullState)
        assertRejected(#"{"e":""}"#, onlyNullState)
    }


    // MARK: - Lifecycle

    func testResetRestoresInitialState() throws {
        var state = JSONGrammarState(schema: try canarySchema())
        XCTAssertTrue(feed(#"{"status":"ok"}"#, into: &state))
        XCTAssertEqual(state.status, .complete)
        state.reset()
        XCTAssertEqual(state.status, .inProgress)
        XCTAssertTrue(feed(#"{"status":"ok"}"#, into: &state))
        XCTAssertEqual(state.status, .complete)

        var failed = freeState()
        XCTAssertFalse(feed("Hello", into: &failed))
        XCTAssertEqual(failed.status, .failed)
        failed.reset()
        XCTAssertEqual(failed.status, .inProgress)
        XCTAssertTrue(feed(#"{"a": 1}"#, into: &failed))
    }

    func testStateCopiesAreIndependent() throws {
        var original = JSONGrammarState(schema: try canarySchema())
        XCTAssertTrue(feed(#"{"status""#, into: &original))
        var copy = original
        XCTAssertTrue(feed(#": "ok"}"#, into: &copy))
        XCTAssertEqual(copy.status, .complete)
        XCTAssertEqual(original.status, .inProgress)
        XCTAssertTrue(feed(#": "ok"}"#, into: &original))
        XCTAssertEqual(original.status, .complete)
    }

    func testTextFormatHasNoGrammar() {
        XCTAssertThrowsError(try JSONGrammarState(format: .text)) { error in
            XCTAssertEqual(error as? JSONGrammarError, .unconstrainedFormat)
        }
    }
}
