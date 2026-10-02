import XCTest
@testable import MeiCore

/// Contract tests for the Chat Completions `response_format` request field —
/// slice 1 of the structured-generation plan (freeze the request/error
/// contract; no generation behavior yet).
///
/// Pinned here:
/// - decoding of `text` / `json_object` / strict `json_schema` payloads,
///   including the exact CoCore structured-output canary body;
/// - explicit typed rejection (→ HTTP 400, `param: "response_format"`) of
///   missing/wrong `type`, missing `json_schema`/`name`/`schema`, non-object
///   schemas, and non-strict `strict`;
/// - byte-compatible behavior for requests without the field and for the
///   existing error envelope (no `code`/`param` added to old errors);
/// - the pre-generation validation contract that replaced the intermediate
///   fail-closed gate: structured formats compile before generation (a
///   compiler rejection is HTTP 400 with `param: "response_format"`) and
///   structured output + `tools` is rejected explicitly (see
///   `StructuredGenerationTests` for the wiring contract);
/// - legacy `/v1/completions` stays out of scope.
final class ResponseFormatTests: XCTestCase {

    // MARK: - Helpers

    private func chatBody(responseFormat raw: String?) -> Data {
        let field = raw.map { "\"response_format\": \($0), " } ?? ""
        let json =
            "{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \(field)\"temperature\": 0}"
        return Data(json.utf8)
    }

    private func decodeFormat(_ raw: String) throws -> ResponseFormat {
        try ChatRequest(json: chatBody(responseFormat: raw)).responseFormat
    }

    /// The HTTP contract the router produces for a thrown error.
    private func httpError(_ error: Error) -> (status: Int, body: [String: Any]) {
        let result = Router.errorResult(error, serializer: ResponseSerializer())
        guard case .plain(let status, _, let body) = result else {
            XCTFail("expected a plain error response, got \(result)")
            return (0, [:])
        }
        let parsed = (try? JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any]) ?? [:]
        return (Int(status.code), parsed)
    }

    private func errorObject(_ error: Error) -> [String: Any] {
        let (_, body) = httpError(error)
        return body["error"] as? [String: Any] ?? [:]
    }

    // MARK: - Valid payloads

    func testAbsentResponseFormatDefaultsToText() throws {
        let request = try ChatRequest(json: chatBody(responseFormat: nil))
        XCTAssertEqual(request.responseFormat, .text)
    }

    func testExplicitTextFormatDecodesToText() throws {
        XCTAssertEqual(try decodeFormat(#"{"type": "text"}"#), .text)
        // Unknown sibling keys stay inert (house rule: unknown fields never error).
        XCTAssertEqual(try decodeFormat(#"{"type": "text", "json_schema": {"name": "x"}}"#), .text)
    }

    func testExplicitNullResponseFormatIsTreatedAsAbsent() throws {
        // Some JSON encoders emit null for an unset optional field; treating
        // it as absent keeps those clients working.
        XCTAssertEqual(try decodeFormat("null"), .text)
    }

    func testJSONObjectFormatDecodes() throws {
        XCTAssertEqual(try decodeFormat(#"{"type": "json_object"}"#), .jsonObject)
    }

    func testStrictJSONSchemaFormatDecodesPreservingSchemaValues() throws {
        let format = try decodeFormat(
            #"{"type": "json_schema", "json_schema": {"name": "thing", "description": "why", "strict": true, "schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "additionalProperties": false}}}"#
        )
        guard case .jsonSchema(let schema) = format else {
            return XCTFail("expected json_schema, got \(format)")
        }
        XCTAssertEqual(schema.name, "thing")
        XCTAssertTrue(schema.strict)
        guard case .object(let root) = schema.schema else {
            return XCTFail("schema must be preserved as a JSON object")
        }
        XCTAssertEqual(root["type"], .string("object"))
        guard case .object(let properties)? = root["properties"] else {
            return XCTFail("properties must be preserved")
        }
        XCTAssertEqual(properties["n"], .object(["type": .string("integer")]))
        XCTAssertEqual(root["required"], .array([.string("n")]))
        XCTAssertEqual(root["additionalProperties"], .bool(false))
    }

    /// The exact request CoCore's attached engine sends at startup to prove
    /// structured output (graze-social/cocore PR #237,
    /// `engines/openai_http.rs::structured_output_canary_body`, pinned in
    /// docs/COCORE.md §2). The prompt deliberately begs for prose so a server
    /// that drops `response_format` fails the canary instead of passing by
    /// luck.
    func testCoCoreStructuredOutputCanaryBodyDecodes() throws {
        let json = #"""
        {
          "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
          "messages": [
            {"role": "system", "content": "You are a friendly assistant who always answers in two or three warm, conversational sentences."},
            {"role": "user", "content": "Say hello and tell me how you are doing today."}
          ],
          "response_format": {
            "type": "json_schema",
            "json_schema": {
              "name": "canary_status",
              "strict": true,
              "schema": {
                "type": "object",
                "properties": {"status": {"type": "string", "enum": ["ok"]}},
                "required": ["status"],
                "additionalProperties": false
              }
            }
          },
          "max_tokens": 64,
          "temperature": 0
        }
        """#
        let request = try ChatRequest(json: Data(json.utf8))
        XCTAssertEqual(request.messages.count, 2)
        XCTAssertEqual(request.maxTokens, 64)
        XCTAssertEqual(request.temperature, 0)
        guard case .jsonSchema(let schema) = request.responseFormat else {
            return XCTFail("canary must decode to json_schema, got \(request.responseFormat)")
        }
        XCTAssertEqual(schema.name, "canary_status")
        XCTAssertTrue(schema.strict)
        guard case .object(let root) = schema.schema,
            case .object(let properties)? = root["properties"],
            case .object(let status)? = properties["status"]
        else {
            return XCTFail("canary schema must preserve the status property")
        }
        XCTAssertEqual(status["enum"], .array([.string("ok")]))
        XCTAssertEqual(root["required"], .array([.string("status")]))
        XCTAssertEqual(root["additionalProperties"], .bool(false))
    }

    /// Schema *subset* validation is the model-free compiler's job (slice 2),
    /// deliberately not HTTP decoding: an out-of-subset schema still decodes
    /// here and is rejected later by the compiler / request gate.
    func testSchemaSubsetValidationIsNotPerformedAtDecode() throws {
        let format = try decodeFormat(
            #"{"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": {"type": "array", "items": {"type": "string"}}}}"#
        )
        guard case .jsonSchema(let schema) = format else {
            return XCTFail("expected json_schema, got \(format)")
        }
        XCTAssertEqual(schema.schema, .object(["type": .string("array"), "items": .object(["type": .string("string")])]))
    }

    // MARK: - Malformed / unsupported payloads

    func testNonObjectResponseFormatRejected() throws {
        for raw in [#""text""#, "5", "[]", "true"] {
            XCTAssertThrowsError(try decodeFormat(raw), raw) { error in
                XCTAssertEqual(error as? ResponseFormatError, .notObject, raw)
            }
        }
    }

    func testMissingTypeRejected() throws {
        for raw in [#"{}"#, #"{"json_schema": {"name": "x"}}"#] {
            XCTAssertThrowsError(try decodeFormat(raw), raw) { error in
                XCTAssertEqual(error as? ResponseFormatError, .missingType, raw)
            }
        }
    }

    func testNonStringTypeRejected() throws {
        XCTAssertThrowsError(try decodeFormat(#"{"type": 5}"#)) { error in
            XCTAssertEqual(error as? ResponseFormatError, .invalidType)
        }
    }

    func testUnknownTypeRejected() throws {
        XCTAssertThrowsError(try decodeFormat(#"{"type": "audio"}"#)) { error in
            XCTAssertEqual(error as? ResponseFormatError, .unsupportedType("audio"))
        }
    }

    func testMissingJSONSchemaRejected() throws {
        for raw in [#"{"type": "json_schema"}"#, #"{"type": "json_schema", "json_schema": "x"}"#, #"{"type": "json_schema", "json_schema": null}"#] {
            XCTAssertThrowsError(try decodeFormat(raw), raw) { error in
                XCTAssertEqual(error as? ResponseFormatError, .missingJSONSchema, raw)
            }
        }
    }

    func testMissingNameRejected() throws {
        let raws = [
            #"{"type": "json_schema", "json_schema": {"strict": true, "schema": {}}}"#,
            #"{"type": "json_schema", "json_schema": {"name": "", "strict": true, "schema": {}}}"#,
            #"{"type": "json_schema", "json_schema": {"name": 5, "strict": true, "schema": {}}}"#,
        ]
        for raw in raws {
            XCTAssertThrowsError(try decodeFormat(raw), raw) { error in
                XCTAssertEqual(error as? ResponseFormatError, .missingName, raw)
            }
        }
    }

    func testMissingOrMalformedSchemaRejected() throws {
        let missing = [
            #"{"type": "json_schema", "json_schema": {"name": "x", "strict": true}}"#,
            #"{"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": null}}"#,
        ]
        for raw in missing {
            XCTAssertThrowsError(try decodeFormat(raw), raw) { error in
                XCTAssertEqual(error as? ResponseFormatError, .missingSchema, raw)
            }
        }
        let malformed = [
            #"{"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": "nope"}}"#,
            #"{"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": [1]}}"#,
        ]
        for raw in malformed {
            XCTAssertThrowsError(try decodeFormat(raw), raw) { error in
                XCTAssertEqual(error as? ResponseFormatError, .invalidSchema, raw)
            }
        }
    }

    func testInvalidStrictRejected() throws {
        // Only explicitly strict schemas are supported; an absent, non-bool,
        // or false `strict` is an explicit 400, never a silent downgrade.
        let raws = [
            #"{"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}"#,
            #"{"type": "json_schema", "json_schema": {"name": "x", "strict": false, "schema": {}}}"#,
            #"{"type": "json_schema", "json_schema": {"name": "x", "strict": "true", "schema": {}}}"#,
            #"{"type": "json_schema", "json_schema": {"name": "x", "strict": 1, "schema": {}}}"#,
        ]
        for raw in raws {
            XCTAssertThrowsError(try decodeFormat(raw), raw) { error in
                XCTAssertEqual(error as? ResponseFormatError, .invalidStrict, raw)
            }
        }
    }

    // MARK: - HTTP error contract

    func testResponseFormatErrorsCarryHTTP400Contract() throws {
        let cases: [(String, ResponseFormatError)] = [
            (#"{}"#, .missingType),
            (#"{"type": 5}"#, .invalidType),
            (#"{"type": "audio"}"#, .unsupportedType("audio")),
            (#"{"type": "json_schema"}"#, .missingJSONSchema),
            (#"{"type": "json_schema", "json_schema": {"strict": true, "schema": {}}}"#, .missingName),
            (#"{"type": "json_schema", "json_schema": {"name": "x", "strict": true}}"#, .missingSchema),
            (#"{"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": []}}"#, .invalidSchema),
            (#"{"type": "json_schema", "json_schema": {"name": "x", "strict": false, "schema": {}}}"#, .invalidStrict),
        ]
        for (raw, expected) in cases {
            XCTAssertThrowsError(try decodeFormat(raw), raw) { error in
                XCTAssertEqual(error as? ResponseFormatError, expected, raw)
            }
            let (status, body) = httpError(expected)
            XCTAssertEqual(status, 400, raw)
            let error = body["error"] as? [String: Any] ?? [:]
            XCTAssertEqual(error["type"] as? String, "invalid_request_error", raw)
            XCTAssertEqual(error["code"] as? String, "invalid_response_format", raw)
            XCTAssertEqual(error["param"] as? String, "response_format", raw)
            let message = error["message"] as? String ?? ""
            XCTAssertFalse(message.isEmpty, raw)
            XCTAssertTrue(message.contains("response_format"), "message must name the field: \(message)")
        }
    }

    func testErrorEnvelopeOmitsNilCodeAndParam() throws {
        struct PlainFailure: Error, LocalizedError {
            var errorDescription: String? { "plain failure" }
        }
        let (status, body) = httpError(PlainFailure())
        XCTAssertEqual(status, 400)
        let error = body["error"] as? [String: Any] ?? [:]
        XCTAssertEqual(error["message"] as? String, "plain failure")
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertNil(error["code"], "existing 400 decode errors carry no code — keep the envelope byte-compatible")
        XCTAssertNil(error["param"], "param must be omitted unless the error names a request parameter")
    }

    // MARK: - Structured requests are wired (gate replaced)

    func testStructuredRequestsPassPreGenerationValidation() throws {
        // The temporary fail-closed gate (`Router.structuredFormatRejection`)
        // was removed when the constrained-decoding processor was wired in:
        // a well-formed structured request is no longer rejected up front —
        // it is answered under a token-level constraint — while malformed
        // payloads keep their explicit 400s (pinned above) and structured +
        // tools is rejected (`StructuredGenerationTests`).
        let canary =
            #"{"type": "json_schema", "json_schema": {"name": "canary_status", "strict": true, "schema": {"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}}, "required": ["status"], "additionalProperties": false}}}"#
        guard case .jsonSchema = try decodeFormat(canary) else {
            return XCTFail("the canary schema must decode to a json_schema format")
        }
        for raw in [#"{"type": "json_object"}"#, canary] {
            let request = try ChatRequest(json: chatBody(responseFormat: raw))
            XCTAssertNoThrow(try Router.validateStructuredRequest(request), raw)
        }
        let absent = try ChatRequest(json: chatBody(responseFormat: nil))
        XCTAssertEqual(absent.responseFormat, .text)
        XCTAssertNoThrow(
            try Router.validateStructuredRequest(absent),
            "an absent response_format keeps the ordinary path")
    }

    // MARK: - Out of scope

    func testRawCompletionsRemainOutOfScope() throws {
        // Legacy /v1/completions is out of the P0 contract: its DTO must not
        // gain a response_format surface, and the field stays inert there.
        let json = #"""
        {"model": "m", "prompt": "hello", "response_format": {"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": false}}}}
        """#
        let request = try CompletionRequest(json: Data(json.utf8))
        XCTAssertEqual(request.prompt, "hello")
    }
}
