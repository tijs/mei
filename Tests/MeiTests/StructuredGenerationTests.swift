import Foundation
import MLX
import XCTest

@testable import MeiCore

/// Model-free tests for the Engine seam that builds structured-generation
/// plans (slice 4 of the structured-generation plan), the Router-side
/// pre-generation validation that replaced the temporary fail-closed gate,
/// and the thinking-off policy for structured requests.
///
/// These pin the wiring contract without loading any model weights: the
/// tokenizer is a fake, the vocabulary size comes from a temporary
/// `config.json`, and the HTTP mapping is the pure `Router.errorResult`.
final class StructuredGenerationTests: XCTestCase {

    // MARK: - Helpers

    private func canarySchemaJSON() -> String {
        #"{"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}}, "required": ["status"], "additionalProperties": false}"#
    }

    private func canaryFormat() throws -> ResponseFormat {
        let schema = try JSONDecoder().decode(MeiJSONValue.self, from: Data(canarySchemaJSON().utf8))
        return .jsonSchema(JSONSchemaFormat(name: "canary_status", strict: true, schema: schema))
    }

    private func toyTable() -> StaticTokenFragmentTable {
        StaticTokenFragmentTable(
            strings: ["{\"", "status\":", " \"", "ok\"", "}", " ", nil],
            endOfSequenceTokenIds: [6])
    }

    private func chatRequest(
        responseFormat: String? = nil,
        tools: String? = nil,
        reasoningEffort: String? = nil
    ) throws -> ChatRequest {
        var fields: [String] = [
            #""model": "m""#,
            #""messages": [{"role": "user", "content": "hi"}]"#,
        ]
        if let responseFormat { fields.append(#""response_format": \#(responseFormat)"#) }
        if let tools { fields.append(#""tools": \#(tools)"#) }
        if let reasoningEffort { fields.append(#""reasoning_effort": "\#(reasoningEffort)""#) }
        return try ChatRequest(json: Data("{\(fields.joined(separator: ", "))}".utf8))
    }

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
        httpError(error).body["error"] as? [String: Any] ?? [:]
    }

    // MARK: - Plan construction

    func testTextFormatHasNoPlan() throws {
        XCTAssertNil(try StructuredGeneration.plan(for: .text, table: toyTable()))
    }

    func testAbsentResponseFormatHasNoPlan() throws {
        let request = try chatRequest()
        XCTAssertEqual(request.responseFormat, .text)
        XCTAssertNil(try StructuredGeneration.plan(for: request.responseFormat, table: toyTable()))
    }

    func testJSONObjectPlanUsesTheCompiledConstraint() throws {
        let plan = try XCTUnwrap(StructuredGeneration.plan(for: .jsonObject, table: toyTable()))
        XCTAssertEqual(plan.format, .jsonObject)
        XCTAssertEqual(plan.processor.constraintKey, "json_object")
        XCTAssertFalse(plan.runRecord.isFailed)
        XCTAssertTrue(plan.runRecord === plan.processor.runRecord)
    }

    func testJSONSchemaPlanCompilesTheStrictSubset() throws {
        let plan = try XCTUnwrap(StructuredGeneration.plan(for: try canaryFormat(), table: toyTable()))
        guard case .jsonSchema(let compiled) = plan.format else {
            return XCTFail("expected a compiled schema, got \(plan.format)")
        }
        XCTAssertEqual(compiled.name, "canary_status")
        XCTAssertEqual(compiled.required, ["status"])
    }

    func testUnsupportedSchemaSurfacesTheCompilerError() throws {
        let schema = try JSONDecoder().decode(
            MeiJSONValue.self,
            from: Data(
                #"{"type": "object", "properties": {"nested": {"type": "object"}}, "required": ["nested"], "additionalProperties": false}"#
                    .utf8))
        let format = ResponseFormat.jsonSchema(
            JSONSchemaFormat(name: "x", strict: true, schema: schema))
        XCTAssertThrowsError(try StructuredGeneration.plan(for: format, table: toyTable())) { error in
            XCTAssertEqual(
                error as? JSONSchemaCompileError,
                .propertyTypeUnsupported("nested", "object"))
        }
    }

    func testPlanProcessorMasksThroughTheSuppliedTable() throws {
        let plan = try XCTUnwrap(StructuredGeneration.plan(for: try canaryFormat(), table: toyTable()))
        let masked = plan.processor.process(
            logits: MLXArray(Array(repeating: Float(1), count: 7)))
        let values = masked.asArray(Float.self)
        XCTAssertEqual(values[0], 1, "`{\"` can start the document")
        XCTAssertEqual(values[5], 1, "whitespace is allowed")
        XCTAssertEqual(values[1], -Float.infinity)
        XCTAssertEqual(values[6], -Float.infinity, "premature EOS")
    }

    // MARK: - Model vocabulary size

    private func makeModelDirectory(configJSON: String?) throws -> String {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-structured-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        if let configJSON {
            try Data(configJSON.utf8).write(to: directory.appendingPathComponent("config.json"))
        }
        return directory.path
    }

    func testModelVocabularySizeReadsRootAndNestedTextConfig() throws {
        let root = try makeModelDirectory(configJSON: #"{"model_type": "qwen3", "vocab_size": 151936}"#)
        XCTAssertEqual(try StructuredGeneration.modelVocabularySize(modelDirectory: root), 151936)

        let nested = try makeModelDirectory(
            configJSON: #"{"model_type": "qwen3_5_moe", "text_config": {"vocab_size": 248320}}"#)
        XCTAssertEqual(try StructuredGeneration.modelVocabularySize(modelDirectory: nested), 248320)
    }

    func testModelVocabularySizeFailsClosed() throws {
        let missing = try makeModelDirectory(configJSON: nil)
        XCTAssertThrowsError(try StructuredGeneration.modelVocabularySize(modelDirectory: missing)) { error in
            XCTAssertEqual(error as? StructuredGenerationError, .modelConfigUnreadable(missing))
        }

        let absent = try makeModelDirectory(configJSON: #"{"model_type": "qwen3"}"#)
        XCTAssertThrowsError(try StructuredGeneration.modelVocabularySize(modelDirectory: absent)) { error in
            XCTAssertEqual(error as? StructuredGenerationError, .modelVocabularySizeMissing(absent))
        }

        for raw in [#"{"vocab_size": "big"}"#, #"{"vocab_size": 0}"#, #"{"vocab_size": true}"#] {
            let directory = try makeModelDirectory(configJSON: raw)
            XCTAssertThrowsError(try StructuredGeneration.modelVocabularySize(modelDirectory: directory)) { error in
                guard case .modelVocabularySizeInvalid = error as? StructuredGenerationError else {
                    return XCTFail("expected invalid size for \(raw), got \(error)")
                }
            }
        }
    }

    // MARK: - Thinking policy

    func testStructuredRequestsForceThinkingOff() throws {
        let structured = try chatRequest(
            responseFormat: #"{"type": "json_object"}"#, reasoningEffort: "high")
        XCTAssertEqual(
            StructuredGeneration.enableThinking(request: structured, configEnableThinking: nil),
            false, "structured requests must not reason")
        XCTAssertEqual(
            StructuredGeneration.enableThinking(request: structured, configEnableThinking: true),
            false, "even an explicit server-side thinking enable is overridden")
    }

    func testTextRequestsKeepTheExistingThinkingPrecedence() throws {
        let effort = try chatRequest(reasoningEffort: "high")
        XCTAssertEqual(StructuredGeneration.enableThinking(request: effort, configEnableThinking: nil), true)
        let none = try chatRequest(reasoningEffort: "none")
        XCTAssertEqual(StructuredGeneration.enableThinking(request: none, configEnableThinking: nil), false)
        let plain = try chatRequest()
        XCTAssertNil(StructuredGeneration.enableThinking(request: plain, configEnableThinking: nil))
        XCTAssertEqual(StructuredGeneration.enableThinking(request: plain, configEnableThinking: true), true)
        XCTAssertEqual(
            StructuredGeneration.enableThinking(request: effort, configEnableThinking: false), false,
            "an explicit server-side value still wins for text")
    }

    // MARK: - Decode-path gate

    func testStructuredRequestsStayOffTheCompiledDecodePath() throws {
        // The plan's decode-path gate: structured requests ride the ordinary
        // single-sequence, non-compiled decode path until the processor state
        // is proven safe for graph-traced replay. An operator's
        // `--compiled-decode true` must not pull a constrained request onto it.
        let structured = try chatRequest(responseFormat: #"{"type": "json_object"}"#)
        XCTAssertFalse(
            StructuredGeneration.enableCompiledDecode(request: structured, configEnabled: true),
            "a structured request must not ride compiled decode")
        let plain = try chatRequest()
        XCTAssertTrue(StructuredGeneration.enableCompiledDecode(request: plain, configEnabled: true))
        XCTAssertFalse(StructuredGeneration.enableCompiledDecode(request: plain, configEnabled: false))
    }

    // MARK: - Failure mapping (HTTP 500)

    func testCapturedFailureMapsToGenerationFailed() {
        let error = StructuredGeneration.generationFailure(.noLegalContinuation)
        guard case .generationFailed(let message) = error else {
            return XCTFail("expected generationFailed, got \(error)")
        }
        XCTAssertTrue(
            message.contains("no token in the vocabulary can advance the grammar"),
            "the grammar failure must be named: \(message)")
        XCTAssertEqual(Router.errorStatus(error), .internalServerError)
        let (status, body) = httpError(error)
        XCTAssertEqual(status, 500)
        let object = body["error"] as? [String: Any] ?? [:]
        XCTAssertEqual(object["type"] as? String, "invalid_request_error")
        XCTAssertEqual(object["code"] as? String, "engine_error")
    }

    func testStructuredConstructionFailuresAreServerErrors() {
        // Unreadable model config / no identifiable EOS / mismatched
        // vocabulary are server-side: well-formed requests cannot fix them,
        // so they must not masquerade as client 400s.
        let (status, body) = httpError(StructuredGenerationError.modelConfigUnreadable("/nonexistent"))
        XCTAssertEqual(status, 500)
        XCTAssertEqual(
            (body["error"] as? [String: Any])?["code"] as? String, "engine_error")

        let (tableStatus, tableBody) = httpError(
            TokenizerFragmentTableError.missingEndOfSequenceToken)
        XCTAssertEqual(tableStatus, 500)
        XCTAssertEqual(
            (tableBody["error"] as? [String: Any])?["code"] as? String, "engine_error")
    }

    // MARK: - Router pre-generation validation

    func testSupportedStructuredRequestsPassValidation() throws {
        let jsonObject = try chatRequest(responseFormat: #"{"type": "json_object"}"#)
        XCTAssertNoThrow(try Router.validateStructuredRequest(jsonObject))

        let schema = try chatRequest(
            responseFormat: #"{"type": "json_schema", "json_schema": {"name": "canary_status", "strict": true, "schema": \#(canarySchemaJSON())}}"#)
        XCTAssertNoThrow(try Router.validateStructuredRequest(schema))
    }

    func testTextRequestsSkipStructuredValidation() throws {
        let request = try chatRequest(
            tools: #"[{"type": "function", "function": {"name": "f", "parameters": {}}}]"#)
        XCTAssertEqual(request.responseFormat, .text)
        XCTAssertNoThrow(
            try Router.validateStructuredRequest(request),
            "tools without response_format keep the ordinary path")
    }

    func testStructuredRequestsWithToolsAreRejected() throws {
        let request = try chatRequest(
            responseFormat: #"{"type": "json_object"}"#,
            tools: #"[{"type": "function", "function": {"name": "f", "parameters": {}}}]"#)
        XCTAssertThrowsError(try Router.validateStructuredRequest(request)) { error in
            XCTAssertEqual(error as? ResponseFormatError, .structuredToolsUnsupported)
        }
        let object = errorObject(ResponseFormatError.structuredToolsUnsupported)
        XCTAssertEqual(object["code"] as? String, "response_format_unsupported")
        XCTAssertEqual(object["param"] as? String, "response_format")
        XCTAssertEqual(httpError(ResponseFormatError.structuredToolsUnsupported).status, 400)
    }

    func testEmptyToolsArrayDoesNotRejectStructuredRequests() throws {
        let request = try chatRequest(
            responseFormat: #"{"type": "json_object"}"#, tools: "[]")
        XCTAssertNoThrow(try Router.validateStructuredRequest(request))
    }

    func testCompilerErrorsAreHTTP400NamingResponseFormat() throws {
        // A well-formed request whose schema uses an unsupported construct:
        // the compiler rejects it BEFORE generation with a precise 400.
        let request = try chatRequest(
            responseFormat: #"{"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": {"type": "object", "properties": {"nested": {"type": "object"}}, "required": ["nested"], "additionalProperties": false}}}"#)
        XCTAssertThrowsError(try Router.validateStructuredRequest(request)) { error in
            guard let compileError = error as? JSONSchemaCompileError else {
                return XCTFail("expected a compiler error, got \(error)")
            }
            XCTAssertEqual(compileError, .propertyTypeUnsupported("nested", "object"))
            let object = errorObject(compileError)
            XCTAssertEqual(object["type"] as? String, "invalid_request_error")
            XCTAssertEqual(object["code"] as? String, "response_format_unsupported")
            XCTAssertEqual(object["param"] as? String, "response_format")
            let message = object["message"] as? String ?? ""
            XCTAssertTrue(
                message.contains("response_format.json_schema.schema"),
                "the message must name the offending field: \(message)")
            XCTAssertTrue(message.contains("nested"), "the compiler message must survive: \(message)")
            XCTAssertEqual(httpError(compileError).status, 400)
        }
    }

    func testRootLevelCompilerRejectionsAlsoMapTo400() throws {
        // A schema whose ROOT is unsupported (missing additionalProperties:
        // false) still maps through the same precise 400 shape.
        let request = try chatRequest(
            responseFormat: #"{"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": {"type": "object", "properties": {}, "required": []}}}"#)
        XCTAssertThrowsError(try Router.validateStructuredRequest(request)) { error in
            XCTAssertEqual(
                error as? JSONSchemaCompileError, .additionalPropertiesMustBeFalse)
        }
    }
}
