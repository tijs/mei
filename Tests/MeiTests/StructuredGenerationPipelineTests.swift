import MLX
import MLXLMCommon
import XCTest

@testable import MeiCore

/// Model-free end-to-end pipeline tests for structured generation (slices 4–6
/// of the structured-generation plan): the exact CoCore canary request is
/// decoded, compiled, run through a deterministic scripted "model" driving the
/// real `JSONGrammarLogitProcessor` in the same call order as the vmlx
/// `TokenIterator` (`process(logits:) -> sampler -> didSample(token:)`), and
/// the assembled run is pushed through the same OpenAI response DTOs the
/// engine feeds — `Router.completionResponse` for the buffered path and
/// `Router.chunkSSEData` / `Router.finishSSEData` for the streaming path.
///
/// No test here may pass merely because the prompt asked for JSON: the
/// scripted sampler can only produce output the constraint mask admits, and
/// the negatives prove that a captured grammar failure or an incomplete run
/// cannot be reported as a successful structured response.
final class StructuredGenerationPipelineTests: XCTestCase {

    private let model = "test-model"
    private let serializer = ResponseSerializer()

    // MARK: - Toy vocabularies

    /// Tokens for the exact CoCore canary document `{"status": "ok"}` plus
    /// the tokens the canary prompt begs for (prose) and the tokens a model
    /// would try for the wrong schema (wrong enum).
    private enum CanaryToken: Int, CaseIterable {
        case openBraceQuote  // `{"`
        case keyTailColon  // `status":`
        case valueOpenQuote  // ` "`
        case valueOKCloseQuote  // `ok"`
        case closeBrace  // `}`
        case space  // ` `
        case prose  // `Hello` — what the unconstrained model wants
        case wrongEnum  // `nope"` — a wrong enum value
        case eos  // no fragment

        var fragment: [UInt8]? {
            func bytes(_ string: String) -> [UInt8] { Array(string.utf8) }
            switch self {
            case .openBraceQuote: return bytes("{\"")
            case .keyTailColon: return bytes("status\":")
            case .valueOpenQuote: return bytes(" \"")
            case .valueOKCloseQuote: return bytes("ok\"")
            case .closeBrace: return bytes("}")
            case .space: return bytes(" ")
            case .prose: return bytes("Hello")
            case .wrongEnum: return bytes("nope\"")
            case .eos: return nil
            }
        }
    }

    /// Tokens for a free `json_object` document `{"items": [1,5]}`.
    private enum JSONObjectToken: Int, CaseIterable {
        case openItemsBracket  // `{"items": [`
        case one  // `1`
        case comma  // `,`
        case five  // `5`
        case closeBracket  // `]`
        case closeBrace  // `}`
        case space  // ` `
        case eos  // no fragment

        var fragment: [UInt8]? {
            func bytes(_ string: String) -> [UInt8] { Array(string.utf8) }
            switch self {
            case .openItemsBracket: return bytes("{\"items\": [")
            case .one: return bytes("1")
            case .comma: return bytes(",")
            case .five: return bytes("5")
            case .closeBracket: return bytes("]")
            case .closeBrace: return bytes("}")
            case .space: return bytes(" ")
            case .eos: return nil
            }
        }
    }

    /// Tokens for the scalar-type matrix schema document
    /// `{"name":"ok","count":3,"ratio":1.5,"flag":true}` plus decoys for
    /// every type boundary (fraction under `integer`, wrong enum, duplicate
    /// key, boolean/string confusion).
    private enum MatrixToken: Int, CaseIterable {
        case openNameQuote  // `{"name":"`
        case nameOKQuote  // `ok"`
        case commaCountQuote  // `,"count":`
        case three  // `3`
        case commaRatioQuote  // `,"ratio":`
        case onePointFive  // `1.5`
        case commaFlagQuote  // `,"flag":`
        case trueLiteral  // `true`
        case closeBrace  // `}`
        case commaNameQuote  // `,"name":` — duplicate key
        case threePointFive  // `3.5` — fraction under integer
        case quotedNope  // `"nope"` — wrong enum value
        case falseLiteral  // `false`
        case nullLiteral  // `null`
        case space  // ` `
        case eos  // no fragment

        var fragment: [UInt8]? {
            func bytes(_ string: String) -> [UInt8] { Array(string.utf8) }
            switch self {
            case .openNameQuote: return bytes("{\"name\":\"")
            case .nameOKQuote: return bytes("ok\"")
            case .commaCountQuote: return bytes(",\"count\":")
            case .three: return bytes("3")
            case .commaRatioQuote: return bytes(",\"ratio\":")
            case .onePointFive: return bytes("1.5")
            case .commaFlagQuote: return bytes(",\"flag\":")
            case .trueLiteral: return bytes("true")
            case .closeBrace: return bytes("}")
            case .commaNameQuote: return bytes(",\"name\":")
            case .threePointFive: return bytes("3.5")
            case .quotedNope: return bytes("\"nope\"")
            case .falseLiteral: return bytes("false")
            case .nullLiteral: return bytes("null")
            case .space: return bytes(" ")
            case .eos: return nil
            }
        }
    }

    private func canaryTable() -> StaticTokenFragmentTable {
        StaticTokenFragmentTable(
            fragments: CanaryToken.allCases.map { $0.fragment },
            endOfSequenceTokenIds: [CanaryToken.eos.rawValue])
    }

    private func jsonObjectTable() -> StaticTokenFragmentTable {
        StaticTokenFragmentTable(
            fragments: JSONObjectToken.allCases.map { $0.fragment },
            endOfSequenceTokenIds: [JSONObjectToken.eos.rawValue])
    }

    private func matrixTable() -> StaticTokenFragmentTable {
        StaticTokenFragmentTable(
            fragments: MatrixToken.allCases.map { $0.fragment },
            endOfSequenceTokenIds: [MatrixToken.eos.rawValue])
    }

    /// The exact document tokens (decoys excluded), then EOS.
    private var canaryScript: [Int] {
        [CanaryToken.openBraceQuote, .keyTailColon, .valueOpenQuote, .valueOKCloseQuote, .closeBrace]
            .map(\.rawValue) + [CanaryToken.eos.rawValue]
    }

    private var jsonObjectScript: [Int] {
        [JSONObjectToken.openItemsBracket, .one, .comma, .five, .closeBracket, .closeBrace]
            .map(\.rawValue) + [JSONObjectToken.eos.rawValue]
    }

    // MARK: - Scripted model seam

    /// Drives the plan's processor in the vmlx `TokenIterator` order:
    /// `process(logits:) -> sampler -> didSample(token:)`. The `script` is the
    /// model's fixed token preference; every scripted token must survive the
    /// mask or the test fails — a passing run therefore proves the constraint
    /// admitted exactly this output, not that the prompt asked nicely.
    ///
    /// The processor is copied by value exactly like the engine's copy that
    /// the iterator stores, so completion/failure observations must travel
    /// through the shared run record.
    private func scriptedGenerate(
        plan: StructuredGeneration.Plan,
        table: any TokenFragmentTable,
        script: [Int],
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> String {
        var processor = plan.processor
        var bytes: [UInt8] = []
        for tokenId in script {
            let masked = processor.process(
                logits: MLXArray(Array(repeating: Float(0), count: table.vocabularySize)))
            let row = masked.asArray(Float.self)
            XCTAssertNotEqual(
                row[tokenId], -Float.infinity,
                "scripted token \(tokenId) must survive the constraint mask", file: file, line: line)
            processor.didSample(token: MLXArray([Int32(tokenId)]))
            bytes += table.fragment(forTokenId: tokenId) ?? []
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// A sampler that ignores the mask (models a sampler bug or a run whose
    /// constraint was bypassed): the token is consumed unconditionally.
    private func forceSample(_ processor: inout JSONGrammarLogitProcessor, tokenId: Int) {
        processor.didSample(token: MLXArray([Int32(tokenId)]))
    }

    // MARK: - Request/response helpers

    private func canaryRequest() throws -> ChatRequest {
        try ChatRequest(json: CoCoreCanary.structuredOutputBodyData(model: model))
    }

    private func makeRun(
        text: String,
        finishReason: String,
        completionTokens: Int,
        promptTokens: Int = 24
    ) -> GenerationRun {
        var run = GenerationRun()
        run.text = text
        run.finishReason = finishReason
        run.promptTokenCount = promptTokens
        run.completionTokenCount = completionTokens
        run.cachedTokenCount = 0
        run.decodeTokensPerSecond = 42.5
        run.promptTokensPerSecond = 100
        run.prefillMilliseconds = 12
        run.generateMilliseconds = 34
        return run
    }

    private func jsonObject(_ string: String) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: Data(string.utf8)) as? [String: Any]) ?? [:]
    }

    private func responseBody(_ run: GenerationRun) -> [String: Any] {
        jsonObject(serializer.json(
            Router.completionResponse(run: run, model: model, emitReasoning: false)))
    }

    private func httpError(_ error: Error) -> (status: Int, body: [String: Any]) {
        let result = Router.errorResult(error, serializer: serializer)
        guard case .plain(let status, _, let body) = result else {
            XCTFail("expected a plain error response, got \(result)")
            return (0, [:])
        }
        return (Int(status.code), jsonObject(body))
    }

    /// The streaming terminal frames exactly as `ResponseWriter.streamSSE`
    /// ships them: `data: <json>\n\n` per frame.
    private func finishFrames(_ run: GenerationRun, includeUsage: Bool) -> [String] {
        Router.finishSSEData(
            id: "chatcmpl-pipeline-test", run: run, model: model,
            created: 1_700_000_000, includeUsage: includeUsage
        ).map { "data: \($0)\n\n" }
    }

    private func chunkFrames(_ chunks: [String]) -> [String] {
        chunks.map {
            Router.chunkSSEData(
                text: $0, id: "chatcmpl-pipeline-test", model: model, created: 1_700_000_000)
        }
    }

    /// Assembles SSE frames the way a client does: concatenate delta content,
    /// take the last finish_reason, keep the usage chunk, note [DONE].
    private func assembleSSE(
        _ frames: [String]
    ) -> (content: String, finishReason: String?, usage: [String: Any]?, sawDone: Bool) {
        var content = ""
        var finishReason: String?
        var usage: [String: Any]?
        var sawDone = false
        for frame in frames {
            for rawLine in frame.split(separator: "\n", omittingEmptySubsequences: false) {
                let line = rawLine.trimmingCharacters(in: .whitespaces)
                guard line.hasPrefix("data:") else { continue }
                let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                if payload == "[DONE]" {
                    sawDone = true
                    continue
                }
                guard let object = jsonObject(String(payload)) as [String: Any]?,
                    !object.isEmpty
                else { continue }
                if let chunkUsage = object["usage"] as? [String: Any] { usage = chunkUsage }
                for choice in object["choices"] as? [[String: Any]] ?? [] {
                    if let delta = choice["delta"] as? [String: Any],
                        let text = delta["content"] as? String
                    {
                        content += text
                    }
                    if let reason = choice["finish_reason"] as? String { finishReason = reason }
                }
            }
        }
        return (content, finishReason, usage, sawDone)
    }

    // MARK: - Exact CoCore canary: buffered path

    func testBufferedCanaryRunYieldsTheExactCoCoreResponse() throws {
        let request = try canaryRequest()
        XCTAssertNoThrow(try Router.validateStructuredRequest(request))
        let table = canaryTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: request.responseFormat, table: table))

        let text = scriptedGenerate(plan: plan, table: table, script: canaryScript)
        XCTAssertEqual(text, #"{"status": "ok"}"#)
        XCTAssertNil(plan.runRecord.failure)
        XCTAssertTrue(
            plan.runRecord.rootValueCompleted,
            "the shared run record must observe completion through the iterator's copy")

        let run = makeRun(text: text, finishReason: "stop", completionTokens: 6)
        let body = responseBody(run)
        let choices = body["choices"] as? [[String: Any]] ?? []
        XCTAssertEqual(choices.count, 1)
        XCTAssertEqual(choices.first?["finish_reason"] as? String, "stop")
        let content = (choices.first?["message"] as? [String: Any])?["content"] as? String
        XCTAssertEqual(content, text)
        XCTAssertTrue(
            CoCoreCanary.structuredOutputPassed(responseBody: body),
            "the exact CoCore canary must pass on this response")

        // The post-generation invariant agrees: a normal stop with a
        // completed root value is reportable.
        XCTAssertNil(
            StructuredGeneration.postGenerationError(
                plan: plan, stopReason: .stop, toolCallCount: 0))
    }

    // MARK: - Exact CoCore canary: streaming path

    func testStreamingCanaryRunMatchesTheBufferedResponse() throws {
        let request = try canaryRequest()
        let table = canaryTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: request.responseFormat, table: table))
        let text = scriptedGenerate(plan: plan, table: table, script: canaryScript)
        let run = makeRun(text: text, finishReason: "stop", completionTokens: 6)

        // Per-token chunks, exactly how the engine forwards vmlx `.chunk`.
        let chunks = [CanaryToken.openBraceQuote, .keyTailColon, .valueOpenQuote, .valueOKCloseQuote, .closeBrace]
            .map { String(decoding: CanaryToken(rawValue: $0.rawValue)!.fragment ?? [], as: UTF8.self) }
        let assembled = assembleSSE(chunkFrames(chunks) + finishFrames(run, includeUsage: true))

        XCTAssertEqual(assembled.content, text, "SSE deltas must assemble to the buffered content")
        XCTAssertEqual(assembled.finishReason, "stop")
        XCTAssertTrue(assembled.sawDone, "[DONE] must terminate the stream")
        XCTAssertTrue(
            CoCoreCanary.structuredOutputPassed(content: assembled.content),
            "the assembled canary stream must pass the exact CoCore condition")

        // Usage parity: the streamed usage chunk carries the same numbers as
        // the buffered response's usage block.
        let bufferedUsage = Router.usage(run: run)
        let usage = try XCTUnwrap(assembled.usage)
        XCTAssertEqual(usage["prompt_tokens"] as? Int, bufferedUsage.promptTokens)
        XCTAssertEqual(usage["completion_tokens"] as? Int, bufferedUsage.completionTokens)
        XCTAssertEqual(usage["total_tokens"] as? Int, bufferedUsage.totalTokens)
    }

    func testSplitJSONChunksReassembleToTheSameContent() throws {
        let request = try canaryRequest()
        let table = canaryTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: request.responseFormat, table: table))
        let text = scriptedGenerate(plan: plan, table: table, script: canaryScript)
        let run = makeRun(text: text, finishReason: "stop", completionTokens: 6)

        // Split mid-fragment (one character per chunk) so the JSON arrives
        // split at arbitrary boundaries, not only at token boundaries.
        let chunks = text.map { String($0) }
        let assembled = assembleSSE(chunkFrames(chunks) + finishFrames(run, includeUsage: false))
        XCTAssertEqual(assembled.content, text)
        XCTAssertEqual(assembled.finishReason, "stop")
        XCTAssertTrue(CoCoreCanary.structuredOutputPassed(content: assembled.content))
    }

    // MARK: - json_object tracer bullet

    func testJSONObjectPipelineProducesParseableJSON() throws {
        let table = jsonObjectTable()
        let plan = try XCTUnwrap(try StructuredGeneration.plan(for: .jsonObject, table: table))
        let text = scriptedGenerate(plan: plan, table: table, script: jsonObjectScript)
        XCTAssertEqual(text, #"{"items": [1,5]}"#)
        XCTAssertTrue(plan.runRecord.rootValueCompleted)

        // Buffered path: content parses as JSON and finishes with stop.
        let run = makeRun(text: text, finishReason: "stop", completionTokens: 6)
        let body = responseBody(run)
        let content = ((body["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String
        XCTAssertEqual(content, text)
        XCTAssertEqual((body["choices"] as? [[String: Any]])?.first?["finish_reason"] as? String, "stop")
        let parsed = try XCTUnwrap(
            (try? JSONSerialization.jsonObject(with: Data((content ?? "").utf8))) as? [String: Any])
        XCTAssertEqual(parsed["items"] as? [Int], [1, 5])

        // Streaming path: same content, same finish reason.
        let assembled = assembleSSE(chunkFrames(text.map { String($0) }) + finishFrames(run, includeUsage: true))
        XCTAssertEqual(assembled.content, text)
        XCTAssertEqual(assembled.finishReason, "stop")
        XCTAssertTrue(assembled.sawDone)
    }

    // MARK: - Negatives: prose, wrong enum, impossible state

    func testProseAndWrongEnumAreMaskedOutUnderTheCanaryConstraint() throws {
        let request = try canaryRequest()
        let table = canaryTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: request.responseFormat, table: table))
        var processor = plan.processor

        // The canary prompt begs for prose; the constraint must not admit it.
        let initial = processor.process(
            logits: MLXArray(Array(repeating: Float(0), count: table.vocabularySize)))
        let initialRow = initial.asArray(Float.self)
        XCTAssertNotEqual(initialRow[CanaryToken.openBraceQuote.rawValue], -Float.infinity)
        XCTAssertEqual(initialRow[CanaryToken.prose.rawValue], -Float.infinity, "prose must be masked")
        XCTAssertEqual(
            initialRow[CanaryToken.wrongEnum.rawValue], -Float.infinity,
            "a wrong enum value must be masked")

        // Same at the enum value position.
        forceSample(&processor, tokenId: CanaryToken.openBraceQuote.rawValue)
        forceSample(&processor, tokenId: CanaryToken.keyTailColon.rawValue)
        forceSample(&processor, tokenId: CanaryToken.valueOpenQuote.rawValue)
        let valuePosition = processor.process(
            logits: MLXArray(Array(repeating: Float(0), count: table.vocabularySize)))
        let valueRow = valuePosition.asArray(Float.self)
        XCTAssertNotEqual(valueRow[CanaryToken.valueOKCloseQuote.rawValue], -Float.infinity)
        XCTAssertEqual(valueRow[CanaryToken.wrongEnum.rawValue], -Float.infinity)
        XCTAssertEqual(valueRow[CanaryToken.prose.rawValue], -Float.infinity)
        XCTAssertEqual(
            valueRow[CanaryToken.eos.rawValue], -Float.infinity, "premature EOS must be masked")
    }

    func testCapturedGrammarFailureCannotYieldASuccessFinish() throws {
        let request = try canaryRequest()
        let table = canaryTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: request.responseFormat, table: table))
        var processor = plan.processor
        for token in [CanaryToken.openBraceQuote, .keyTailColon, .valueOpenQuote] {
            forceSample(&processor, tokenId: token.rawValue)
        }
        // A sampler that ignores the mask picks prose: the constraint records
        // the failure and the request must fail closed.
        forceSample(&processor, tokenId: CanaryToken.prose.rawValue)
        XCTAssertEqual(plan.runRecord.failure, .illegalToken(CanaryToken.prose.rawValue))

        let error = try XCTUnwrap(
            StructuredGeneration.postGenerationError(
                plan: plan, stopReason: .stop, toolCallCount: 0))
        guard case .generationFailed(let message) = error else {
            return XCTFail("expected generationFailed, got \(error)")
        }
        XCTAssertTrue(message.contains("structured output"), message)
        let (status, body) = httpError(error)
        XCTAssertEqual(status, 500)
        let object = body["error"] as? [String: Any] ?? [:]
        XCTAssertEqual(object["type"] as? String, "invalid_request_error")
        XCTAssertEqual(object["code"] as? String, "engine_error")
        XCTAssertNil(body["choices"], "a failed constraint must not carry a completion")
    }

    func testIncompleteRunAtStopFailsClosed() throws {
        let request = try canaryRequest()
        let table = canaryTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: request.responseFormat, table: table))
        // Valid prefix, but the run stops (no EOS) before the object closes.
        _ = scriptedGenerate(
            plan: plan, table: table,
            script: [CanaryToken.openBraceQuote, .keyTailColon, .valueOpenQuote, .valueOKCloseQuote]
                .map(\.rawValue))
        XCTAssertNil(plan.runRecord.failure, "no token was illegal — the run is just incomplete")
        XCTAssertFalse(plan.runRecord.rootValueCompleted)

        for stopReason: GenerateStopReason in [.stop, .cancelled] {
            let error = try XCTUnwrap(
                StructuredGeneration.postGenerationError(
                    plan: plan, stopReason: stopReason, toolCallCount: 0),
                "a \(stopReason) stop with an incomplete grammar must fail closed")
            guard case .generationFailed(let message) = error else {
                return XCTFail("expected generationFailed, got \(error)")
            }
            XCTAssertTrue(message.contains("complete JSON value"), message)
            XCTAssertEqual(httpError(error).status, 500)
        }
        XCTAssertNotNil(
            StructuredGeneration.postGenerationError(plan: plan, stopReason: nil, toolCallCount: 0),
            "a missing stop reason defaults to a reported stop and must fail closed")

        // A structured response with an incomplete root is invalid even when
        // the producer reports a length stop: do not return a successful
        // partial JSON payload.
        let lengthError = try XCTUnwrap(
            StructuredGeneration.postGenerationError(
                plan: plan, stopReason: .length, toolCallCount: 0))
        guard case .generationFailed(let lengthMessage) = lengthError else {
            return XCTFail("expected generationFailed for length truncation, got \(lengthError)")
        }
        XCTAssertTrue(lengthMessage.contains("complete JSON value"), lengthMessage)
        XCTAssertEqual(httpError(lengthError).status, 500)
        XCTAssertEqual(Engine.mapStopReason(.length, toolCallCount: 0), "length")
    }

    func testCompletedRunWithTrailingWhitespaceStillReportable() throws {
        let request = try canaryRequest()
        let table = canaryTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: request.responseFormat, table: table))
        // Complete the root, then let the model emit whitespace before EOS —
        // a normal vmlx completion.
        _ = scriptedGenerate(
            plan: plan, table: table,
            script: [CanaryToken.openBraceQuote, .keyTailColon, .valueOpenQuote, .valueOKCloseQuote, .closeBrace, .space, .eos]
                .map(\.rawValue))
        XCTAssertTrue(plan.runRecord.rootValueCompleted)
        XCTAssertNil(plan.runRecord.failure)
        XCTAssertNil(
            StructuredGeneration.postGenerationError(
                plan: plan, stopReason: .stop, toolCallCount: 0))
    }

    // MARK: - Streaming error mapping

    func testStreamErrorFrameCarriesStructuredFailure() throws {
        let request = try canaryRequest()
        let table = canaryTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: request.responseFormat, table: table))
        _ = scriptedGenerate(
            plan: plan, table: table,
            script: [CanaryToken.openBraceQuote, .keyTailColon].map(\.rawValue))
        let error = try XCTUnwrap(
            StructuredGeneration.postGenerationError(plan: plan, stopReason: .stop, toolCallCount: 0))

        let frame = Router.streamErrorSSEData(error)
        XCTAssertTrue(frame.hasPrefix("data: "), frame)
        XCTAssertTrue(frame.hasSuffix("\n\n"), frame)
        XCTAssertFalse(frame.contains("finish_reason"), "a failed stream must not carry a finish frame")
        let payload = frame
            .dropFirst("data: ".count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let object = jsonObject(String(payload))
        let envelope = object["error"] as? [String: Any] ?? [:]
        XCTAssertEqual(envelope["code"] as? String, "stream_error")
        let message = envelope["message"] as? String ?? ""
        XCTAssertTrue(message.contains("structured output"), message)
    }

    // MARK: - Scalar-type schema matrix slice

    func testScalarTypeMatrixMasksAndProducesTheSchemaObject() throws {
        let format = ResponseFormat.jsonSchema(
            JSONSchemaFormat(
                name: "matrix", strict: true,
                schema: try decode(
                    #"{"type": "object", "properties": {"name": {"type": "string", "enum": ["ok", "fine"]}, "count": {"type": "integer"}, "ratio": {"type": "number"}, "flag": {"type": "boolean"}}, "required": ["name", "count", "ratio", "flag"], "additionalProperties": false}"#
                )))
        _ = try JSONSchemaCompiler.compile(format)
        let table = matrixTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: format, table: table))
        var processor = plan.processor

        func row(_ processor: inout JSONGrammarLogitProcessor) -> [Float] {
            processor.process(logits: MLXArray(Array(repeating: Float(0), count: table.vocabularySize)))
                .asArray(Float.self)
        }
        func allowed(_ processor: inout JSONGrammarLogitProcessor, _ token: MatrixToken) -> Bool {
            row(&processor)[token.rawValue] != -Float.infinity
        }

        // Document start: only a `{`-leading token can begin the object.
        XCTAssertTrue(allowed(&processor, .openNameQuote))
        XCTAssertFalse(allowed(&processor, .nameOKQuote))
        XCTAssertFalse(allowed(&processor, .three))
        XCTAssertFalse(allowed(&processor, .nullLiteral))
        XCTAssertFalse(allowed(&processor, .eos))

        // String enum position: only prefix-compatible enum values.
        forceSample(&processor, tokenId: MatrixToken.openNameQuote.rawValue)
        XCTAssertTrue(allowed(&processor, .nameOKQuote))
        XCTAssertFalse(allowed(&processor, .quotedNope))
        XCTAssertFalse(allowed(&processor, .nullLiteral))

        // After the enum value: another key, but not a duplicate, and not a
        // close while required keys are missing.
        forceSample(&processor, tokenId: MatrixToken.nameOKQuote.rawValue)
        XCTAssertTrue(allowed(&processor, .commaCountQuote))
        XCTAssertTrue(allowed(&processor, .commaRatioQuote))
        XCTAssertFalse(allowed(&processor, .commaNameQuote), "a key may appear at most once")
        XCTAssertFalse(allowed(&processor, .three))
        XCTAssertFalse(allowed(&processor, .closeBrace), "required keys are still missing")

        // Integer position: no fraction, no exponent, no boolean.
        forceSample(&processor, tokenId: MatrixToken.commaCountQuote.rawValue)
        XCTAssertTrue(allowed(&processor, .three))
        XCTAssertFalse(allowed(&processor, .onePointFive), "a fraction is not an integer")
        XCTAssertFalse(allowed(&processor, .threePointFive), "a fraction is not an integer")
        XCTAssertFalse(allowed(&processor, .trueLiteral))

        // Number position: integers are valid numbers.
        forceSample(&processor, tokenId: MatrixToken.three.rawValue)
        forceSample(&processor, tokenId: MatrixToken.commaRatioQuote.rawValue)
        XCTAssertTrue(allowed(&processor, .onePointFive))
        XCTAssertTrue(allowed(&processor, .three))
        XCTAssertFalse(allowed(&processor, .trueLiteral))
        XCTAssertFalse(allowed(&processor, .quotedNope))

        // Boolean position: only true/false.
        forceSample(&processor, tokenId: MatrixToken.onePointFive.rawValue)
        forceSample(&processor, tokenId: MatrixToken.commaFlagQuote.rawValue)
        XCTAssertTrue(allowed(&processor, .trueLiteral))
        XCTAssertTrue(allowed(&processor, .falseLiteral))
        XCTAssertFalse(allowed(&processor, .three))

        // Close: only after every required key, then EOS.
        forceSample(&processor, tokenId: MatrixToken.trueLiteral.rawValue)
        XCTAssertTrue(allowed(&processor, .closeBrace))
        XCTAssertFalse(allowed(&processor, .eos))
        forceSample(&processor, tokenId: MatrixToken.closeBrace.rawValue)
        XCTAssertTrue(allowed(&processor, .eos))
        XCTAssertTrue(allowed(&processor, .space))
        forceSample(&processor, tokenId: MatrixToken.eos.rawValue)

        XCTAssertTrue(plan.runRecord.rootValueCompleted)
        XCTAssertNil(plan.runRecord.failure)
    }

    func testScalarTypeMatrixDocumentThroughBothResponsePaths() throws {
        let format = ResponseFormat.jsonSchema(
            JSONSchemaFormat(
                name: "matrix", strict: true,
                schema: try decode(
                    #"{"type": "object", "properties": {"name": {"type": "string", "enum": ["ok", "fine"]}, "count": {"type": "integer"}, "ratio": {"type": "number"}, "flag": {"type": "boolean"}}, "required": ["name", "count", "ratio", "flag"], "additionalProperties": false}"#
                )))
        guard case .jsonSchema(let schema) = try JSONSchemaCompiler.compile(format) else {
            return XCTFail("matrix schema must compile")
        }
        let table = matrixTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: format, table: table))
        let script = [
            MatrixToken.openNameQuote, .nameOKQuote, .commaCountQuote, .three,
            .commaRatioQuote, .onePointFive, .commaFlagQuote, .trueLiteral,
            .closeBrace, .eos,
        ].map(\.rawValue)
        let text = scriptedGenerate(plan: plan, table: table, script: script)
        XCTAssertEqual(text, #"{"name":"ok","count":3,"ratio":1.5,"flag":true}"#)
        XCTAssertTrue(plan.runRecord.rootValueCompleted)

        let run = makeRun(text: text, finishReason: "stop", completionTokens: script.count)
        let body = responseBody(run)
        let content = ((body["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String
        let parsed = try XCTUnwrap(
            (try? JSONSerialization.jsonObject(with: Data((content ?? "").utf8))) as? [String: Any])
        XCTAssertEqual(parsed["name"] as? String, "ok")
        XCTAssertEqual((parsed["count"] as? NSNumber)?.intValue, 3)
        XCTAssertEqual((parsed["ratio"] as? NSNumber)?.doubleValue, 1.5)
        XCTAssertEqual(parsed["flag"] as? Bool, true)
        XCTAssertTrue(schema.validate(try decode(content ?? "{}")), "the compiled validator agrees")

        let assembled = assembleSSE(chunkFrames(text.map { String($0) }) + finishFrames(run, includeUsage: true))
        XCTAssertEqual(assembled.content, text)
        XCTAssertEqual(assembled.finishReason, "stop")
        XCTAssertNil(
            StructuredGeneration.postGenerationError(plan: plan, stopReason: .stop, toolCallCount: 0))
    }

    // MARK: - CoCore checker oracle

    func testCoCoreCanaryCheckerMatchesTheRustOracle() {
        // Positives (cocore's own test cases).
        XCTAssertTrue(CoCoreCanary.structuredOutputPassed(content: #"{"status":"ok"}"#))
        XCTAssertTrue(CoCoreCanary.structuredOutputPassed(content: "  {\"status\": \"ok\"}\n"))
        // Negatives.
        XCTAssertFalse(CoCoreCanary.structuredOutputPassed(content: "Hello! I'm doing great today, thanks for asking."))
        XCTAssertFalse(CoCoreCanary.structuredOutputPassed(content: #"{"status":"fine"}"#))
        XCTAssertFalse(CoCoreCanary.structuredOutputPassed(content: #"{"status":"ok","mood":"good"}"#))
        XCTAssertFalse(CoCoreCanary.structuredOutputPassed(content: "Sure! {\"status\":\"ok\"}"))
        XCTAssertFalse(CoCoreCanary.structuredOutputPassed(content: nil))
        XCTAssertFalse(CoCoreCanary.structuredOutputPassed(content: #"["status","ok"]"#))
        XCTAssertFalse(CoCoreCanary.structuredOutputPassed(content: #"{"status":42}"#))
        XCTAssertFalse(CoCoreCanary.structuredOutputPassed(responseBody: ["choices": []]))

        // The response-body form reads choices[0].message.content.
        XCTAssertTrue(
            CoCoreCanary.structuredOutputPassed(
                responseBody: CoCoreCanary.responseBody(content: CoCoreCanary.passingContent)))
        XCTAssertFalse(
            CoCoreCanary.structuredOutputPassed(
                responseBody: CoCoreCanary.responseBody(content: "Hello!")))
    }

    func testCoCoreCanaryRequestBodyIsTheExactShape() throws {
        let request = try canaryRequest()
        XCTAssertEqual(request.maxTokens, 64)
        XCTAssertEqual(request.temperature, 0)
        XCTAssertEqual(request.messages.count, 2)
        guard case .jsonSchema(let schema) = request.responseFormat else {
            return XCTFail("canary must decode to json_schema, got \(request.responseFormat)")
        }
        XCTAssertEqual(schema.name, "canary_status")
        XCTAssertTrue(schema.strict)
        // The compiler accepts the exact canary schema.
        let compiled = try JSONSchemaCompiler.compile(schema)
        XCTAssertEqual(compiled.required, ["status"])
    }

    // MARK: - Helpers

    private func decode(_ json: String) throws -> MeiJSONValue {
        try JSONDecoder().decode(MeiJSONValue.self, from: Data(json.utf8))
    }
}
