import MLX
import MLXLMCommon
import XCTest

@testable import MeiCore

/// Model-free tests for the Mei-owned `LogitProcessor` wrapper around the
/// grammar core (slice 4 of the structured-generation plan).
///
/// `LogitProcessor.process` cannot throw, so the wrapper must turn every
/// grammar failure into (a) a deterministic, request-scoped, inspectable
/// failure record and (b) an all-`-inf` mask — never unmasked logits.
final class JSONGrammarLogitProcessorTests: XCTestCase {

    // MARK: - Toy vocabulary

    /// `{"status": "ok"}` + EOS over a small fragment table.
    private enum Toy: Int, CaseIterable {
        case openBraceQuote
        case keyTail
        case valueOpenQuote
        case valueOKCloseQuote
        case closeBrace
        case space
        case eos

        var fragment: [UInt8]? {
            switch self {
            case .openBraceQuote: return Array("{\"".utf8)
            case .keyTail: return Array("status\":".utf8)
            case .valueOpenQuote: return Array(" \"".utf8)
            case .valueOKCloseQuote: return Array("ok\"".utf8)
            case .closeBrace: return Array("}".utf8)
            case .space: return Array(" ".utf8)
            case .eos: return nil
            }
        }
    }

    private func toyTable() -> StaticTokenFragmentTable {
        StaticTokenFragmentTable(
            fragments: Toy.allCases.map { $0.fragment },
            endOfSequenceTokenIds: [Toy.eos.rawValue])
    }

    private func canarySchema() throws -> CompiledJSONSchema {
        let schema = try JSONDecoder().decode(
            MeiJSONValue.self,
            from: Data(
                #"{"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}}, "required": ["status"], "additionalProperties": false}"#
                    .utf8))
        return try JSONSchemaCompiler.compile(
            JSONSchemaFormat(name: "canary_status", strict: true, schema: schema))
    }

    private func canaryProcessor(
        handle: JSONGrammarRunRecord = JSONGrammarRunRecord()
    ) throws -> JSONGrammarLogitProcessor {
        try JSONGrammarLogitProcessor(
            format: .jsonSchema(try canarySchema()), table: toyTable(),
            runRecord: handle)
    }

    private func logits(_ values: [Float]) -> MLXArray { MLXArray(values) }

    private func sample(_ id: Int) -> MLXArray { MLXArray([Int32(id)]) }

    // MARK: - Masking

    func testProcessMasksDisallowedTokensAndKeepsAllowedValues() throws {
        let processor = try canaryProcessor()
        let input = logits(Array(repeating: Float(1), count: Toy.allCases.count))
        let masked = processor.process(logits: input)
        let values = masked.asArray(Float.self)

        XCTAssertEqual(values[Toy.openBraceQuote.rawValue], 1)
        XCTAssertEqual(values[Toy.space.rawValue], 1)
        XCTAssertEqual(values[Toy.keyTail.rawValue], -Float.infinity)
        XCTAssertEqual(values[Toy.eos.rawValue], -Float.infinity, "premature EOS must be masked")
        XCTAssertEqual(
            input.asArray(Float.self)[Toy.keyTail.rawValue], 1,
            "the input logits must not be modified")
        XCTAssertFalse(processor.runRecord.isFailed)
        XCTAssertEqual(processor.status, .inProgress)
    }

    // MARK: - Failure capture (process cannot throw)

    func testNoLegalContinuationReturnsAllNegativeInfinityAndCapturesFailure() throws {
        // A vocabulary that can spell `{"status":"ok"` but cannot close the
        // object: after the value, no token can advance the grammar.
        let table = StaticTokenFragmentTable(
            strings: ["{\"", "status\":", "\"ok\"", nil], endOfSequenceTokenIds: [3])
        let handle = JSONGrammarRunRecord()
        var processor = try JSONGrammarLogitProcessor(
            format: .jsonSchema(try canarySchema()), table: table, runRecord: handle)
        processor.didSample(token: sample(0))
        processor.didSample(token: sample(1))
        processor.didSample(token: sample(2))
        XCTAssertFalse(handle.isFailed)

        let masked = processor.process(logits: logits(Array(repeating: Float(1), count: 4)))
        let values = masked.asArray(Float.self)
        XCTAssertEqual(values.count, 4)
        for (index, value) in values.enumerated() {
            XCTAssertEqual(value, -Float.infinity, "index \(index) must be masked")
        }
        XCTAssertEqual(handle.failure, .noLegalContinuation)
        XCTAssertTrue(handle.isFailed)
    }

    func testIllegalSampledTokenIsCapturedAndProcessStaysAllMasked() throws {
        let handle = JSONGrammarRunRecord()
        var processor = try canaryProcessor(handle: handle)
        // `status":` cannot start a document.
        processor.didSample(token: sample(Toy.keyTail.rawValue))
        XCTAssertEqual(handle.failure, .illegalToken(Toy.keyTail.rawValue))

        let masked = processor.process(logits: logits(Array(repeating: Float(1), count: Toy.allCases.count)))
        XCTAssertTrue(masked.asArray(Float.self).allSatisfy { $0 == -Float.infinity })
        XCTAssertEqual(handle.failure, .illegalToken(Toy.keyTail.rawValue), "the FIRST failure is deterministic")
    }

    func testFirstFailureWinsDeterministically() throws {
        let handle = JSONGrammarRunRecord()
        var processor = try canaryProcessor(handle: handle)
        processor.didSample(token: sample(Toy.keyTail.rawValue))  // illegalToken
        processor.didSample(token: sample(Toy.openBraceQuote.rawValue))  // grammarFailed (state poisoned)
        XCTAssertEqual(handle.failure, .illegalToken(Toy.keyTail.rawValue))
    }

    func testVocabularyMismatchFailsClosed() throws {
        let handle = JSONGrammarRunRecord()
        let processor = try canaryProcessor(handle: handle)
        let masked = processor.process(logits: logits([1, 1, 1]))
        XCTAssertTrue(masked.asArray(Float.self).allSatisfy { $0 == -Float.infinity })
        XCTAssertEqual(
            handle.failure,
            .vocabularyMismatch(expected: Toy.allCases.count, actual: 3))
    }

    func testPrematureEndOfSequenceIsCapturedWithoutPoisoningTheState() throws {
        let handle = JSONGrammarRunRecord()
        var processor = try canaryProcessor(handle: handle)
        processor.didSample(token: sample(Toy.eos.rawValue))
        XCTAssertEqual(handle.failure, .prematureEndOfSequence)
        // The rejected EOS must not have advanced or poisoned the grammar:
        // a fresh, valid step still masks correctly.
        XCTAssertEqual(processor.status, .inProgress)
        processor.didSample(token: sample(Toy.openBraceQuote.rawValue))
        XCTAssertEqual(processor.status, .inProgress)
    }

    // MARK: - Normal completion

    func testAcceptingGrammarAllowsEOSAndFinishesNormally() throws {
        let handle = JSONGrammarRunRecord()
        var processor = try canaryProcessor(handle: handle)
        for token in [
            Toy.openBraceQuote, .keyTail, .valueOpenQuote, .valueOKCloseQuote, .closeBrace,
        ] {
            XCTAssertTrue(
                processor.process(logits: logits(Array(repeating: Float(1), count: Toy.allCases.count)))
                    .asArray(Float.self)[token.rawValue] != -Float.infinity,
                "\(token) must be allowed")
            processor.didSample(token: sample(token.rawValue))
        }
        XCTAssertEqual(processor.status, .complete)
        XCTAssertTrue(processor.isAccepting)

        let masked = processor.process(logits: logits(Array(repeating: Float(1), count: Toy.allCases.count)))
        XCTAssertEqual(masked.asArray(Float.self)[Toy.eos.rawValue], 1, "EOS must be allowed at completion")
        processor.didSample(token: sample(Toy.eos.rawValue))
        XCTAssertEqual(processor.status, .finished)
        XCTAssertTrue(
            processor.isAccepting,
            "a complete root value stays accepting after EOS: the answer is satisfied")
        XCTAssertTrue(
            handle.rootValueCompleted,
            "the run record must observe completion for the engine's post-generation check")
        XCTAssertFalse(handle.isFailed, "a completed document with EOS is a normal completion")
    }

    func testGenerationLookaheadAfterEOSDoesNotPoisonACompletedRun() throws {
        let handle = JSONGrammarRunRecord()
        var processor = try canaryProcessor(handle: handle)
        for token in [
            Toy.openBraceQuote, Toy.keyTail, Toy.valueOpenQuote, Toy.valueOKCloseQuote, Toy.closeBrace,
            Toy.eos,
        ] {
            processor.didSample(token: sample(token.rawValue))
        }
        XCTAssertEqual(processor.status, .finished)

        // TokenIterator.next() primes the following model step before it
        // forwards the previously sampled token. Therefore vmlx can invoke
        // process/didSample once after EOS; that lookahead is invisible to the
        // client and must not turn a valid completion into HTTP 500.
        _ = processor.process(logits: logits(Array(repeating: Float(1), count: Toy.allCases.count)))
        processor.didSample(token: sample(Toy.space.rawValue))

        XCTAssertFalse(handle.isFailed, "post-EOS lookahead is benign after a valid completion")
        XCTAssertTrue(handle.rootValueCompleted)
    }

    // MARK: - Lifecycle

    func testPromptResetsTheGrammar() throws {
        var processor = try canaryProcessor()
        processor.didSample(token: sample(Toy.openBraceQuote.rawValue))
        processor.didSample(token: sample(Toy.keyTail.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Toy.openBraceQuote.rawValue))

        processor.prompt(MLXArray([Int32(1), Int32(2), Int32(3)]))
        XCTAssertEqual(processor.status, .inProgress)
        XCTAssertTrue(processor.isAllowed(tokenId: Toy.openBraceQuote.rawValue))
    }

    func testIndependentCopyIsolatesStateAndSharesTheFailureHandle() throws {
        let handle = JSONGrammarRunRecord()
        var original = try canaryProcessor(handle: handle)
        original.didSample(token: sample(Toy.openBraceQuote.rawValue))

        var copy = original.independentCopy()
        XCTAssertTrue(copy.runRecord === handle, "the request-scoped handle is shared")
        XCTAssertTrue(copy.runRecord === original.runRecord)

        // Advancing the copy must not advance the original.
        copy.didSample(token: sample(Toy.keyTail.rawValue))
        copy.didSample(token: sample(Toy.valueOpenQuote.rawValue))
        XCTAssertTrue(copy.isAllowed(tokenId: Toy.valueOKCloseQuote.rawValue))
        XCTAssertTrue(original.isAllowed(tokenId: Toy.keyTail.rawValue))
        XCTAssertFalse(original.isAllowed(tokenId: Toy.valueOpenQuote.rawValue))

        // A failure recorded through the copy is visible to the request owner.
        copy.didSample(token: sample(Toy.eos.rawValue))
        XCTAssertEqual(original.runRecord.failure, .prematureEndOfSequence)

        // The original's own masking still reflects its own state.
        let masked = original.process(logits: logits(Array(repeating: Float(1), count: Toy.allCases.count)))
        XCTAssertEqual(masked.asArray(Float.self)[Toy.keyTail.rawValue], 1)
    }

    func testAllMaskedFallbackPreservesShapeAndDtype() throws {
        let table = StaticTokenFragmentTable(
            strings: ["{\"", "status\":", "\"ok\"", nil], endOfSequenceTokenIds: [3])
        var processor = try JSONGrammarLogitProcessor(
            format: .jsonSchema(try canarySchema()), table: table)
        processor.didSample(token: sample(0))
        processor.didSample(token: sample(1))
        processor.didSample(token: sample(2))

        let input = MLXArray(Array(repeating: Float(0.5), count: 4)).reshaped(1, 4).asType(.float16)
        let masked = processor.process(logits: input)
        XCTAssertEqual(masked.shape, [1, 4])
        XCTAssertEqual(masked.dtype, .float16)
        XCTAssertTrue(masked.asArray(Float16.self).allSatisfy { $0 == -Float16.infinity })
    }

    // MARK: - Recursive schema (nested objects, arrays, nullable unions)

    func testRecursiveSchemaCompletesThroughTheLogitProcessor() throws {
        let table = RecursiveSchemaFixture.table()
        let handle = JSONGrammarRunRecord()
        var processor = try JSONGrammarLogitProcessor(
            format: .jsonSchema(try RecursiveSchemaFixture.schema()), table: table,
            runRecord: handle)
        for tokenId in RecursiveSchemaFixture.documentScript {
            let masked = processor.process(
                logits: logits(Array(repeating: Float(1), count: table.vocabularySize)))
            XCTAssertNotEqual(
                masked.asArray(Float.self)[tokenId], -Float.infinity,
                "scripted token \(tokenId) must survive the constraint mask")
            processor.didSample(token: sample(tokenId))
        }
        XCTAssertEqual(processor.status, .finished)
        XCTAssertTrue(processor.isAccepting)
        XCTAssertFalse(handle.isFailed)
        XCTAssertTrue(handle.rootValueCompleted, "the shared run record must observe completion")
    }

    func testIllegalNestedTokenIsCapturedAndProcessStaysAllMasked() throws {
        let table = RecursiveSchemaFixture.table()
        let handle = JSONGrammarRunRecord()
        var processor = try JSONGrammarLogitProcessor(
            format: .jsonSchema(try RecursiveSchemaFixture.schema()), table: table,
            runRecord: handle)
        // `"tags":` is a root key; it cannot open inside `meta` after `{"meta":{`.
        processor.didSample(token: sample(RecursiveSchemaFixture.Token.openMetaNested.rawValue))
        processor.didSample(token: sample(RecursiveSchemaFixture.Token.tagsKey.rawValue))
        XCTAssertEqual(handle.failure, .illegalToken(RecursiveSchemaFixture.Token.tagsKey.rawValue))

        let masked = processor.process(logits: logits(Array(repeating: Float(1), count: table.vocabularySize)))
        XCTAssertTrue(masked.asArray(Float.self).allSatisfy { $0 == -Float.infinity })
        XCTAssertEqual(
            handle.failure, .illegalToken(RecursiveSchemaFixture.Token.tagsKey.rawValue),
            "the FIRST failure is deterministic")
    }

    func testRecursiveSchemaMasksNestedBoundaries() throws {
        typealias Token = RecursiveSchemaFixture.Token
        let table = RecursiveSchemaFixture.table()
        var processor = try JSONGrammarLogitProcessor(
            format: .jsonSchema(try RecursiveSchemaFixture.schema()), table: table)

        func masked(_ token: Token) -> Float {
            processor.process(logits: logits(Array(repeating: Float(1), count: table.vocabularySize)))
                .asArray(Float.self)[token.rawValue]
        }

        // Inside the nested object only its own key can open.
        processor.didSample(token: sample(Token.openMetaNested.rawValue))
        XCTAssertNotEqual(masked(.idKey), -Float.infinity)
        XCTAssertEqual(masked(.tagsKey), -Float.infinity)
        XCTAssertEqual(masked(.closeNested), -Float.infinity, "the nested required key is missing")

        // Nullable union: null and strings are allowed, other types are not.
        processor.didSample(token: sample(Token.idKey.rawValue))
        processor.didSample(token: sample(Token.seven.rawValue))
        processor.didSample(token: sample(Token.closeNested.rawValue))
        processor.didSample(token: sample(Token.tagsArrayOpen.rawValue))
        processor.didSample(token: sample(Token.closeBracket.rawValue))
        processor.didSample(token: sample(Token.noteKey.rawValue))
        XCTAssertNotEqual(masked(.nullLiteral), -Float.infinity)
        XCTAssertNotEqual(masked(.stringX), -Float.infinity)
        XCTAssertEqual(masked(.trueLiteral), -Float.infinity)
        XCTAssertEqual(masked(.eos), -Float.infinity, "premature EOS must be masked")
    }

    // MARK: - End to end with the tokenizer adapter

    func testEndToEndFromTokenizerVocabularyToNormalCompletion() throws {
        let tokenizer = TokenizerFragmentTableTests.FakeVocabularyTokenizer(
            rawTokens: [
                0: "{\"", 1: "status\":", 2: " \"", 3: "ok\"", 4: "}", 5: "<|eos|>",
            ],
            specialIds: [5],
            eosToken: "<|eos|>")
        let table = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 6)
        let handle = JSONGrammarRunRecord()
        var processor = try JSONGrammarLogitProcessor(
            format: .jsonSchema(try canarySchema()), table: table, runRecord: handle)

        for id in [0, 1, 2, 3, 4] {
            let masked = processor.process(logits: logits(Array(repeating: Float(1), count: 6)))
            XCTAssertNotEqual(masked.asArray(Float.self)[id], -Float.infinity, "token \(id)")
            processor.didSample(token: sample(id))
        }
        XCTAssertEqual(processor.status, .complete)
        let masked = processor.process(logits: logits(Array(repeating: Float(1), count: 6)))
        XCTAssertNotEqual(masked.asArray(Float.self)[5], -Float.infinity)
        processor.didSample(token: sample(5))
        XCTAssertEqual(processor.status, .finished)
        XCTAssertFalse(handle.isFailed)
    }
}
