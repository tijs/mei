import MLX
import XCTest
@testable import MeiCore

/// Model-free tests for the token-level constrained-decoding processor — the
/// second half of the constrained-decoding core (slice 3 of the
/// structured-generation plan).
///
/// The processor consumes a compiled response format plus a caller-supplied
/// tokenizer vocabulary/token-fragment table (the toy table below stands in
/// for a real tokenizer), tracks grammar state across prompt reset and sampled
/// token fragments, and exposes allowed-token decisions plus a logits mask
/// suitable for a vmlx `LogitProcessor`. Production tokenizer/model wiring is
/// deliberately out of scope here; these tests pin the model-free contract,
/// including fail-closed behavior when no token can advance the grammar.
final class JSONGrammarProcessorTests: XCTestCase {

    // MARK: - Toy vocabulary

    /// A toy tokenizer: a mix of single-character, multi-character, and
    /// multi-byte fragments so token boundaries inside strings, numbers, keys,
    /// and across `{`/`"`/`:` are all exercised.
    private enum ToyToken: Int, CaseIterable {
        case openBrace
        case closeBrace
        case openBracket
        case closeBracket
        case colon
        case comma
        case quote
        case keyStatus
        case keyX
        case valueOK
        case wrongEnumValue
        case space
        case newline
        case trueLiteral
        case falseLiteral
        case nullLiteral
        case zero
        case one
        case five
        case minus
        case dot
        case e
        case x
        case hello
        case openBraceKeyPrefix
        case keyTailColon
        case keyStatusColon
        case valueOKCloseBrace
        case closeBraceNewline
        case enumPrefixO
        case enumEscapedOK
        case enumEscapedUpper
        case surrogatePairString
        case loneHighSurrogateString
        case lowSurrogateClose
        case wrongSurrogateClose
        case loneLowSurrogateString
        case invalidUTF8String
        case eAcute
        case emojiHead
        case emojiTail
        case commaSpace
        case wholeCanary
        case wholeCanaryPlusProse
        case eos

        var fragment: [UInt8]? {
            func bytes(_ string: String) -> [UInt8] { Array(string.utf8) }
            switch self {
            case .openBrace: return bytes("{")
            case .closeBrace: return bytes("}")
            case .openBracket: return bytes("[")
            case .closeBracket: return bytes("]")
            case .colon: return bytes(":")
            case .comma: return bytes(",")
            case .quote: return bytes("\"")
            case .keyStatus: return bytes("\"status\"")
            case .keyX: return bytes("\"x\"")
            case .valueOK: return bytes("\"ok\"")
            case .wrongEnumValue: return bytes("\"nope\"")
            case .space: return bytes(" ")
            case .newline: return bytes("\n")
            case .trueLiteral: return bytes("true")
            case .falseLiteral: return bytes("false")
            case .nullLiteral: return bytes("null")
            case .zero: return bytes("0")
            case .one: return bytes("1")
            case .five: return bytes("5")
            case .minus: return bytes("-")
            case .dot: return bytes(".")
            case .e: return bytes("e")
            case .x: return bytes("x")
            case .hello: return bytes("Hello")
            case .openBraceKeyPrefix: return bytes("{\"")
            case .keyTailColon: return bytes("status\":")
            case .keyStatusColon: return bytes("\"status\":")
            case .valueOKCloseBrace: return bytes("\"ok\"}")
            case .closeBraceNewline: return bytes("}\n")
            case .enumPrefixO: return bytes("\"o")
            case .enumEscapedOK: return bytes("\"\\u006f\\u006b\"")
            case .enumEscapedUpper: return bytes("\"\\u004F\\u004B\"")
            case .surrogatePairString: return bytes("\"\\uD83D\\uDE00\"")
            case .loneHighSurrogateString: return bytes("\"\\uD83D")
            case .lowSurrogateClose: return bytes("\\uDE00\"")
            case .wrongSurrogateClose: return bytes("\\u0041\"")
            case .loneLowSurrogateString: return bytes("\"\\uDE00\"")
            case .invalidUTF8String: return bytes("\"") + [0xFF]
            case .eAcute: return bytes("é")
            case .emojiHead: return bytes("\"") + [0xF0, 0x9F]
            case .emojiTail: return [0x98, 0x80] + bytes("\"")
            case .commaSpace: return bytes(", ")
            case .wholeCanary: return bytes("{\"status\": \"ok\"}")
            case .wholeCanaryPlusProse: return bytes("{\"status\": \"ok\"}!")
            case .eos: return nil
            }
        }
    }

    private func toyTable() -> StaticTokenFragmentTable {
        StaticTokenFragmentTable(
            fragments: ToyToken.allCases.map { $0.fragment },
            endOfSequenceTokenIds: [ToyToken.eos.rawValue])
    }

    // MARK: - Helpers

    private func mei(_ json: String) throws -> MeiJSONValue {
        try JSONDecoder().decode(MeiJSONValue.self, from: Data(json.utf8))
    }

    private func canarySchema() throws -> CompiledJSONSchema {
        try JSONSchemaCompiler.compile(
            JSONSchemaFormat(
                name: "canary_status", strict: true,
                schema: try mei(
                    #"{"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}}, "required": ["status"], "additionalProperties": false}"#
                )))
    }

    private func canaryProcessor() throws -> JSONGrammarProcessor {
        try JSONGrammarProcessor(format: .jsonSchema(try canarySchema()), table: toyTable())
    }

    private func jsonObjectProcessor() throws -> JSONGrammarProcessor {
        try JSONGrammarProcessor(format: .jsonObject, table: toyTable())
    }

    // MARK: - Allowed-token decisions

    func testJSONObjectInitialAllowedTokens() throws {
        let processor = try jsonObjectProcessor()
        let allowed = Set(processor.allowedTokenIds())
        let expected: Set<ToyToken> = [
            .openBrace, .openBracket, .quote,
            .keyStatus, .keyX, .valueOK, .wrongEnumValue,
            .space, .newline,
            .trueLiteral, .falseLiteral, .nullLiteral,
            .zero, .one, .five, .minus,
            .openBraceKeyPrefix, .wholeCanary,
            .enumPrefixO, .enumEscapedOK, .enumEscapedUpper,
            .surrogatePairString, .loneHighSurrogateString, .emojiHead,
        ]
        XCTAssertEqual(allowed, Set(expected.map(\.rawValue)))
    }

    func testSchemaInitialAllowedTokens() throws {
        let processor = try canaryProcessor()
        let allowed = Set(processor.allowedTokenIds())
        XCTAssertEqual(
            allowed,
            Set(
                [ToyToken.openBrace, .openBraceKeyPrefix, .space, .newline, .wholeCanary]
                    .map(\.rawValue)))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.eos.rawValue))
    }

    // MARK: - Canary flow

    func testCanarySchemaTokenFlow() throws {
        var processor = try canaryProcessor()
        try processor.consume(tokenId: ToyToken.openBraceKeyPrefix.rawValue)  // `{"`
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.keyTailColon.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.keyStatusColon.rawValue), "the key is already open")
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.keyX.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.hello.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.eos.rawValue))

        try processor.consume(tokenId: ToyToken.keyTailColon.rawValue)  // `status":`
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.valueOK.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.enumPrefixO.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.enumEscapedOK.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.space.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.wrongEnumValue.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.enumEscapedUpper.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.surrogatePairString.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.emojiHead.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.eos.rawValue))

        try processor.consume(tokenId: ToyToken.valueOK.rawValue)
        XCTAssertEqual(processor.status, .inProgress)
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.eos.rawValue), "premature EOS")
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.closeBrace.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.closeBraceNewline.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.space.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.comma.rawValue), "all properties are used")
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.openBrace.rawValue))

        try processor.consume(tokenId: ToyToken.closeBraceNewline.rawValue)
        XCTAssertEqual(processor.status, .complete)
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.eos.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.space.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.closeBrace.rawValue))

        try processor.consume(tokenId: ToyToken.eos.rawValue)
        XCTAssertEqual(processor.status, .finished)
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.eos.rawValue))
    }

    func testWholeDocumentTokenCompletesTheGrammar() throws {
        var processor = try canaryProcessor()
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.wholeCanary.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.wholeCanaryPlusProse.rawValue))
        try processor.consume(tokenId: ToyToken.wholeCanary.rawValue)
        XCTAssertEqual(processor.status, .complete)
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.eos.rawValue))
    }

    func testCanaryFlowProducesExactlyTheSchemaObject() throws {
        let table = toyTable()
        var processor = try canaryProcessor()
        var output: [UInt8] = []
        let steps: [ToyToken] = [.openBraceKeyPrefix, .keyTailColon, .space, .valueOK, .closeBraceNewline]
        for token in steps {
            XCTAssertTrue(processor.isAllowed(tokenId: token.rawValue), "\(token) must be allowed")
            try processor.consume(tokenId: token.rawValue)
            output += table.fragment(forTokenId: token.rawValue) ?? []
        }
        XCTAssertEqual(processor.status, .complete)
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.eos.rawValue))
        let decoded = try JSONDecoder().decode(MeiJSONValue.self, from: Data(output))
        XCTAssertEqual(decoded, .object(["status": .string("ok")]))
    }

    func testJSONObjectFlowProducesParseableNestedValue() throws {
        let table = toyTable()
        var processor = try jsonObjectProcessor()
        var output: [UInt8] = []
        let steps: [ToyToken] = [
            .openBraceKeyPrefix, .keyTailColon, .space,
            .openBracket, .one, .comma, .five, .closeBracket,
            .closeBrace,
        ]
        for token in steps {
            XCTAssertTrue(processor.isAllowed(tokenId: token.rawValue), "\(token) must be allowed")
            try processor.consume(tokenId: token.rawValue)
            output += table.fragment(forTokenId: token.rawValue) ?? []
        }
        XCTAssertEqual(processor.status, .complete)
        let decoded = try JSONDecoder().decode(MeiJSONValue.self, from: Data(output))
        XCTAssertEqual(decoded, .object(["status": .array([.number(1), .number(5)])]))
    }

    // MARK: - Recursive schema flow (nested objects, arrays, nullable unions)

    private func recursiveProcessor() throws -> JSONGrammarProcessor {
        try JSONGrammarProcessor(
            format: .jsonSchema(try RecursiveSchemaFixture.schema()),
            table: RecursiveSchemaFixture.table())
    }

    func testRecursiveSchemaInitialAllowedTokens() throws {
        let processor = try recursiveProcessor()
        let allowed = Set(processor.allowedTokenIds())
        XCTAssertEqual(
            allowed,
            Set(
                [RecursiveSchemaFixture.Token.openMetaNested, .space, .wholeDocument]
                    .map(\.rawValue)))
        XCTAssertFalse(processor.isAllowed(tokenId: RecursiveSchemaFixture.Token.eos.rawValue))
    }

    func testRecursiveSchemaTokenFlow() throws {
        var processor = try recursiveProcessor()
        typealias Token = RecursiveSchemaFixture.Token
        try processor.consume(tokenId: Token.openMetaNested.rawValue)

        // Inside the nested object only its own key can open, and the nested
        // required key must be present before it can close.
        XCTAssertTrue(processor.isAllowed(tokenId: Token.idKey.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.tagsKey.rawValue), "`tags` is not a key of `meta`")
        XCTAssertFalse(processor.isAllowed(tokenId: Token.closeNested.rawValue), "the nested required key is missing")
        XCTAssertFalse(processor.isAllowed(tokenId: Token.seven.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.eos.rawValue))

        try processor.consume(tokenId: Token.idKey.rawValue)
        XCTAssertTrue(processor.isAllowed(tokenId: Token.seven.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.stringX.rawValue), "the nested value is an integer")
        XCTAssertFalse(processor.isAllowed(tokenId: Token.nullLiteral.rawValue))

        try processor.consume(tokenId: Token.seven.rawValue)
        XCTAssertTrue(processor.isAllowed(tokenId: Token.closeNested.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.comma.rawValue), "no second key in `meta`")
        try processor.consume(tokenId: Token.closeNested.rawValue)

        // Back at the root object: root keys are available, nested-only keys are not.
        XCTAssertTrue(processor.isAllowed(tokenId: Token.tagsArrayOpen.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.idKey.rawValue), "`id` is not a root key")
        XCTAssertFalse(processor.isAllowed(tokenId: Token.closeRoot.rawValue), "root required keys are missing")

        try processor.consume(tokenId: Token.tagsArrayOpen.rawValue)
        // Array item position: strings only, and the empty array may close.
        XCTAssertTrue(processor.isAllowed(tokenId: Token.stringA.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: Token.closeBracket.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.seven.rawValue), "an integer is not a string item")
        XCTAssertFalse(processor.isAllowed(tokenId: Token.nullLiteral.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.closeNested.rawValue), "the wrong closer")

        try processor.consume(tokenId: Token.stringA.rawValue)
        XCTAssertTrue(processor.isAllowed(tokenId: Token.comma.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: Token.closeBracket.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.stringB.rawValue), "items need a comma first")
        try processor.consume(tokenId: Token.comma.rawValue)
        XCTAssertFalse(processor.isAllowed(tokenId: Token.closeBracket.rawValue), "a comma requires another item")
        XCTAssertTrue(processor.isAllowed(tokenId: Token.stringB.rawValue))
        try processor.consume(tokenId: Token.stringB.rawValue)
        try processor.consume(tokenId: Token.closeBracket.rawValue)

        // Nullable union position: null and a string are both allowed; other
        // types are not.
        XCTAssertTrue(processor.isAllowed(tokenId: Token.noteKey.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.closeRoot.rawValue))
        try processor.consume(tokenId: Token.noteKey.rawValue)
        XCTAssertTrue(processor.isAllowed(tokenId: Token.nullLiteral.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: Token.stringX.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.seven.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.trueLiteral.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: Token.eos.rawValue))

        try processor.consume(tokenId: Token.nullLiteral.rawValue)
        XCTAssertEqual(processor.status, .inProgress)
        XCTAssertFalse(processor.isAllowed(tokenId: Token.eos.rawValue), "the root object is still open")
        XCTAssertTrue(processor.isAllowed(tokenId: Token.closeRoot.rawValue))
        try processor.consume(tokenId: Token.closeRoot.rawValue)
        XCTAssertEqual(processor.status, .complete)
        XCTAssertTrue(processor.isAllowed(tokenId: Token.eos.rawValue))
    }

    func testWholeRecursiveDocumentTokenCompletesTheGrammar() throws {
        var processor = try recursiveProcessor()
        XCTAssertTrue(processor.isAllowed(tokenId: RecursiveSchemaFixture.Token.wholeDocument.rawValue))
        try processor.consume(tokenId: RecursiveSchemaFixture.Token.wholeDocument.rawValue)
        XCTAssertEqual(processor.status, .complete)
        XCTAssertTrue(processor.isAllowed(tokenId: RecursiveSchemaFixture.Token.eos.rawValue))
    }

    func testRecursiveSchemaFlowProducesExactlyTheSchemaObject() throws {
        let table = RecursiveSchemaFixture.table()
        var processor = try recursiveProcessor()
        var output: [UInt8] = []
        for tokenId in RecursiveSchemaFixture.documentScript {
            XCTAssertTrue(processor.isAllowed(tokenId: tokenId), "token \(tokenId) must be allowed")
            try processor.consume(tokenId: tokenId)
            output += table.fragment(forTokenId: tokenId) ?? []
        }
        XCTAssertEqual(processor.status, .finished)
        let decoded = try JSONDecoder().decode(MeiJSONValue.self, from: Data(output))
        XCTAssertEqual(
            decoded,
            .object([
                "meta": .object(["id": .number(7)]),
                "tags": .array([.string("a"), .string("b")]),
                "note": .null,
            ]))
    }

    // MARK: - Fail-closed behavior

    func testPrematureEndOfSequenceIsRejected() throws {
        var processor = try canaryProcessor()
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.eos.rawValue))
        XCTAssertThrowsError(try processor.consume(tokenId: ToyToken.eos.rawValue)) { error in
            XCTAssertEqual(error as? JSONGrammarError, .prematureEndOfSequence)
        }
        XCTAssertEqual(processor.status, .inProgress, "a rejected EOS must not poison the grammar state")
        try processor.consume(tokenId: ToyToken.openBraceKeyPrefix.rawValue)
        XCTAssertEqual(processor.status, .inProgress)
    }

    func testProseTokensAreRejectedAndFailClosed() throws {
        var processor = try jsonObjectProcessor()
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.hello.rawValue))
        XCTAssertThrowsError(try processor.consume(tokenId: ToyToken.hello.rawValue)) { error in
            XCTAssertEqual(error as? JSONGrammarError, .illegalToken(ToyToken.hello.rawValue))
        }
        XCTAssertEqual(processor.status, .failed)
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.openBrace.rawValue))
        XCTAssertThrowsError(try processor.maskedLogits(MLXArray(Array(repeating: Float(0), count: toyTable().vocabularySize)))) {
            error in
            XCTAssertEqual(error as? JSONGrammarError, .grammarFailed)
        }
        processor.reset()
        XCTAssertEqual(processor.status, .inProgress)
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.openBrace.rawValue))
    }

    func testNoLegalContinuationFailsClosed() throws {
        // A vocabulary that can spell `{"status":"ok"` but has no way to close
        // the object: after the value, no token can advance the grammar.
        let table = StaticTokenFragmentTable(
            strings: ["{\"", "status\":", "\"ok\"", nil],
            endOfSequenceTokenIds: [3])
        var processor = try JSONGrammarProcessor(format: .jsonSchema(try canarySchema()), table: table)
        try processor.consume(tokenId: 0)
        try processor.consume(tokenId: 1)
        try processor.consume(tokenId: 2)
        XCTAssertTrue(processor.allowedTokenIds().isEmpty)
        XCTAssertFalse(processor.isAllowed(tokenId: 3), "EOS must not be allowed before the root is complete")
        XCTAssertThrowsError(try processor.maskedLogits(MLXArray(Array(repeating: Float(0), count: 4)))) { error in
            XCTAssertEqual(error as? JSONGrammarError, .noLegalContinuation)
        }
    }

    func testWhitespaceOnlyTokensAreAllowedAfterCompleteRoot() throws {
        var processor = try canaryProcessor()
        try processor.consume(tokenId: ToyToken.wholeCanary.rawValue)
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.space.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.newline.rawValue))
        try processor.consume(tokenId: ToyToken.space.rawValue)
        try processor.consume(tokenId: ToyToken.newline.rawValue)
        XCTAssertEqual(processor.status, .complete)
        XCTAssertThrowsError(try processor.consume(tokenId: ToyToken.closeBrace.rawValue)) { error in
            XCTAssertEqual(error as? JSONGrammarError, .illegalToken(ToyToken.closeBrace.rawValue))
        }
        XCTAssertEqual(processor.status, .failed, "trailing prose fails closed")
    }

    func testUnknownTokensAreRejected() throws {
        var processor = try jsonObjectProcessor()
        let outOfRange = toyTable().vocabularySize + 5
        XCTAssertFalse(processor.isAllowed(tokenId: outOfRange))
        XCTAssertThrowsError(try processor.consume(tokenId: outOfRange)) { error in
            XCTAssertEqual(error as? JSONGrammarError, .unknownToken(outOfRange))
        }
        XCTAssertEqual(processor.status, .inProgress)
    }

    func testTextFormatCannotBeConstrained() {
        XCTAssertThrowsError(try JSONGrammarProcessor(format: .text, table: toyTable())) { error in
            XCTAssertEqual(error as? JSONGrammarError, .unconstrainedFormat)
        }
    }

    // MARK: - Logits mask

    func testMaskedLogitsSetsDisallowedTokensToNegativeInfinity() throws {
        let table = toyTable()
        let processor = try jsonObjectProcessor()
        let logits = MLXArray(Array(repeating: Float(1), count: table.vocabularySize))
        let masked = try processor.maskedLogits(logits)
        let values = masked.asArray(Float.self)
        XCTAssertEqual(values.count, table.vocabularySize)
        XCTAssertEqual(values[ToyToken.openBrace.rawValue], 1)
        XCTAssertEqual(values[ToyToken.wholeCanary.rawValue], 1)
        XCTAssertEqual(values[ToyToken.hello.rawValue], -Float.infinity)
        XCTAssertEqual(values[ToyToken.eos.rawValue], -Float.infinity)
        XCTAssertEqual(logits.asArray(Float.self)[ToyToken.hello.rawValue], 1, "the input logits must not be modified")

        let batched = try processor.maskedLogits(logits.reshaped(1, -1))
        XCTAssertEqual(batched.shape, [1, table.vocabularySize])
        XCTAssertEqual(batched.asArray(Float.self)[ToyToken.eos.rawValue], -Float.infinity)
    }

    func testMaskedLogitsRejectsVocabularyMismatch() throws {
        let processor = try jsonObjectProcessor()
        XCTAssertThrowsError(try processor.maskedLogits(MLXArray(Array(repeating: Float(0), count: 3)))) { error in
            XCTAssertEqual(
                error as? JSONGrammarError,
                .vocabularyMismatch(expected: toyTable().vocabularySize, actual: 3))
        }
    }

    func testMaskedLogitsAgreesWithAllowedTokenIds() throws {
        var processor = try canaryProcessor()
        try processor.consume(tokenId: ToyToken.openBraceKeyPrefix.rawValue)
        try processor.consume(tokenId: ToyToken.keyTailColon.rawValue)
        let allowed = Set(processor.allowedTokenIds())
        let masked = try processor.maskedLogits(MLXArray(Array(repeating: Float(1), count: toyTable().vocabularySize)))
        let values = masked.asArray(Float.self)
        let maskedAllowed = Set(values.indices.filter { values[$0] != -Float.infinity })
        XCTAssertEqual(maskedAllowed, allowed)
        XCTAssertFalse(maskedAllowed.isEmpty)
    }

    func testMaskedLogitsPreservesLogitsDtype() throws {
        let table = toyTable()
        let processor = try jsonObjectProcessor()
        let logits = MLXArray(Array(repeating: Float(1), count: table.vocabularySize)).asType(.float16)
        let masked = try processor.maskedLogits(logits)
        XCTAssertEqual(masked.dtype, .float16)
        let values = masked.asArray(Float16.self)
        XCTAssertEqual(values[ToyToken.openBrace.rawValue], 1)
        XCTAssertEqual(values[ToyToken.hello.rawValue], -Float16.infinity)
    }

    // MARK: - Multi-byte fragments

    func testMultiByteFragmentsAcrossTokenBoundaries() throws {
        var processor = try jsonObjectProcessor()
        // `"𝄞"` split as `"` + partial UTF-8, then the rest + closing quote.
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.emojiHead.rawValue))
        try processor.consume(tokenId: ToyToken.emojiHead.rawValue)
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.eos.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.emojiTail.rawValue))
        try processor.consume(tokenId: ToyToken.emojiTail.rawValue)
        XCTAssertEqual(processor.status, .complete)

        // Invalid UTF-8 inside a string is never an allowed fragment.
        let fresh = try jsonObjectProcessor()
        XCTAssertFalse(fresh.isAllowed(tokenId: ToyToken.invalidUTF8String.rawValue))

        // A lone high surrogate escape is a legal prefix in plain strings.
        var surrogate = try jsonObjectProcessor()
        try surrogate.consume(tokenId: ToyToken.loneHighSurrogateString.rawValue)
        XCTAssertTrue(surrogate.isAllowed(tokenId: ToyToken.lowSurrogateClose.rawValue))
        XCTAssertFalse(surrogate.isAllowed(tokenId: ToyToken.wrongSurrogateClose.rawValue))
        XCTAssertFalse(surrogate.isAllowed(tokenId: ToyToken.eos.rawValue))
        try surrogate.consume(tokenId: ToyToken.lowSurrogateClose.rawValue)
        XCTAssertEqual(surrogate.status, .complete)

        // A lone low surrogate escape is never valid.
        XCTAssertFalse(try jsonObjectProcessor().isAllowed(tokenId: ToyToken.loneLowSurrogateString.rawValue))
    }

    func testSchemaKeyAndEnumMatchingUsesDecodedScalars() throws {
        var processor = try canaryProcessor()
        try processor.consume(tokenId: ToyToken.openBraceKeyPrefix.rawValue)
        // Escaped key spelling of "status" is accepted; a lone surrogate is not.
        let escapedKeyTable = StaticTokenFragmentTable(
            strings: ["{\"", "\\u0073tatus\":", "\"ok\"", "}", nil],
            endOfSequenceTokenIds: [4])
        var escapedKey = try JSONGrammarProcessor(format: .jsonSchema(try canarySchema()), table: escapedKeyTable)
        try escapedKey.consume(tokenId: 0)
        XCTAssertTrue(escapedKey.isAllowed(tokenId: 1), "a decoded key of \"status\" must match")
        try escapedKey.consume(tokenId: 1)
        try escapedKey.consume(tokenId: 2)
        try escapedKey.consume(tokenId: 3)
        XCTAssertEqual(escapedKey.status, .complete)

        // Enum values are matched on decoded scalars: `"\u006f\u006b"` is "ok".
        try processor.consume(tokenId: ToyToken.keyTailColon.rawValue)
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.enumEscapedOK.rawValue))
        XCTAssertFalse(processor.isAllowed(tokenId: ToyToken.enumEscapedUpper.rawValue))
        try processor.consume(tokenId: ToyToken.enumEscapedOK.rawValue)
        XCTAssertTrue(processor.isAllowed(tokenId: ToyToken.closeBrace.rawValue))
    }

    // MARK: - Lifecycle

    func testResetRestoresInitialTokenSet() throws {
        var processor = try canaryProcessor()
        let initial = processor.allowedTokenIds()
        try processor.consume(tokenId: ToyToken.openBraceKeyPrefix.rawValue)
        XCTAssertNotEqual(processor.allowedTokenIds(), initial)
        processor.reset()
        XCTAssertEqual(processor.allowedTokenIds(), initial)
        XCTAssertEqual(processor.status, .inProgress)

        // A finished processor is reusable for the next request after reset.
        try processor.consume(tokenId: ToyToken.wholeCanary.rawValue)
        try processor.consume(tokenId: ToyToken.eos.rawValue)
        XCTAssertEqual(processor.status, .finished)
        processor.reset()
        XCTAssertEqual(processor.allowedTokenIds(), initial)
        XCTAssertEqual(processor.status, .inProgress)
    }

    func testProcessorCopiesAreIndependent() throws {
        var original = try canaryProcessor()
        try original.consume(tokenId: ToyToken.openBraceKeyPrefix.rawValue)
        var copy = original.independentCopy()
        try copy.consume(tokenId: ToyToken.keyTailColon.rawValue)
        try copy.consume(tokenId: ToyToken.valueOK.rawValue)
        XCTAssertTrue(copy.isAllowed(tokenId: ToyToken.closeBrace.rawValue))
        XCTAssertTrue(original.isAllowed(tokenId: ToyToken.keyTailColon.rawValue))
        XCTAssertFalse(original.isAllowed(tokenId: ToyToken.valueOK.rawValue))

        var plainCopy = original
        try plainCopy.consume(tokenId: ToyToken.keyTailColon.rawValue)
        XCTAssertTrue(plainCopy.isAllowed(tokenId: ToyToken.valueOK.rawValue))
        XCTAssertFalse(original.isAllowed(tokenId: ToyToken.valueOK.rawValue))
    }

    func testProcessorExposesTheCompiledConstraintKey() throws {
        let objectProcessor = try jsonObjectProcessor()
        XCTAssertEqual(objectProcessor.constraintKey, "json_object")
        let schemaProcessor = try canaryProcessor()
        XCTAssertEqual(schemaProcessor.constraintKey, try canarySchema().constraintKey)
    }
}
