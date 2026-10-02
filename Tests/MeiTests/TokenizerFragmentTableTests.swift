import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MeiCore

/// Model-free tests for the production tokenizer adapter — the bridge from
/// `MLXLMCommon.Tokenizer` (plus the model's explicit vocabulary size) to the
/// immutable `TokenFragmentTable` the constrained-decoding core consumes
/// (slice 4 of the structured-generation plan).
///
/// The double below stands in for the production tokenizer bridge: it models
/// exactly the parts of the public `Tokenizer` API the adapter may use —
/// `convertIdToToken`, `convertTokenToId`, and `decode(tokenIds:skipSpecialTokens:)`
/// with a pluggable decoder chain — so the adapter's fragment semantics can be
/// pinned without a model.
final class TokenizerFragmentTableTests: XCTestCase {

    // MARK: - Fake tokenizer

    /// A model-free stand-in for the swift-transformers bridge. Raw vocabulary
    /// strings are joined through a pluggable `decoder` closure that mimics the
    /// real decoder chain (byte-level identity, SentencePiece leading-space
    /// stripping, byte fallback). Special ids are dropped by
    /// `skipSpecialTokens: true` exactly as the real bridge drops them.
    final class FakeVocabularyTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
        typealias Decoder = @Sendable ([String]) -> String

        let rawTokens: [Int: String]
        let specialIds: Set<Int>
        let bosToken: String?
        let eosToken: String?
        let unknownToken: String?
        /// Forward-only resolutions: model an inconsistent tokenizer whose
        /// `convertTokenToId` resolves an id with no reverse vocabulary entry.
        let tokenToIdOverrides: [String: Int]
        private let decoder: Decoder

        init(
            rawTokens: [Int: String],
            specialIds: Set<Int> = [],
            bosToken: String? = nil,
            eosToken: String? = nil,
            unknownToken: String? = nil,
            tokenToIdOverrides: [String: Int] = [:],
            decoder: @escaping Decoder = { $0.joined() }
        ) {
            self.rawTokens = rawTokens
            self.specialIds = specialIds
            self.bosToken = bosToken
            self.eosToken = eosToken
            self.unknownToken = unknownToken
            self.tokenToIdOverrides = tokenToIdOverrides
            self.decoder = decoder
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            let ids = skipSpecialTokens
                ? tokenIds.filter { !specialIds.contains($0) }
                : tokenIds
            return decoder(ids.compactMap { rawTokens[$0] })
        }

        func convertTokenToId(_ token: String) -> Int? {
            if let override = tokenToIdOverrides[token] { return override }
            return rawTokens.first { $0.value == token }?.key
        }

        func convertIdToToken(_ id: Int) -> String? { rawTokens[id] }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// Byte-level-BPE-like: raw vocabulary strings are already text.
    static let identityDecoder: FakeVocabularyTokenizer.Decoder = { $0.joined() }

    /// SentencePiece/Llama-like: the metaspace marker renders as a space and
    /// the FIRST token of a decode call loses one leading space (Metaspace
    /// `add_prefix_space` / `Strip`-after-`Fuse`).
    static let stripLeadingSpaceDecoder: FakeVocabularyTokenizer.Decoder = { tokens in
        let joined = tokens.map { $0.replacingOccurrences(of: "▁", with: " ") }.joined()
        return joined.hasPrefix(" ") ? String(joined.dropFirst()) : joined
    }

    /// Byte-fallback-like: `<0xXX>` tokens are raw bytes and an incomplete
    /// sequence renders as U+FFFD, exactly as `String(decoding:as:)` does in
    /// the real byte-fallback decoder.
    static let byteFallbackDecoder: FakeVocabularyTokenizer.Decoder = { tokens in
        var output = ""
        var pending: [UInt8] = []
        func flush() {
            output += String(decoding: pending, as: UTF8.self)
            pending = []
        }
        for token in tokens {
            if token.hasPrefix("<0x"), token.hasSuffix(">"), token.count == 6,
                let byte = UInt8(token.dropFirst(3).dropLast(), radix: 16)
            {
                pending.append(byte)
            } else {
                flush()
                output += token
            }
        }
        flush()
        return output
    }

    // MARK: - Helpers

    private func canarySchema() throws -> CompiledJSONSchema {
        let schema = try JSONDecoder().decode(
            MeiJSONValue.self,
            from: Data(
                #"{"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}}, "required": ["status"], "additionalProperties": false}"#
                    .utf8))
        return try JSONSchemaCompiler.compile(
            JSONSchemaFormat(name: "canary_status", strict: true, schema: schema))
    }

    // MARK: - Fragments come from the decode entry point

    func testFragmentsAreTheTokensDecodedText() throws {
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [
                0: "{\"",
                1: "\"status\":",
                2: "\"ok\"",
                3: "}",
                4: " ",
                5: "<|eos|>",
            ],
            specialIds: [5],
            eosToken: "<|eos|>")
        let table = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 6)

        XCTAssertEqual(table.vocabularySize, 6)
        XCTAssertEqual(table.fragment(forTokenId: 0), Array("{\"".utf8))
        XCTAssertEqual(table.fragment(forTokenId: 1), Array("\"status\":".utf8))
        XCTAssertEqual(table.fragment(forTokenId: 2), Array("\"ok\"".utf8))
        XCTAssertEqual(table.fragment(forTokenId: 3), Array("}".utf8))
        XCTAssertEqual(table.fragment(forTokenId: 4), Array(" ".utf8))
        XCTAssertNil(table.fragment(forTokenId: 5), "a special token is not text")
        XCTAssertEqual(table.endOfSequenceTokenIds, [5])
        XCTAssertEqual(table.diagnostics.textFragmentCount, 5)
        XCTAssertEqual(table.diagnostics.specialTokenIdCount, 1)
    }

    func testSpecialNonTextIdsAreIdentifiedNotJustEOS() throws {
        // Chat markers are specials too: they must never be consumable as
        // JSON text even though `skipSpecialTokens: false` would decode them
        // to their literal marker string.
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "{", 1: "<|im_start|>", 2: "<|im_end|>", 3: "}"],
            specialIds: [1, 2],
            eosToken: "<|im_end|>")
        let table = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 4)

        XCTAssertEqual(table.fragment(forTokenId: 0), Array("{".utf8))
        XCTAssertNil(table.fragment(forTokenId: 1), "a chat marker is not text")
        XCTAssertNil(table.fragment(forTokenId: 2))
        XCTAssertEqual(table.fragment(forTokenId: 3), Array("}".utf8))
        XCTAssertEqual(table.endOfSequenceTokenIds, [2])
        XCTAssertEqual(table.diagnostics.specialTokenIdCount, 2)
    }

    // MARK: - Tokenizer-boundary correction (the documented assumption)

    func testLeadingSpaceStrippingIsCorrectedWithTheSeedProbe() throws {
        // `▁hello` contributes " hello" mid-stream, but a standalone
        // `decode(["▁hello"])` strips the leading space. The adapter probes
        // with a seed token (BOS) so the fragment matches the in-context
        // contribution as far as the public API permits.
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "▁hello", 1: "▁world", 2: "<s>", 3: "<|eos|>"],
            specialIds: [2, 3],
            bosToken: "<s>",
            eosToken: "<|eos|>",
            decoder: Self.stripLeadingSpaceDecoder)
        let table = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 4)

        XCTAssertEqual(
            tokenizer.decode(tokenIds: [0], skipSpecialTokens: false), "hello",
            "precondition: the standalone decode strips the space")
        XCTAssertEqual(table.fragment(forTokenId: 0), Array(" hello".utf8))
        XCTAssertEqual(table.fragment(forTokenId: 1), Array(" world".utf8))
        XCTAssertEqual(table.diagnostics.seedTokenId, 2)
        XCTAssertEqual(table.diagnostics.seedCorrectionCount, 2)
        XCTAssertNil(table.fragment(forTokenId: 3))
    }

    func testStandaloneDecodeIsUsedWhenNoSeedTokenExists() throws {
        // No BOS/EOS/unknown string resolves: the adapter falls back to the
        // standalone decode and documents the leading-space assumption.
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "▁hello", 1: "<|eos|>"],
            specialIds: [1],
            decoder: Self.stripLeadingSpaceDecoder)
        let table = try TokenizerFragmentTable(
            tokenizer: tokenizer, vocabularySize: 2,
            additionalEndOfSequenceTokenIds: [1])

        XCTAssertNil(table.diagnostics.seedTokenId)
        XCTAssertEqual(table.diagnostics.seedCorrectionCount, 0)
        XCTAssertEqual(table.fragment(forTokenId: 0), Array("hello".utf8))
        XCTAssertEqual(table.endOfSequenceTokenIds, [1])
    }

    func testByteFallbackPartialSequencesFailClosed() throws {
        // `<0xE4>` alone decodes to U+FFFD: it cannot be represented as a
        // context-free fragment, so it is masked (never allowed) instead of
        // being guessed.
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "<0xE4>", 1: "é", 2: "<s>", 3: "<|eos|>"],
            specialIds: [2, 3],
            bosToken: "<s>",
            eosToken: "<|eos|>",
            decoder: Self.byteFallbackDecoder)
        let table = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 4)

        XCTAssertNil(table.fragment(forTokenId: 0))
        XCTAssertEqual(table.fragment(forTokenId: 1), Array("é".utf8))
        XCTAssertEqual(table.diagnostics.replacementCharacterIdCount, 1)
        XCTAssertEqual(table.diagnostics.textFragmentCount, 1)
    }

    // MARK: - Fail-closed vocabulary handling

    func testMissingVocabularyEntriesFailClosed() throws {
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "{", 1: "}", 3: "<|eos|>"],
            specialIds: [3],
            eosToken: "<|eos|>")
        // The declared model vocabulary is larger than the tokenizer's entries.
        let table = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 5)

        XCTAssertNil(table.fragment(forTokenId: 2), "a missing entry must never be allowed")
        XCTAssertNil(table.fragment(forTokenId: 4))
        XCTAssertNil(table.fragment(forTokenId: 5), "out-of-range ids are not entries")
        XCTAssertEqual(table.diagnostics.missingEntryCount, 2)
        XCTAssertEqual(table.diagnostics.textFragmentCount, 2)
    }

    func testEmptyDecodeTokensAreNotText() throws {
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "{", 1: "▁", 2: "<|eos|>"],
            specialIds: [2],
            eosToken: "<|eos|>",
            decoder: Self.stripLeadingSpaceDecoder)
        let table = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 3)

        // "▁" standalone decodes to "" (the space is stripped); whatever the
        // cause, an empty decode is not a usable fragment.
        XCTAssertNil(table.fragment(forTokenId: 1))
        XCTAssertEqual(table.diagnostics.emptyDecodeCount, 1)
    }

    func testThrowsWhenNoEndOfSequenceTokenIsIdentified() {
        let tokenizer = FakeVocabularyTokenizer(rawTokens: [0: "{", 1: "}"])
        XCTAssertThrowsError(try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 2)) { error in
            XCTAssertEqual(error as? TokenizerFragmentTableError, .missingEndOfSequenceToken)
        }
    }

    func testThrowsWhenEndOfSequenceIdIsOutOfVocabulary() {
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "{", 1: "}", 9: "<|eos|>"],
            specialIds: [9],
            eosToken: "<|eos|>")
        XCTAssertThrowsError(try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 3)) { error in
            XCTAssertEqual(
                error as? TokenizerFragmentTableError,
                .endOfSequenceTokenNotInVocabulary(9, vocabularySize: 3))
        }

        // The declared (model configuration) ids are held to the same rule.
        let declared = FakeVocabularyTokenizer(
            rawTokens: [0: "{", 1: "}", 2: "<|eos|>"],
            specialIds: [2],
            eosToken: "<|eos|>")
        XCTAssertThrowsError(
            try TokenizerFragmentTable(
                tokenizer: declared, vocabularySize: 3,
                additionalEndOfSequenceTokenIds: [7])
        ) { error in
            XCTAssertEqual(
                error as? TokenizerFragmentTableError,
                .endOfSequenceTokenNotInVocabulary(7, vocabularySize: 3))
        }
    }

    func testThrowsWhenEndOfSequenceIdHasNoVocabularyEntry() {
        // `convertTokenToId` resolves 5 and 5 is in range, but the vocabulary
        // has no reverse entry for it: an inconsistent tokenizer fails closed.
        let inconsistent = FakeVocabularyTokenizer(
            rawTokens: [0: "{", 1: "}"],
            eosToken: "<|eos|>",
            tokenToIdOverrides: ["<|eos|>": 5])
        XCTAssertThrowsError(
            try TokenizerFragmentTable(tokenizer: inconsistent, vocabularySize: 10)
        ) { error in
            XCTAssertEqual(
                error as? TokenizerFragmentTableError,
                .endOfSequenceTokenNotInVocabulary(5, vocabularySize: 10))
        }
    }

    func testThrowsWhenNoUsableTextFragmentExists() {
        // A totally mismatched vocabulary: every in-range id is special or
        // missing. The table cannot constrain anything, so construction fails
        // instead of producing an all-masked mask at generation time.
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "<|eos|>"],
            specialIds: [0],
            eosToken: "<|eos|>")
        XCTAssertThrowsError(try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 4)) { error in
            XCTAssertEqual(
                error as? TokenizerFragmentTableError,
                .noUsableTextFragments(vocabularySize: 4))
        }
    }

    func testThrowsForInvalidVocabularySize() {
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "{", 1: "}", 2: "<|eos|>"],
            specialIds: [2],
            eosToken: "<|eos|>")
        for size in [0, -1] {
            XCTAssertThrowsError(try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: size)) { error in
                XCTAssertEqual(error as? TokenizerFragmentTableError, .invalidVocabularySize(size))
            }
        }
    }

    func testDeclaredEOSIdsAreUnionedWithTheTokenizerEOS() throws {
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "{", 1: "}", 2: "<|eos|>", 3: "<|endoftext|>"],
            specialIds: [2, 3],
            eosToken: "<|eos|>")
        let table = try TokenizerFragmentTable(
            tokenizer: tokenizer, vocabularySize: 4,
            additionalEndOfSequenceTokenIds: [3])
        XCTAssertEqual(table.endOfSequenceTokenIds, [2, 3])
    }

    // MARK: - Determinism and the production decode path

    func testConstructionIsDeterministicAndImmutable() throws {
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [0: "▁hello", 1: "▁world", 2: "<s>", 3: "<|eos|>"],
            specialIds: [2, 3],
            bosToken: "<s>",
            eosToken: "<|eos|>",
            decoder: Self.stripLeadingSpaceDecoder)
        let first = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 4)
        let second = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 4)
        XCTAssertEqual(first.diagnostics, second.diagnostics)
        for id in 0..<4 {
            XCTAssertEqual(first.fragment(forTokenId: id), second.fragment(forTokenId: id), "id \(id)")
        }
    }

    func testTableDrivesTheGrammarProcessorThroughAWholeDocument() throws {
        // The adapter's fragments must actually spell the document the schema
        // requires: `{"status": "ok"}` built from multi-character tokens.
        let tokenizer = FakeVocabularyTokenizer(
            rawTokens: [
                0: "{\"", 1: "status\":", 2: " \"", 3: "ok\"", 4: "}", 5: "<|eos|>",
            ],
            specialIds: [5],
            eosToken: "<|eos|>")
        let table = try TokenizerFragmentTable(tokenizer: tokenizer, vocabularySize: 6)
        var processor = try JSONGrammarProcessor(
            format: .jsonSchema(try canarySchema()), table: table)
        for id in [0, 1, 2, 3, 4] {
            XCTAssertTrue(processor.isAllowed(tokenId: id), "token \(id) must be allowed")
            try processor.consume(tokenId: id)
        }
        XCTAssertEqual(processor.status, .complete)
        XCTAssertTrue(processor.isAllowed(tokenId: 5))
        try processor.consume(tokenId: 5)
        XCTAssertEqual(processor.status, .finished)
    }
}
