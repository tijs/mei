import MLX
import MLXLMCommon
import XCTest

@testable import MeiCore

/// Model-compat regressions for strict structured output, pinned to the live
/// failures observed on the Qwen3.6 and Ornith checkpoints (Mei 0.7.0,
/// 2026-10-04):
///
/// - **Ornith aligned** streamed `{"s\u00e` and then had NO legal
///   continuation: the grammar admitted the intermediate `\uXXXX` nibble `e`
///   even though every completion of `\u00e…` is rejected by the key matcher
///   for `status`, so the next mask was empty and the request failed closed
///   with `noLegalContinuation`. The same defect class exists for the
///   low-surrogate half of a `\uD83D\u…` pair.
/// - **Qwen3.6 text-only / vision** preferred whitespace tokens over `{` and
///   padded the entire 64-token budget without ever starting the value,
///   failing closed with `incompleteAtStop`. The grammar legally admits JSON
///   whitespace at every structural position, so the token-level mask must
///   bound whitespace runs to force structural progress.
///
/// Both are token-level mask defects — not prompt or model problems. The
/// tests below reproduce the live token paths with model-free tables and pin
/// the fixed contract; none of them may pass merely because a prompt asked
/// for JSON.
final class StructuredOutputModelCompatTests: XCTestCase {

    // MARK: - Canary schema

    private func canaryFormat() throws -> JSONSchemaFormat {
        let schema = try JSONDecoder().decode(
            MeiJSONValue.self,
            from: Data(
                #"{"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}}, "required": ["status"], "additionalProperties": false}"#
                    .utf8))
        return JSONSchemaFormat(name: "canary_status", strict: true, schema: schema)
    }

    private func canarySchema() throws -> CompiledJSONSchema {
        try JSONSchemaCompiler.compile(try canaryFormat())
    }

    // MARK: - Toy vocabulary: the live Ornith escape prefix

    /// Single-character fragments for the exact tokens the live Ornith run
    /// emitted (`{`, `"`, `s`, `\`, `u`, `0`, `0`, `e`, …) plus the tokens
    /// needed to finish the canary. The vocabulary is deliberately tiny so a
    /// dead-end cannot be blamed on tokenization coverage.
    private enum EscapeToken: Int, CaseIterable {
        case openBrace  // `{`
        case quote  // `"`
        case s  // `s`
        case backslash  // `\`
        case u  // `u`
        case zero  // `0`
        case e  // `e`
        case seven  // `7`
        case four  // `4`
        case t  // `t`
        case a  // `a`
        case colon  // `:`
        case o  // `o`
        case k  // `k`
        case closeBrace  // `}`
        case space  // ` `
        case eos

        var fragment: [UInt8]? {
            func bytes(_ string: String) -> [UInt8] { Array(string.utf8) }
            switch self {
            case .openBrace: return bytes("{")
            case .quote: return bytes("\"")
            case .s: return bytes("s")
            case .backslash: return bytes("\\")
            case .u: return bytes("u")
            case .zero: return bytes("0")
            case .e: return bytes("e")
            case .seven: return bytes("7")
            case .four: return bytes("4")
            case .t: return bytes("t")
            case .a: return bytes("a")
            case .colon: return bytes(":")
            case .o: return bytes("o")
            case .k: return bytes("k")
            case .closeBrace: return bytes("}")
            case .space: return bytes(" ")
            case .eos: return nil
            }
        }
    }

    private func escapeTable() -> StaticTokenFragmentTable {
        StaticTokenFragmentTable(
            fragments: EscapeToken.allCases.map { $0.fragment },
            endOfSequenceTokenIds: [EscapeToken.eos.rawValue])
    }

    // MARK: - Escape nibble liveness (the Ornith `noLegalContinuation`)

    func testEscapeNibbleThatCanNeverCompleteTheKeyIsMasked() throws {
        var processor = try JSONGrammarProcessor(
            format: .jsonSchema(try canarySchema()), table: escapeTable())

        // The live Ornith prefix `{"s\u00`: every token of it was admissible.
        for token in [EscapeToken.openBrace, .quote, .s, .backslash, .u, .zero, .zero] {
            XCTAssertTrue(
                processor.isAllowed(tokenId: token.rawValue),
                "the live prefix token \(token) must stay admissible")
            try processor.consume(tokenId: token.rawValue)
        }
        XCTAssertFalse(
            processor.allowedTokenIds().isEmpty,
            "the mask must never be empty at a state reached by admissible tokens")

        // `\u00e…` (0x00E0...0x00EF) can never continue the key matcher for
        // `status` after `s` (the next scalar must be `t`). Admitting `e`
        // dead-ends the grammar; the live run failed closed exactly here.
        XCTAssertFalse(
            processor.isAllowed(tokenId: EscapeToken.e.rawValue),
            "nibble `e` must be masked: no completion of `\\u00e` keeps the key viable")
        XCTAssertTrue(
            processor.isAllowed(tokenId: EscapeToken.seven.rawValue),
            "`\\u007…` is the only feasible third nibble on the way to `t`")

        // A sampler that ignores the mask must fail closed instead of walking
        // into the dead-end state.
        var poisoned = processor.independentCopy()
        XCTAssertThrowsError(try poisoned.consume(tokenId: EscapeToken.e.rawValue)) { error in
            XCTAssertEqual(error as? JSONGrammarError, .illegalToken(EscapeToken.e.rawValue))
        }
        XCTAssertEqual(poisoned.status, .failed)

        try processor.consume(tokenId: EscapeToken.seven.rawValue)
        XCTAssertTrue(
            processor.isAllowed(tokenId: EscapeToken.four.rawValue),
            "`\\u0074` completes `t`")
        XCTAssertFalse(
            processor.isAllowed(tokenId: EscapeToken.e.rawValue),
            "`\\u007e` would be `~`, not `t`")
        try processor.consume(tokenId: EscapeToken.four.rawValue)
        XCTAssertTrue(
            processor.isAllowed(tokenId: EscapeToken.a.rawValue),
            "after the escaped `t` the key matcher continues with `a`")
        XCTAssertFalse(processor.allowedTokenIds().isEmpty)
    }

    // MARK: - Low-surrogate nibble liveness

    /// The low-surrogate half of a pair has the same defect class: after
    /// `\uD83D\uD`, completions 0xD000...0xDFFF are feasible only if they can
    /// still reach the candidate's exact low surrogate.
    private enum EmojiEscapeToken: Int, CaseIterable {
        case openBrace  // `{`
        case keyX  // `"x"`
        case colon  // `:`
        case quote  // `"`
        case backslash  // `\`
        case u  // `u`
        case d  // `D`
        case eight  // `8`
        case three  // `3`
        case e  // `E`
        case c  // `C`
        case zero  // `0`
        case one  // `1`
        case closeQuoteBrace  // `"}`
        case eos

        var fragment: [UInt8]? {
            func bytes(_ string: String) -> [UInt8] { Array(string.utf8) }
            switch self {
            case .openBrace: return bytes("{")
            case .keyX: return bytes("\"x\"")
            case .colon: return bytes(":")
            case .quote: return bytes("\"")
            case .backslash: return bytes("\\")
            case .u: return bytes("u")
            case .d: return bytes("D")
            case .eight: return bytes("8")
            case .three: return bytes("3")
            case .e: return bytes("E")
            case .c: return bytes("C")
            case .zero: return bytes("0")
            case .one: return bytes("1")
            case .closeQuoteBrace: return bytes("\"}")
            case .eos: return nil
            }
        }
    }

    private func emojiSchema() throws -> CompiledJSONSchema {
        let schema = try JSONDecoder().decode(
            MeiJSONValue.self,
            from: Data(
                #"{"type": "object", "properties": {"x": {"type": "string", "enum": ["😀"]}}, "required": ["x"], "additionalProperties": false}"#
                    .utf8))
        return try JSONSchemaCompiler.compile(
            JSONSchemaFormat(name: "emoji_v1", strict: true, schema: schema))
    }

    func testLowSurrogateNibbleThatCanNeverCompleteThePairIsMasked() throws {
        let table = StaticTokenFragmentTable(
            fragments: EmojiEscapeToken.allCases.map { $0.fragment },
            endOfSequenceTokenIds: [EmojiEscapeToken.eos.rawValue])
        var processor = try JSONGrammarProcessor(
            format: .jsonSchema(try emojiSchema()), table: table)

        // `{"x":"\uD83D` — the high surrogate of U+1F600 is admissible.
        for token in [
            EmojiEscapeToken.openBrace, .keyX, .colon, .quote,
            .backslash, .u, .d, .eight, .three, .d,
            .backslash, .u, .d,
        ] {
            XCTAssertTrue(processor.isAllowed(tokenId: token.rawValue), "\(token) must be admissible")
            try processor.consume(tokenId: token.rawValue)
        }

        // U+1F600 pairs 0xD83D with 0xDE00. After `\uD` only completions in
        // 0xDExx can still reach 0xDE00; `C` (0xDCxx) can never.
        XCTAssertTrue(processor.isAllowed(tokenId: EmojiEscapeToken.e.rawValue))
        XCTAssertFalse(
            processor.isAllowed(tokenId: EmojiEscapeToken.c.rawValue),
            "`\\uDC…` can never complete the pair for U+1F600 and must be masked")

        try processor.consume(tokenId: EmojiEscapeToken.e.rawValue)
        // `\uDE`: the low surrogate is now 0xDE00...0xDEFF; only `0` can keep
        // the exact 0xDE00 required by the pair.
        XCTAssertTrue(processor.isAllowed(tokenId: EmojiEscapeToken.zero.rawValue))
        XCTAssertFalse(
            processor.isAllowed(tokenId: EmojiEscapeToken.one.rawValue),
            "`\\uDE1…` can never complete the pair for U+1F600")

        try processor.consume(tokenId: EmojiEscapeToken.zero.rawValue)
        try processor.consume(tokenId: EmojiEscapeToken.zero.rawValue)
        XCTAssertTrue(
            processor.isAllowed(tokenId: EmojiEscapeToken.closeQuoteBrace.rawValue),
            "the completed scalar satisfies the enum value")
        try processor.consume(tokenId: EmojiEscapeToken.closeQuoteBrace.rawValue)
        XCTAssertEqual(processor.status, .complete)
    }

    // MARK: - Whitespace stall (the Qwen3.6 `incompleteAtStop`)

    /// The exact live Qwen3.6 shape: a vocabulary whose whitespace tokens are
    /// admissible at the root (legal JSON) and a model that prefers them over
    /// `{`. Before the fix the walk below emits whitespace for the whole
    /// 64-token budget and never starts the value.
    private enum StallToken: Int, CaseIterable {
        case openBrace  // `{`
        case openBraceQuote  // `{"`
        case quote  // `"`
        case status  // `status`
        case quoteColonQuote  // `":"`
        case ok  // `ok`
        case quoteCloseBrace  // `"}`
        case space  // ` `
        case newline  // `\n`
        case doubleSpace  // `  `
        case newlineNewlineSpace  // `\n\n `
        case colon  // `:`
        case closeBrace  // `}`
        case eos

        var fragment: [UInt8]? {
            func bytes(_ string: String) -> [UInt8] { Array(string.utf8) }
            switch self {
            case .openBrace: return bytes("{")
            case .openBraceQuote: return bytes("{\"")
            case .quote: return bytes("\"")
            case .status: return bytes("status")
            case .quoteColonQuote: return bytes("\":\"")
            case .ok: return bytes("ok")
            case .quoteCloseBrace: return bytes("\"}")
            case .space: return bytes(" ")
            case .newline: return bytes("\n")
            case .doubleSpace: return bytes("  ")
            case .newlineNewlineSpace: return bytes("\n\n ")
            case .colon: return bytes(":")
            case .closeBrace: return bytes("}")
            case .eos: return nil
            }
        }
    }

    private func stallTable() -> StaticTokenFragmentTable {
        StaticTokenFragmentTable(
            fragments: StallToken.allCases.map { $0.fragment },
            endOfSequenceTokenIds: [StallToken.eos.rawValue])
    }

    func testConsecutiveWhitespaceOnlyTokensAreMaskedWhileIncomplete() throws {
        var processor = try JSONGrammarProcessor(
            format: .jsonSchema(try canarySchema()), table: stallTable())

        // The first whitespace token stays legal (JSON padding, and pinned by
        // the existing suite)…
        XCTAssertTrue(processor.isAllowed(tokenId: StallToken.space.rawValue))
        try processor.consume(tokenId: StallToken.space.rawValue)
        // …but a second consecutive whitespace-only token may not stall the
        // value: the Qwen3.6 checkpoints padded 64 tokens exactly this way.
        for token in [StallToken.space, .newline, .doubleSpace, .newlineNewlineSpace] {
            XCTAssertFalse(
                processor.isAllowed(tokenId: token.rawValue),
                "a second consecutive whitespace-only token (\(token)) must be masked")
        }
        XCTAssertTrue(
            processor.isAllowed(tokenId: StallToken.openBrace.rawValue),
            "structural progress must remain available")
        try processor.consume(tokenId: StallToken.openBrace.rawValue)

        // After a structural byte, one whitespace token is legal again — the
        // cap is a run bound, not a whitespace ban.
        XCTAssertTrue(processor.isAllowed(tokenId: StallToken.space.rawValue))
        try processor.consume(tokenId: StallToken.space.rawValue)
        XCTAssertFalse(processor.isAllowed(tokenId: StallToken.space.rawValue))
        XCTAssertTrue(processor.isAllowed(tokenId: StallToken.quote.rawValue))
    }

    func testWhitespacePreferringWalkCompletesTheCanaryWithinBudget() throws {
        let table = stallTable()
        let plan = try XCTUnwrap(
            try StructuredGeneration.plan(for: .jsonSchema(try canaryFormat()), table: table))
        var processor = plan.processor
        let whitespaceIds = [
            StallToken.space, .newline, .doubleSpace, .newlineNewlineSpace,
        ].map(\.rawValue)
        var bytes: [UInt8] = []
        var steps = 0
        // The live budget: the CoCore canary requests max_tokens 64.
        while steps < 64, !plan.runRecord.rootValueCompleted {
            let allowed = (0..<table.vocabularySize).filter { processor.isAllowed(tokenId: $0) }
            guard let pick = whitespaceIds.first(where: { allowed.contains($0) }) ?? allowed.min()
            else {
                return XCTFail("the mask must never be empty")
            }
            processor.didSample(token: MLXArray([Int32(pick)]))
            bytes += table.fragment(forTokenId: pick) ?? []
            steps += 1
        }
        XCTAssertTrue(
            plan.runRecord.rootValueCompleted,
            "a whitespace-preferring model must still complete the canary within 64 tokens")
        XCTAssertNil(plan.runRecord.failure)
        XCTAssertTrue(
            CoCoreCanary.structuredOutputPassed(content: String(decoding: bytes, as: UTF8.self)),
            "the walk must emit the exact canary document, got \(String(decoding: bytes, as: UTF8.self))")
        XCTAssertNil(
            StructuredGeneration.postGenerationError(plan: plan, stopReason: .stop, toolCallCount: 0))
    }
}
