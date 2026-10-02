import Foundation
import MLXLMCommon

// MARK: - Production tokenizer adapter

/// Construction failures of the production `TokenFragmentTable` adapter.
///
/// Every case is fail-closed: the table is either built with a complete,
/// consistent view of the vocabulary, or the structured request is rejected
/// before generation instead of being constrained by a guess.
public enum TokenizerFragmentTableError: Error, Sendable, Equatable {
    /// The declared model vocabulary size is not a positive number.
    case invalidVocabularySize(Int)
    /// Neither the tokenizer nor the model configuration identifies an
    /// end-of-sequence token, so normal structured completion is impossible.
    case missingEndOfSequenceToken
    /// A declared EOS id is outside the model vocabulary or has no vocabulary
    /// entry (an inconsistent tokenizer/model pairing).
    case endOfSequenceTokenNotInVocabulary(Int, vocabularySize: Int)
    /// No in-range token produced a usable text fragment: the declared
    /// vocabulary and the tokenizer's vocabulary do not line up.
    case noUsableTextFragments(vocabularySize: Int)

    public var message: String {
        switch self {
        case .invalidVocabularySize(let size):
            return "model vocabulary size \(size) is not a positive token count"
        case .missingEndOfSequenceToken:
            return
                "the tokenizer does not identify an end-of-sequence token; structured generation cannot complete normally"
        case .endOfSequenceTokenNotInVocabulary(let id, let vocabularySize):
            return
                "end-of-sequence token id \(id) is not an entry in the declared model vocabulary of size \(vocabularySize)"
        case .noUsableTextFragments(let vocabularySize):
            return
                "none of the \(vocabularySize) declared vocabulary entries yields a usable text fragment; the tokenizer and the model vocabulary do not match"
        }
    }
}

extension TokenizerFragmentTableError: LocalizedError {
    public var errorDescription: String? { message }
}

/// A production `TokenFragmentTable` built from an `MLXLMCommon.Tokenizer`
/// plus the model's explicit vocabulary size (the logits width).
///
/// Fragments are derived with `tokenizer.decode(tokenIds: [id],
/// skipSpecialTokens: false)` — the same decode entry point vmlx's
/// `NaiveStreamingDetokenizer` uses to emit text — so a token's fragment is
/// the text that token contributes on the wire.
///
/// ## Tokenizer-boundary assumption
///
/// The public `Tokenizer` API only decodes token LISTS, while the streaming
/// detokenizer emits the difference between cumulative decodes. The adapter
/// approximates a token's in-context contribution as closely as that API
/// permits: it probes `decode([seed, id])` against `decode([seed])` (seed =
/// BOS/EOS/unknown, whichever resolves) and uses the difference when it is a
/// clean prefix extension. That corrects the one systematic, observable
/// boundary effect — decoders that strip the FIRST token's leading space per
/// call (`Metaspace` `add_prefix_space`, Llama-style `Strip`-after-`Fuse`).
/// Residual assumptions, all fail-closed or whitespace-bounded:
/// - byte-fallback partial sequences only materialize when adjacent byte
///   tokens are decoded together; standalone they decode to U+FFFD, so the
///   adapter refuses them (nil fragment → never allowed);
/// - when no seed token resolves, or the probe is not a clean prefix
///   extension, the adapter falls back to the standalone decode (the same
///   weaker assumption, documented, still the vmlx decode entry point);
/// - fragments are context-free: they cannot model reassembly that spans two
///   arbitrary neighbouring tokens. The grammar treats such tokens as
///   unavailable (fail closed) rather than guessing their bytes.
///
/// ## Fail-closed vocabulary handling
///
/// - a missing vocabulary entry (`convertIdToToken` nil) or an empty decode
///   is never text: nil fragment, counted in `diagnostics`;
/// - special tokens (dropped by `skipSpecialTokens: true`) are never text,
///   including chat markers, BOS, and EOS;
/// - every declared EOS id must be an in-range vocabulary entry, otherwise
///   construction throws;
/// - a vocabulary where nothing is usable throws instead of producing a
///   table that masks everything at generation time.
public struct TokenizerFragmentTable: TokenFragmentTable {

    /// What the adapter saw while deriving the table. Deterministic and
    /// inspectable; useful for logs and for pinning behavior in tests.
    public struct Diagnostics: Sendable, Equatable {
        /// Ids with a usable text fragment.
        public let textFragmentCount: Int
        /// Ids dropped as special non-text tokens.
        public let specialTokenIdCount: Int
        /// Ids with no vocabulary entry at all.
        public let missingEntryCount: Int
        /// Ids whose decode is empty (no text contribution).
        public let emptyDecodeCount: Int
        /// Ids whose decode is not representable (U+FFFD: a partial byte
        /// sequence that only materializes in context).
        public let replacementCharacterIdCount: Int
        /// The probe seed token, when one resolved.
        public let seedTokenId: Int?
        /// Fragments that the seed probe corrected away from the standalone
        /// decode (the leading-space boundary effect).
        public let seedCorrectionCount: Int

        public init(
            textFragmentCount: Int,
            specialTokenIdCount: Int,
            missingEntryCount: Int,
            emptyDecodeCount: Int,
            replacementCharacterIdCount: Int,
            seedTokenId: Int?,
            seedCorrectionCount: Int
        ) {
            self.textFragmentCount = textFragmentCount
            self.specialTokenIdCount = specialTokenIdCount
            self.missingEntryCount = missingEntryCount
            self.emptyDecodeCount = emptyDecodeCount
            self.replacementCharacterIdCount = replacementCharacterIdCount
            self.seedTokenId = seedTokenId
            self.seedCorrectionCount = seedCorrectionCount
        }
    }

    public let vocabularySize: Int
    public let endOfSequenceTokenIds: Set<Int>
    public let diagnostics: Diagnostics
    private let fragments: [[UInt8]?]

    /// Builds the table for `vocabularySize` token ids.
    ///
    /// - Parameters:
    ///   - tokenizer: the production tokenizer bridge.
    ///   - vocabularySize: the model's logits width (from its configuration).
    ///     Ids at or beyond it are not entries and can never be allowed.
    ///   - additionalEndOfSequenceTokenIds: EOS ids declared by the model
    ///     configuration (e.g. `generation_config.json`), unioned with the
    ///     tokenizer's own EOS.
    public init(
        tokenizer: any MLXLMCommon.Tokenizer,
        vocabularySize: Int,
        additionalEndOfSequenceTokenIds: Set<Int> = []
    ) throws {
        guard vocabularySize > 0 else {
            throw TokenizerFragmentTableError.invalidVocabularySize(vocabularySize)
        }

        // Resolve the EOS set first: no structured request can complete
        // normally without a known EOS, and a declared id that is not a real
        // vocabulary entry means the tokenizer/model pairing is inconsistent.
        var endOfSequenceTokenIds = additionalEndOfSequenceTokenIds
        if let eosToken = tokenizer.eosToken, let id = tokenizer.convertTokenToId(eosToken) {
            endOfSequenceTokenIds.insert(id)
        }
        guard !endOfSequenceTokenIds.isEmpty else {
            throw TokenizerFragmentTableError.missingEndOfSequenceToken
        }
        for id in endOfSequenceTokenIds.sorted() {
            guard id >= 0, id < vocabularySize, tokenizer.convertIdToToken(id) != nil else {
                throw TokenizerFragmentTableError.endOfSequenceTokenNotInVocabulary(
                    id, vocabularySize: vocabularySize)
            }
        }

        // Seed probe: the first token of a decode call is the only position
        // affected by call-leading reassembly in the decoders this stack
        // ships (Metaspace add_prefix_space, Strip-after-Fuse).
        let seedTokenId: Int? = [tokenizer.bosToken, tokenizer.eosToken, tokenizer.unknownToken]
            .compactMap { $0 }
            .compactMap { tokenizer.convertTokenToId($0) }
            .first { id in
                guard id >= 0, id < vocabularySize, tokenizer.convertIdToToken(id) != nil else {
                    return false
                }
                let text = tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
                return !text.isEmpty && !Self.containsReplacementCharacter(text)
            }
        let seedText = seedTokenId.map {
            tokenizer.decode(tokenIds: [$0], skipSpecialTokens: false)
        }

        var fragments = [[UInt8]?](repeating: nil, count: vocabularySize)
        var textFragmentCount = 0
        var specialTokenIdCount = 0
        var missingEntryCount = 0
        var emptyDecodeCount = 0
        var replacementCharacterIdCount = 0
        var seedCorrectionCount = 0

        for id in 0..<vocabularySize {
            guard tokenizer.convertIdToToken(id) != nil else {
                missingEntryCount += 1
                continue
            }
            let standalone = tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
            let withoutSpecials = tokenizer.decode(tokenIds: [id], skipSpecialTokens: true)
            if !standalone.isEmpty, withoutSpecials.isEmpty {
                specialTokenIdCount += 1
                continue
            }
            if standalone.isEmpty {
                emptyDecodeCount += 1
                continue
            }
            if Self.containsReplacementCharacter(standalone) {
                replacementCharacterIdCount += 1
                continue
            }

            if let seedTokenId, let seedText, !seedText.isEmpty {
                let combined = tokenizer.decode(
                    tokenIds: [seedTokenId, id], skipSpecialTokens: false)
                if combined.hasPrefix(seedText) {
                    let contribution = String(combined.dropFirst(seedText.count))
                    if contribution.isEmpty {
                        // The probe says this token contributes nothing after
                        // a real token: trusting the standalone decode could
                        // admit bytes the model never emits. Fail closed.
                        emptyDecodeCount += 1
                        continue
                    }
                    if Self.containsReplacementCharacter(contribution) {
                        replacementCharacterIdCount += 1
                        continue
                    }
                    if contribution != standalone {
                        seedCorrectionCount += 1
                    }
                    fragments[id] = Array(contribution.utf8)
                    textFragmentCount += 1
                    continue
                }
                // Not a clean prefix extension: fall back to the standalone
                // decode (the documented weaker assumption).
            }

            fragments[id] = Array(standalone.utf8)
            textFragmentCount += 1
        }

        guard textFragmentCount > 0 else {
            throw TokenizerFragmentTableError.noUsableTextFragments(vocabularySize: vocabularySize)
        }

        self.vocabularySize = vocabularySize
        self.endOfSequenceTokenIds = endOfSequenceTokenIds
        self.fragments = fragments
        self.diagnostics = Diagnostics(
            textFragmentCount: textFragmentCount,
            specialTokenIdCount: specialTokenIdCount,
            missingEntryCount: missingEntryCount,
            emptyDecodeCount: emptyDecodeCount,
            replacementCharacterIdCount: replacementCharacterIdCount,
            seedTokenId: seedTokenId,
            seedCorrectionCount: seedCorrectionCount)
    }

    public func fragment(forTokenId tokenId: Int) -> [UInt8]? {
        guard fragments.indices.contains(tokenId) else { return nil }
        return fragments[tokenId]
    }

    private static func containsReplacementCharacter(_ string: String) -> Bool {
        string.unicodeScalars.contains { $0 == "\u{FFFD}" }
    }
}
