import Foundation
import MLX

// MARK: - Caller-supplied tokenizer vocabulary

/// A tokenizer vocabulary as seen by the constrained-decoding core: token ids
/// map to the byte fragments they contribute to the output text.
///
/// The caller — the engine's `StructuredGeneration` seam, backed in production
/// by `TokenizerFragmentTable` — supplies this from the real tokenizer; the
/// core never touches a tokenizer itself. Implementations must be immutable
/// after construction — the processor copies share them — and must keep
/// fragment lookup allocation-light for the per-step mask scan.
public protocol TokenFragmentTable: Sendable {
    /// Token ids are `0 ..< vocabularySize`.
    var vocabularySize: Int { get }
    /// Fragment bytes for `tokenId`, or nil for tokens without text (special
    /// tokens such as BOS/EOS).
    func fragment(forTokenId tokenId: Int) -> [UInt8]?
    /// Token ids that terminate generation.
    var endOfSequenceTokenIds: Set<Int> { get }
}

/// Simple immutable table over an array of fragments (index = token id).
/// Useful for tests and for callers that can materialize the vocabulary.
public struct StaticTokenFragmentTable: TokenFragmentTable {
    public let vocabularySize: Int
    public let endOfSequenceTokenIds: Set<Int>
    private let fragments: [[UInt8]?]

    public init(fragments: [[UInt8]?], endOfSequenceTokenIds: Set<Int> = []) {
        self.fragments = fragments
        self.vocabularySize = fragments.count
        self.endOfSequenceTokenIds = endOfSequenceTokenIds
    }

    /// Convenience initializer over UTF-8 strings (`nil` = no text fragment).
    public init(strings: [String?], endOfSequenceTokenIds: Set<Int> = []) {
        self.init(
            fragments: strings.map { $0.map { Array($0.utf8) } },
            endOfSequenceTokenIds: endOfSequenceTokenIds)
    }

    public func fragment(forTokenId tokenId: Int) -> [UInt8]? {
        guard fragments.indices.contains(tokenId) else { return nil }
        return fragments[tokenId]
    }
}

// MARK: - Token-level constrained decoding processor

/// Token-level constrained-decoding processor: the model-free core the vmlx
/// `LogitProcessor` wiring drives (via `JSONGrammarLogitProcessor`).
///
/// It combines a compiled response format with a caller-supplied
/// `TokenFragmentTable` and exposes, per generation step:
///
/// - `isAllowed(tokenId:)` / `allowedTokenIds()` — allowed-token decisions;
/// - `maskedLogits(_:)` — a logits mask (disallowed tokens set to `-inf`)
///   suitable for a `LogitProcessor.process(logits:)` implementation;
/// - `consume(tokenId:)` — state advancement for `didSample(token:)`;
/// - `reset()` — the prompt-reset hook;
/// - `independentCopy()` — the isolation hook (all mutable state is a value
///   type, so copies never share request state).
///
/// Fail-closed contract: a grammar state from which no token can advance
/// throws `noLegalContinuation` instead of returning unmasked logits; an
/// illegal sampled token throws and poisons the state (`grammarFailed`); a
/// premature EOS throws (`prematureEndOfSequence`). None of these may be
/// reported as a successful structured response.
public struct JSONGrammarProcessor: Sendable {

    /// The compiled format this processor enforces.
    public let format: CompiledResponseFormat

    /// Deterministic cache key for the compiled constraint (vocabulary
    /// mappings and compiled grammars are cacheable per model/schema).
    public var constraintKey: String { format.constraintKey }

    private let table: any TokenFragmentTable
    private var state: JSONGrammarState

    /// Builds a processor for a compiled format. `.text` has no grammar and
    /// throws `unconstrainedFormat`.
    public init(format: CompiledResponseFormat, table: any TokenFragmentTable) throws {
        switch format {
        case .text:
            throw JSONGrammarError.unconstrainedFormat
        case .jsonObject, .jsonSchema:
            break
        }
        self.format = format
        self.table = table
        self.state = try JSONGrammarState(format: format)
    }

    // MARK: Status

    /// Grammar status at the current position.
    public var status: JSONGrammarStatus { state.status }

    /// True when a complete root value has been consumed — `.complete` (EOS
    /// may still be sampled) or `.finished` (EOS already consumed). The
    /// constrained answer is satisfied in both states; a completed-then-EOS
    /// run must not be mistaken for an incomplete one.
    public var isAccepting: Bool {
        state.status == .complete || state.status == .finished
    }

    // MARK: Lifecycle

    /// Prompt-reset hook: starts a fresh grammar for a new request.
    public mutating func reset() {
        state.reset()
    }

    /// A copy whose mutable state is independent of the receiver's. All
    /// mutable state is a value type, so this is an ordinary copy; the
    /// explicit entry point mirrors the vmlx `LogitProcessor` contract.
    public func independentCopy() -> JSONGrammarProcessor {
        self
    }

    // MARK: Allowed-token decisions

    /// Whether `tokenId` can be sampled in the current state. EOS is allowed
    /// only after a complete root value.
    public func isAllowed(tokenId: Int) -> Bool {
        guard tokenId >= 0, tokenId < table.vocabularySize else { return false }
        if table.endOfSequenceTokenIds.contains(tokenId) {
            return state.status == .complete
        }
        guard let fragment = table.fragment(forTokenId: tokenId), !fragment.isEmpty else { return false }
        var trial = state
        return trial.consume(fragment: fragment)
    }

    /// Every allowed token id in ascending order, EOS last when allowed.
    ///
    /// Diagnostic/test helper: it scans the whole vocabulary. Production
    /// masking should use `maskedLogits(_:)`, which performs the same scan
    /// while building the mask.
    public func allowedTokenIds() -> [Int] {
        (0..<table.vocabularySize).filter { isAllowed(tokenId: $0) }
    }

    // MARK: Mask

    /// Returns a copy of `logits` with every token that cannot advance the
    /// grammar set to `-inf` (the vmlx masking convention).
    ///
    /// Throws instead of masking everything when the state is already failed,
    /// already finished, or has no legal continuation — the caller must abort
    /// the generation rather than sample from an all-masked distribution.
    public func maskedLogits(_ logits: MLXArray) throws -> MLXArray {
        switch state.status {
        case .failed:
            throw JSONGrammarError.grammarFailed
        case .finished:
            throw JSONGrammarError.alreadyFinished
        case .inProgress, .complete:
            break
        }
        let vocabularySize = logits.dim(-1)
        guard vocabularySize == table.vocabularySize else {
            throw JSONGrammarError.vocabularyMismatch(expected: table.vocabularySize, actual: vocabularySize)
        }
        var allowed = [Bool](repeating: false, count: vocabularySize)
        var anyAllowed = false
        for tokenId in 0..<vocabularySize where isAllowed(tokenId: tokenId) {
            allowed[tokenId] = true
            anyAllowed = true
        }
        guard anyAllowed else { throw JSONGrammarError.noLegalContinuation }
        let condition = MLXArray(allowed.map { Int32($0 ? 1 : 0) }).asType(.bool)
        return MLX.where(condition, logits, MLXArray(-Float.infinity, dtype: logits.dtype))
    }

    // MARK: State advancement

    /// Advances the grammar by one sampled token.
    ///
    /// - EOS: allowed only when the root value is complete; marks the
    ///   processor finished.
    /// - Any other token: its fragment must be fully consumable. After a
    ///   complete root, only whitespace-only fragments are consumable.
    ///
    /// Throws (and fails closed) instead of accepting a token the mask should
    /// have excluded.
    public mutating func consume(tokenId: Int) throws {
        guard tokenId >= 0, tokenId < table.vocabularySize else {
            throw JSONGrammarError.unknownToken(tokenId)
        }
        switch state.status {
        case .failed:
            throw JSONGrammarError.grammarFailed
        case .finished:
            throw JSONGrammarError.alreadyFinished
        case .complete:
            if table.endOfSequenceTokenIds.contains(tokenId) {
                state.markEndOfSequence()
                return
            }
        case .inProgress:
            if table.endOfSequenceTokenIds.contains(tokenId) {
                throw JSONGrammarError.prematureEndOfSequence
            }
        }
        guard let fragment = table.fragment(forTokenId: tokenId), !fragment.isEmpty else {
            throw JSONGrammarError.illegalToken(tokenId)
        }
        guard state.consume(fragment: fragment) else {
            throw JSONGrammarError.illegalToken(tokenId)
        }
    }
}
