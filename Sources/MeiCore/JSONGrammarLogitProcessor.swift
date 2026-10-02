import Foundation
import MLX
import MLXLMCommon

// MARK: - Request-scoped run record

/// Request-scoped, inspectable record of one constrained run: the FIRST
/// grammar failure, and whether the grammar ever consumed a complete root
/// value.
///
/// `LogitProcessor.process` and `didSample` cannot throw, so grammar failures
/// are captured here while generation continues with an all-`-inf` mask. The
/// engine inspects the record after the generation task and turns a failure —
/// or a stop that contradicts the constraint (no complete root value, yet the
/// response would say `stop`) — into `EngineError.generationFailed` instead of
/// returning an answer that does not satisfy the requested constraint.
///
/// Reference-typed on purpose: the processor is copied by value along the
/// vmlx processor pipeline (and `independentCopy` must keep reporting into the
/// same record), while its grammar state stays value-isolated. Every per-run
/// observation the engine needs therefore travels through this shared record.
public final class JSONGrammarRunRecord: @unchecked Sendable {
    private let lock = NSLock()
    private var first: JSONGrammarError?
    private var completedRootValue = false

    public init() {}

    /// The first failure observed for this request, or nil while the
    /// constraint is satisfied.
    public var failure: JSONGrammarError? {
        lock.lock()
        defer { lock.unlock() }
        return first
    }

    public var isFailed: Bool { failure != nil }

    /// True once the grammar consumed a complete root JSON value
    /// (`.complete` or `.finished`): the run's answer satisfies the
    /// constraint. Monotonic — the grammar cannot leave completion; a later
    /// failure is reported separately through `failure`.
    public var rootValueCompleted: Bool {
        lock.lock()
        defer { lock.unlock() }
        return completedRootValue
    }

    func record(_ error: JSONGrammarError) {
        lock.lock()
        defer { lock.unlock() }
        if first == nil { first = error }
    }

    /// Records that the grammar reached an accepting state. Only ever sets
    /// the flag.
    func recordRootValueCompleted() {
        lock.lock()
        defer { lock.unlock() }
        completedRootValue = true
    }
}

// MARK: - Mei-owned LogitProcessor

/// Mei-owned `LogitProcessor` wrapper around the model-free
/// `JSONGrammarProcessor` (the vmlx wiring seam's consumer).
///
/// Contract:
/// - `prompt(_:)` resets the grammar for a new request (the prompt tokens
///   themselves are not constrained);
/// - `process(logits:)` returns the grammar mask (disallowed tokens at
///   `-inf`); it cannot throw through the vmlx protocol, so a failure is
///   recorded in the request-scoped `runRecord` and the returned mask is
///   all-`-inf` — NEVER unmasked logits;
/// - `didSample(token:)` advances the grammar, records completion once the
///   root value is complete, and captures any failure (illegal token,
///   premature EOS, or the sticky failed state) the same way;
/// - `independentCopy()` returns a copy whose grammar state is independent
///   (all mutable state is a value type) while the run record — and therefore
///   the reporting path — stays request-scoped and shared.
public struct JSONGrammarLogitProcessor: LogitProcessor, Sendable {

    /// The request-scoped run record. The engine holds the same instance the
    /// iterator's processor copies report into.
    public let runRecord: JSONGrammarRunRecord

    private var processor: JSONGrammarProcessor

    public init(
        format: CompiledResponseFormat,
        table: any TokenFragmentTable,
        runRecord: JSONGrammarRunRecord = JSONGrammarRunRecord()
    ) throws {
        self.processor = try JSONGrammarProcessor(format: format, table: table)
        self.runRecord = runRecord
    }

    // MARK: Status

    /// Grammar status at the current position (diagnostics/tests).
    public var status: JSONGrammarStatus { processor.status }

    /// True when a complete root value has been consumed — `.complete` (EOS
    /// may still be sampled) or `.finished` (EOS already consumed).
    public var isAccepting: Bool { processor.isAccepting }

    /// Deterministic cache key of the compiled constraint.
    public var constraintKey: String { processor.constraintKey }

    /// Whether `tokenId` can be sampled in the current state
    /// (diagnostics/tests; the mask itself is built by `process(logits:)`).
    public func isAllowed(tokenId: Int) -> Bool { processor.isAllowed(tokenId: tokenId) }

    // MARK: LogitProcessor

    /// Prompt-reset hook: starts a fresh grammar for the new request.
    public mutating func prompt(_ prompt: MLXArray) {
        processor.reset()
    }

    public func process(logits: MLXArray) -> MLXArray {
        // TokenIterator.next() primes one decode step before forwarding the
        // previously sampled token. After EOS, that discarded lookahead can
        // call process once more while the grammar is `.finished`; it is a
        // benign consequence of the iterator's one-token latency, not a
        // constraint failure. Preserve the logits and let didSample ignore
        // the equally discarded token below.
        if processor.status == .finished { return logits }
        do {
            return try processor.maskedLogits(logits)
        } catch let error as JSONGrammarError {
            runRecord.record(error)
            return Self.allMasked(logits)
        } catch {
            // `maskedLogits` only throws `JSONGrammarError`; kept so no
            // unknown error can ever fall through to unmasked logits.
            runRecord.record(.grammarFailed)
            return Self.allMasked(logits)
        }
    }

    public mutating func didSample(token: MLXArray) {
        // See process(logits:): the iterator has already accepted EOS, but
        // its priming step may still report one discarded sample before the
        // caller observes the stop token. It cannot affect the response and
        // must not become an `alreadyFinished` failure.
        guard processor.status != .finished else { return }
        guard token.size > 0 else { return }
        let tokenId = token.reshaped(-1)[0].item(Int.self)
        do {
            try processor.consume(tokenId: tokenId)
            // Completion must be observable through the shared record: the
            // engine's copy of this processor is never advanced (the
            // iterator mutates its own copy).
            if processor.isAccepting {
                runRecord.recordRootValueCompleted()
            }
        } catch let error as JSONGrammarError {
            runRecord.record(error)
        } catch {
            runRecord.record(.illegalToken(tokenId))
        }
    }

    /// A copy whose grammar state is independent of the receiver's (all
    /// mutable state is a value type). The run record is intentionally
    /// shared: copies — including speculative throwaway copies, should one
    /// ever be made — must report failures to the same request-scoped record.
    public func independentCopy() -> Self {
        self
    }

    // MARK: Helpers

    /// The fail-closed fallback: every token at `-inf`, same shape and dtype
    /// as the incoming logits. A masked row is never silently replaced by an
    /// unmasked one.
    private static func allMasked(_ logits: MLXArray) -> MLXArray {
        broadcast(MLXArray(-Float.infinity, dtype: logits.dtype), to: logits.shape)
    }
}
