import Foundation
import MLXLMCommon

// MARK: - Structured-generation Engine seam

/// Construction failures of the structured-generation Engine seam.
public enum StructuredGenerationError: Error, Sendable, Equatable {
    /// `config.json` is missing or unreadable in the model directory.
    case modelConfigUnreadable(String)
    /// `config.json` declares no `vocab_size` (root or nested `text_config`).
    case modelVocabularySizeMissing(String)
    /// `config.json` declares a `vocab_size` that is not a positive integer.
    case modelVocabularySizeInvalid(String)

    public var message: String {
        switch self {
        case .modelConfigUnreadable(let directory):
            return "cannot read config.json in model directory \(directory)"
        case .modelVocabularySizeMissing(let directory):
            return "config.json in \(directory) declares no vocab_size (root or text_config)"
        case .modelVocabularySizeInvalid(let raw):
            return "config.json declares an invalid vocab_size: \(raw)"
        }
    }
}

extension StructuredGenerationError: LocalizedError {
    public var errorDescription: String? { message }
}

/// Deterministic, model-free construction of the structured-generation
/// pipeline for one chat request — the Engine seam.
///
/// The Engine supplies the real tokenizer, the model's vocabulary size, and
/// the model configuration's declared EOS ids; everything else here is pure,
/// so tests exercise the whole construction path without loading weights.
/// The Engine caches the immutable fragment table per model; the grammar
/// state itself is per request (a fresh `JSONGrammarLogitProcessor`).
public enum StructuredGeneration {

    /// One request's structured-generation plan.
    public struct Plan: Sendable {
        /// The compiled constraint (deterministic `constraintKey`).
        public let format: CompiledResponseFormat
        /// The `LogitProcessor` to hand to the ordinary `TokenIterator`
        /// via `additionalProcessor:`.
        public let processor: JSONGrammarLogitProcessor
        /// The request-scoped run record the engine inspects after the
        /// generation task. The `processor` above is copied by value into
        /// the iterator (which mutates its own copy), so per-run
        /// observations — the first failure and whether a complete root
        /// value was consumed — must come from this shared record.
        public var runRecord: JSONGrammarRunRecord { processor.runRecord }

        init(format: CompiledResponseFormat, processor: JSONGrammarLogitProcessor) {
            self.format = format
            self.processor = processor
        }
    }

    /// The structured plan for a decoded request format, or nil for `.text`
    /// (the byte-compatible ordinary path: no processor is ever built).
    ///
    /// Compiles the format BEFORE generation; compiler rejections propagate
    /// as `JSONSchemaCompileError` for the HTTP layer to map.
    public static func plan(
        for format: ResponseFormat,
        table: any TokenFragmentTable
    ) throws -> Plan? {
        switch format {
        case .text:
            return nil
        case .jsonObject, .jsonSchema:
            let compiled = try JSONSchemaCompiler.compile(format)
            let processor = try JSONGrammarLogitProcessor(format: compiled, table: table)
            return Plan(format: compiled, processor: processor)
        }
    }

    /// The model's logits width from the bundle's `config.json` (`vocab_size`
    /// at the root, or nested under `text_config` — the same convention the
    /// model loader uses). Fail-closed: unreadable or absent metadata throws
    /// instead of guessing, and the runtime vocabulary check in the mask
    /// (`vocabularyMismatch`) is the backstop if this ever disagrees with the
    /// model's actual output width.
    public static func modelVocabularySize(modelDirectory: String) throws -> Int {
        let url = URL(fileURLWithPath: modelDirectory, isDirectory: true)
            .appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
            let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            throw StructuredGenerationError.modelConfigUnreadable(modelDirectory)
        }
        var raw = root["vocab_size"]
        if raw == nil, let textConfig = root["text_config"] as? [String: Any] {
            raw = textConfig["vocab_size"]
        }
        guard let raw else {
            throw StructuredGenerationError.modelVocabularySizeMissing(modelDirectory)
        }
        guard !(raw is Bool), let number = raw as? NSNumber else {
            throw StructuredGenerationError.modelVocabularySizeInvalid(String(describing: raw))
        }
        let size = number.intValue
        guard size > 0, Double(size) == number.doubleValue else {
            throw StructuredGenerationError.modelVocabularySizeInvalid(String(describing: raw))
        }
        return size
    }

    /// The thinking-enablement decision for a chat request.
    ///
    /// Structured requests force thinking OFF: the JSON grammar constrains
    /// the first sampled token to start a JSON value, so any reasoning
    /// preamble (or a template-level think prefill) would leave the grammar
    /// with no legal continuation. Ordinary text requests keep the existing
    /// precedence (request `reasoning_effort` when the operator configured
    /// nothing, otherwise the operator's explicit value).
    public static func enableThinking(request: ChatRequest, configEnableThinking: Bool?) -> Bool? {
        if request.responseFormat != .text { return false }
        if let reasoningEffort = request.reasoningEffort, configEnableThinking == nil {
            return reasoningEffort != "none"
        }
        return configEnableThinking
    }

    /// The compiled-decode flag for one chat request.
    ///
    /// Structured requests stay on the ordinary single-sequence,
    /// non-compiled decode path until the constrained processor's state is
    /// proven safe for graph-traced replay — the plan's decode-path gate
    /// (speculative/MTP, compiled, and batched paths stay disabled for
    /// structured requests). An operator's `--compiled-decode true` therefore
    /// applies to text requests only; text requests keep the configured value
    /// byte-compatibly.
    public static func enableCompiledDecode(request: ChatRequest, configEnabled: Bool) -> Bool {
        request.responseFormat == .text ? configEnabled : false
    }

    /// Maps a captured grammar failure onto the engine error the HTTP layer
    /// surfaces as a 500: a request whose constraint failed must never be
    /// answered with a "successful" structured response.
    public static func generationFailure(_ failure: JSONGrammarError) -> EngineError {
        .generationFailed("structured output constraint failed: \(failure.message)")
    }

    /// The post-generation invariant for one structured run: returns the
    /// engine error that must fail the request instead of answering it, or
    /// nil when the run may be reported.
    ///
    /// Two conditions fail closed:
    /// 1. the run record captured a constraint failure (illegal token,
    ///    premature EOS, no legal continuation, vocabulary mismatch) — the
    ///    answer is not the constrained one;
    /// 2. the run did not consume a complete root value — a truncated or
    ///    aborted answer must never masquerade as a successful structured
    ///    response, regardless of whether the producer reports `stop`,
    ///    `length`, or cancellation. The normal stop-reason mapping remains
    ///    unchanged for complete runs.
    ///
    /// A nil `stopReason` is still accepted here because the incomplete-root
    /// invariant is independent of the producer's reported reason.
    public static func postGenerationError(
        plan: Plan?,
        stopReason: GenerateStopReason?,
        toolCallCount: Int
    ) -> EngineError? {
        guard let plan else { return nil }
        if let failure = plan.runRecord.failure {
            return generationFailure(failure)
        }
        if !plan.runRecord.rootValueCompleted {
            return generationFailure(.incompleteAtStop)
        }
        return nil
    }
}
