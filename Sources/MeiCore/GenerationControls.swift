import Foundation

/// The request-time generation controls of one run, resolved against server
/// defaults. Pure and deterministic so the precedence rules are unit-tested
/// without a model: every request value wins when present; otherwise the
/// server default applies. This mirrors exactly the precedence the engine
/// applied inline (Engine.makeParameters / makeCompletionParameters) before
/// this type existed — it is a refactor seam, not a behavior change.
///
/// The `maxTokens` arithmetic is also pinned here: the request's (or the
/// server's) generation cap is clamped to the KV capacity remaining after the
/// prompt, with a floor of 1 so a full context never turns into a
/// 0-token generation.
public struct GenerationControlSelection: Sendable, Equatable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float?
    public var presencePenalty: Float?
    public var frequencyPenalty: Float?
    public var randomSeed: UInt64?
    public var extraStopStrings: [String]?
    public var maxTokens: Int

    public init(
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        repetitionPenalty: Float?,
        presencePenalty: Float?,
        frequencyPenalty: Float?,
        randomSeed: UInt64?,
        extraStopStrings: [String]?,
        maxTokens: Int
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.randomSeed = randomSeed
        self.extraStopStrings = extraStopStrings
        self.maxTokens = maxTokens
    }

    /// Resolve a chat request against the server configuration.
    public static func resolve(
        request: ChatRequest,
        config: ServerConfig,
        promptTokenCount: Int
    ) -> GenerationControlSelection {
        GenerationControlSelection(
            temperature: Float(request.temperature ?? Double(config.temperature)),
            topP: Float(request.topP ?? Double(config.topP)),
            topK: request.topK ?? config.topK,
            minP: Float(request.minP ?? Double(config.minP)),
            repetitionPenalty: (request.repetitionPenalty ?? config.repetitionPenalty.map(Double.init)).map(Float.init),
            presencePenalty: (request.presencePenalty ?? config.presencePenalty.map(Double.init)).map(Float.init),
            frequencyPenalty: (request.frequencyPenalty ?? config.frequencyPenalty.map(Double.init)).map(Float.init),
            randomSeed: request.seed,
            extraStopStrings: (request.stop?.isEmpty == false) ? request.stop : nil,
            maxTokens: resolveMaxTokens(
                requested: request.maxTokens ?? config.maxTokensDefault,
                promptTokenCount: promptTokenCount,
                kvCapacity: config.maxKVSize))
    }

    /// Resolve a legacy /v1/completions request against the server
    /// configuration.
    public static func resolve(
        request: CompletionRequest,
        config: ServerConfig,
        promptTokenCount: Int
    ) -> GenerationControlSelection {
        GenerationControlSelection(
            temperature: Float(request.temperature ?? Double(config.temperature)),
            topP: Float(request.topP ?? Double(config.topP)),
            topK: request.topK ?? config.topK,
            minP: Float(request.minP ?? Double(config.minP)),
            repetitionPenalty: (request.repetitionPenalty ?? config.repetitionPenalty.map(Double.init)).map(Float.init),
            presencePenalty: (request.presencePenalty ?? config.presencePenalty.map(Double.init)).map(Float.init),
            frequencyPenalty: (request.frequencyPenalty ?? config.frequencyPenalty.map(Double.init)).map(Float.init),
            randomSeed: request.seed,
            extraStopStrings: (request.stop?.isEmpty == false) ? request.stop : nil,
            maxTokens: resolveMaxTokens(
                requested: request.maxTokens ?? config.maxTokensDefault,
                promptTokenCount: promptTokenCount,
                kvCapacity: config.maxKVSize))
    }

    /// The generation token budget: `requested` clamped to the KV capacity
    /// remaining after the prompt, floored at 1.
    public static func resolveMaxTokens(
        requested: Int,
        promptTokenCount: Int,
        kvCapacity: Int
    ) -> Int {
        min(requested, max(1, kvCapacity - promptTokenCount))
    }
}