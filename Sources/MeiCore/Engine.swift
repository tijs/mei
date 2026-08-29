import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXHuggingFace
import VMLXTokenizers
import os

/// Minimal unchecked box for crossing non-Sendable model references out of
/// the container actor, mirroring vmlx-swift's own `SendableBox` idiom.
final class MeiBox<Value>: @unchecked Sendable {
    var value: Value
    init(_ value: Value) { self.value = value }
}

public enum EngineError: LocalizedError, Sendable {
    case modelDirectoryMissing(String)
    case overContextCap(promptTokens: Int, cap: Int)
    case emptyPrompt
    case generationFailed(String)
    case modelNotLoaded

    public var errorDescription: String? {
        switch self {
        case .modelDirectoryMissing(let path):
            "model directory does not exist: \(path)"
        case .overContextCap(let promptTokens, let cap):
            "request exceeded context cap: \(promptTokens) prompt tokens > \(cap) allowed"
        case .emptyPrompt:
            "request prompt was empty after tokenization"
        case .generationFailed(let message):
            "generation failed: \(message)"
        case .modelNotLoaded:
            "model is not loaded"
        }
    }
}

/// One generation event, exposed by the streaming path and consumed by the
/// router's SSE mapper. The non-streaming path assembles these internally.
public enum StreamEvent: Sendable {
    case chunk(String)
    case reasoning(String)
    case prefill(completed: Int, total: Int)
    case toolCall(GenerationRun.ToolCallEmitting)
    case finish(GenerationRun)
}

/// Engine is a strict single-flight inference actor: one generation at a
/// time, slot state mutated only by generations that acquire the flight
/// token, matching the benchmark's "never launch two servers concurrently"
/// discipline and keeping KV-reuse semantics deterministic.
public actor Engine {
    private let container: ModelContainer
    public let config: ServerConfig
    private let logger = Logger(subsystem: "mei.engine", category: "engine")

    // In-process prefix cache slot. The [KVCache] array mutates in place as
    // generations append to it.
    private var slotTokens: [Int] = []
    private var slotCache: [KVCache]? = nil

    /// Serializes all engine work onto one FIFO queue.
    private var queueTail: Task<Void, Never>?

    public init(container: ModelContainer, config: ServerConfig) {
        self.container = container
        self.config = config
    }

    /// Load a model from a local directory (config.json + safetensors +
    /// tokenizer files), using the pinned vmlx-swift stack.
    public static func load(config: ServerConfig) async throws -> Engine {
        let directory = URL(fileURLWithPath: config.modelDirectory, isDirectory: true)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw EngineError.modelDirectoryMissing(config.modelDirectory)
        }
        let container = try await loadModelContainer(
            from: directory,
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: (try? LoadConfiguration()) ?? LoadConfiguration()
        )
        return Engine(container: container, config: config)
    }

    public var servedModelID: String { config.servedModelID }

    // MARK: - Serialized execution

    private func serialized<T: Sendable>(
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            let previous = queueTail
            queueTail = Task {
                _ = await previous?.value
                do {
                    let result = try await operation()
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Chat completion

    public func chatRun(request: ChatRequest) async throws -> GenerationRun {
        try await serialized {
            try await self.chatRunLocked(request: request)
        }
    }

    public func chatRunStreaming(request: ChatRequest) -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.serialized {
                        try await self.runStreamingLocked(request: request, continuation: continuation)
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func chatRunLocked(request: ChatRequest) async throws -> GenerationRun {
        let (template, tokens, context) = try await renderChatTemplate(
            messages: request.messages,
            tools: request.tools,
            enableThinking: requestEnableThinking(request),
            reasoningEffort: request.reasoningEffort,
            toolChoice: request.toolChoice)
        let parameters = try await makeParameters(
            tokens: tokens, request: request, templateCount: template.count, context: context)
        let slot = SlotDecision.make(
            newTokens: tokens,
            slotTokens: config.cacheReuse ? slotTokens : [],
            maxSlotTokens: config.maxCacheSlotTokens)
        let run = try await generateLocked(
            tokens: tokens,
            reuseSlot: slot,
            parameters: parameters,
            tools: request.tools)
        adoptSlot(tokens: tokens, cache: iterationCache!, reuseApplied: slot.reuseApplied)
        return run
    }

    private func requestEnableThinking(_ request: ChatRequest) -> Bool? {
        if let reasoningEffort = request.reasoningEffort, config.enableThinking == nil {
            return reasoningEffort != "none"
        }
        return config.enableThinking
    }

    private func adoptSlot(tokens: [Int], cache: [KVCache], reuseApplied: Bool) {
        guard tokens.count <= config.maxCacheSlotTokens else {
            slotTokens = []
            slotCache = nil
            return
        }
        slotTokens = tokens
        slotCache = cache
    }

    /// The cache used by the most recent generation, captured by
    /// `generateLocked`/`runStreamingLocked` for slot adoption.
    private var iterationCache: [KVCache]?


    /// Serialize a parsed vmlx ToolCall's arguments to a compact JSON string,
    /// preferring the exact protocol-bytes when the parser preserved them.
    private static func toolArgumentsJSON(_ call: ToolCall) -> String {
        if let raw = call.function.rawArgumentsJSON, !raw.isEmpty { return raw }
        let anyDict = call.function.arguments.mapValues { $0.anyValue }
        if let data = try? JSONSerialization.data(withJSONObject: anyDict),
            let string = String(data: data, encoding: .utf8)
        {
            return string
        }
        return "{}"
    }

    private func renderChatTemplate(
        messages: [APIMessage],
        tools: [MeiJSONValue]?,
        enableThinking: Bool?,
        reasoningEffort: String?,
        toolChoice: MeiJSONValue?
    ) async throws -> (template: [[String: any Sendable]], tokens: [Int], context: [String: any Sendable]?) {
        let template = messages.map { MessageMapping.templateDictionary(from: $0) }
        let templateTools = MessageMapping.templateTools(tools)
        let context = MessageMapping.additionalContext(
            enableThinking: enableThinking ?? config.enableThinking,
            reasoningEffort: reasoningEffort ?? config.reasoningEffort,
            toolChoice: toolChoice)
        let tokenizer = await container.tokenizer
        let tokens = try tokenizer.applyChatTemplate(
            messages: template, tools: templateTools, additionalContext: context)
        return (template, tokens, context)
    }

    private func generateLocked(
        tokens: [Int],
        reuseSlot: SlotDecision.Result,
        parameters: GenerateParameters,
        tools: [MeiJSONValue]?
    ) async throws -> GenerationRun {
        let tokenizer = await container.tokenizer
        let modelConfiguration = await container.configuration

        let inputTokens: [Int] = reuseSlot.reuseApplied
            ? Array(tokens.suffix(tokens.count - reuseSlot.cachedTokenCount))
            : tokens
        let input = LMInput(
            tokens: MLXArray(inputTokens),
            tokenIds: inputTokens,
            toolSchemas: MessageMapping.templateTools(tools))

        let modelBox: MeiBox<any LanguageModel> = await container.perform { context in
            MeiBox(context.model)
        }
        let model = modelBox.value

        let reuseCache = reuseSlot.reuseApplied ? slotCache : nil
        let cache: [KVCache]
        if let reuseCache {
            cache = reuseCache
        } else {
            cache = model.newCache(parameters: parameters)
        }
        iterationCache = cache

        let iterationStart = Date()
        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: cache,
            parameters: parameters)
        let prefillWall = Date().timeIntervalSince(iterationStart)

        let (stream, task) = MLXLMCommon.generateTask(
            promptTokenCount: inputTokens.count,
            modelConfiguration: modelConfiguration,
            tokenizer: tokenizer,
            iterator: iterator,
            toolSchemas: MessageMapping.templateTools(tools))

        var run = GenerationRun()
        run.cachedTokenCount = reuseSlot.reuseApplied ? reuseSlot.cachedTokenCount : 0
        var info: GenerateCompletionInfo?
        for await item in stream {
            switch item {
            case .chunk(let chunk):
                run.text += chunk
            case .reasoning(let reason):
                run.reasoning += reason
            case .toolCall(let call):
                run.toolCalls.append(.init(
                    id: call.id ?? "call_\(UUID().uuidString.lowercased().prefix(12))",
                    name: call.function.name,
                    argumentsJSON: Self.toolArgumentsJSON(call)))
            case .toolCallProgress:
                break  // assembled into .toolCall on envelope close
            case .prefillProgress(let progress):
                if config.logRequests {
                    logger.info("prefill \(progress.stage.rawValue, privacy: .public) \(progress.completedUnitCount)/\(progress.totalUnitCount)")
                }
            case .info(let completionInfo):
                info = completionInfo
            }
        }
        await task.value

        if let info {
            run.promptTokenCount = info.promptTokenCount
            run.completionTokenCount = info.generationTokenCount
            run.decodeTokensPerSecond = info.tokensPerSecond
            run.promptTokensPerSecond = info.promptTokensPerSecond
            run.prefillMilliseconds = info.promptTime * 1000
            run.generateMilliseconds = info.generateTime * 1000
            run.finishReason = Self.mapStopReason(info.stopReason, toolCallCount: run.toolCalls.count)
        } else {
            run.promptTokenCount = inputTokens.count
            run.completionTokenCount = run.text.isEmpty ? 0 : 1
        }
        if run.prefillMilliseconds == 0 { run.prefillMilliseconds = prefillWall * 1000 }
        run.wallMilliseconds = Date().timeIntervalSince(iterationStart) * 1000
        run.cacheHit = reuseSlot.reuseApplied
        run.text = run.text.trimmingCharacters(in: .whitespacesAndNewlines)

        return run
    }

    private func runStreamingLocked(
        request: ChatRequest,
        continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation
    ) async throws {
        let (template, tokens, context) = try await renderChatTemplate(
            messages: request.messages,
            tools: request.tools,
            enableThinking: requestEnableThinking(request),
            reasoningEffort: request.reasoningEffort,
            toolChoice: request.toolChoice)
        let parameters = try await makeParameters(
            tokens: tokens, request: request, templateCount: template.count, context: context)
        let slot = SlotDecision.make(
            newTokens: tokens,
            slotTokens: config.cacheReuse ? slotTokens : [],
            maxSlotTokens: config.maxCacheSlotTokens)
        let tokenizer = await container.tokenizer
        let modelConfiguration = await container.configuration

        let inputTokens: [Int] = slot.reuseApplied
            ? Array(tokens.suffix(tokens.count - slot.cachedTokenCount))
            : tokens
        let input = LMInput(
            tokens: MLXArray(inputTokens),
            tokenIds: inputTokens,
            toolSchemas: MessageMapping.templateTools(request.tools))

        let modelBox: MeiBox<any LanguageModel> = await container.perform { context in
            MeiBox(context.model)
        }
        let model = modelBox.value

        let reuseCache = slot.reuseApplied ? slotCache : nil
        let cache: [KVCache]
        if let reuseCache {
            cache = reuseCache
        } else {
            cache = model.newCache(parameters: parameters)
        }
        iterationCache = cache

        let iterationStart = Date()
        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: cache,
            parameters: parameters)
        let prefillWall = Date().timeIntervalSince(iterationStart)

        let (stream, task) = MLXLMCommon.generateTask(
            promptTokenCount: inputTokens.count,
            modelConfiguration: modelConfiguration,
            tokenizer: tokenizer,
            iterator: iterator,
            toolSchemas: MessageMapping.templateTools(request.tools))

        var run = GenerationRun()
        run.cachedTokenCount = slot.reuseApplied ? slot.cachedTokenCount : 0
        var info: GenerateCompletionInfo?
        for await item in stream {
            switch item {
            case .chunk(let chunk):
                run.text += chunk
                continuation.yield(.chunk(chunk))
            case .reasoning(let reason):
                run.reasoning += reason
                continuation.yield(.reasoning(reason))
            case .toolCall(let call):
                let emitting = GenerationRun.ToolCallEmitting(
                    id: call.id ?? "call_\(UUID().uuidString.lowercased().prefix(12))",
                    name: call.function.name,
                    argumentsJSON: Self.toolArgumentsJSON(call))
                run.toolCalls.append(emitting)
                continuation.yield(.toolCall(emitting))
            case .toolCallProgress:
                break
            case .prefillProgress(let progress):
                if config.logRequests {
                    logger.info("prefill \(progress.stage.rawValue, privacy: .public) \(progress.completedUnitCount)/\(progress.totalUnitCount)")
                }
                continuation.yield(.prefill(completed: progress.completedUnitCount, total: progress.totalUnitCount))
            case .info(let completionInfo):
                info = completionInfo
            }
        }
        await task.value

        if let info {
            run.promptTokenCount = info.promptTokenCount
            run.completionTokenCount = info.generationTokenCount
            run.decodeTokensPerSecond = info.tokensPerSecond
            run.promptTokensPerSecond = info.promptTokensPerSecond
            run.prefillMilliseconds = info.promptTime * 1000
            run.generateMilliseconds = info.generateTime * 1000
            run.finishReason = Self.mapStopReason(info.stopReason, toolCallCount: run.toolCalls.count)
        } else {
            run.promptTokenCount = inputTokens.count
            run.completionTokenCount = run.text.isEmpty ? 0 : 1
        }
        if run.prefillMilliseconds == 0 { run.prefillMilliseconds = prefillWall * 1000 }
        run.wallMilliseconds = Date().timeIntervalSince(iterationStart) * 1000
        run.cacheHit = slot.reuseApplied
        run.text = run.text.trimmingCharacters(in: .whitespacesAndNewlines)

        continuation.yield(.finish(run))
        adoptSlot(tokens: tokens, cache: cache, reuseApplied: slot.reuseApplied)
        continuation.finish()
    }

    // MARK: - Text completion (/v1/completions)

    public func completionRun(request: CompletionRequest) async throws -> GenerationRun {
        try await serialized {
            try await self.completionRunLocked(request: request)
        }
    }

    private func completionRunLocked(request: CompletionRequest) async throws -> GenerationRun {
        let tokenizer = await container.tokenizer
        let tokens = tokenizer.encode(text: request.prompt)
        guard !tokens.isEmpty else { throw EngineError.emptyPrompt }
        let parameters = try await makeCompletionParameters(tokens: tokens, request: request)

        // Raw completion: no slot reuse; a fresh cache per request.
        let input = LMInput(tokens: MLXArray(tokens), tokenIds: tokens)
        let modelConfiguration = await container.configuration
        let modelBox: MeiBox<any LanguageModel> = await container.perform { context in
            MeiBox(context.model)
        }
        let model = modelBox.value
        let cache = model.newCache(parameters: parameters)
        iterationCache = cache

        let iterator = try TokenIterator(
            input: input, model: model, cache: cache, parameters: parameters)
        let (stream, task) = MLXLMCommon.generateTask(
            promptTokenCount: tokens.count,
            modelConfiguration: modelConfiguration,
            tokenizer: tokenizer,
            iterator: iterator,
            toolSchemas: nil)

        var run = GenerationRun()
        var info: GenerateCompletionInfo?
        for await item in stream {
            switch item {
            case .chunk(let chunk): run.text += chunk
            case .toolCall(let call):
                run.toolCalls.append(.init(
                    id: call.id ?? "call_\(UUID().uuidString.lowercased().prefix(12))",
                    name: call.function.name,
                    argumentsJSON: Self.toolArgumentsJSON(call)))
            case .info(let completionInfo): info = completionInfo
            default: break
            }
        }
        await task.value
        if let info {
            run.promptTokenCount = info.promptTokenCount
            run.completionTokenCount = info.generationTokenCount
            run.decodeTokensPerSecond = info.tokensPerSecond
            run.promptTokensPerSecond = info.promptTokensPerSecond
            run.prefillMilliseconds = info.promptTime * 1000
            run.generateMilliseconds = info.generateTime * 1000
            run.finishReason = Self.mapStopReason(info.stopReason, toolCallCount: run.toolCalls.count)
        } else {
            run.promptTokenCount = tokens.count
        }
        run.text = run.text.trimmingCharacters(in: .whitespacesAndNewlines)
        return run
    }

    // MARK: - Parameters

    private func makeParameters(
        tokens: [Int],
        request: ChatRequest,
        templateCount: Int,
        context: [String: any Sendable]?
    ) async throws -> GenerateParameters {
        guard tokens.count <= config.contextCap else {
            throw EngineError.overContextCap(promptTokens: tokens.count, cap: config.contextCap)
        }
        var parameters = GenerateParameters()
        parameters.prefillStepSize = config.prefillStepSize
        parameters.maxKVSize = config.maxKVSize
        if let kvBits = config.kvBits {
            parameters.kvBits = kvBits
            parameters.kvGroupSize = config.kvGroupSize
            parameters.quantizedKVStart = config.quantizedKVStart
        }
        parameters.temperature = Float(request.temperature ?? Double(config.temperature))
        parameters.topP = Float(request.topP ?? Double(config.topP))
        parameters.topK = request.topK ?? config.topK
        parameters.minP = Float(request.minP ?? Double(config.minP))
        if let penalty = request.repetitionPenalty ?? config.repetitionPenalty.map(Double.init) {
            parameters.repetitionPenalty = Float(penalty)
        }
        if let penalty = request.presencePenalty ?? config.presencePenalty.map(Double.init) {
            parameters.presencePenalty = Float(penalty)
        }
        if let penalty = request.frequencyPenalty ?? config.frequencyPenalty.map(Double.init) {
            parameters.frequencyPenalty = Float(penalty)
        }
        if let seed = request.seed {
            parameters.randomSeed = seed
        }
        if let stop = request.stop, !stop.isEmpty {
            parameters.extraStopStrings = stop
        }
        let requestedMax = request.maxTokens ?? config.maxTokensDefault
        let capacity = config.maxKVSize - tokens.count
        parameters.maxTokens = min(requestedMax, max(1, capacity))
        return parameters
    }

    private func makeCompletionParameters(
        tokens: [Int],
        request: CompletionRequest
    ) async throws -> GenerateParameters {
        guard tokens.count <= config.contextCap else {
            throw EngineError.overContextCap(promptTokens: tokens.count, cap: config.contextCap)
        }
        var parameters = GenerateParameters()
        parameters.prefillStepSize = config.prefillStepSize
        parameters.maxKVSize = config.maxKVSize
        if let kvBits = config.kvBits {
            parameters.kvBits = kvBits
            parameters.kvGroupSize = config.kvGroupSize
            parameters.quantizedKVStart = config.quantizedKVStart
        }
        parameters.temperature = Float(request.temperature ?? Double(config.temperature))
        parameters.topP = Float(request.topP ?? Double(config.topP))
        parameters.topK = request.topK ?? config.topK
        parameters.minP = Float(request.minP ?? Double(config.minP))
        if let penalty = request.repetitionPenalty ?? config.repetitionPenalty.map(Double.init) {
            parameters.repetitionPenalty = Float(penalty)
        }
        if let penalty = request.presencePenalty ?? config.presencePenalty.map(Double.init) {
            parameters.presencePenalty = Float(penalty)
        }
        if let penalty = request.frequencyPenalty ?? config.frequencyPenalty.map(Double.init) {
            parameters.frequencyPenalty = Float(penalty)
        }
        if let seed = request.seed {
            parameters.randomSeed = seed
        }
        if let stop = request.stop, !stop.isEmpty {
            parameters.extraStopStrings = stop
        }
        let requestedMax = request.maxTokens ?? config.maxTokensDefault
        let capacity = config.maxKVSize - tokens.count
        parameters.maxTokens = min(requestedMax, max(1, capacity))
        return parameters
    }

    // MARK: - Stop reason mapping

    public static func mapStopReason(_ reason: GenerateStopReason, toolCallCount: Int) -> String {
        if toolCallCount > 0 { return "tool_calls" }
        switch reason {
        case .stop: return "stop"
        case .length: return "length"
        case .cancelled: return "stop"
        @unknown default: return "stop"
        }
    }
}

/// Cache-reuse decision: whether the new rendered token sequence exactly
/// extends the previous slot, and how many prefix tokens the cache already
/// covers. Pure logic, kept out of the actor for direct unit testing.
public enum SlotDecision {
    public struct Result: Sendable, Equatable {
        public var reuseApplied: Bool
        public var cachedTokenCount: Int

        public init(reuseApplied: Bool, cachedTokenCount: Int) {
            self.reuseApplied = reuseApplied
            self.cachedTokenCount = cachedTokenCount
        }
    }

    public static func make(
        newTokens: [Int],
        slotTokens: [Int],
        maxSlotTokens: Int
    ) -> Result {
        // 1) The slot must exist and be non-empty.
        // 2) The new render must strictly extend it (more tokens + identical
        //    prefix). Exact extension only: any divergence at the seam means
        //    the transcript changed and prefix KV is no longer positionally
        //    valid, so fall back to a fresh prefill (always correct).
        // 3) The result must stay inside the shared budget.
        guard !slotTokens.isEmpty,
            newTokens.count > slotTokens.count,
            newTokens.count <= maxSlotTokens,
            newTokens.prefix(slotTokens.count).elementsEqual(slotTokens)
        else {
            return Result(reuseApplied: false, cachedTokenCount: 0)
        }
        return Result(reuseApplied: true, cachedTokenCount: slotTokens.count)
    }
}