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

    // In-process prefix reuse is owned by the container's CacheCoordinator
    // (see Engine.load): hash-chained paged KV blocks plus hybrid companion
    // state (Ornith GatedDelta), with the chat template's generation-prompt
    // suffix stripped at store time so growing transcripts hit.
    private var queueTail: Task<Void, Never>?

    public init(container: ModelContainer, config: ServerConfig, loadMemory: Memory.Snapshot? = nil) {
        self.container = container
        self.config = config
        self.loadMemory = loadMemory
    }

    /// Load a model from a local directory (config.json + safetensors +
    /// tokenizer files), using the pinned vmlx-swift stack.
    public static func load(config: ServerConfig) async throws -> Engine {
        let directory = URL(fileURLWithPath: config.modelDirectory, isDirectory: true)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw EngineError.modelDirectoryMissing(config.modelDirectory)
        }
        // A reported memory limit below the model's working set makes MLX
        // alloc calls wait on scheduled tasks instead of failing — the
        // "hang" failure mode. If the operator configured an explicit limit
        // (bytes), apply it before the allocator starts seeing load traffic.
        if config.memoryLimitBytes > 0 {
            Memory.memoryLimit = config.memoryLimitBytes
            if config.cacheLimitBytes > 0 {
                Memory.cacheLimit = config.cacheLimitBytes
            }
        }
        let container = try await loadModelContainer(
            from: directory,
            using: #huggingFaceTokenizerLoader(),
            loadConfiguration: LoadConfiguration(useMmapSafetensors: config.useMmapSafetensors)
        )
        // In-process multi-tier KV reuse (the benchmark's growing-transcript
        // pattern): the coordinator owns prefix matching, block hashing, and
        // hybrid companion-state (GatedDelta/SSM) restore across requests —
        // the machinery the hand-rolled exact-extension slot could not
        // provide for hybrid models. Paged (in-memory) tier on, disk tier
        // off: this server is single-process and the plan explicitly defers
        // cross-restart persistence. pagedBlockSize 16 follows upstream's
        // hybrid guidance (short system-only prefixes still store blocks).
        if config.cacheReuse {
            let diskEnabled = !config.kvCacheDir.isEmpty
            let diskCacheDir: URL? = diskEnabled
                ? URL(fileURLWithPath: config.kvCacheDir, isDirectory: true)
                : nil
            if diskEnabled {
                try? FileManager.default.createDirectory(
                    at: diskCacheDir!, withIntermediateDirectories: true)
            }
            await container.enableCachingAsync(config: CacheCoordinatorConfig(
                usePagedCache: true,
                enableDiskCache: diskEnabled,
                pagedBlockSize: 16,
                maxCacheBlocks: 8192,
                diskCacheDir: diskCacheDir,
                enableSSMReDerive: config.enableSSMReDerive,
                modelKey: config.servedModelID))
            let topology = await container.cacheTopologySnapshot()
            let tier = diskEnabled ? "disk at \(config.kvCacheDir)" : "no disk"
            print("mei: prefix cache enabled (paged in-memory + \(tier)); topology \(topology.topologyTags.joined(separator: " "))")
        } else {
            print("mei: prefix cache disabled (--cache-reuse false)")
        }
        fflush(stdout)
        // Peak memory is program-wide; reset it after weights are resident so
        // per-run peak numbers describe inference, not tokenizer/model init.
        Memory.peakMemory = 0
        // loadModelContainer's LoadConfiguration applies its own memoryLimit
        // (default .fraction(0.70) of physical RAM, ~22.4GB here) over any
        // limit set before load. Re-assert the operator's explicit limit so
        // an explicit --memory-limit-bytes >= working set actually governs
        // inference (otherwise the 35B at 24GB active would hang alloca on
        // the 22.4GB default).
        if config.memoryLimitBytes > 0 {
            Memory.memoryLimit = config.memoryLimitBytes
            if config.cacheLimitBytes > 0 {
                Memory.cacheLimit = config.cacheLimitBytes
            }
        }
        return Engine(container: container, config: config, loadMemory: Memory.snapshot())
    }

    public var servedModelID: String { config.servedModelID }

    /// MLX allocator snapshot immediately after the weights are resident.
    public private(set) var loadMemory: Memory.Snapshot?

    /// Current allocator + device report for /v1/mei/status. Uses the MLX
    /// `Memory`/`GPU` APIs (the nonexistent `get_physical_memory` Cmlx entry
    /// is NOT used).
    /// Nonisolated: reads are cheap thread-safe globals, and routing this
    /// through the actor would starve during a long generation (the actor
    /// spends long synchronous stretches inside MLX eval calls).
    public nonisolated static func liveMemoryReport() -> MeiMemoryReport {
        let snapshot = Memory.snapshot()
        let info = GPU.deviceInfo()
        return MeiMemoryReport(
            activeBytes: snapshot.activeMemory,
            cacheBytes: snapshot.cacheMemory,
            peakBytes: snapshot.peakMemory,
            memoryLimitBytes: Memory.memoryLimit,
            cacheLimitBytes: Memory.cacheLimit,
            recommendedWorkingSetBytes: GPU.maxRecommendedWorkingSetBytes(),
            device: .init(architecture: info.architecture, memoryBytes: info.memorySize))
    }

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

    /// Live coordinator counters for /v1/mei/status (nil when cache reuse is
    /// disabled): paged hit/miss/eviction and SSM companion stats.
    public func cacheStats() -> CacheCoordinatorStatsSnapshot? {
        container.cacheCoordinator?.snapshotStats()
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
        let anchors = try await ssmAnchorOffsets(
            template: template, tools: request.tools,
            context: context, fullTokenCount: tokens.count)
        let parameters = try await makeParameters(
            tokens: tokens, request: request, templateCount: template.count,
            context: context, anchorOffsets: anchors)
        if config.logRequests {
            print("mei: chat request tokens \(tokens.count)")
            fflush(stdout)
        }
        let run = try await generateLocked(
            tokens: tokens,
            parameters: parameters,
            tools: request.tools)
        return run
    }

    private func requestEnableThinking(_ request: ChatRequest) -> Bool? {
        if let reasoningEffort = request.reasoningEffort, config.enableThinking == nil {
            return reasoningEffort != "none"
        }
        return config.enableThinking
    }


    /// Serialize a parsed vmlx ToolCall's arguments to a compact JSON string,
    /// applying schema-aware normalization so `number`/`integer` fields are
    /// reported as JSON numbers even when the model emitted them as numeric
    /// strings (Gemma-4). Builds from the parser's typed dictionary rather than
    /// the verbatim raw bytes so numeric typing is deterministic across model
    /// paths, and falls back to the raw protocol text only when the parser
    /// produced no dict.
    private static func toolArgumentsJSON(
        _ call: ToolCall,
        tools: [MeiJSONValue]?
    ) -> String {
        var parsed: MeiJSONValue
        if !call.function.arguments.isEmpty {
            let arguments = call.function.arguments.mapValues { MeiJSONValue.from($0.anyValue) }
            parsed = .object(arguments)
        } else if let raw = call.function.rawArgumentsJSON, !raw.isEmpty,
            let rawParsed = MeiJSONValue.parseObject(from: raw) {
            parsed = rawParsed
        } else {
            return "{}"
        }
        let schema = Self.toolParametersSchema(for: call.function.name, tools: tools)
        let normalized = ToolArgumentNormalizer.normalize(
            arguments: parsed, parametersSchema: schema)
        return (try? normalized.jsonString()) ?? "{}"
    }

    /// Look up a tool's `parameters` schema object by function name in the
    /// request's `tools` array, so argument normalization is schema-driven.
    private static func toolParametersSchema(
        for name: String,
        tools: [MeiJSONValue]?
    ) -> MeiJSONValue? {
        guard let tools else { return nil }
        for tool in tools {
            guard case .object(let toolObject) = tool,
                case .object(let function)? = toolObject["function"],
                case .string(let functionName)? = function["name"]
            else { continue }
            if functionName == name {
                return function["parameters"]
            }
        }
        return nil
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
        parameters: GenerateParameters,
        tools: [MeiJSONValue]?
    ) async throws -> GenerationRun {
        // Emit the token array as `[1, T]` (batch-first), matching the raw
        // completions path and the vmlx cache-restore rebuild. A multimodal
        // bundle routed through the loader's VLM-first registry (Gemma4 with a
        // bundled vision tower, Qwen3.5/3.8 with vision_config) runs its VLM
        // `prepare`, which embeds `tokens` directly; a 1-D array yields a 2-D
        // embedding whose `dim(1)` is the hidden size and chunked prefill then
        // slices out of rank -> precondition crash (Gemma4.prepare measured
        // 2026-09-03, 19-token chat prompt). Text-only models take the
        // rank-safe LLM default prepare, which flattens [1, T] and only
        // rejects batch > 1, so [1, T] is safe for every model class.
        let input = LMInput(
            tokens: MLXArray(tokens).expandedDimensions(axis: 0),
            tokenIds: tokens,
            toolSchemas: MessageMapping.templateTools(tools))

        let modelBox: MeiBox<any LanguageModel> = await container.perform { context in
            MeiBox(context.model)
        }
        let model = modelBox.value
        let restoreBox = RestoreBox()
            let logProgress = self.config.logRequests
        let iterationStart = Date()
        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: model.newCache(parameters: parameters),
            parameters: parameters,
            cacheCoordinator: container.cacheCoordinator,
            prefillProgressHandler: { progress in
            if logProgress {
                print("mei: pp stage=\(progress.stage.rawValue) completed=\(progress.completedUnitCount) total=\(progress.totalUnitCount)")
                fflush(stdout)
            }
            restoreBox.tracker.observe(progress) })
        let (stream, task) = MLXLMCommon.generateTask(
            promptTokenCount: tokens.count,
            modelConfiguration: await container.configuration,
            tokenizer: await container.tokenizer,
            iterator: iterator,
            toolSchemas: MessageMapping.templateTools(tools))

        var run = GenerationRun()
        var restoreTracker = restoreBox.tracker
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
                    argumentsJSON: Self.toolArgumentsJSON(call, tools: tools)))
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
            run.promptTokenCount = tokens.count
            run.completionTokenCount = run.text.isEmpty ? 0 : 1
        }
        run.cachedTokenCount = restoreTracker.restoredTokens
        run.cacheHit = restoreTracker.isCacheHit
        if run.prefillMilliseconds == 0 { run.prefillMilliseconds = Date().timeIntervalSince(iterationStart) * 1000 }
        run.wallMilliseconds = Date().timeIntervalSince(iterationStart) * 1000
        run.text = run.text.trimmingCharacters(in: .whitespacesAndNewlines)
        captureRunMemory(&run)

        if config.logRequests {
            print("mei: run tokens \(run.promptTokenCount) cached \(run.cachedTokenCount) decode \(String(format: "%.1f", run.decodeTokensPerSecond)) tok/s")
            fflush(stdout)
        }
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
        let anchors = try await ssmAnchorOffsets(
            template: template, tools: request.tools,
            context: context, fullTokenCount: tokens.count)
        let parameters = try await makeParameters(
            tokens: tokens, request: request, templateCount: template.count,
            context: context, anchorOffsets: anchors)
        // Batch-first `[1, T]` tokens like generateLocked and the raw path — see
        // the comment there; Gemma4's VLM prepare crashed on 1-D chat tokens.
        let input = LMInput(
            tokens: MLXArray(tokens).expandedDimensions(axis: 0),
            tokenIds: tokens,
            toolSchemas: MessageMapping.templateTools(request.tools))

        let modelBox: MeiBox<any LanguageModel> = await container.perform { context in
            MeiBox(context.model)
        }
        let model = modelBox.value
        let restoreBox = RestoreBox()
            let logProgress = self.config.logRequests
        let iterationStart = Date()
        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: model.newCache(parameters: parameters),
            parameters: parameters,
            cacheCoordinator: container.cacheCoordinator,
            prefillProgressHandler: { progress in
            if logProgress {
                print("mei: pp stage=\(progress.stage.rawValue) completed=\(progress.completedUnitCount) total=\(progress.totalUnitCount)")
                fflush(stdout)
            }
            restoreBox.tracker.observe(progress) })
        let (stream, task) = MLXLMCommon.generateTask(
            promptTokenCount: tokens.count,
            modelConfiguration: await container.configuration,
            tokenizer: await container.tokenizer,
            iterator: iterator,
            toolSchemas: MessageMapping.templateTools(request.tools))

        var run = GenerationRun()
        var restoreTracker = restoreBox.tracker
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
                    argumentsJSON: Self.toolArgumentsJSON(call, tools: request.tools))
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
            run.promptTokenCount = tokens.count
            run.completionTokenCount = run.text.isEmpty ? 0 : 1
        }
        run.cachedTokenCount = restoreTracker.restoredTokens
        run.cacheHit = restoreTracker.isCacheHit
        if run.prefillMilliseconds == 0 { run.prefillMilliseconds = Date().timeIntervalSince(iterationStart) * 1000 }
        run.wallMilliseconds = Date().timeIntervalSince(iterationStart) * 1000
        run.text = run.text.trimmingCharacters(in: .whitespacesAndNewlines)
        captureRunMemory(&run)

        continuation.yield(.finish(run))
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

        // Raw completions participate in the same paged prefix cache as chat:
        // any request whose token stream extends a stored prefix resumes from
        // the coordinator's KV and only the new tokens are prefilled.
        //
        // Emit the token array as `[1, T]` (batch-first), matching the shape
        // the cache-restore path already rebuilds for VLM-routed bundles
        // (vmlx Evaluate.swift "Rebuild inputForPrepare with tokens shaped as
        // [1, T]"). A multimodal bundle such as mlx-community/Qwen3.8-27B-4bit
        // (vision_config + processor files) routes to MLXVLM.Qwen35 through
        // the loader's VLM-first factory registry, and its `prepare` reads
        // `tokens.dim(1)` unconditionally — a 1-D token array dies there with
        // `Fatal error: SmallVector out of range` (mlx/c/array.cpp:335) at ANY
        // prompt length (measured 11/60/30k/65k tokens). Text-only bundles
        // (Mei-produced 5-bit) load the rank-safe MLXLLM default prepare,
        // which flattens and only rejects batch > 1, so [1, T] is safe for
        // both model classes.
        let input = LMInput(
            tokens: MLXArray(tokens).expandedDimensions(axis: 0),
            tokenIds: tokens)

        let modelBox: MeiBox<any LanguageModel> = await container.perform { context in
            MeiBox(context.model)
        }
        let model = modelBox.value
        let restoreBox = RestoreBox()
            let logProgress = self.config.logRequests
        let iterationStart = Date()
        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: model.newCache(parameters: parameters),
            parameters: parameters,
            cacheCoordinator: container.cacheCoordinator,
            prefillProgressHandler: { progress in
            if logProgress {
                print("mei: pp stage=\(progress.stage.rawValue) completed=\(progress.completedUnitCount) total=\(progress.totalUnitCount)")
                fflush(stdout)
            }
            restoreBox.tracker.observe(progress) })
        let (stream, task) = MLXLMCommon.generateTask(
            promptTokenCount: tokens.count,
            modelConfiguration: await container.configuration,
            tokenizer: await container.tokenizer,
            iterator: iterator,
            toolSchemas: nil)

        var run = GenerationRun()
        var restoreTracker = restoreBox.tracker
        var info: GenerateCompletionInfo?
        for await item in stream {
            switch item {
            case .chunk(let chunk): run.text += chunk
            case .toolCall(let call):
                run.toolCalls.append(.init(
                    id: call.id ?? "call_\(UUID().uuidString.lowercased().prefix(12))",
                    name: call.function.name,
                    argumentsJSON: Self.toolArgumentsJSON(call, tools: nil)))
            case .prefillProgress(let progress):
                if config.logRequests {
                    logger.info("prefill \(progress.stage.rawValue, privacy: .public) \(progress.completedUnitCount)/\(progress.totalUnitCount)")
                }
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
            run.completionTokenCount = run.text.isEmpty ? 0 : 1
        }
        run.cachedTokenCount = restoreTracker.restoredTokens
        run.cacheHit = restoreTracker.isCacheHit
        if run.prefillMilliseconds == 0 { run.prefillMilliseconds = Date().timeIntervalSince(iterationStart) * 1000 }
        run.wallMilliseconds = Date().timeIntervalSince(iterationStart) * 1000
        run.text = run.text.trimmingCharacters(in: .whitespacesAndNewlines)
        captureRunMemory(&run)
        return run
    }

    // MARK: - Parameters

    /// Deterministic early role-turn anchor offsets for the SSM companion
    /// store (patch 0005; default [] = upstream behavior). Computed from
    /// the request's OWN rendering path: same tokenizer, same tool schema,
    /// same additional context — so the additivity self-check inside
    /// `SSMAnchorBoundaries.compute` reproduces the request tokens
    /// exactly. Non-additive transcripts fall back to [] (always correct).
    private func ssmAnchorOffsets(
        template: [[String: any Sendable]],
        tools: [MeiJSONValue]?,
        context: [String: any Sendable]?,
        fullTokenCount: Int
    ) async throws -> [Int] {
        let k = config.ssmAnchorBoundaryCount
        guard k > 0 else { return [] }
        let tokenizer = await container.tokenizer
        let templateTools = MessageMapping.templateTools(tools)
        let result = try SSMAnchorBoundaries.compute(
            template: template,
            fullTokenCount: fullTokenCount,
            k: k
        ) { prefixCount in
            try tokenizer.applyChatTemplate(
                messages: Array(template.prefix(prefixCount)),
                tools: templateTools,
                additionalContext: context
            ).count
        }
        if let warning = result.warning {
            print("mei: ssm-anchor-boundaries disabled for this transcript: \(warning)")
        } else if !result.offsets.isEmpty {
            print("mei: ssm anchor boundaries (k=\(k)): \(result.offsets)")
        }
        fflush(stdout)
        return result.offsets
    }

    private func makeParameters(
        tokens: [Int],
        request: ChatRequest,
        templateCount: Int,
        context: [String: any Sendable]?,
        anchorOffsets: [Int] = []
    ) async throws -> GenerateParameters {
        guard tokens.count <= config.contextCap else {
            throw EngineError.overContextCap(promptTokens: tokens.count, cap: config.contextCap)
        }
        var parameters = GenerateParameters()
        parameters.prefillStepSize = config.prefillStepSize
        parameters.maxKVSize = config.maxKVSize
        parameters.enableCompiledDecode = config.enableCompiledDecode
        parameters.compiledDecodeMaxPromptOffset = config.compiledDecodeMaxPromptOffset
        if config.maxKVWindowSize > 0 {
            parameters.maxKVWindowSize = config.maxKVWindowSize
        }
        if !anchorOffsets.isEmpty {
            parameters.ssmAnchorBoundaries = anchorOffsets
        }
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
        parameters.enableCompiledDecode = config.enableCompiledDecode
        parameters.compiledDecodeMaxPromptOffset = config.compiledDecodeMaxPromptOffset
        if config.maxKVWindowSize > 0 {
            parameters.maxKVWindowSize = config.maxKVWindowSize
        }
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

    /// Patch a run with the MLX allocator snapshot captured at completion.
    private func captureRunMemory(_ run: inout GenerationRun) {
        let snapshot = Memory.snapshot()
        run.memoryActiveBytes = snapshot.activeMemory
        run.memoryCacheBytes = snapshot.cacheMemory
        run.memoryPeakBytes = snapshot.peakMemory
    }

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

/// Thread-safe box for the per-request cache-restore tracker, driven by
/// `TokenIterator`'s prefillProgressHandler (the solo generate path emits
/// no `.prefillProgress` stream events; the handler receives them directly).
final class RestoreBox: @unchecked Sendable {
    var tracker = CacheRestoreTracker()
}

/// Tracks how many prompt tokens were restored from the coordinator's prefix
/// cache during one generation, from the `.prefillProgress` event stream.
/// Pure logic, kept out of the actor for direct unit testing.
///
/// The coordinator emits a `.cacheRestore`-stage frame carrying the matched
/// prefix length, then `.prefill` frames, then `.complete`. On a full
/// (uncached) prefill the first `.prefill` frame reports 0 completed tokens,
/// so no hit is inferred.
public struct CacheRestoreTracker: Sendable {
    public private(set) var restoredTokens = 0
    private var firstPrefillFrameSeen = false

    public mutating func observe(_ progress: PrefillProgress) {
        switch progress.stage {
        case .cacheRestore:
            restoredTokens = max(restoredTokens, progress.completedUnitCount)
        case .prefill:
            if !firstPrefillFrameSeen {
                firstPrefillFrameSeen = true
                if progress.completedUnitCount > 0 {
                    restoredTokens = max(restoredTokens, progress.completedUnitCount)
                }
            }
        default:
            break
        }
    }

    /// Drive the tracker from plain stage names (used by unit tests; the
    /// server path calls `observe(_ progress: PrefillProgress)`).
    public mutating func observe(stage: String, completed: Int) {
        switch stage {
        case "cacheRestore":
            restoredTokens = max(restoredTokens, completed)
        case "prefill":
            if !firstPrefillFrameSeen {
                firstPrefillFrameSeen = true
                if completed > 0 {
                    restoredTokens = max(restoredTokens, completed)
                }
            }
        default:
            break
        }
    }

    public var isCacheHit: Bool { restoredTokens > 0 }
}
