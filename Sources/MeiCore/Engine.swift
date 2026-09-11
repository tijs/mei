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
    case toolCall(index: Int, call: GenerationRun.ToolCallEmitting)
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
    /// The tokens this chat template appends after the history to open the
    /// assistant turn, learned from the first request that renders both forms.
    /// Lets every later turn find its history boundary with array work instead
    /// of a second full render. See `historyBoundary`.
    private var generationPromptSuffix: [Int] = []
    /// Memo for `ssmAnchorOffsets`, keyed by the token prefix the offsets were
    /// derived from, so a continuing conversation never recomputes them.
    private var anchorMemo: (prefixLength: Int, hash: Int, offsets: [Int])?
    /// The longest prefix every sizeable prompt so far has agreed on. Converges
    /// to the caller's stable header. See `adaptiveStableBoundary`.
    private var previousPromptTokens: [Int] = []
    /// A boundary must cover at least this many tokens to be worth storing: the
    /// snapshot costs disk and a store, and reusing a short prefix saves less
    /// than the bookkeeping costs.
    private static let adaptiveStableMinimumTokens = 1024
    /// Boundaries are rounded DOWN to a multiple of this so that prompts whose
    /// shared prefixes differ slightly still agree on one boundary. Measured on
    /// the coding suites, the longest common prefix at consecutive task
    /// transitions was 5770, 5681, 5752, 6216, 5729, 5670, 5718, 5674, 5663,
    /// 5683, 5756 — every pair genuinely shares over 5.6k tokens, but no two
    /// agree on the exact number, so an unrounded boundary is stored at one
    /// offset and probed at another and never matches. At 512, ten of those
    /// eleven collapse onto 5632.
    /// EXPERIMENT (2026-09-11): 1 disables quantization.
    ///
    /// Quantization was added so that store and probe agreed despite the
    /// longest-common-prefix differing per PAIR (5770, 5681, 5752 … at real
    /// task transitions). Converging a candidate across all prompts already
    /// stabilises that value, so the rounding may now be redundant — and it is
    /// the step that moves the boundary off the point where content actually
    /// diverges and onto an arbitrary offset. Arbitrary boundaries restore
    /// unfaithfully (0/5) while structural ones do not (5/5, and still exact
    /// with 422 tokens of tail), so the rounding is the prime suspect.
    private static let adaptiveStableQuantum = 1
    /// How far short of the previous prompt the shared prefix must fall before
    /// this counts as a new conversation rather than another turn of the same
    /// one. Only needs to exceed the chat template's generation-prompt suffix
    /// (5 tokens on Ornith); 64 leaves room without risking a real divergence
    /// being read as a continuation.
    /// Prompts below this are ignored when converging the candidate. Between
    /// coding tasks hermes issues a ~500-token call of its own, and comparing
    /// against that instead of the previous real conversation is what made an
    /// earlier version of this never fire: EVERY cold prefill in a traced run
    /// was immediately preceded by a request of 470-516 tokens.
    private static let adaptiveStableFloorTokens = 2048
    /// The adaptive boundary already written to the cache. Re-offering it for
    /// STORING is very expensive: a restoring request starts prefill AT the
    /// boundary, so the prefill capture never crosses it, and the post-answer
    /// store falls back to rederiving it — measured at 9.8 s per task on a
    /// 3,072-token boundary, against the 7.2 s of prefill the restore saved.
    /// It stays in the FETCH list so later conversations still find it.
    private var storedAdaptiveBoundary: Int?

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
            loadConfiguration: LoadConfiguration(
                // MLXPress axis E (cold-weight tier). `.default` is
                // `.auto(envFallback: true)`: it honours an explicit `MLXPRESS=N`
                // (`0` disables; N in [0,95] sets coldFraction N/100) and otherwise
                // self-enables at coldFraction 0.70 only when the bundle is routed
                // (MoE) AND its raw bytes exceed 50% of physical memory — which is
                // exactly the 32 GB host / ~18 GiB qwen3_5_moe case.
                //
                // This was previously the init default `.disabled`, which
                // short-circuits before any environment lookup — so `MLXPRESS=…`
                // was silently inert on every Mei build to date.
                //
                // Runtime lifetime is handled: `ModelContext` stores
                // `jangPressRuntime`, so holding the container holds the tiers.
                jangPress: .default,
                useMmapSafetensors: config.useMmapSafetensors)
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
        var anchors = try await ssmAnchorOffsets(
            template: template, tools: request.tools,
            context: context, tokens: tokens)
        // Offer the prefix this prompt shares with earlier ones as an extra
        // boundary, so a client that varies its system-prompt tail per session
        // still gets cross-session reuse. See adaptiveStableBoundary.
        //
        // It goes in the FETCH list every time, but in the STORE list only
        // until it has actually been written once. Re-offering it for storing
        // costs a rederive on every later conversation — a restoring request
        // begins prefill at the boundary, so the prefill capture never crosses
        // it — measured at 9.8 s per task against the 7.2 s the restore saved.
        var adaptiveBoundary: Int? = nil
        var adaptiveNeedsStore = false
        if let shared = adaptiveStableBoundary(tokens: tokens), !anchors.contains(shared) {
            anchors = (anchors + [shared]).sorted()
            adaptiveBoundary = shared
            // ALWAYS keep it in the stable list. vmlx both stores AND probes
            // only boundaries listed there, so dropping it after the first
            // write silently disabled the FETCH: a coding stage went from 2
            // cold prefills to 17, worse than having no adaptive boundary at
            // all. The dedupe existed to avoid a rederive on every later
            // conversation, and vmlx 6c807ec6 removed that cost by making the
            // capture fire after a restore — so there is nothing left to
            // dedupe.
            adaptiveNeedsStore = true
            if config.logRequests {
                print("mei: adaptive boundary \(shared) (prompt \(tokens.count)"
                    + ", store=\(adaptiveNeedsStore))")
                fflush(stdout)
            }
        }
        let parameters = try await makeParameters(
            tokens: tokens, request: request, templateCount: template.count,
            context: context, anchorOffsets: anchors)
        let prefixCounts = await canonicalPrefixBoundaries(
            template: template, tools: request.tools, context: context,
            tokens: tokens, anchors: parameters.ssmAnchorBoundaries)
        if config.logRequests {
            print("mei: chat request tokens \(tokens.count)")
            fflush(stdout)
        }
        // Drop the adaptive boundary from the STORE list once it is written.
        let stableCounts: [Int]? = (adaptiveBoundary != nil && !adaptiveNeedsStore)
            ? anchors.filter { $0 != adaptiveBoundary! } : nil
        if adaptiveNeedsStore { storedAdaptiveBoundary = adaptiveBoundary }
        let run = try await generateLocked(
            tokens: tokens,
            parameters: parameters,
            tools: request.tools,
            cachePrefixCounts: prefixCounts,
            stablePrefixCounts: stableCounts)
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
        tools: [MeiJSONValue]?,
        cachePrefixCounts: [Int]? = nil,
        stablePrefixCounts: [Int]? = nil
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
        // SSM anchor offsets are the stable system+tools prefix boundaries
        // (see ssmAnchorOffsets). vmlx's post-answer boundary store loop
        // iterates `cachePrefixTokenCounts`, and for a hybrid cache it stores
        // ONLY boundaries also listed in `cacheStablePrefixTokenCounts` — the
        // field documented as "deliberately persisted for reuse by unrelated
        // new chat sessions". Passing neither is why every new conversation
        // re-prefilled the whole shared prefix: the only stored boundary was
        // the generation-stripped one, which already contains this turn's own
        // user tokens, so its key never matched another conversation.
        let input = LMInput(
            tokens: MLXArray(tokens).expandedDimensions(axis: 0),
            tokenIds: tokens,
            cachePrefixTokenCounts: cachePrefixCounts ?? parameters.ssmAnchorBoundaries,
            cacheStablePrefixTokenCounts: stablePrefixCounts ?? parameters.ssmAnchorBoundaries,
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
        RequestLog.record(run, kind: "chat")

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
        var anchors = try await ssmAnchorOffsets(
            template: template, tools: request.tools,
            context: context, tokens: tokens)
        // Offer the prefix this prompt shares with earlier ones as an extra
        // boundary, so a client that varies its system-prompt tail per session
        // still gets cross-session reuse. See adaptiveStableBoundary.
        //
        // It goes in the FETCH list every time, but in the STORE list only
        // until it has actually been written once. Re-offering it for storing
        // costs a rederive on every later conversation — a restoring request
        // begins prefill at the boundary, so the prefill capture never crosses
        // it — measured at 9.8 s per task against the 7.2 s the restore saved.
        var adaptiveBoundary: Int? = nil
        var adaptiveNeedsStore = false
        if let shared = adaptiveStableBoundary(tokens: tokens), !anchors.contains(shared) {
            anchors = (anchors + [shared]).sorted()
            adaptiveBoundary = shared
            // ALWAYS keep it in the stable list. vmlx both stores AND probes
            // only boundaries listed there, so dropping it after the first
            // write silently disabled the FETCH: a coding stage went from 2
            // cold prefills to 17, worse than having no adaptive boundary at
            // all. The dedupe existed to avoid a rederive on every later
            // conversation, and vmlx 6c807ec6 removed that cost by making the
            // capture fire after a restore — so there is nothing left to
            // dedupe.
            adaptiveNeedsStore = true
            if config.logRequests {
                print("mei: adaptive boundary \(shared) (prompt \(tokens.count)"
                    + ", store=\(adaptiveNeedsStore))")
                fflush(stdout)
            }
        }
        let parameters = try await makeParameters(
            tokens: tokens, request: request, templateCount: template.count,
            context: context, anchorOffsets: anchors)
        // Batch-first `[1, T]` tokens like generateLocked and the raw path — see
        // the comment there; Gemma4's VLM prepare crashed on 1-D chat tokens.
        // See the note in generateLocked: these two fields are what let the
        // shared system+tools prefix be stored for other conversations.
        // Same rule as the non-streaming path: keep the adaptive boundary for
        // fetching, drop it from the store list once written.
        let streamStableCounts: [Int] = (adaptiveBoundary != nil && !adaptiveNeedsStore)
            ? parameters.ssmAnchorBoundaries.filter { $0 != adaptiveBoundary! }
            : parameters.ssmAnchorBoundaries
        if adaptiveNeedsStore { storedAdaptiveBoundary = adaptiveBoundary }
        let streamPrefixCounts = await canonicalPrefixBoundaries(
            template: template, tools: request.tools, context: context,
            tokens: tokens, anchors: parameters.ssmAnchorBoundaries)
        let input = LMInput(
            tokens: MLXArray(tokens).expandedDimensions(axis: 0),
            tokenIds: tokens,
            cachePrefixTokenCounts: streamPrefixCounts,
            cacheStablePrefixTokenCounts: streamStableCounts,
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
                // Distinct tool calls must stream under distinct OpenAI indexes;
                // clients merge deltas keyed by index, so two calls at index 0
                // would otherwise concatenate name/arguments.
                let index = run.toolCalls.count
                run.toolCalls.append(emitting)
                continuation.yield(.toolCall(index: index, call: emitting))
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
        RequestLog.record(run, kind: "chat_stream")

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
        RequestLog.record(run, kind: "completion")
        return run
    }

    // MARK: - Parameters

    /// Deterministic early role-turn anchor offsets for the SSM companion
    /// store (patch 0005; default [] = upstream behavior). Computed from
    /// the request's OWN rendering path: same tokenizer, same tool schema,
    /// same additional context — so the additivity self-check inside
    /// `SSMAnchorBoundaries.compute` reproduces the request tokens
    /// exactly. Non-additive transcripts fall back to [] (always correct).
    /// The advancing per-turn history boundary, merged with the fixed SSM
    /// anchors, for `LMInput.cachePrefixTokenCounts`.
    ///
    /// vmlx documents two distinct lists. `cacheStablePrefixTokenCounts` is the
    /// FIXED system+tools prefix a different conversation can reuse. The other,
    /// `cachePrefixTokenCounts`, must contain the canonical no-generation-prompt
    /// history boundary of THIS turn, because `hybridStripBoundaryIndex` takes
    /// its maximum as the boundary to store after the answer.
    ///
    /// Mei passed the anchor list to both. The anchor never moves, so from the
    /// second turn on the stored boundary froze at the anchor and every
    /// subsequent turn re-prefilled the whole growing tail: measured on a
    /// 40k-token conversation, prefill 5.4 s, 10.0 s, 15.0 s, 20.1 s, 25.1 s,
    /// 30.7 s on turns 2-7, against a flat 5.4 s with anchors off. That is the
    /// multi-turn regression that kept `--ssm-anchor-boundaries` from being
    /// adopted, and it was never a trade-off — just this contract violation.
    ///
    /// Every boundary vmlx returns here is proven by token equality to be a
    /// real prefix of the active prompt, so a template that reorders or
    /// rewrites its history fails closed and contributes nothing.
    private func canonicalPrefixBoundaries(
        template: [[String: any Sendable]],
        tools: [MeiJSONValue]?,
        context: [String: any Sendable]?,
        tokens: [Int],
        anchors: [Int]
    ) async -> [Int] {
        guard !anchors.isEmpty else { return [] }
        let tokenizer = await container.tokenizer
        // Deliberately NOT canonicalChatCacheBoundaries(): that helper derives
        // several boundaries (static-system hints, trailing-continuation and
        // assistant-continuation probes) and renders the whole transcript four
        // or more times to do it. Measured on a 40k-token conversation, calling
        // it added 4.4 s to every turn -- the per-turn interval went from 1.45 s
        // to 5.8 s. Only ONE of its boundaries is needed here: the canonical
        // no-generation-prompt history boundary, which is a single render plus
        // the same exact-token-prefix proof the helper applies.
        let boundaries = historyBoundary(
            tokenizer: tokenizer,
            messages: template,
            tools: MessageMapping.templateTools(tools),
            additionalContext: context,
            promptTokens: tokens)
        if boundaries.isEmpty {
            // No advancing boundary available: the tokenizer cannot render
            // without a generation prompt, or the template is not
            // prefix-additive. Anchors alone would freeze the stored boundary,
            // so say so rather than silently regressing every later turn.
            print("mei: no canonical history boundary for this transcript; "
                + "anchors will not advance across turns")
            fflush(stdout)
            return anchors
        }
        return Set(anchors + boundaries).sorted()
    }

    /// The canonical no-generation-prompt history boundary, or `[]`.
    ///
    /// The boundary is the prompt minus whatever the template appends to open
    /// the assistant turn. The obvious way to find it is to render the
    /// transcript a second time with that rail suppressed, and that is what
    /// this did at first — but a render plus tokenize of a growing transcript
    /// is not cheap. Measured on a 40k-token conversation, the interval between
    /// one answer and the next request reaching the engine scaled almost
    /// exactly linearly with the number of full renders per turn: 0.49 s with
    /// none, 1.95 s with one, 5.75 s with four, i.e. ~1.45 s per render.
    ///
    /// So render only ONCE, on the first request, and learn the suffix from the
    /// difference. Every later turn just checks whether the prompt ends with
    /// that same suffix and subtracts its length.
    ///
    /// The tail check is the proof, and it is re-done on every turn: a template
    /// that renders a different rail for some turn shape simply fails it and
    /// falls back to rendering. Nothing is assumed about the template, and a
    /// wrong boundary could at worst cost a cache miss, never a bad restore —
    /// the stored KV state is a snapshot taken at that position of THIS prompt
    /// and is keyed by exactly those tokens, so it always describes itself.
    private func historyBoundary(
        tokenizer: any MLXLMCommon.Tokenizer,
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        promptTokens: [Int]
    ) -> [Int] {
        if !generationPromptSuffix.isEmpty,
            promptTokens.count > generationPromptSuffix.count,
            promptTokens.suffix(generationPromptSuffix.count)
                .elementsEqual(generationPromptSuffix)
        {
            return [promptTokens.count - generationPromptSuffix.count]
        }
        guard let controllable = tokenizer as? any GenerationPromptControllableTokenizer,
            let rendered = try? controllable.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
                addGenerationPrompt: false),
            !rendered.isEmpty,
            rendered.count < promptTokens.count,
            promptTokens.prefix(rendered.count).elementsEqual(rendered)
        else { return [] }
        generationPromptSuffix = Array(promptTokens.suffix(promptTokens.count - rendered.count))
        return [rendered.count]
    }

    /// The longest prefix this prompt shares with the previous one, when that
    /// is worth storing a snapshot for.
    ///
    /// WHY THIS EXISTS. Anchors are derived by divergence across message
    /// variants WITHIN one request, so the boundary always lands after the
    /// whole system message. When a client varies the TAIL of its system
    /// prompt per session — hermes appends `cwd` and `session_id` after its
    /// stable section — every new session's prompt shares a long head with the
    /// last one but hashes differently at the anchor, so nothing is ever
    /// reused. Measured on the coding suites: 16 cache misses, one per task,
    /// ~5,700 tokens each, 46.7% of all prefill time.
    ///
    /// The obvious fix — match the longest common prefix at fetch time and trim
    /// the stored state down to it — is impossible for this topology. Trimming
    /// means rewinding, and the 30 GatedDelta layers hold a recurrent state
    /// that is not invertible: `BaseKVCache.isTrimmable` is false and neither
    /// `ArraysCache` nor `MambaCache` overrides it. The state has to be STORED
    /// at a boundary later prompts will share, which means discovering that
    /// boundary before storing rather than after.
    ///
    /// So: compare against the previous prompt and offer their common prefix as
    /// an extra stable boundary. The first pair of sessions still pays a cold
    /// prefill — nothing is stored there yet — and every session after reuses.
    /// The split is discovered, not declared, so no client cooperation is
    /// needed.
    private func adaptiveStableBoundary(tokens: [Int]) -> Int? {
        // Ignore the small interleaved calls a client makes between
        // conversations; they share almost nothing and would collapse the
        // candidate to nothing.
        guard tokens.count >= Self.adaptiveStableFloorTokens else { return nil }

        if previousPromptTokens.isEmpty {
            previousPromptTokens = tokens
            return nil
        }

        // Intersect: the candidate can only ever shrink, so it converges on the
        // prefix EVERY sizeable prompt agrees about — the caller's stable
        // header. Continuations of a conversation share that header and leave
        // it untouched, so intra-task turns cost nothing and cannot drag the
        // boundary along with the growing transcript.
        var shared = 0
        let limit = min(previousPromptTokens.count, tokens.count)
        while shared < limit, previousPromptTokens[shared] == tokens[shared] {
            shared += 1
        }
        if shared < previousPromptTokens.count {
            previousPromptTokens = Array(previousPromptTokens[..<shared])
        }

        // Round DOWN, never up: the boundary must stay inside the prefix the
        // prompts actually share, or the stored content will not match what a
        // later probe hashes. Rounding down also makes the result transitive —
        // if every prompt agrees on at least Q tokens, they all agree on the
        // same first Q — which is what lets one stored entry serve every later
        // conversation. Measured across 14 real task transitions the shared
        // prefix was 5663-6216 tokens, no two identical; at 512 thirteen of the
        // fourteen collapse onto 5632.
        let quantized = (shared / Self.adaptiveStableQuantum) * Self.adaptiveStableQuantum
        if config.logRequests {
            print("mei: [adaptive] prompt=\(tokens.count) shared=\(shared) "
                + "quantized=\(quantized) candidate=\(previousPromptTokens.count)")
            fflush(stdout)
        }
        guard quantized >= Self.adaptiveStableMinimumTokens, quantized < tokens.count
        else { return nil }
        return quantized
    }

    private func ssmAnchorOffsets(
        template: [[String: any Sendable]],
        tools: [MeiJSONValue]?,
        context: [String: any Sendable]?,
        tokens: [Int]
    ) async throws -> [Int] {
        let k = config.ssmAnchorBoundaryCount
        guard k > 0 else { return [] }

        // Reuse the offsets when this prompt still begins with the exact token
        // prefix they were derived from.
        //
        // Anchors mark the shared system+tools prefix, which cannot move while
        // a conversation grows -- but computing them renders the whole chat
        // template once per anchor, and that is not free on a long transcript.
        // Measured on a 40k-token conversation, the interval between one answer
        // and the next request reaching the engine was 0.50 s with anchors off,
        // 1.00 s at k=1 and 1.49 s at k=2: about 0.49 s per render, paid again
        // on every single turn.
        //
        // The key is the token prefix itself, not a message count or a session
        // id, so an unrelated conversation with a different system prompt
        // cannot collide with it -- it simply fails the comparison and
        // recomputes. Hashing 40k integers costs microseconds against ~0.5 s
        // per render.
        //
        // A side effect worth stating: the declared list stops growing when a
        // second user message appears (k=2 used to publish a second anchor
        // then). That anchor sat after the first user turn of THIS
        // conversation, so no other conversation could ever match it -- it was
        // cost without benefit. Keeping the list stable across a conversation
        // is also what the store path wants.
        func prefixHash(_ length: Int) -> Int {
            var hasher = Hasher()
            for token in tokens.prefix(length) { hasher.combine(token) }
            return hasher.finalize()
        }
        if let memo = anchorMemo, tokens.count > memo.prefixLength,
            prefixHash(memo.prefixLength) == memo.hash
        {
            return memo.offsets
        }

        let fullTokenCount = tokens.count
        let tokenizer = await container.tokenizer
        let templateTools = MessageMapping.templateTools(tools)
        // Divergence method first: exact, template-agnostic (see
        // SSMAnchorBoundaries.computeByDivergence). The additive method
        // below stays as the fallback and as the self-checked reference.
        let divergent = try SSMAnchorBoundaries.computeByDivergence(
            template: template, fullTokens: tokens, k: k
        ) { variant in
            try tokenizer.applyChatTemplate(
                messages: variant, tools: templateTools, additionalContext: context)
        }
        if !divergent.offsets.isEmpty {
            print("mei: ssm anchor boundaries (k=\(k), divergence): \(divergent.offsets)")
            fflush(stdout)
            if let longest = divergent.offsets.max(), longest > 0, longest < tokens.count {
                anchorMemo = (longest, prefixHash(longest), divergent.offsets)
            }
            return divergent.offsets
        }
        let trace = ProcessInfo.processInfo.environment["MEI_ANCHOR_TRACE"] == "1"
        if trace {
            let roles = template.map { ($0["role"] as? String) ?? "<\(type(of: $0["role"] as Any))>" }
            print("mei: [anchor-trace] k=\(k) full=\(fullTokenCount) roles=\(roles) tools=\(templateTools?.count ?? -1)")
            fflush(stdout)
        }
        let result = try SSMAnchorBoundaries.compute(
            template: template,
            fullTokenCount: fullTokenCount,
            k: k
        ) { prefixCount in
            let n = try tokenizer.applyChatTemplate(
                messages: Array(template.prefix(prefixCount)),
                tools: templateTools,
                additionalContext: context
            ).count
            if trace {
                print("mei: [anchor-trace] prefix(\(prefixCount)) -> \(n) tokens")
                fflush(stdout)
            }
            return n
        }
        if trace {
            print("mei: [anchor-trace] result offsets=\(result.offsets) warning=\(result.warning ?? "nil")")
            fflush(stdout)
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
