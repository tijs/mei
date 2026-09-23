import Foundation
import NIOCore
import NIOHTTP1

/// JSON + SSE response assembly shared by the HTTP handlers.
public final class ResponseSerializer: @unchecked Sendable {
    private let encoder: JSONEncoder

    public init() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        self.encoder = encoder
    }

    public func json<T: Encodable>(_ value: T) -> String {
        guard let data = try? encoder.encode(value),
            let string = String(data: data, encoding: .utf8)
        else {
            return #"{"error":{"message":"encoding failure","type":"internal_error"}}"#
        }
        return string
    }

    public func errorPayload(_ message: String, type: String = "invalid_request_error", code: String? = nil) -> String {
        json(APIErrorEnvelope(error: .init(message: message, type: type, code: code)))
    }
}

/// OpenAI-shape API router. Pure request→response mapping; the engine owns
/// all inference state.
public final class Router: @unchecked Sendable {
    /// Optional so unit tests can exercise the pure routing/serialization
    /// seams without a model; the HTTP server always wires a loaded engine.
    /// `route()` fails closed with a 500 when nil.
    public let engine: Engine?
    public let config: ServerConfig
    public let serializer = ResponseSerializer()
    public let startedAt: Date

    public init(engine: Engine?, config: ServerConfig) {
        self.engine = engine
        self.config = config
        self.startedAt = Date()
    }

    public enum RouteResult {
        case plain(status: HTTPResponseStatus, contentType: String, body: String)
        case stream(request: ChatRequest)
        case notFound
    }

    /// Synchronous dispatch for non-streaming routes. Streaming routes
    /// return `.stream` immediately; the HTTP handler feeds them.
    public func route(method: HTTPMethod, uri: String, body: Data) async -> RouteResult {
        guard let engine else {
            return .plain(
                status: .internalServerError, contentType: "application/json",
                body: serializer.errorPayload("engine is not loaded", type: "engine_error", code: "engine_error"))
        }
        switch (method, uri) {
        case (.GET, "/v1/models"), (.GET, "/v1/models/"):
            return models()
        case (.GET, "/healthz"), (.GET, "/health"):
            return .plain(status: .ok, contentType: "application/json", body: #"{"status":"ok"}"#)
        case (.GET, "/v1/mei/status"):
            return await status(engine: engine)
        case (.POST, "/v1/chat/completions"):
            return await chat(body: body, engine: engine)
        case (.POST, "/v1/completions"):
            return await completion(body: body, engine: engine)
        default:
            return .notFound
        }
    }

    private func models() -> RouteResult {
        let response = ModelsResponse(data: [
            .init(id: config.servedModelID, created: Int(startedAt.timeIntervalSince1970))
        ])
        return .plain(status: .ok, contentType: "application/json", body: serializer.json(response))
    }

    private func status(engine: Engine) async -> RouteResult {
        let report = Engine.liveMemoryReport()
        let cache = await engine.cacheStats().map { stats in
            MeiCacheStatus(
                pagedEnabled: stats.pagedEnabled,
                pagedHits: stats.pagedStats?.cacheHits ?? 0,
                pagedMisses: stats.pagedStats?.cacheMisses ?? 0,
                pagedEvictions: stats.pagedStats?.evictions ?? 0,
                ssmHits: stats.ssmStats.hits,
                ssmMisses: stats.ssmStats.misses,
                isHybrid: stats.isHybrid)
        }
        let response = MeiStatusResponse(
            status: "ok",
            model: config.servedModelID,
            contextCap: config.contextCap,
            prefillStepSize: config.prefillStepSize,
            maxTokens: config.maxTokensDefault,
            kvBits: config.kvBits,
            cacheReuse: config.cacheReuse,
            uptimeSeconds: Int(Date().timeIntervalSince(startedAt)),
            memory: report,
            cache: cache)
        return .plain(status: .ok, contentType: "application/json", body: serializer.json(response))
    }

    private func chat(body: Data, engine: Engine) async -> RouteResult {
        do {
            let request = try ChatRequest(json: body)
            if request.stream {
                // Validate BEFORE the SSE response starts: a request that can
                // never generate (over context cap, template-rejected
                // message) must fail with a clean JSON 400, not an error
                // frame after the 200 headers are on the wire.
                try await engine.preflightChat(request: request)
                return .stream(request: request)
            }
            let run = try await engine.chatRun(request: request)
            return .plain(
                status: .ok, contentType: "application/json",
                body: serializer.json(Self.completionResponse(run: run, model: config.servedModelID, emitReasoning: config.emitReasoning)))
        } catch let error as EngineError {
            return .plain(status: Self.errorStatus(error), contentType: "application/json", body: serializer.errorPayload(error.localizedDescription, code: "engine_error"))
        } catch {
            return .plain(status: .badRequest, contentType: "application/json", body: serializer.errorPayload(error.localizedDescription))
        }
    }

    private func completion(body: Data, engine: Engine) async -> RouteResult {
        do {
            let request = try CompletionRequest(json: body)
            if request.stream {
                // The minimal legacy path is non-streaming only. Accepting
                // stream=true and answering with a plain JSON response would
                // be exactly the silent semantic drift the P0 contract
                // forbids, so it is a loud 400 instead.
                throw APIRequestError.invalidField(
                    "stream: streaming is not supported on /v1/completions in the Mei P0 contract")
            }
            let run = try await engine.completionRun(request: request)
            return .plain(
                status: .ok, contentType: "application/json",
                body: serializer.json(Self.legacyCompletionResponse(run: run, model: config.servedModelID)))
        } catch let error as EngineError {
            return .plain(status: Self.errorStatus(error), contentType: "application/json", body: serializer.errorPayload(error.localizedDescription, code: "engine_error"))
        } catch {
            return .plain(status: .badRequest, contentType: "application/json", body: serializer.errorPayload(error.localizedDescription))
        }
    }

    public static func errorStatus(_ error: EngineError) -> HTTPResponseStatus {
        switch error {
        case .overContextCap: return .badRequest
        case .modelDirectoryMissing, .modelNotLoaded, .generationFailed: return .internalServerError
        case .emptyPrompt: return .badRequest
        }
    }

    /// Usage block shared by chat/completions, chat/completions streaming
    /// finish, and /v1/completions — carries engine-reported tok/s and the
    /// MLX allocator footprint so the benchmark never has to guess.
    public static func usage(run: GenerationRun) -> ChatCompletionResponse.Usage {
        ChatCompletionResponse.Usage(
            promptTokens: run.promptTokenCount,
            completionTokens: run.completionTokenCount,
            totalTokens: run.promptTokenCount + run.completionTokenCount,
            promptTokensDetails: .init(cachedTokens: run.cachedTokenCount),
            tokensPerSecond: run.decodeTokensPerSecond > 0 ? run.decodeTokensPerSecond : nil,
            promptTokensPerSecond: run.promptTokensPerSecond > 0 ? run.promptTokensPerSecond : nil,
            prefillMilliseconds: run.prefillMilliseconds > 0 ? run.prefillMilliseconds : nil,
            generateMilliseconds: run.generateMilliseconds > 0 ? run.generateMilliseconds : nil,
            memoryActiveBytes: run.memoryActiveBytes > 0 ? run.memoryActiveBytes : nil,
            memoryCacheBytes: run.memoryCacheBytes > 0 ? run.memoryCacheBytes : nil,
            memoryPeakBytes: run.memoryPeakBytes > 0 ? run.memoryPeakBytes : nil)
    }

    public static func completionResponse(run: GenerationRun, model: String, emitReasoning: Bool) -> ChatCompletionResponse {
        let message = ChatCompletionResponse.ResponseMessage(
            content: run.text.isEmpty && !run.toolCalls.isEmpty ? nil : run.text,
            toolCalls: run.toolCalls.isEmpty ? nil : run.toolCalls.map { call in
                .init(id: call.id ?? "call_unknown", function: .init(name: call.name, arguments: call.argumentsJSON))
            },
            reasoningContent: emitReasoning && !run.reasoning.isEmpty ? run.reasoning : nil)
        let usage = Self.usage(run: run)
        let id = "chatcmpl-\(UUID().uuidString.lowercased().prefix(24))"
        return ChatCompletionResponse(
            id: id,
            created: Int(Date().timeIntervalSince1970),
            model: model,
            choices: [.init(message: message, finishReason: run.finishReason)],
            usage: usage)
    }

    /// The minimal legacy /v1/completions response: OpenAI `text_completion`
    /// shape with the same usage block the chat path reports.
    public static func legacyCompletionResponse(run: GenerationRun, model: String) -> TextCompletionResponse {
        TextCompletionResponse(
            id: "cmpl-\(UUID().uuidString.lowercased().prefix(24))",
            created: Int(Date().timeIntervalSince1970),
            model: model,
            choices: [.init(text: run.text, finishReason: run.finishReason)],
            usage: Self.usage(run: run))
    }

    /// The JSON payloads for a streaming run's terminal frames: a
    /// `finish_reason` chunk, then a usage chunk iff `includeUsage` (the
    /// OpenAI `stream_options.include_usage` contract), then `[DONE]`.
    /// Pure/static so unit tests can pin usage presence/absence and count
    /// parity without a live engine.
    public static func finishSSEData(
        id: String,
        run: GenerationRun,
        model: String,
        created: Int,
        includeUsage: Bool
    ) -> [String] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let finishChunk = SSEChatChunk(
            id: id, created: created, model: model,
            choices: [.init(delta: .init(), finishReason: run.finishReason)],
            usage: nil)
        var frames: [String] = []
        if let json = try? encoder.encode(finishChunk), let string = String(data: json, encoding: .utf8) {
            frames.append(string)
        }
        if includeUsage {
            let usageChunk = SSEChatChunk(
                id: id, created: created, model: model,
                choices: [], usage: Self.usage(run: run))
            if let json = try? encoder.encode(usageChunk), let string = String(data: json, encoding: .utf8) {
                frames.append(string)
            }
        }
        frames.append("[DONE]")
        return frames
    }

    /// Serialize one tool-call StreamEvent into a single SSE `data: {...}\n\n`
    /// frame, exactly as the streaming handler ships over the wire. Static and
    /// pure so it can be unit-tested without loading a model: the streaming
    /// `.toolCall` case in `sseFrame` delegates here with the call's index.
    public static func toolCallSSEData(
        call: GenerationRun.ToolCallEmitting,
        index: Int,
        id: String,
        model: String,
        created: Int
    ) -> String {
        let chunk = SSEChatChunk(
            id: id, created: created, model: model,
            choices: [.init(delta: .init(toolCalls: [
                .init(
                    index: index,
                    id: call.id,
                    type: "function",
                    function: .init(name: call.name, arguments: call.argumentsJSON))
            ]))],
            usage: nil)
        return "data: \(ResponseSerializer().json(chunk))\n\n"
    }

    /// SSE frame for one stream event. Returns "" for events that should not
    /// produce visible frames.
    public func sseFrame(
        id: String,
        event: StreamEvent,
        model: String,
        emitReasoning: Bool,
        includeUsage: Bool
    ) -> String {
        switch event {
        case .chunk(let text):
            // role rides on content deltas. OpenAI emits it once, on the
            // first chunk; Mei repeats it because the router is stateless and
            // every streaming client merges the role idempotently. The
            // contract documents this repetition.
            let chunk = SSEChatChunk(
                id: id, created: Int(Date().timeIntervalSince1970), model: model,
                choices: [.init(delta: .init(role: "assistant", content: text))],
                usage: nil)
            return "data: \(serializer.json(chunk))\n\n"
        case .reasoning(let reason):
            guard emitReasoning else { return "" }
            let chunk = SSEChatChunk(
                id: id, created: Int(Date().timeIntervalSince1970), model: model,
                choices: [.init(delta: .init(reasoningContent: reason))],
                usage: nil)
            return "data: \(serializer.json(chunk))\n\n"
        case .toolCall(let index, let call):
            return Self.toolCallSSEData(
                call: call, index: index, id: id, model: model,
                created: Int(Date().timeIntervalSince1970))
        case .prefill:
            return ""
        case .finish(let run):
            let frames = Self.finishSSEData(
                id: id, run: run, model: model,
                created: Int(Date().timeIntervalSince1970),
                includeUsage: includeUsage)
            return frames.map { "data: \($0)\n\n" }.joined()
        }
    }

    public func sseHeaders() -> HTTPHeaders {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        headers.add(name: "x-accel-buffering", value: "no")
        return headers
    }

    /// The terminal frame of a stream that failed mid-generation: a single
    /// `data:` frame carrying the error envelope (`code: "stream_error"`).
    /// Pinned contract: an errored stream emits NO finish frame, NO usage
    /// chunk, and NO `[DONE]` — `[DONE]` is the success terminator only
    /// (`finishSSEData`), so clients treat the error frame (or EOF without
    /// `[DONE]`) as the end of a failed stream. Pinned by
    /// RouterSSEFrameTests and probe_mei.py --self-test.
    public func errorSSEFrame(message: String) -> String {
        "data: \(serializer.errorPayload(message, code: "stream_error"))\n\n"
    }
}