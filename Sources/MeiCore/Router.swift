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
    public let engine: Engine
    public let config: ServerConfig
    public let serializer = ResponseSerializer()
    public let startedAt: Date

    public init(engine: Engine, config: ServerConfig) {
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
        switch (method, uri) {
        case (.GET, "/v1/models"), (.GET, "/v1/models/"):
            return models()
        case (.GET, "/healthz"), (.GET, "/health"):
            return .plain(status: .ok, contentType: "application/json", body: #"{"status":"ok"}"#)
        case (.GET, "/v1/mei/status"):
            return await status()
        case (.POST, "/v1/chat/completions"):
            return await chat(body: body)
        case (.POST, "/v1/completions"):
            return await completion(body: body)
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

    private func status() async -> RouteResult {
        let report = await engine.memoryReport()
        let response = MeiStatusResponse(
            status: "ok",
            model: config.servedModelID,
            contextCap: config.contextCap,
            prefillStepSize: config.prefillStepSize,
            maxTokens: config.maxTokensDefault,
            kvBits: config.kvBits,
            cacheReuse: config.cacheReuse,
            uptimeSeconds: Int(Date().timeIntervalSince(startedAt)),
            memory: report)
        return .plain(status: .ok, contentType: "application/json", body: serializer.json(response))
    }

    private func chat(body: Data) async -> RouteResult {
        do {
            let request = try ChatRequest(json: body)
            if request.stream {
                return .stream(request: request)
            }
            let run = try await engine.chatRun(request: request)
            return .plain(
                status: .ok, contentType: "application/json",
                body: serializer.json(Self.completionResponse(run: run, model: config.servedModelID, emitReasoning: config.emitReasoning)))
        } catch let error as EngineError {
            return .plain(status: errorStatus(error), contentType: "application/json", body: serializer.errorPayload(error.localizedDescription, code: "engine_error"))
        } catch {
            return .plain(status: .badRequest, contentType: "application/json", body: serializer.errorPayload(error.localizedDescription))
        }
    }

    private func completion(body: Data) async -> RouteResult {
        do {
            let request = try CompletionRequest(json: body)
            let run = try await engine.completionRun(request: request)
            return .plain(
                status: .ok, contentType: "application/json",
                body: serializer.json(Self.completionResponse(run: run, model: config.servedModelID, emitReasoning: config.emitReasoning)))
        } catch let error as EngineError {
            return .plain(status: errorStatus(error), contentType: "application/json", body: serializer.errorPayload(error.localizedDescription, code: "engine_error"))
        } catch {
            return .plain(status: .badRequest, contentType: "application/json", body: serializer.errorPayload(error.localizedDescription))
        }
    }

    public func errorStatus(_ error: EngineError) -> HTTPResponseStatus {
        switch error {
        case .overContextCap: return .badRequest
        case .modelDirectoryMissing, .modelNotLoaded, .generationFailed: return .internalServerError
        case .emptyPrompt: return .badRequest
        }
    }

    public static func completionResponse(run: GenerationRun, model: String, emitReasoning: Bool) -> ChatCompletionResponse {
        let message = ChatCompletionResponse.ResponseMessage(
            content: run.text.isEmpty && !run.toolCalls.isEmpty ? nil : run.text,
            toolCalls: run.toolCalls.isEmpty ? nil : run.toolCalls.map { call in
                .init(id: call.id ?? "call_unknown", function: .init(name: call.name, arguments: call.argumentsJSON))
            },
            reasoningContent: emitReasoning && !run.reasoning.isEmpty ? run.reasoning : nil)
        let usage = ChatCompletionResponse.Usage(
            promptTokens: run.promptTokenCount,
            completionTokens: run.completionTokenCount,
            totalTokens: run.promptTokenCount + run.completionTokenCount,
            promptTokensDetails: .init(cachedTokens: run.cachedTokenCount))
        let id = "chatcmpl-\(UUID().uuidString.lowercased().prefix(24))"
        return ChatCompletionResponse(
            id: id,
            created: Int(Date().timeIntervalSince1970),
            model: model,
            choices: [.init(message: message, finishReason: run.finishReason)],
            usage: usage)
    }

    /// SSE frame for one stream event. Returns "" for events that should not
    /// produce visible frames.
    public func sseFrame(
        id: String,
        event: StreamEvent,
        model: String,
        emitReasoning: Bool
    ) -> String {
        switch event {
        case .chunk(let text):
            let chunk = SSEChatChunk(
                id: id, created: Int(Date().timeIntervalSince1970), model: model,
                choices: [.init(delta: .init(role: nil, content: text))],
                usage: nil)
            return "data: \(serializer.json(chunk))\n\n"
        case .reasoning(let reason):
            guard emitReasoning else { return "" }
            let chunk = SSEChatChunk(
                id: id, created: Int(Date().timeIntervalSince1970), model: model,
                choices: [.init(delta: .init(reasoningContent: reason))],
                usage: nil)
            return "data: \(serializer.json(chunk))\n\n"
        case .toolCall(let call):
            let chunk = SSEChatChunk(
                id: id, created: Int(Date().timeIntervalSince1970), model: model,
                choices: [.init(delta: .init(toolCalls: [
                    .init(
                        index: 0,
                        id: call.id,
                        type: "function",
                        function: .init(name: call.name, arguments: call.argumentsJSON))
                ]))],
                usage: nil)
            return "data: \(serializer.json(chunk))\n\n"
        case .prefill:
            return ""
        case .finish(let run):
            var lines: [String] = []
            let finishChunk = SSEChatChunk(
                id: id, created: Int(Date().timeIntervalSince1970), model: model,
                choices: [.init(delta: .init(), finishReason: run.finishReason)],
                usage: nil)
            lines.append("data: \(serializer.json(finishChunk))")
            let usage = ChatCompletionResponse.Usage(
                promptTokens: run.promptTokenCount,
                completionTokens: run.completionTokenCount,
                totalTokens: run.promptTokenCount + run.completionTokenCount,
                promptTokensDetails: .init(cachedTokens: run.cachedTokenCount))
            let usageChunk = SSEChatChunk(
                id: id, created: Int(Date().timeIntervalSince1970), model: model,
                choices: [], usage: usage)
            lines.append("data: \(serializer.json(usageChunk))")
            lines.append("data: [DONE]")
            return lines.map { $0 + "\n\n" }.joined()
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
}