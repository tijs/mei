import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix

/// Thread-safe writer over a NIO Channel: HTTP response parts can be written
/// from any thread (Channel.writeAndFlush is thread-safe), which is what lets
/// the router's async generation tasks stream SSE frames directly.
final class ResponseWriter: @unchecked Sendable {
    private let channel: Channel
    private let router: Router
    private var chunkID: String

    init(channel: Channel, router: Router) {
        self.channel = channel
        self.router = router
        self.chunkID = "chatcmpl-\(UUID().uuidString.lowercased().prefix(24))"
    }

    private func write(_ part: HTTPServerResponsePart) {
        channel.writeAndFlush(NIOAny(part), promise: nil)
    }

    func begin(_ status: HTTPResponseStatus, headers: HTTPHeaders) {
        write(.head(.init(version: .http1_1, status: status, headers: headers)))
    }

    func body(_ string: String) {
        var buffer = channel.allocator.buffer(capacity: string.utf8.count)
        buffer.writeString(string)
        write(.body(.byteBuffer(buffer)))
    }

    func end() {
        write(.end(nil))
    }

    func close() {
        channel.close(promise: nil)
    }

    func jsonResponse(status: HTTPResponseStatus, contentType: String, body: String) {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: contentType)
        headers.add(name: "content-length", value: String(body.utf8.count))
        begin(status, headers: headers)
        self.body(body)
        end()
    }

    func streamSSE(request: ChatRequest, engine: Engine, config: ServerConfig) async {
        begin(.ok, headers: router.sseHeaders())
        let stream = await engine.chatRunStreaming(request: request)
        do {
            for try await event in stream {
                let frame = router.sseFrame(
                    id: chunkID, event: event,
                    model: config.servedModelID,
                    emitReasoning: config.emitReasoning,
                    includeUsage: request.includeUsage)
                if !frame.isEmpty {
                    body(frame)
                }
            }
        } catch {
            var buffer = channel.allocator.buffer(capacity: 256)
            buffer.writeString(
                "data: \(router.serializer.errorPayload(error.localizedDescription, code: "stream_error"))\n\n")
            write(.body(.byteBuffer(buffer)))
        }
        end()
        close()
    }
}

/// One per-connection NIO handler: parses HTTP/1.1 requests, dispatches to
/// the router, and writes responses (including SSE) without blocking the
/// event loop.
public final class MeiHTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    public typealias InboundIn = HTTPServerRequestPart
    public typealias OutboundOut = HTTPServerResponsePart

    private let router: Router
    private let config: ServerConfig
    private var requestHead: HTTPRequestHead?
    private var requestBody: ByteBuffer?
    private var pendingTask: Task<Void, Never>?

    public init(router: Router, config: ServerConfig) {
        self.router = router
        self.config = config
    }

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            requestHead = head
            requestBody = context.channel.allocator.buffer(capacity: 0)
        case .body(var part):
            requestBody?.writeBuffer(&part)
        case .end:
            guard let head = requestHead, let body = requestBody else { return }
            requestHead = nil
            requestBody = nil
            let channel = context.channel
            let task = Task { [weak self] in
                if let self {
                    await self.respond(channel: channel, head: head, body: body)
                }
            }
            pendingTask = task
        }
    }

    public func channelInactive(context: ChannelHandlerContext) {
        pendingTask?.cancel()
        pendingTask = nil
    }

    private func respond(channel: Channel, head: HTTPRequestHead, body: ByteBuffer) async {
        let writer = ResponseWriter(channel: channel, router: router)
        let data = body.getData(at: body.readerIndex, length: body.readableBytes) ?? Data()
        do {
            try await HTTPRequestLimiter.ensure(uri: head.uri, body: data)
        } catch {
            writer.jsonResponse(
                status: .payloadTooLarge, contentType: "application/json",
                body: router.serializer.errorPayload(error.localizedDescription, code: "payload_too_large"))
            writer.close()
            return
        }

        switch await router.route(method: head.method, uri: head.uri, body: data) {
        case .plain(let status, let contentType, let body):
            writer.jsonResponse(status: status, contentType: contentType, body: body)
            writer.close()
        case .stream(let request):
            await writer.streamSSE(request: request, engine: router.engine, config: config)
        case .notFound:
            writer.jsonResponse(
                status: .notFound, contentType: "application/json",
                body: router.serializer.errorPayload("no such route: \(head.method) \(head.uri)", code: "not_found"))
            writer.close()
        }
    }

    public func errorCaught(context: ChannelHandlerContext, error: Error) {
        let writer = ResponseWriter(channel: context.channel, router: router)
        writer.jsonResponse(
            status: .internalServerError, contentType: "application/json",
            body: router.serializer.errorPayload(error.localizedDescription, code: "internal_error"))
        writer.close()
    }
}

/// Bounded request size guard so a malformed client cannot feed an unbounded
/// body into the decoder's memory.
public enum HTTPRequestLimiter {
    public static let maxBodyBytes = 64 * 1024 * 1024

    public enum LimitError: LocalizedError {
        case bodyTooLarge(Int)
        public var errorDescription: String? {
            switch self {
            case .bodyTooLarge(let size):
                "request body too large: \(size) bytes (limit \(maxBodyBytes))"
            }
        }
    }

    public static func ensure(uri: String, body: Data) throws {
        guard body.count <= maxBodyBytes else {
            throw LimitError.bodyTooLarge(body.count)
        }
    }
}

/// Lightweight HTTP/1.1 server on NIO.
public final class MeiHTTPServer: @unchecked Sendable {
    private let group: MultiThreadedEventLoopGroup
    private let channel: Channel?

    public var boundPort: Int? {
        channel?.localAddress?.port
    }

    public init(router: Router, config: ServerConfig) throws {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 2)
        self.group = group
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                channel.pipeline.addHandler(ByteToMessageHandler(HTTPRequestDecoder())).flatMap {
                    channel.pipeline.addHandler(HTTPResponseEncoder())
                }.flatMap {
                    channel.pipeline.addHandler(MeiHTTPHandler(router: router, config: config))
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
            .childChannelOption(ChannelOptions.recvAllocator, value: AdaptiveRecvByteBufferAllocator())
        self.channel = try bootstrap.bind(host: config.host, port: config.port).wait()
    }

    public func run() async throws {
        guard let channel else { return }
        try await channel.closeFuture.get()
    }

    public func shutdown() {
        channel?.close(promise: nil)
        try? group.syncShutdownGracefully()
    }
}