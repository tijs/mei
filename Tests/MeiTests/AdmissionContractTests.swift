import XCTest

@testable import MeiCore

/// Contract tests for the Mei-local admission surface: the `mei_priority`
/// request extension, the overload error mapping, the streaming failure
/// delivery decision, and the scheduler block of `/v1/mei/status`. Pure
/// decoding and mapping — no Engine or model.
final class AdmissionContractTests: XCTestCase {

    // MARK: - mei_priority request extension

    private func chatBody(_ extra: String = "") -> Data {
        Data("""
        {"model": "mei/test", "messages": [{"role": "user", "content": "hi"}]\(extra)}
        """.utf8)
    }

    func testChatRequestDefaultsToInteractive() throws {
        let request = try ChatRequest(json: chatBody())
        XCTAssertEqual(request.priority, .interactive)
    }

    func testChatRequestDecodesExplicitClasses() throws {
        let background = try ChatRequest(json: chatBody(", \"mei_priority\": \"background\""))
        XCTAssertEqual(background.priority, .background)
        let interactive = try ChatRequest(json: chatBody(", \"mei_priority\": \"interactive\""))
        XCTAssertEqual(interactive.priority, .interactive)
    }

    func testNullPriorityDefaultsToInteractive() throws {
        let request = try ChatRequest(json: chatBody(", \"mei_priority\": null"))
        XCTAssertEqual(request.priority, .interactive)
    }

    func testUnknownPriorityValueIsRejected() throws {
        XCTAssertThrowsError(try ChatRequest(json: chatBody(", \"mei_priority\": \"urgent\""))) { error in
            guard case RequestFieldError.invalidPriority(let raw) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(raw, "\"urgent\"")
        }
    }

    func testNonStringPriorityIsRejected() throws {
        XCTAssertThrowsError(try ChatRequest(json: chatBody(", \"mei_priority\": 3"))) { error in
            guard case RequestFieldError.invalidPriority(let raw) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(raw, "3")
        }
    }

    func testStandardPriorityFieldIsIgnored() throws {
        // The unqualified `priority` field means an integer scheduling
        // priority in other OpenAI-compatible servers (vLLM): it must be
        // ignored as an unknown field, never mistaken for the Mei extension —
        // whatever its value or type.
        let integer = try ChatRequest(json: chatBody(", \"priority\": 5"))
        XCTAssertEqual(integer.priority, .interactive)
        let string = try ChatRequest(json: chatBody(", \"priority\": \"urgent\""))
        XCTAssertEqual(string.priority, .interactive)
        let completion = try CompletionRequest(json: Data(
            #"{"model": "mei/test", "prompt": "hi", "priority": 3}"#.utf8))
        XCTAssertEqual(completion.priority, .interactive)
    }

    func testCompletionRequestDecodesPriority() throws {
        let background = try CompletionRequest(json: Data(
            #"{"model": "mei/test", "prompt": "hi", "mei_priority": "background"}"#.utf8))
        XCTAssertEqual(background.priority, .background)
        let defaulted = try CompletionRequest(json: Data(
            #"{"model": "mei/test", "prompt": "hi"}"#.utf8))
        XCTAssertEqual(defaulted.priority, .interactive)
        XCTAssertThrowsError(try CompletionRequest(json: Data(
            #"{"model": "mei/test", "prompt": "hi", "mei_priority": "later"}"#.utf8)))
    }

    // MARK: - overload error mapping

    private func errorEnvelope(_ error: Error) throws -> [String: Any] {
        let plain = Router.errorPlain(error, serializer: ResponseSerializer())
        XCTAssertEqual(plain.contentType, "application/json")
        let body = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(plain.body.utf8)) as? [String: Any])
        return try XCTUnwrap(body["error"] as? [String: Any])
    }

    func testQueueFullMapsTo429RateLimitError() throws {
        let plain = Router.errorPlain(AdmissionError.queueFull(limit: 7), serializer: ResponseSerializer())
        XCTAssertEqual(plain.status, .tooManyRequests)
        let error = try errorEnvelope(AdmissionError.queueFull(limit: 7))
        XCTAssertEqual(error["type"] as? String, "rate_limit_error")
        XCTAssertEqual(error["code"] as? String, "queue_full")
        XCTAssertTrue((error["message"] as? String)?.contains("full") == true)
        XCTAssertNil(error["param"])
    }

    func testQueueWaitTimeoutMapsTo503() throws {
        let plain = Router.errorPlain(
            AdmissionError.queueWaitTimeout(milliseconds: 250), serializer: ResponseSerializer())
        XCTAssertEqual(plain.status, .serviceUnavailable)
        let error = try errorEnvelope(AdmissionError.queueWaitTimeout(milliseconds: 250))
        XCTAssertEqual(error["type"] as? String, "server_error")
        XCTAssertEqual(error["code"] as? String, "queue_timeout")
        XCTAssertTrue((error["message"] as? String)?.contains("250 ms") == true)
        XCTAssertNil(error["param"])
    }

    func testInvalidPriorityMapsTo400WithParam() throws {
        let plain = Router.errorPlain(
            RequestFieldError.invalidPriority("\"urgent\""), serializer: ResponseSerializer())
        XCTAssertEqual(plain.status, .badRequest)
        let error = try errorEnvelope(RequestFieldError.invalidPriority("\"urgent\""))
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertEqual(error["code"] as? String, "invalid_priority")
        XCTAssertEqual(error["param"] as? String, "mei_priority")
        XCTAssertTrue(
            (error["message"] as? String)?.contains("mei_priority") == true,
            "the message must name the offending field")
    }

    func testErrorResultStillWrapsErrorPlain() throws {
        guard case .plain(let status, let contentType, let body) =
            Router.errorResult(AdmissionError.queueFull(limit: 1), serializer: ResponseSerializer())
        else {
            return XCTFail("expected a plain result")
        }
        XCTAssertEqual(status, .tooManyRequests)
        XCTAssertEqual(contentType, "application/json")
        XCTAssertTrue(body.contains("queue_full"))
    }

    // MARK: - streaming failure delivery

    /// The streaming handler's status decision, model-free: before the
    /// stream's first event the SSE head has not been written, so any failure
    /// is delivered as the real HTTP status (the same mapping non-streaming
    /// requests use); once the stream has started the 200 head is committed
    /// and the failure travels in-band as a `stream_error` frame; a cancelled
    /// request (client gone) is answered with nothing at all.
    func testPreFirstEventFailureGetsHTTPResult() throws {
        let serializer = ResponseSerializer()

        // Admission refusals: the same 429/503 mapping as non-streaming.
        let refusal = ResponseWriter.failureResponse(
            for: AdmissionError.queueFull(limit: 3), streamStarted: false, serializer: serializer)
        guard case .httpError(let refusalStatus, let refusalType, let refusalBody) = refusal else {
            return XCTFail("expected an HTTP error for a pre-first-event refusal: \(refusal)")
        }
        XCTAssertEqual(refusalStatus, .tooManyRequests)
        XCTAssertEqual(refusalType, "application/json")
        XCTAssertTrue(refusalBody.contains("queue_full"))

        let timeout = ResponseWriter.failureResponse(
            for: AdmissionError.queueWaitTimeout(milliseconds: 10),
            streamStarted: false, serializer: serializer)
        guard case .httpError(let timeoutStatus, _, let timeoutBody) = timeout else {
            return XCTFail("expected an HTTP error for a pre-first-event timeout: \(timeout)")
        }
        XCTAssertEqual(timeoutStatus, .serviceUnavailable)
        XCTAssertTrue(timeoutBody.contains("queue_timeout"))

        // Pre-generation engine failures: the same 400/500 mapping.
        guard case .httpError(let capStatus, _, let capBody) = ResponseWriter.failureResponse(
            for: EngineError.overContextCap(promptTokens: 99, cap: 8),
            streamStarted: false, serializer: serializer)
        else {
            return XCTFail("expected an HTTP error for an over-context prompt")
        }
        XCTAssertEqual(capStatus, .badRequest)
        XCTAssertTrue(capBody.contains("engine_error"))
        guard case .httpError(let structuredStatus, _, let structuredBody) = ResponseWriter.failureResponse(
            for: EngineError.generationFailed("structured output constraint failed: boom"),
            streamStarted: false, serializer: serializer)
        else {
            return XCTFail("expected an HTTP error for a pre-first-event structured failure")
        }
        XCTAssertEqual(structuredStatus, .internalServerError)
        XCTAssertTrue(structuredBody.contains("engine_error"))
    }

    func testPostStartFailureStaysAnSSEErrorFrame() {
        let errors: [Error] = [
            AdmissionError.queueFull(limit: 1),
            EngineError.generationFailed("structured output constraint failed: boom"),
        ]
        for error in errors {
            let result = ResponseWriter.failureResponse(
                for: error, streamStarted: true, serializer: ResponseSerializer())
            guard case .sseErrorFrame(let frame) = result else {
                return XCTFail("a failure after the first event must stay in-band: \(result)")
            }
            XCTAssertEqual(frame, Router.streamErrorSSEData(error))
            XCTAssertTrue(frame.contains("\"code\":\"stream_error\""))
        }
    }

    func testCancellationIsAnsweredWithNothing() {
        for streamStarted in [false, true] {
            XCTAssertEqual(
                ResponseWriter.failureResponse(
                    for: CancellationError(), streamStarted: streamStarted,
                    serializer: ResponseSerializer()),
                .noResponse,
                "a cancelled request must never be reported as an HTTP error or an error frame")
        }
    }

    // MARK: - status surface

    func testSchedulerStatusMappingMatchesSnapshot() {
        var counters = RequestScheduler.Counters()
        counters.admitted = 3
        counters.completed = 2
        counters.cancelledActive = 1
        counters.slotReleases = 3
        counters.lastCancelToReleaseMilliseconds = 12.5
        counters.maxCancelToReleaseMilliseconds = 20
        let snapshot = RequestScheduler.Snapshot(
            running: true,
            queueDepth: 2,
            queuedInteractive: 1,
            queuedBackground: 1,
            limits: .init(maxQueueDepth: 9, queueWaitTimeoutMilliseconds: 500),
            counters: counters)

        let status = Router.schedulerStatus(snapshot)
        XCTAssertTrue(status.running)
        XCTAssertEqual(status.queueDepth, 2)
        XCTAssertEqual(status.queuedInteractive, 1)
        XCTAssertEqual(status.queuedBackground, 1)
        XCTAssertEqual(status.maxQueueDepth, 9)
        XCTAssertEqual(status.queueWaitTimeoutMs, 500)
        XCTAssertEqual(status.admitted, 3)
        XCTAssertEqual(status.completed, 2)
        XCTAssertEqual(status.cancelledActive, 1)
        XCTAssertEqual(status.slotReleases, 3)
        XCTAssertEqual(status.lastCancelToReleaseMs, 12.5)
        XCTAssertEqual(status.maxCancelToReleaseMs, 20)
    }

    func testSchedulerStatusEncodesAndOmitsNilTimings() throws {
        let status = MeiSchedulerStatus(
            running: false,
            queueDepth: 0,
            queuedInteractive: 0,
            queuedBackground: 0,
            maxQueueDepth: 32,
            queueWaitTimeoutMs: 600_000,
            admitted: 0,
            completed: 0,
            failed: 0,
            cancelledQueued: 0,
            cancelledActive: 0,
            rejectedQueueFull: 0,
            rejectedTimeout: 0,
            slotReleases: 0,
            lastQueueWaitMs: 0,
            maxQueueWaitMs: 0,
            lastCancelToReleaseMs: nil,
            maxCancelToReleaseMs: nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(status)
        let dict = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(dict["queueDepth"] as? Int, 0)
        XCTAssertEqual(dict["maxQueueDepth"] as? Int, 32)
        XCTAssertEqual(dict["queueWaitTimeoutMs"] as? Int, 600_000)
        XCTAssertNil(dict["lastCancelToReleaseMs"], "nil timings stay absent")
        XCTAssertNil(dict["maxCancelToReleaseMs"], "nil timings stay absent")
    }
}
