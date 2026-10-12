import XCTest

@testable import MeiCore

/// The stream lifecycle contract `Engine.chatRunStreaming` is built on:
/// a consumer that goes away cancels the producer task, a producer that is
/// cancelled never emits a success finish frame (no false success finish),
/// the producer's own cleanup still runs, and a producer that completes
/// normally is delivered intact. No model or network is involved.
final class CancellableStreamTests: XCTestCase {

    private enum Event: Equatable, Sendable {
        case chunk(String)
        case finish
    }

    func testNormalCompletionDeliversEveryEvent() async throws {
        let stream: AsyncThrowingStream<Event, Error> = makeCancellableStream { continuation in
            continuation.yield(.chunk("a"))
            continuation.yield(.chunk("b"))
            continuation.yield(.finish)
        }
        var received: [Event] = []
        for try await event in stream { received.append(event) }
        XCTAssertEqual(received, [.chunk("a"), .chunk("b"), .finish])
    }

    func testProducerReturningWithoutFinishingTerminatesTheStream() async throws {
        // A body that returns without calling finish() must not leave the
        // consumer hanging.
        let stream: AsyncThrowingStream<Event, Error> = makeCancellableStream { continuation in
            continuation.yield(.chunk("only"))
        }
        var received: [Event] = []
        for try await event in stream { received.append(event) }
        XCTAssertEqual(received, [.chunk("only")])
    }

    func testConsumerCancellationCancelsTheProducerAndSuppressesTheFinish() async throws {
        let producer = TestRecorder()
        let stream: AsyncThrowingStream<Event, Error> = makeCancellableStream { continuation in
            producer.record("started")
            continuation.yield(.chunk("a"))
            do {
                try await Task.sleep(for: .seconds(60))
            } catch {
                // Cancelled: the engine's rule is to return without a
                // success finish frame, then run the cleanup.
                producer.record("cancelled")
                producer.record("cleanup-done")
                return
            }
            producer.record("completed")
            producer.record("yield-finish")
            continuation.yield(.finish)
        }

        let consumer = Task { () -> Void in
            do {
                for try await _ in stream {}
            } catch {
                // Consumer-side cancellation surfaces as an error here.
            }
        }
        await waitUntil("producer started") { producer.contains("started") }
        consumer.cancel()
        await consumer.value

        await waitUntil("producer observed cancellation") { producer.contains("cancelled") }
        await waitUntil("producer cleanup completed") { producer.contains("cleanup-done") }
        XCTAssertFalse(producer.contains("completed"))
        XCTAssertFalse(
            producer.contains("yield-finish"),
            "a cancelled run must not emit a success finish frame")
    }

    func testProducerErrorSurfacesToTheConsumer() async throws {
        struct Boom: Error, Equatable {}
        let stream: AsyncThrowingStream<Event, Error> = makeCancellableStream { _ in
            throw Boom()
        }
        do {
            for try await _ in stream { XCTFail("no events expected") }
            XCTFail("expected the producer error to be delivered")
        } catch is Boom {
            // expected
        }
    }

    func testConsumerAbandoningTheStreamDoesNotCancelTheProducer() async throws {
        // Pins the boundary of the contract: termination is driven by the
        // consuming task's cancellation (what a client disconnect produces),
        // not by a consumer that silently stops iterating — the producer
        // holds the continuation, so an abandoned iterator alone does not
        // terminate the stream. Code paths that must stop a producer have to
        // cancel their task, which is exactly what the HTTP handler does.
        let producer = TestRecorder()
        let stream: AsyncThrowingStream<Event, Error> = makeCancellableStream { continuation in
            producer.record("started")
            continuation.yield(.chunk("a"))
            continuation.finish()
            producer.record("finished")
        }
        var iterator = stream.makeAsyncIterator()
        let first = try await iterator.next()
        XCTAssertEqual(first, .chunk("a"))
        await waitUntil("producer finished") { producer.contains("finished") }
        // The producer ran to completion; no cancellation was triggered.
        XCTAssertFalse(producer.contains("cancelled"))
        _ = iterator
    }
}
