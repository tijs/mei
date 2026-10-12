import Foundation

/// Wrap a streaming producer in an `AsyncThrowingStream` whose producer task
/// is cancelled when the consumer goes away.
///
/// `AsyncThrowingStream` alone does not cancel its producer when the consumer
/// goes away: the producer task keeps running, and for an inference server
/// that means the single execution slot stays held by a generation whose
/// client has already disconnected. This helper wires the missing
/// propagation: the producer runs in its own task and is cancelled when the
/// stream terminates with `.cancelled`, which is what a consuming task's
/// cancellation produces (a client disconnect, or any caller that cancels
/// its request task). Note that merely abandoning the iterator without
/// cancelling the task does not terminate the stream — the producer holds
/// the continuation — so callers that must stop a producer have to cancel
/// their task, which the HTTP handler already does on channel close.
///
/// Only `.cancelled` termination cancels the producer. A producer-initiated
/// `finish()` must not cancel it: the producer still has post-finish work
/// (request logging, cache bookkeeping), and cancelling it there would make
/// its own `Task.isCancelled` checks report a cancelled run for a request
/// that completed normally. This mirrors the contract vmlx's `generateTask`
/// applies to its own producer task, one level up.
///
/// Cancellation of the producer does not skip the producer's own cleanup:
/// the body observes cancellation cooperatively (e.g. by cancelling the
/// underlying generation task), finishes the stream, and only then returns —
/// which is what lets `RequestScheduler` release the slot after all GPU
/// work for the request is done.
public func makeCancellableStream<Event: Sendable>(
    _ produce: @escaping @Sendable (AsyncThrowingStream<Event, Error>.Continuation) async throws -> Void
) -> AsyncThrowingStream<Event, Error> {
    AsyncThrowingStream { continuation in
        let producer = Task {
            do {
                try await produce(continuation)
                // The producer may already have finished the stream (the
                // engine's streaming path finishes it at `.info`); finish()
                // is idempotent, so this only covers a body that returned
                // without terminating its own stream.
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable termination in
            if case .cancelled = termination {
                producer.cancel()
            }
        }
    }
}
