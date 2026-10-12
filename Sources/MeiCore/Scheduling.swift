import Foundation

/// Generic scheduling class for one inference request.
///
/// Peix owns semantic priority — which agent's work matters, retries, and
/// checkpoints. Mei only needs the coarse class that decides queue order:
/// work a human is waiting on (`interactive`) versus durable background work
/// (`background`). A queued interactive request outranks every queued
/// background request; neither class can preempt a generation that is
/// already running, because priority orders the queue and nothing else (see
/// `docs/INFERENCE-SCHEDULING.md`).
public enum RequestPriority: String, Sendable, CaseIterable, Equatable {
    /// Default class. Interactive work: a person or a synchronous agent turn
    /// is waiting for the answer.
    case interactive
    /// Opt-in background class: durable agent jobs that can wait. A queued
    /// background request may be overtaken by later-arriving interactive
    /// requests; it never interrupts a running generation.
    case background

    /// Lower rank is served first among queued requests.
    var rank: Int { self == .interactive ? 0 : 1 }
}

/// Admission-layer refusals, thrown before any generation work starts.
///
/// Distinct from `EngineError` so the router can map them to their own
/// deterministic statuses: both mean "the server is busy, retry later",
/// never "the request was bad". See `Router.errorPlain`.
public enum AdmissionError: LocalizedError, Sendable, Equatable {
    /// The bounded queue was already full when this request arrived; it was
    /// rejected immediately and never enqueued.
    case queueFull(limit: Int)
    /// The request waited the configured budget for the execution slot and
    /// was removed from the queue without starting.
    case queueWaitTimeout(milliseconds: Int)

    public var errorDescription: String? {
        switch self {
        case .queueFull(let limit):
            "admission queue is full (\(limit) waiting); the request was not started, retry later"
        case .queueWaitTimeout(let milliseconds):
            "request waited \(milliseconds) ms for the inference slot without being admitted; retry later"
        }
    }

    /// Stable machine-readable code for the error envelope.
    public var code: String {
        switch self {
        case .queueFull: "queue_full"
        case .queueWaitTimeout: "queue_timeout"
        }
    }
}

/// One granted execution slot, handed to the operation by
/// `RequestScheduler.submit`. `queueWaitMilliseconds` is 0 when the slot was
/// idle on arrival; otherwise it is the time this request spent queued.
public struct SlotGrant: Sendable, Equatable {
    public var queueWaitMilliseconds: Double
    public var priority: RequestPriority

    public init(queueWaitMilliseconds: Double, priority: RequestPriority) {
        self.queueWaitMilliseconds = queueWaitMilliseconds
        self.priority = priority
    }
}

/// The instant a cancellation was first observed for a granted request.
/// Written from the synchronous cancellation handler (which runs on the
/// cancelling thread) and read by the scheduler actor afterwards, hence the
/// lock. Used to report cancellation-to-slot-release timing.
final class CancellationTimestamp: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedAt: Date?

    func record(_ date: Date = Date()) {
        lock.lock()
        if recordedAt == nil { recordedAt = date }
        lock.unlock()
    }

    var recorded: Date? {
        lock.lock()
        defer { lock.unlock() }
        return recordedAt
    }
}

/// Bounded admission in front of Mei's single execution slot.
///
/// Kiem plan `820814e1` (Mei inference scheduling for Peixinho), stages 3–4:
/// Mei keeps strict single-flight execution — one generation at a time — and
/// gains a bounded queue with deterministic overload behavior plus a
/// cancellable lifecycle. This actor does not run generations itself: it
/// hands out exactly one slot, in a deterministic order, and the caller runs
/// the operation while holding it.
///
/// Contract (the wire-level version is in `docs/INFERENCE-SCHEDULING.md`):
/// - Queue capacity is bounded (`maxQueueDepth` waiting requests). A request
///   arriving while the queue is full is rejected immediately with
///   `AdmissionError.queueFull` — never enqueued, never run.
/// - A queued request waits at most `queueWaitTimeoutMilliseconds` (0 =
///   unlimited) and is then rejected with `.queueWaitTimeout` without
///   running.
/// - Ordering is deterministic: interactive before background, FIFO within
///   a class. Priority orders the queue only; it cannot preempt the running
///   generation.
/// - Cancelling the caller's task while queued removes the request without
///   running it. Cancelling while active reaches the operation (whose own
///   cancellation handling stops the generation) and the slot is released
///   exactly once, after the operation returns — so a cancelled generation
///   still holds the slot until its cleanup (GPU drain, cache store) is
///   done.
public actor RequestScheduler {
    /// Admission bounds, snapshotted from `ServerConfig` at startup.
    public struct Limits: Sendable, Equatable {
        /// Maximum number of requests waiting for the slot (0 = no queue:
        /// reject immediately whenever the slot is busy).
        public var maxQueueDepth: Int
        /// Queue wait budget in milliseconds (0 = unlimited).
        public var queueWaitTimeoutMilliseconds: Int

        public init(maxQueueDepth: Int, queueWaitTimeoutMilliseconds: Int) {
            self.maxQueueDepth = maxQueueDepth
            self.queueWaitTimeoutMilliseconds = queueWaitTimeoutMilliseconds
        }
    }

    /// Cumulative scheduling counters, surfaced by `/v1/mei/status`.
    ///
    /// - `admitted`: requests that started running (granted a slot and not
    ///   cancelled before start).
    /// - `completed` / `failed`: runs that ended without cancellation, by
    ///   success or thrown error.
    /// - `cancelledQueued`: cancelled while waiting, or granted and
    ///   cancelled before the operation started. No generation work happened.
    /// - `cancelledActive`: cancelled after the operation started.
    /// - `rejectedQueueFull` / `rejectedTimeout`: admission refusals.
    /// - `slotReleases`: every release of the execution slot (the
    ///   "slot-release signal": a client watching this counter advance knows
    ///   the slot changed hands).
    /// - `lastQueueWaitMilliseconds` / `maxQueueWaitMilliseconds`: queue wait
    ///   of the most recently granted request, and the running maximum.
    /// - `lastCancelToReleaseMilliseconds` /
    ///   `maxCancelToReleaseMilliseconds`: time from the first observed
    ///   cancellation of an active request to the slot release. Nil until
    ///   the first active cancellation.
    public struct Counters: Sendable, Equatable {
        public var admitted = 0
        public var completed = 0
        public var failed = 0
        public var cancelledQueued = 0
        public var cancelledActive = 0
        public var rejectedQueueFull = 0
        public var rejectedTimeout = 0
        public var slotReleases = 0
        public var lastQueueWaitMilliseconds: Double = 0
        public var maxQueueWaitMilliseconds: Double = 0
        public var lastCancelToReleaseMilliseconds: Double?
        public var maxCancelToReleaseMilliseconds: Double?
    }

    /// One consistent read of the scheduler's state.
    public struct Snapshot: Sendable, Equatable {
        public var running: Bool
        public var queueDepth: Int
        public var queuedInteractive: Int
        public var queuedBackground: Int
        public var limits: Limits
        public var counters: Counters
    }

    /// How a granted slot was given back.
    private enum ReleaseOutcome {
        case completed
        case failed
        case cancelled(observedAt: Date)
        /// Granted, then cancelled before the operation started.
        case cancelledBeforeStart
    }

    /// One queued request's wait state. Confined to the actor. `continuation`
    /// is non-nil exactly while the request is waiting, and is cleared
    /// (exactly once) by whichever path settles it first: grant, cancel, or
    /// timeout — the exactly-once guarantee for the slot's wait side.
    private final class Waiter {
        let id: UInt64
        let priority: RequestPriority
        let enqueuedAt: Date
        var continuation: CheckedContinuation<SlotGrant, Error>?
        var timeoutTask: Task<Void, Never>?

        init(id: UInt64, priority: RequestPriority, enqueuedAt: Date) {
            self.id = id
            self.priority = priority
            self.enqueuedAt = enqueuedAt
        }
    }

    private let limits: Limits
    private var running = false
    private var waiters: [Waiter] = []
    private var nextWaiterID: UInt64 = 0
    private var counters = Counters()

    public init(limits: Limits) {
        self.limits = limits
    }

    /// The live scheduling snapshot for `/v1/mei/status`.
    public func snapshot() -> Snapshot {
        Snapshot(
            running: running,
            queueDepth: waiters.count,
            queuedInteractive: waiters.filter { $0.priority == .interactive }.count,
            queuedBackground: waiters.filter { $0.priority == .background }.count,
            limits: limits,
            counters: counters)
    }

    /// Run `operation` while holding the single execution slot, after
    /// bounded admission.
    ///
    /// Throws `AdmissionError` when the queue is full or the wait budget
    /// expires, and `CancellationError` when the caller's task is cancelled
    /// while queued (the operation then never runs). The slot is released
    /// exactly once — on the operation's return or throw — so a cancelled
    /// generation still holds the slot until its own cleanup is complete.
    public func submit<T: Sendable>(
        priority: RequestPriority,
        operation: @escaping @Sendable (SlotGrant) async throws -> T
    ) async throws -> T {
        let grant = try await acquire(priority: priority)
        // From here on the slot is held. A cancellation between the grant
        // and the operation start must not run the operation at all.
        if Task.isCancelled {
            release(grant: grant, outcome: .cancelledBeforeStart)
            throw CancellationError()
        }
        counters.admitted += 1
        let cancellationObservedAt = CancellationTimestamp()
        do {
            let result = try await withTaskCancellationHandler {
                try await operation(grant)
            } onCancel: {
                cancellationObservedAt.record()
            }
            release(
                grant: grant,
                outcome: releaseOutcome(
                    error: nil, cancellationObservedAt: cancellationObservedAt))
            return result
        } catch {
            release(
                grant: grant,
                outcome: releaseOutcome(
                    error: error, cancellationObservedAt: cancellationObservedAt))
            throw error
        }
    }

    // MARK: - Slot acquisition

    /// Wait for the execution slot, or fail fast per the admission limits.
    ///
    /// All state changes happen synchronously inside the continuation body
    /// (or in the settle paths), so admission decisions are atomic on the
    /// actor: the idle fast path, the queue-full rejection, and the enqueue
    /// cannot interleave with another request.
    private func acquire(priority: RequestPriority) async throws -> SlotGrant {
        let waiterID = nextWaiterID
        nextWaiterID += 1
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<SlotGrant, Error>) in
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                if !running && waiters.isEmpty {
                    running = true
                    continuation.resume(returning: SlotGrant(
                        queueWaitMilliseconds: 0, priority: priority))
                    return
                }
                // Bounded queue: reject immediately (never enqueue) when
                // full. This is the deterministic overload path.
                guard waiters.count < limits.maxQueueDepth else {
                    counters.rejectedQueueFull += 1
                    continuation.resume(throwing: AdmissionError.queueFull(
                        limit: limits.maxQueueDepth))
                    return
                }
                let waiter = Waiter(id: waiterID, priority: priority, enqueuedAt: Date())
                waiter.continuation = continuation
                waiters.append(waiter)
                if limits.queueWaitTimeoutMilliseconds > 0 {
                    let budget = limits.queueWaitTimeoutMilliseconds
                    waiter.timeoutTask = Task { [weak self] in
                        try? await Task.sleep(for: .milliseconds(budget))
                        guard !Task.isCancelled else { return }
                        await self?.expireWaiter(id: waiterID)
                    }
                }
            }
        } onCancel: {
            Task { [weak self] in await self?.cancelWaiter(id: waiterID) }
        }
    }

    /// Settle a waiting request whose caller's task was cancelled. No-op when
    /// the request was already granted (the grant path's cancellation check
    /// settles that case) or already settled.
    private func cancelWaiter(id: UInt64) {
        guard let index = waiters.firstIndex(where: { $0.id == id }),
            let continuation = waiters[index].continuation
        else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation = nil
        waiter.timeoutTask?.cancel()
        counters.cancelledQueued += 1
        continuation.resume(throwing: CancellationError())
    }

    /// Settle a waiting request whose queue-wait budget expired.
    private func expireWaiter(id: UInt64) {
        guard let index = waiters.firstIndex(where: { $0.id == id }),
            let continuation = waiters[index].continuation
        else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation = nil
        counters.rejectedTimeout += 1
        continuation.resume(throwing: AdmissionError.queueWaitTimeout(
            milliseconds: limits.queueWaitTimeoutMilliseconds))
    }

    /// Hand the slot to the best waiting request: interactive before
    /// background, FIFO within a class. Deterministic by construction —
    /// waiter ids increase with arrival and the scan picks the lowest
    /// (rank, id) pair.
    private func grantNextWaiter() {
        while !waiters.isEmpty {
            var bestIndex = 0
            for index in waiters.indices.dropFirst() {
                let candidate = waiters[index]
                let best = waiters[bestIndex]
                if candidate.priority.rank < best.priority.rank
                    || (candidate.priority.rank == best.priority.rank
                        && candidate.id < best.id)
                {
                    bestIndex = index
                }
            }
            let waiter = waiters.remove(at: bestIndex)
            waiter.timeoutTask?.cancel()
            waiter.timeoutTask = nil
            guard let continuation = waiter.continuation else { continue }
            waiter.continuation = nil
            running = true
            let queueWait = Date().timeIntervalSince(waiter.enqueuedAt) * 1000
            counters.lastQueueWaitMilliseconds = queueWait
            counters.maxQueueWaitMilliseconds = max(counters.maxQueueWaitMilliseconds, queueWait)
            continuation.resume(returning: SlotGrant(
                queueWaitMilliseconds: queueWait, priority: waiter.priority))
            return
        }
    }

    // MARK: - Slot release

    /// Classify how a granted slot was given back. Exactly one of
    /// `submit`'s success/catch paths calls this per granted request.
    private func releaseOutcome(
        error: Error?, cancellationObservedAt: CancellationTimestamp
    ) -> ReleaseOutcome {
        if Task.isCancelled || error is CancellationError {
            return .cancelled(observedAt: cancellationObservedAt.recorded ?? Date())
        }
        return error == nil ? .completed : .failed
    }

    /// Release the slot exactly once and hand it to the next waiter.
    private func release(grant: SlotGrant, outcome: ReleaseOutcome) {
        counters.slotReleases += 1
        switch outcome {
        case .completed:
            counters.completed += 1
        case .failed:
            counters.failed += 1
        case .cancelled(let observedAt):
            counters.cancelledActive += 1
            let delta = Date().timeIntervalSince(observedAt) * 1000
            counters.lastCancelToReleaseMilliseconds = delta
            counters.maxCancelToReleaseMilliseconds = max(
                counters.maxCancelToReleaseMilliseconds ?? 0, delta)
        case .cancelledBeforeStart:
            // Granted but never started: counted with queued cancellation —
            // no generation work happened, and no cancel-to-release timing
            // sample is meaningful.
            counters.cancelledQueued += 1
        }
        running = false
        grantNextWaiter()
    }
}
