import XCTest

@testable import MeiCore

/// Deterministic unit tests for bounded admission in front of Mei's single
/// execution slot (`RequestScheduler`).
///
/// Kiem plan `820814e1` stages 3–4. Every ordering assertion is exact: the
/// tests hold the slot open with explicit latches instead of sleeps, and
/// only the "wait until this state exists" helpers are time-based. What is
/// pinned here:
/// - idle admission and queue wait accounting,
/// - FIFO within a priority class and interactive-before-background across
///   classes (arrival order interleaved on purpose),
/// - queue-full refusal that never starts the operation,
/// - queue-wait timeout refusal that never starts the operation,
/// - queued cancellation removing the request without running it,
/// - active cancellation stopping the operation and releasing the slot
///   exactly once, with cancellation-to-release timing recorded,
/// - the snapshot the status endpoint exposes.
final class RequestSchedulerTests: XCTestCase {

    private func scheduler(maxQueueDepth: Int = 8, waitMs: Int = 0) -> RequestScheduler {
        RequestScheduler(limits: .init(
            maxQueueDepth: maxQueueDepth,
            queueWaitTimeoutMilliseconds: waitMs))
    }

    // MARK: - Idle admission

    func testIdleSlotAdmitsImmediatelyAndReleasesExactlyOnce() async throws {
        let scheduler = self.scheduler()
        let grant = try await scheduler.submit(priority: .interactive) { $0 }
        XCTAssertEqual(grant.priority, .interactive)
        XCTAssertEqual(grant.queueWaitMilliseconds, 0)

        let snapshot = await scheduler.snapshot()
        XCTAssertFalse(snapshot.running)
        XCTAssertEqual(snapshot.queueDepth, 0)
        XCTAssertEqual(snapshot.counters.admitted, 1)
        XCTAssertEqual(snapshot.counters.completed, 1)
        XCTAssertEqual(snapshot.counters.slotReleases, 1)
    }

    func testFailedOperationIsCountedAndReleasesTheSlot() async throws {
        let scheduler = self.scheduler()
        struct Boom: Error {}
        do {
            _ = try await scheduler.submit(priority: .interactive) { _ -> Int in throw Boom() }
            XCTFail("expected the operation's error to propagate")
        } catch is Boom {
        }
        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.counters.admitted, 1)
        XCTAssertEqual(snapshot.counters.failed, 1)
        XCTAssertEqual(snapshot.counters.completed, 0)
        XCTAssertEqual(snapshot.counters.slotReleases, 1)
        XCTAssertFalse(snapshot.running)

        // The slot is immediately reusable.
        let grant = try await scheduler.submit(priority: .interactive) { $0 }
        XCTAssertEqual(grant.priority, .interactive)
    }

    // MARK: - Deterministic ordering

    func testQueuedRequestsRunFIFOWithinOnePriorityClass() async throws {
        let scheduler = self.scheduler()
        let firstStarted = TestLatch()
        let releaseFirst = TestLatch()
        let order = TestRecorder()

        let first = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                order.record("first")
                firstStarted.open()
                await releaseFirst.wait()
            }
        }
        await firstStarted.wait()

        let second = Task {
            try await scheduler.submit(priority: .interactive) { _ in order.record("second") }
        }
        await waitUntil("second queued") { await scheduler.snapshot().queueDepth == 1 }
        let third = Task {
            try await scheduler.submit(priority: .interactive) { _ in order.record("third") }
        }
        await waitUntil("third queued") { await scheduler.snapshot().queueDepth == 2 }

        releaseFirst.open()
        try await first.value
        try await second.value
        try await third.value
        XCTAssertEqual(order.recorded, ["first", "second", "third"])
    }

    func testQueuedInteractiveOutranksQueuedBackground() async throws {
        let scheduler = self.scheduler()
        let firstStarted = TestLatch()
        let releaseFirst = TestLatch()
        let order = TestRecorder()

        // The running request is background so the class of the *active*
        // request cannot influence the queue order under test.
        let first = Task {
            try await scheduler.submit(priority: .background) { _ in
                order.record("running")
                firstStarted.open()
                await releaseFirst.wait()
            }
        }
        await firstStarted.wait()

        // Arrival order interleaves the classes: background, interactive,
        // background. Execution order must be interactive first, then the
        // two background requests in arrival order.
        let bg1 = Task {
            try await scheduler.submit(priority: .background) { _ in order.record("bg1") }
        }
        await waitUntil("bg1 queued") { await scheduler.snapshot().queueDepth == 1 }
        let int1 = Task {
            try await scheduler.submit(priority: .interactive) { _ in order.record("int1") }
        }
        await waitUntil("int1 queued") { await scheduler.snapshot().queueDepth == 2 }
        let bg2 = Task {
            try await scheduler.submit(priority: .background) { _ in order.record("bg2") }
        }
        await waitUntil("bg2 queued") { await scheduler.snapshot().queueDepth == 3 }

        releaseFirst.open()
        try await first.value
        try await bg1.value
        try await int1.value
        try await bg2.value
        XCTAssertEqual(order.recorded, ["running", "int1", "bg1", "bg2"])
    }

    func testQueuedRequestReportsItsQueueWaitOnGrant() async throws {
        let scheduler = self.scheduler()
        let firstStarted = TestLatch()
        let releaseFirst = TestLatch()

        let first = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                firstStarted.open()
                await releaseFirst.wait()
            }
        }
        await firstStarted.wait()

        let second = Task { () -> SlotGrant in
            try await scheduler.submit(priority: .interactive) { $0 }
        }
        await waitUntil("second queued") { await scheduler.snapshot().queueDepth == 1 }
        try await Task.sleep(for: .milliseconds(30))
        releaseFirst.open()

        let grant = try await second.value
        XCTAssertGreaterThanOrEqual(grant.queueWaitMilliseconds, 20)
        try await first.value

        let snapshot = await scheduler.snapshot()
        XCTAssertGreaterThanOrEqual(snapshot.counters.maxQueueWaitMilliseconds, 20)
        XCTAssertGreaterThanOrEqual(
            snapshot.counters.maxQueueWaitMilliseconds,
            snapshot.counters.lastQueueWaitMilliseconds)
    }

    // MARK: - Bounded queue and overload refusals

    func testQueueFullRejectsImmediatelyAndNeverStarts() async throws {
        let scheduler = self.scheduler(maxQueueDepth: 1)
        let firstStarted = TestLatch()
        let releaseFirst = TestLatch()
        let order = TestRecorder()

        let first = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                firstStarted.open()
                await releaseFirst.wait()
            }
        }
        await firstStarted.wait()

        let queued = Task {
            try await scheduler.submit(priority: .interactive) { _ in order.record("queued") }
        }
        await waitUntil("one waiter queued") { await scheduler.snapshot().queueDepth == 1 }

        do {
            _ = try await scheduler.submit(priority: .interactive) { _ in order.record("rejected") }
            XCTFail("expected a queueFull refusal")
        } catch let error as AdmissionError {
            XCTAssertEqual(error, .queueFull(limit: 1))
            XCTAssertEqual(error.code, "queue_full")
        }

        releaseFirst.open()
        try await first.value
        try await queued.value
        XCTAssertEqual(order.recorded, ["queued"])

        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.counters.rejectedQueueFull, 1)
        XCTAssertEqual(snapshot.counters.admitted, 2)
        XCTAssertEqual(snapshot.counters.slotReleases, 2)
        XCTAssertEqual(snapshot.queueDepth, 0)
    }

    func testZeroQueueDepthRefusesWheneverTheSlotIsBusy() async throws {
        let scheduler = self.scheduler(maxQueueDepth: 0)
        let firstStarted = TestLatch()
        let releaseFirst = TestLatch()

        let first = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                firstStarted.open()
                await releaseFirst.wait()
            }
        }
        await firstStarted.wait()
        do {
            _ = try await scheduler.submit(priority: .interactive) { $0 }
            XCTFail("expected a queueFull refusal with zero queue depth")
        } catch let error as AdmissionError {
            XCTAssertEqual(error, .queueFull(limit: 0))
        }
        releaseFirst.open()
        try await first.value
        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.counters.rejectedQueueFull, 1)
        XCTAssertEqual(snapshot.counters.slotReleases, 1)
    }

    func testQueueWaitTimeoutRefusesWithoutRunning() async throws {
        let scheduler = self.scheduler(maxQueueDepth: 4, waitMs: 60)
        let firstStarted = TestLatch()
        let releaseFirst = TestLatch()
        let order = TestRecorder()

        let first = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                firstStarted.open()
                await releaseFirst.wait()
            }
        }
        await firstStarted.wait()

        do {
            _ = try await scheduler.submit(priority: .background) { _ in order.record("timed-out") }
            XCTFail("expected a queueWaitTimeout refusal")
        } catch let error as AdmissionError {
            XCTAssertEqual(error, .queueWaitTimeout(milliseconds: 60))
            XCTAssertEqual(error.code, "queue_timeout")
        }
        XCTAssertEqual(order.recorded, [], "a refused request must never start")

        releaseFirst.open()
        try await first.value
        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.counters.rejectedTimeout, 1)
        XCTAssertEqual(snapshot.queueDepth, 0)

        // The slot stays healthy after the refusal.
        let grant = try await scheduler.submit(priority: .interactive) { $0 }
        XCTAssertEqual(grant.priority, .interactive)
    }

    func testUnlimitedQueueWaitIsNotTimedOut() async throws {
        let scheduler = self.scheduler(maxQueueDepth: 4, waitMs: 0)
        let firstStarted = TestLatch()
        let releaseFirst = TestLatch()

        let first = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                firstStarted.open()
                await releaseFirst.wait()
            }
        }
        await firstStarted.wait()

        let queued = Task { () -> SlotGrant in
            try await scheduler.submit(priority: .background) { $0 }
        }
        await waitUntil("queued") { await scheduler.snapshot().queueDepth == 1 }
        try await Task.sleep(for: .milliseconds(150))
        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.counters.rejectedTimeout, 0)
        XCTAssertEqual(snapshot.queueDepth, 1)

        releaseFirst.open()
        let grant = try await queued.value
        XCTAssertEqual(grant.priority, .background)
        try await first.value
    }

    // MARK: - Cancellation

    func testCancellingAQueuedRequestRemovesItWithoutRunning() async throws {
        let scheduler = self.scheduler()
        let firstStarted = TestLatch()
        let releaseFirst = TestLatch()
        let order = TestRecorder()

        let first = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                firstStarted.open()
                await releaseFirst.wait()
            }
        }
        await firstStarted.wait()

        let queued = Task {
            try await scheduler.submit(priority: .interactive) { _ in order.record("cancelled") }
        }
        await waitUntil("queued") { await scheduler.snapshot().queueDepth == 1 }
        queued.cancel()
        do {
            try await queued.value
            XCTFail("expected CancellationError")
        } catch {
            XCTAssertTrue(error is CancellationError, "got \(error)")
        }
        XCTAssertEqual(order.recorded, [], "a cancelled queued request must never start")

        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.counters.cancelledQueued, 1)
        XCTAssertEqual(snapshot.counters.slotReleases, 0)
        XCTAssertEqual(snapshot.queueDepth, 0)

        releaseFirst.open()
        try await first.value
        // The cancelled request consumed no slot: the next request runs.
        try await scheduler.submit(priority: .interactive) { _ in order.record("next") }
        XCTAssertEqual(order.recorded, ["next"])
    }

    func testCancellingAnActiveRequestStopsItAndReleasesTheSlotExactlyOnce() async throws {
        let scheduler = self.scheduler()
        let started = TestLatch()
        let outcome = TestRecorder()

        let active = Task { () -> Error? in
            do {
                try await scheduler.submit(priority: .interactive) { _ in
                    started.open()
                    do {
                        try await Task.sleep(for: .seconds(60))
                        outcome.record("finished")
                    } catch {
                        outcome.record("cancelled")
                        throw error
                    }
                }
                return nil
            } catch {
                return error
            }
        }
        await started.wait()
        active.cancel()
        let error = await active.value
        XCTAssertTrue(error is CancellationError, "got \(String(describing: error))")
        XCTAssertEqual(outcome.recorded, ["cancelled"], "the operation must observe the cancellation")

        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.counters.cancelledActive, 1)
        XCTAssertEqual(snapshot.counters.completed, 0)
        XCTAssertEqual(snapshot.counters.slotReleases, 1)
        XCTAssertFalse(snapshot.running)
        XCTAssertNotNil(snapshot.counters.lastCancelToReleaseMilliseconds)
        XCTAssertNotNil(snapshot.counters.maxCancelToReleaseMilliseconds)

        // The slot is released: the next request runs immediately.
        let grant = try await scheduler.submit(priority: .background) { $0 }
        XCTAssertEqual(grant.priority, .background)
        XCTAssertLessThan(grant.queueWaitMilliseconds, 100)
    }

    func testCancelledBeforeStartNeverRunsTheOperationOrLeaksTheSlot() async throws {
        let scheduler = self.scheduler()
        let ran = TestRecorder()

        let task = Task { () -> Error? in
            do {
                try await scheduler.submit(priority: .interactive) { _ in ran.record("ran") }
                return nil
            } catch {
                return error
            }
        }
        task.cancel()
        let error = await task.value
        XCTAssertTrue(error is CancellationError, "got \(String(describing: error))")
        XCTAssertEqual(ran.recorded, [])

        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.counters.admitted, 0, "the operation never started")
        XCTAssertFalse(snapshot.running, "no slot may leak")

        // The slot is still usable afterwards.
        try await scheduler.submit(priority: .interactive) { _ in ran.record("after") }
        XCTAssertEqual(ran.recorded, ["after"])
    }

    // MARK: - Snapshot

    func testSnapshotReportsQueueMixAndLimits() async throws {
        let scheduler = self.scheduler(maxQueueDepth: 5, waitMs: 1234)
        let firstStarted = TestLatch()
        let releaseFirst = TestLatch()

        let first = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                firstStarted.open()
                await releaseFirst.wait()
            }
        }
        await firstStarted.wait()

        let bg = Task { try await scheduler.submit(priority: .background) { _ in } }
        await waitUntil("bg queued") { await scheduler.snapshot().queueDepth == 1 }
        let int = Task { try await scheduler.submit(priority: .interactive) { _ in } }
        await waitUntil("int queued") { await scheduler.snapshot().queueDepth == 2 }

        let snapshot = await scheduler.snapshot()
        XCTAssertTrue(snapshot.running)
        XCTAssertEqual(snapshot.queueDepth, 2)
        XCTAssertEqual(snapshot.queuedInteractive, 1)
        XCTAssertEqual(snapshot.queuedBackground, 1)
        XCTAssertEqual(snapshot.limits.maxQueueDepth, 5)
        XCTAssertEqual(snapshot.limits.queueWaitTimeoutMilliseconds, 1234)

        releaseFirst.open()
        try await first.value
        try await bg.value
        try await int.value
    }
}
