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

    private func scheduler(
        maxQueueDepth: Int = 8, waitMs: Int = 0, maxActive: Int = 1
    ) -> RequestScheduler {
        RequestScheduler(limits: .init(
            maxQueueDepth: maxQueueDepth,
            queueWaitTimeoutMilliseconds: waitMs,
            maxActiveGenerations: maxActive))
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
        // Strict single-flight is the default: the wire-compatible `running`
        // flag and the new explicit active count agree, and the limit is 1.
        XCTAssertEqual(snapshot.activeGenerations, 1)
        XCTAssertEqual(snapshot.limits.maxActiveGenerations, 1)
        XCTAssertEqual(snapshot.counters.peakActiveGenerations, 1)

        releaseFirst.open()
        try await first.value
        try await bg.value
        try await int.value
    }

    // MARK: - Bounded overlap (opt-in, maxActiveGenerations = 2)

    /// The default stays strict single-flight: with the default limits a
    /// second request queues while the first is active, exactly as 0.8.0
    /// behaved. This is the byte-compatibility guard for the new field.
    func testDefaultRemainsStrictSingleFlight() async throws {
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

        let second = Task { try await scheduler.submit(priority: .interactive) { _ in } }
        await waitUntil("second queues behind the single active slot") {
            await scheduler.snapshot().queueDepth == 1
        }
        let mid = await scheduler.snapshot()
        XCTAssertEqual(mid.activeGenerations, 1)
        XCTAssertEqual(mid.limits.maxActiveGenerations, 1)
        XCTAssertTrue(mid.running)

        releaseFirst.open()
        try await first.value
        try await second.value
        let snapshot = await scheduler.snapshot()
        XCTAssertEqual(snapshot.counters.peakActiveGenerations, 1)
        XCTAssertEqual(snapshot.counters.admitted, 2)
        XCTAssertEqual(snapshot.counters.completed, 2)
    }

    /// Opt-in overlap: two requests run at once, a third queues, and a
    /// release hands the free slot to the queued request — never a third
    /// concurrent generation.
    func testOverlapAdmitsTwoActiveGenerationsAndNeverThree() async throws {
        let scheduler = self.scheduler(maxActive: 2)
        let aStarted = TestLatch()
        let bStarted = TestLatch()
        let releaseA = TestLatch()
        let releaseB = TestLatch()
        let order = TestRecorder()

        let a = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                order.record("a-start")
                aStarted.open()
                await releaseA.wait()
                order.record("a-end")
            }
        }
        await aStarted.wait()

        // Second arrival while one slot is taken: admitted immediately.
        let b = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                order.record("b-start")
                bStarted.open()
                await releaseB.wait()
                order.record("b-end")
            }
        }
        await bStarted.wait()

        let both = await scheduler.snapshot()
        XCTAssertEqual(both.activeGenerations, 2, "both slots must be active")
        XCTAssertEqual(both.counters.peakActiveGenerations, 2)
        XCTAssertEqual(both.queueDepth, 0)

        // Third arrival: every slot is taken, so it queues.
        let cStarted = TestLatch()
        let releaseC = TestLatch()
        let c = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                order.record("c-start")
                cStarted.open()
                await releaseC.wait()
            }
        }
        await waitUntil("third request queues") { await scheduler.snapshot().queueDepth == 1 }
        let queued = await scheduler.snapshot()
        XCTAssertEqual(queued.activeGenerations, 2, "the queued request must not start a third generation")

        // One release admits exactly the queued request.
        releaseA.open()
        try await a.value
        await cStarted.wait()
        let afterGrant = await scheduler.snapshot()
        XCTAssertEqual(afterGrant.activeGenerations, 2, "b + c active")
        XCTAssertEqual(afterGrant.counters.peakActiveGenerations, 2)
        XCTAssertEqual(afterGrant.queueDepth, 0)

        releaseB.open()
        try await b.value
        releaseC.open()
        try await c.value

        let final = await scheduler.snapshot()
        XCTAssertEqual(final.counters.admitted, 3)
        XCTAssertEqual(final.counters.completed, 3)
        XCTAssertEqual(final.counters.slotReleases, 3)
        XCTAssertEqual(final.counters.peakActiveGenerations, 2)
        XCTAssertFalse(final.running)
        XCTAssertEqual(order.recorded, ["a-start", "b-start", "a-end", "c-start", "b-end"])
    }

    /// Overload under overlap: with both slots taken and the queue at its
    /// bound, an arrival is refused immediately (429 path) and never starts.
    func testOverlapQueueFullRefusesWithoutStarting() async throws {
        let scheduler = self.scheduler(maxQueueDepth: 1, maxActive: 2)
        let aStarted = TestLatch()
        let bStarted = TestLatch()
        let releaseA = TestLatch()
        let releaseB = TestLatch()
        let order = TestRecorder()

        let a = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                aStarted.open()
                await releaseA.wait()
            }
        }
        await aStarted.wait()
        let b = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                bStarted.open()
                await releaseB.wait()
            }
        }
        await bStarted.wait()

        let queued = Task {
            try await scheduler.submit(priority: .background) { _ in order.record("queued") }
        }
        await waitUntil("queue reaches its bound") { await scheduler.snapshot().queueDepth == 1 }

        do {
            _ = try await scheduler.submit(priority: .interactive) { _ in order.record("rejected") }
            XCTFail("expected a queueFull refusal with both slots busy")
        } catch let error as AdmissionError {
            XCTAssertEqual(error, .queueFull(limit: 1))
        }

        releaseA.open()
        releaseB.open()
        try await a.value
        try await b.value
        try await queued.value
        XCTAssertEqual(order.recorded, ["queued"], "a refused request must never start")

        let final = await scheduler.snapshot()
        XCTAssertEqual(final.counters.rejectedQueueFull, 1)
        XCTAssertEqual(final.counters.admitted, 3)
        XCTAssertEqual(final.counters.peakActiveGenerations, 2)
    }

    /// Overlap keeps the 0.8.0 ordering contract: a queued interactive
    /// request outranks a queued background request, FIFO within a class,
    /// when slots free up.
    func testOverlapPreservesPriorityOrderingAcrossFreedSlots() async throws {
        let scheduler = self.scheduler(maxActive: 2)
        let aStarted = TestLatch()
        let bStarted = TestLatch()
        let releaseA = TestLatch()
        let releaseB = TestLatch()
        let order = TestRecorder()

        let a = Task {
            try await scheduler.submit(priority: .background) { _ in
                order.record("a-running")
                aStarted.open()
                await releaseA.wait()
            }
        }
        await aStarted.wait()
        let b = Task {
            try await scheduler.submit(priority: .background) { _ in
                order.record("b-running")
                bStarted.open()
                await releaseB.wait()
            }
        }
        await bStarted.wait()

        let bg = Task {
            try await scheduler.submit(priority: .background) { _ in order.record("bg") }
        }
        await waitUntil("bg queued") { await scheduler.snapshot().queueDepth == 1 }
        let int = Task {
            try await scheduler.submit(priority: .interactive) { _ in order.record("int") }
        }
        await waitUntil("int queued") { await scheduler.snapshot().queueDepth == 2 }

        // Free one slot: the queued INTERACTIVE request must take it even
        // though the background request arrived first.
        releaseA.open()
        try await a.value
        try await int.value
        releaseB.open()
        try await b.value
        try await bg.value

        XCTAssertEqual(order.recorded, ["a-running", "b-running", "int", "bg"])
        let final = await scheduler.snapshot()
        XCTAssertEqual(final.counters.admitted, 4)
        XCTAssertEqual(final.counters.completed, 4)
    }

    /// Cancelling one of two active requests releases exactly its own slot;
    /// the other keeps running, and the freed slot is immediately usable.
    func testOverlapCancellationReleasesOnlyItsOwnSlot() async throws {
        let scheduler = self.scheduler(maxActive: 2)
        let aStarted = TestLatch()
        let bStarted = TestLatch()
        let releaseA = TestLatch()
        let outcome = TestRecorder()

        let a = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                aStarted.open()
                await releaseA.wait()
                outcome.record("a-finished")
            }
        }
        await aStarted.wait()

        let b = Task { () -> Error? in
            do {
                try await scheduler.submit(priority: .interactive) { _ in
                    bStarted.open()
                    do {
                        try await Task.sleep(for: .seconds(60))
                        outcome.record("b-finished")
                    } catch {
                        outcome.record("b-cancelled")
                        throw error
                    }
                }
                return nil
            } catch {
                return error
            }
        }
        await bStarted.wait()

        b.cancel()
        let error = await b.value
        XCTAssertTrue(error is CancellationError, "got \(String(describing: error))")
        XCTAssertEqual(outcome.recorded, ["b-cancelled"])

        let mid = await scheduler.snapshot()
        XCTAssertEqual(mid.activeGenerations, 1, "only b's slot may be released")
        XCTAssertTrue(mid.running, "a must still be running")
        XCTAssertEqual(mid.counters.cancelledActive, 1)
        XCTAssertEqual(mid.counters.slotReleases, 1)
        XCTAssertNotNil(mid.counters.lastCancelToReleaseMilliseconds)

        // The freed slot is immediately usable while a still runs.
        let cStarted = TestLatch()
        let releaseC = TestLatch()
        let c = Task {
            try await scheduler.submit(priority: .background) { grant in
                XCTAssertLessThan(grant.queueWaitMilliseconds, 100,
                                  "the freed slot must be granted without queueing")
                cStarted.open()
                await releaseC.wait()
            }
        }
        await cStarted.wait()
        let withC = await scheduler.snapshot()
        XCTAssertEqual(withC.activeGenerations, 2)

        releaseC.open()
        releaseA.open()
        try await a.value
        try await c.value

        let final = await scheduler.snapshot()
        XCTAssertEqual(final.counters.peakActiveGenerations, 2)
        XCTAssertEqual(final.counters.completed, 2)
        XCTAssertFalse(final.running)
    }

    /// Two queued requests are granted in queue order as the two slots free
    /// up, and each reports its real queue wait.
    func testOverlapQueuedRequestsAreGrantedInOrderAsSlotsFree() async throws {
        let scheduler = self.scheduler(maxActive: 2)
        let aStarted = TestLatch()
        let bStarted = TestLatch()
        let releaseA = TestLatch()
        let releaseB = TestLatch()
        let order = TestRecorder()

        let a = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                aStarted.open()
                await releaseA.wait()
            }
        }
        await aStarted.wait()
        let b = Task {
            try await scheduler.submit(priority: .interactive) { _ in
                bStarted.open()
                await releaseB.wait()
            }
        }
        await bStarted.wait()

        let w1 = Task { () -> SlotGrant in
            try await scheduler.submit(priority: .interactive) { grant in
                order.record("w1")
                return grant
            }
        }
        await waitUntil("w1 queued") { await scheduler.snapshot().queueDepth == 1 }
        let w2 = Task { () -> SlotGrant in
            try await scheduler.submit(priority: .interactive) { grant in
                order.record("w2")
                return grant
            }
        }
        await waitUntil("w2 queued") { await scheduler.snapshot().queueDepth == 2 }

        try await Task.sleep(for: .milliseconds(30))
        releaseA.open()
        let g1 = try await w1.value
        XCTAssertGreaterThanOrEqual(g1.queueWaitMilliseconds, 20)

        releaseB.open()
        let g2 = try await w2.value
        // w2 was enqueued after w1 and granted after it; both waited through
        // at least the remaining active generation, so both waits are real.
        XCTAssertGreaterThanOrEqual(g2.queueWaitMilliseconds, 20)

        try await a.value
        try await b.value
        XCTAssertEqual(order.recorded, ["w1", "w2"])
        let final = await scheduler.snapshot()
        XCTAssertEqual(final.counters.admitted, 4)
        XCTAssertEqual(final.counters.completed, 4)
        XCTAssertEqual(final.counters.peakActiveGenerations, 2)
    }
}
