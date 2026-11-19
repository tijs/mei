import Foundation
import XCTest

@testable import MeiCore

private enum ReplayRequest {
  @TaskLocal static var id: Int = -1
}

/// Cancellation-aware manual deadlines. Releasing one request's deadline
/// cannot expire a peer, and cancelled sleeps cannot leave continuations behind.
private actor ReplayClock {
  private var pending: [Int: CheckedContinuation<Void, Error>] = [:]
  func sleep() async throws {
    let id = ReplayRequest.id
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, Error>) in
        if Task.isCancelled {
          continuation.resume(throwing: CancellationError())
        } else {
          pending[id] = continuation
        }
      }
    } onCancel: {
      Task { await self.cancel(id) }
    }
  }
  func contains(_ id: Int) -> Bool { pending[id] != nil }
  func expire(_ id: Int) { pending.removeValue(forKey: id)?.resume() }
  func cancel(_ id: Int) {
    pending.removeValue(forKey: id)?.resume(throwing: CancellationError())
  }
}

private struct SchedulerTraceState: Decodable {
  let phase: [Int]
  let queue: [Int]
  let cancelled: [Int]
  let failed: [Int]
  let releases: Int
  let grants: Int
  let capacity: Int
  let queueLimit: Int
  let event: String
  let request: Int
}
private struct SchedulerTrace: Decodable { let states: [SchedulerTraceState] }

final class SchedulerTraceReplayTests: XCTestCase {
  func testGeneratedQuintTracesAgreeWithRealScheduler() async throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let fixtureRoot =
      ProcessInfo.processInfo.environment["MEI_SCHEDULER_FIXTURE_ROOT"].map {
        URL(fileURLWithPath: $0)
      } ?? root
    let fixture = fixtureRoot.appendingPathComponent("verification/scheduler/traces.json")
    let traces = try JSONDecoder().decode([SchedulerTrace].self, from: Data(contentsOf: fixture))
    XCTAssertGreaterThanOrEqual(traces.count, 32)
    var covered = Set<String>()
    var capacities = Set<Int>()
    for (index, trace) in traces.enumerated() {
      guard let initial = trace.states.first else {
        XCTFail("empty trace")
        continue
      }
      capacities.insert(initial.capacity)
      let clock = ReplayClock()
      let gate = (0..<4).map { _ in TestLatch() }
      let finish = (0..<4).map { _ in TestLatch() }
      let cleanup = (0..<4).map { _ in TestLatch() }
      let started = TestRecorder()
      let cleaning = TestRecorder()
      let failed = TestRecorder()
      let completed = TestRecorder()
      var expected = RequestScheduler.Counters()
      let scheduler = RequestScheduler(
        limits: .init(
          maxQueueDepth: initial.queueLimit,
          queueWaitTimeoutMilliseconds: 1000,
          maxActiveGenerations: initial.capacity),
        timeoutSleep: { _ in try await clock.sleep() },
        beforeOperation: { await gate[ReplayRequest.id].wait() })
      var tasks: [Int: Task<Void, Error>] = [:]
      defer {
        for (id, task) in tasks {
          task.cancel()
          gate[id].open()
          finish[id].open()
          cleanup[id].open()
        }
      }
      func settle() async throws {
        for (id, task) in tasks {
          task.cancel()
          gate[id].open()
          finish[id].open()
          cleanup[id].open()
        }
        for (id, task) in tasks {
          try await eventually("trace \(index) teardown \(id)") { completed.contains("\(id)") }
          _ = await task.result
        }
        let final = await scheduler.snapshot()
        XCTAssertEqual(final.activeGenerations, 0)
        XCTAssertEqual(final.queueDepth, 0)
      }
      do {
        for state in trace.states.dropFirst() {
          let id = state.request
          covered.insert(state.event)
          switch state.event {
          case "arrive":
            tasks[id] = Task {
              defer { completed.record("\(id)") }
              try await ReplayRequest.$id.withValue(id) {
                try await scheduler.submit(priority: id % 2 == 0 ? .interactive : .background) {
                  _ in
                  started.record("\(id)")
                  await finish[id].wait()
                  cleaning.record("\(id)")
                  await cleanup[id].wait()
                  try Task.checkCancellation()
                  if failed.contains("\(id)") { throw ReplayFailure.operation }
                }
              }
            }
            if state.phase[id] == 1 {
              try await eventually("trace \(index) waiter \(id) deadline registered") {
                await clock.contains(id)
              }
            } else if state.phase[id] == 5 {
              expected.rejectedQueueFull += 1
              try await eventually("queue full settles") { completed.contains("\(id)") }
              do {
                try await tasks[id]!.value
                XCTFail("expected queue full")
              } catch let error as AdmissionError {
                XCTAssertEqual(error, .queueFull(limit: initial.queueLimit))
              }
            }
          case "start":
            expected.admitted += 1
            gate[id].open()
            try await eventually("trace \(index) operation start") { started.contains("\(id)") }
          case "cleanup", "fail":
            if state.event == "fail" { failed.record("\(id)") }
            finish[id].open()
            try await eventually("trace \(index) cleanup start") { cleaning.contains("\(id)") }
          case "cancelActive": tasks[id]!.cancel()
          case "cancelQueued", "cancelGranted":
            tasks[id]!.cancel()
            if state.event == "cancelGranted" { gate[id].open() }
            expected.cancelledQueued += 1
            try await eventually("cancellation settles") { completed.contains("\(id)") }
            do {
              try await tasks[id]!.value
              XCTFail("expected cancellation")
            } catch is CancellationError {}
            XCTAssertFalse(started.contains("\(id)"))
          case "timeout":
            await clock.expire(id)
            expected.rejectedTimeout += 1
            try await eventually("timeout settles") { completed.contains("\(id)") }
            do {
              try await tasks[id]!.value
              XCTFail("expected timeout")
            } catch let error as AdmissionError {
              XCTAssertEqual(error, .queueWaitTimeout(milliseconds: 1000))
            }
            XCTAssertFalse(started.contains("\(id)"))
          case "release":
            cleanup[id].open()
            try await eventually("release settles") { completed.contains("\(id)") }
            let outcome = await tasks[id]!.result
            if state.cancelled.contains(id) {
              expected.cancelledActive += 1
              guard case .failure(let error) = outcome, error is CancellationError else {
                XCTFail("modeled cancellation must throw")
                throw ReplayFailure.outcome
              }
            } else if state.failed.contains(id) {
              expected.failed += 1
              guard case .failure(let error) = outcome, error as? ReplayFailure == .operation else {
                XCTFail("modeled operation failure must throw")
                throw ReplayFailure.outcome
              }
            } else {
              expected.completed += 1
              try outcome.get()
            }
          default: XCTFail("unknown action \(state.event)")
          }
          let active = state.phase.filter { [2, 3, 4].contains($0) }.count
          try await eventually("trace \(index) state after \(state.event)(\(id))") {
            let actual = await scheduler.snapshot()
            return actual.activeGenerations == active && actual.queueDepth == state.queue.count
              && actual.counters.slotReleases == state.releases
          }
          let actual = await scheduler.snapshot()
          XCTAssertEqual(actual.counters.slotReleases, state.releases, "trace \(index)")
          XCTAssertEqual(actual.activeGenerations, active, "trace \(index)")
          XCTAssertEqual(actual.queueDepth, state.queue.count, "trace \(index)")
          XCTAssertEqual(actual.queuedInteractive, state.queue.filter { $0 % 2 == 0 }.count)
          XCTAssertLessThanOrEqual(actual.counters.peakActiveGenerations, initial.capacity)
          XCTAssertEqual(actual.counters.admitted, expected.admitted)
          XCTAssertEqual(actual.counters.completed, expected.completed)
          XCTAssertEqual(actual.counters.failed, expected.failed)
          XCTAssertEqual(actual.counters.cancelledActive, expected.cancelledActive)
          XCTAssertEqual(actual.counters.cancelledQueued, expected.cancelledQueued)
          XCTAssertEqual(actual.counters.rejectedTimeout, expected.rejectedTimeout)
          XCTAssertEqual(actual.counters.rejectedQueueFull, expected.rejectedQueueFull)
        }
        try await settle()
        gate[0].open()
        let recovery = Task {
          defer { completed.record("recovery") }
          try await ReplayRequest.$id.withValue(0) {
            try await scheduler.submit(priority: .interactive) { _ in () }
          }
        }
        try await eventually("healthy next request") { completed.contains("recovery") }
        try await recovery.value
      } catch {
        // Even a failed replay attempts bounded cleanup and joins tasks that
        // settle. An implementation that cannot settle fails the watchdog.
        try? await settle()
        throw error
      }
    }
    XCTAssertEqual(capacities, [1, 2])
    XCTAssertTrue(
      covered.isSuperset(of: [
        "arrive", "start", "cancelQueued", "cancelGranted",
        "timeout", "cancelActive", "cleanup", "fail", "release",
      ]))
  }
  private func eventually(_ description: String, _ condition: () async -> Bool) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    while !(await condition()) {
      guard ContinuousClock.now < deadline else {
        XCTFail("replay did not reach: \(description)")
        throw ReplayFailure.timeout
      }
      await Task.yield()
    }
  }
  private enum ReplayFailure: Error { case operation, timeout, outcome }
}
