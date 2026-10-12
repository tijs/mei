import Foundation
import XCTest

/// Shared async test utilities for the admission/scheduling and stream
/// lifecycle suites. No Metal, no model, no network: those tests pin
/// deterministic behavior with plain async operations.
final class TestLatch: @unchecked Sendable {
    private let lock = NSLock()
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Suspends until the latch is opened (returns immediately afterwards).
    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if isOpen {
                lock.unlock()
                continuation.resume()
            } else {
                waiters.append(continuation)
                lock.unlock()
            }
        }
    }

    func open() {
        lock.lock()
        isOpen = true
        let pending = waiters
        waiters = []
        lock.unlock()
        for continuation in pending { continuation.resume() }
    }
}

/// Ordered, thread-safe event recorder used to pin execution order and to
/// prove which paths ran.
final class TestRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [String] = []

    func record(_ event: String) {
        lock.lock()
        events.append(event)
        lock.unlock()
    }

    var recorded: [String] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }

    func contains(_ event: String) -> Bool { recorded.contains(event) }
}

extension XCTestCase {
    /// Wait (bounded) until `condition` holds. Ordering assertions then stay
    /// exact; only the waiting itself is time-based.
    func waitUntil(
        _ description: String,
        timeout: TimeInterval = 10,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ condition: () async -> Bool
    ) async {
        let deadline = Date().addingTimeInterval(timeout)
        while !(await condition()) {
            if Date() >= deadline {
                XCTFail("timed out waiting for: \(description)", file: file, line: line)
                return
            }
            try? await Task.sleep(for: .milliseconds(2))
        }
    }
}
