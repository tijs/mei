import XCTest
import MLXLMCommon
@testable import MeiCore

final class CacheRestoreTrackerTests: XCTestCase {
    private func progress(
        _ stage: PrefillProgress.Stage,
        _ completed: Int,
        total: Int = 100
    ) -> PrefillProgress {
        PrefillProgress(stage: stage, completedUnitCount: completed, totalUnitCount: total)
    }

    func testFullPrefillInfersNoHit() {
        // A cold prefill streams .prefill frames starting at 0 completed.
        var tracker = CacheRestoreTracker()
        tracker.observe(progress(.prefill, 0))
        tracker.observe(progress(.prefill, 25))
        tracker.observe(progress(.complete, 100))
        XCTAssertEqual(tracker.restoredTokens, 0)
        XCTAssertFalse(tracker.isCacheHit)
    }

    func testCacheRestoreStageRecordsMatchedPrefix() {
        var tracker = CacheRestoreTracker()
        tracker.observe(progress(.cacheRestore, 790, total: 804))
        XCTAssertEqual(tracker.restoredTokens, 790)
        XCTAssertTrue(tracker.isCacheHit)
    }

    func testCacheRestoreThenPrefillKeepsMax() {
        var tracker = CacheRestoreTracker()
        tracker.observe(progress(.cacheRestore, 790, total: 804))
        tracker.observe(progress(.prefill, 790, total: 804))
        tracker.observe(progress(.prefill, 795, total: 804))
        tracker.observe(progress(.complete, 804, total: 804))
        XCTAssertEqual(tracker.restoredTokens, 790)
        XCTAssertTrue(tracker.isCacheHit)
    }

    func testQueuedAndCacheLookupStagesIgnored() {
        var tracker = CacheRestoreTracker()
        tracker.observe(progress(.queued, 0))
        tracker.observe(progress(.cacheLookup, 0))
        XCTAssertEqual(tracker.restoredTokens, 0)
        XCTAssertFalse(tracker.isCacheHit)
    }

    func testStringStageDriverMatchesServerPaths() {
        var tracker = CacheRestoreTracker()
        tracker.observe(stage: "prefill", completed: 0)
        tracker.observe(stage: "prefill", completed: 50)
        XCTAssertEqual(tracker.restoredTokens, 0)
        tracker.observe(stage: "cacheRestore", completed: 300)
        XCTAssertEqual(tracker.restoredTokens, 300)
        XCTAssertTrue(tracker.isCacheHit)
    }

    func testStopReasonMapping() {
        XCTAssertEqual(Engine.mapStopReason(.stop, toolCallCount: 0), "stop")
        XCTAssertEqual(Engine.mapStopReason(.length, toolCallCount: 0), "length")
        XCTAssertEqual(Engine.mapStopReason(.cancelled, toolCallCount: 0), "stop")
        XCTAssertEqual(Engine.mapStopReason(.stop, toolCallCount: 2), "tool_calls")
        XCTAssertEqual(Engine.mapStopReason(.length, toolCallCount: 1), "tool_calls")
    }
}