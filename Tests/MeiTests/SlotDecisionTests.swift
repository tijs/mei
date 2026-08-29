import XCTest
@testable import MeiCore

final class SlotDecisionTests: XCTestCase {
    func testNoSlotMeansNoReuse() {
        let result = SlotDecision.make(newTokens: [1, 2, 3], slotTokens: [], maxSlotTokens: 1000)
        XCTAssertFalse(result.reuseApplied)
        XCTAssertEqual(result.cachedTokenCount, 0)
    }

    func testExactExtensionReuses() {
        let result = SlotDecision.make(
            newTokens: [1, 2, 3, 4, 5],
            slotTokens: [1, 2, 3],
            maxSlotTokens: 1000)
        XCTAssertTrue(result.reuseApplied)
        XCTAssertEqual(result.cachedTokenCount, 3)
    }

    func testShrinkingPromptNeverReuses() {
        let result = SlotDecision.make(
            newTokens: [1, 2],
            slotTokens: [1, 2, 3],
            maxSlotTokens: 1000)
        XCTAssertFalse(result.reuseApplied)
    }

    func testDivergedPrefixNeverReuses() {
        let result = SlotDecision.make(
            newTokens: [1, 2, 9, 10],
            slotTokens: [1, 2, 3],
            maxSlotTokens: 1000)
        XCTAssertFalse(result.reuseApplied)
    }

    func testEqualSequenceNeverReuses() {
        // Same token count: the model just regenerated from the same prompt;
        // reusing would resume mid-sequence. Treat as miss (fresh prefill).
        let result = SlotDecision.make(
            newTokens: [1, 2, 3],
            slotTokens: [1, 2, 3],
            maxSlotTokens: 1000)
        XCTAssertFalse(result.reuseApplied)
    }

    func testOverBudgetNeverReuses() {
        let result = SlotDecision.make(
            newTokens: [1, 2, 3, 4, 5, 6],
            slotTokens: [1, 2, 3],
            maxSlotTokens: 5)
        XCTAssertFalse(result.reuseApplied)
    }

    func testRealisticAgenticFlow() {
        // Turn 1: system + user.
        let turn1 = [100, 101, 102, 200, 201]
        // Turn 2 (hermes sends the full transcript): system + user + assistant + tool + user2.
        let turn2 = [100, 101, 102, 200, 201, 300, 301, 400, 500, 501]
        let r1 = SlotDecision.make(newTokens: turn1, slotTokens: [], maxSlotTokens: 100_000)
        XCTAssertFalse(r1.reuseApplied)
        let r2 = SlotDecision.make(newTokens: turn2, slotTokens: turn1, maxSlotTokens: 100_000)
        XCTAssertTrue(r2.reuseApplied)
        XCTAssertEqual(r2.cachedTokenCount, turn1.count)
    }

    func testMultiTurnChain() {
        let t1 = [1, 2, 3]
        let t2 = [1, 2, 3, 4, 5]
        let t3 = [1, 2, 3, 4, 5, 6, 7]
        let r2 = SlotDecision.make(newTokens: t2, slotTokens: t1, maxSlotTokens: 1000)
        XCTAssertTrue(r2.reuseApplied)
        let r3 = SlotDecision.make(newTokens: t3, slotTokens: t2, maxSlotTokens: 1000)
        XCTAssertTrue(r3.reuseApplied)
        XCTAssertEqual(r3.cachedTokenCount, t2.count)
    }

    func testStopReasonMapping() {
        XCTAssertEqual(Engine.mapStopReason(.stop, toolCallCount: 0), "stop")
        XCTAssertEqual(Engine.mapStopReason(.length, toolCallCount: 0), "length")
        XCTAssertEqual(Engine.mapStopReason(.cancelled, toolCallCount: 0), "stop")
        XCTAssertEqual(Engine.mapStopReason(.stop, toolCallCount: 2), "tool_calls")
        XCTAssertEqual(Engine.mapStopReason(.length, toolCallCount: 1), "tool_calls")
    }
}