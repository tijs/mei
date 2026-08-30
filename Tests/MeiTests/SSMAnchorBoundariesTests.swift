import XCTest
@testable import MeiCore

/// Non-Metal unit tests for the patch-0005 SSM anchor-boundary plumbing
/// (Sources/MeiCore/SSMAnchorBoundaries.swift). Pure logic: the tokenizer
/// renderer is injected as a closure, so these run without MLX/Metal.
final class SSMAnchorBoundariesTests: XCTestCase {

    private func template(
        system: Bool = true, turns: Int
    ) -> [[String: any Sendable]] {
        var messages: [[String: any Sendable]] = []
        if system {
            messages.append(["role": "system", "content": "You are helpful."])
        }
        for i in 0..<turns {
            messages.append(["role": "user", "content": "Turn \(i)"])
            messages.append(["role": "assistant", "content": "<tool_call>{}</tool_call>"])
        }
        return messages
    }

    /// Prefix-additive fake renderer: each message contributes 10 tokens,
    /// the full render of N messages = 10N (matches fullTokenCount).
    private func additiveRenderer() -> (Int) throws -> Int {
        { prefix in prefix * 10 }
    }

    func testDefaultOffReturnsEmpty() throws {
        let result = try SSMAnchorBoundaries.compute(
            template: template(turns: 3), fullTokenCount: 70, k: 0,
            renderPrefixCount: additiveRenderer())
        XCTAssertEqual(result.offsets, [])
        XCTAssertNil(result.warning)
    }

    func testEarlyTurnOffsets() throws {
        // template: system + 3 turns = 7 messages; full render = 70 tokens.
        // user indices: 1 (turn 0), 3 (turn 1), 5 (turn 2) -> offsets 10, 30, 50.
        let result = try SSMAnchorBoundaries.compute(
            template: template(turns: 3), fullTokenCount: 70, k: 3,
            renderPrefixCount: additiveRenderer())
        XCTAssertEqual(result.offsets, [10, 30, 50])
        XCTAssertNil(result.warning)
    }

    func testKCapTakesFirstKOnly() throws {
        let result = try SSMAnchorBoundaries.compute(
            template: template(turns: 3), fullTokenCount: 70, k: 2,
            renderPrefixCount: additiveRenderer())
        XCTAssertEqual(result.offsets, [10, 30])
    }

    func testNonAdditiveTemplateFallsBackToEmpty() throws {
        // Renderer lies about the full render: prefix(7) returns 71, but the
        // request's fullTokenCount is 70 -> anchors disabled, never a
        // possibly-misplaced offset.
        let result = try SSMAnchorBoundaries.compute(
            template: template(turns: 3), fullTokenCount: 70, k: 3,
            renderPrefixCount: { _ in 10 })
        XCTAssertEqual(result.offsets, [])
        XCTAssertNotNil(result.warning)
        XCTAssertTrue(result.warning!.contains("not prefix-additive"))
    }

    func testThrowingRendererPropagatesDuringLoop() {
        struct Boom: Error {}
        // A throw while computing a user-boundary prefix (the loop) is a
        // hard failure and must propagate.
        XCTAssertThrowsError(try SSMAnchorBoundaries.compute(
            template: template(turns: 1), fullTokenCount: 30, k: 1,
            renderPrefixCount: { prefix in
                if prefix == 1 { throw Boom() }
                return prefix * 10
            }))
    }

    func testThrowingRendererInSelfCheckFallsBackToEmpty() {
        struct Boom: Error {}
        // A throw in the additivity self-check (full render) is NOT a
        // request failure: anchors are disabled instead (current behavior).
        let result = try? SSMAnchorBoundaries.compute(
            template: template(turns: 1), fullTokenCount: 30, k: 1,
            renderPrefixCount: { prefix in
                if prefix == 3 { throw Boom() }
                return prefix * 10
            })
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.offsets, [])
        XCTAssertNotNil(result?.warning)
        XCTAssertTrue(result!.warning!.contains("prefix render failed"))
    }

    func testZeroAndOutOfRangeOffsetsExcluded() throws {
        // A template that STARTS with a user message: first user index 0
        // -> prefix render count 0 (excluded, engine rejects offsets <= 0).
        let noSystem = template(system: false, turns: 2)
        let result = try SSMAnchorBoundaries.compute(
            template: noSystem, fullTokenCount: 40, k: 2,
            renderPrefixCount: { $0 * 10 })
        // user indices 0 and 2 -> counts 0 (excluded) and 20.
        XCTAssertEqual(result.offsets, [20])
    }

    func testDeterministicAcrossBuilds() throws {
        let a = try SSMAnchorBoundaries.compute(
            template: template(turns: 4), fullTokenCount: 90, k: 4,
            renderPrefixCount: additiveRenderer())
        let b = try SSMAnchorBoundaries.compute(
            template: template(turns: 4), fullTokenCount: 90, k: 4,
            renderPrefixCount: additiveRenderer())
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.offsets, [10, 30, 50, 70])
    }

    func testEmptyTemplateReturnsEmpty() throws {
        let result = try SSMAnchorBoundaries.compute(
            template: [], fullTokenCount: 10, k: 2,
            renderPrefixCount: additiveRenderer())
        XCTAssertEqual(result.offsets, [])
        XCTAssertNil(result.warning)
    }
}