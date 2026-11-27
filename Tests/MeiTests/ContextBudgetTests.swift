import XCTest
@testable import MeiCore

final class ContextBudgetTests: XCTestCase {
    private let profile = ContextProfile(maximum: 1000, autoLimit: 1000, bytesPerToken: 10,
                                         workspace: 100, fallback: 12)
    private let normal = ContextHeadroom(bytes: 1100, pressure: 1)

    func testBoundariesAndClientAllowance() throws {
        var budget = ContextReservations()
        let id = UUID()
        XCTAssertEqual(try budget.admit(id: id, cap: 100, prompt: 99, requested: 12,
                                        profile: profile, headroom: normal), 1)
        budget.release(id)
        XCTAssertThrowsError(try budget.admit(id: id, cap: 100, prompt: 100, requested: 1,
                                              profile: profile, headroom: normal))
        XCTAssertEqual(try budget.admit(id: id, cap: 100, prompt: 10, requested: 2,
                                        profile: profile, headroom: normal), 2)
    }

    func testReservationSurvivesUntilCleanupAndCannotReleaseItsReplacement() throws {
        var budget = ContextReservations()
        let first = UUID(), retry = UUID()
        func admit(_ id: UUID) throws {
            _ = try budget.admit(id: id, cap: 100, prompt: 58, requested: 2, profile: profile, headroom: normal)
        }
        try admit(first)
        XCTAssertThrowsError(try admit(retry))
        budget.release(first)
        try admit(retry)
        budget.release(first)
        XCTAssertEqual(budget.available(cap: 100, profile: profile, headroom: normal), 30)
        budget.release(retry)
        XCTAssertEqual(budget.available(cap: 100, profile: profile, headroom: normal), 100)
    }

    func testLimitsAfterFormattingAndClientDownsizing() throws {
        XCTAssertEqual(try ContextReservations.replyLimit(prompt: 26_000, requested: 1_000, cap: 26_624), 624)
        XCTAssertEqual(try ContextReservations.replyLimit(prompt: 26_000, requested: 32, cap: 26_624), 32)
        XCTAssertThrowsError(try ContextReservations.replyLimit(prompt: 26_624, requested: 1, cap: 26_624))
        XCTAssertThrowsError(try ContextReservations.replyLimit(prompt: 10, requested: 0, cap: 100))
        XCTAssertEqual(try ContextReservations.replyLimit(prompt: 20, requested: Int.max, cap: 100), 80)
    }

    func testExplicitAutoAndFixedLimitsFromModelDirectory() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"model_type":"qwen3_5_moe","text_config":{"max_position_embeddings":98304}}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        func parse(_ flags: [String]) throws -> ServerConfig {
            try ServerConfig.parse(arguments: ["--model-dir", directory.path] + flags, recommendedWorkingSetBytes: nil)
        }
        for flags in [[], ["--context-cap", "auto"], ["--context-cap","8192","--context-cap","auto"]] {
            let config = try parse(flags)
            XCTAssertTrue(config.automaticContext)
            XCTAssertEqual(config.contextCap, ServerConfig.defaultContextCap)
        }
        let fixed = try parse(["--context-cap","auto","--context-cap","72000"])
        XCTAssertFalse(fixed.automaticContext)
        XCTAssertEqual(fixed.contextCap, 72_000)
        XCTAssertThrowsError(try parse(["--context-cap","98305"]))
        let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/AutomaticModelSettings")
        let ceilings = ["groot-19b-a3b-text": 65_536, "qwen3.6-35b-a3b-text": 32_768,
                        "qwen3.6-35b-a3b": 26_624, "ornith-1.5-35b-a3b": 26_624]
        for (name, ceiling) in ceilings {
            let model = fixtures.appendingPathComponent(name)
            XCTAssertEqual(ContextProfile.resolve(directory: model.path, fallback: 26_624).autoLimit, ceiling, name)
        }
        let profile = ContextProfile.resolve(directory: directory.path, fallback: 26_624)
        XCTAssertEqual(ContextReservations().available(cap: fixed.contextCap, profile: profile,
                                                       headroom: .init(bytes: nil, pressure: 0)), 8192)
    }

    func testPressureRecoveryAndMissingTelemetry() {
        let budget = ContextReservations()
        XCTAssertEqual(budget.available(cap: 100, profile: profile, headroom: .init(bytes: nil, pressure: 0)), 12)
        XCTAssertEqual(budget.available(cap: 100, profile: profile, headroom: .init(bytes: 1100, pressure: 2)), 0)
        XCTAssertEqual(budget.available(cap: 100, profile: profile, headroom: normal), 100)
    }

    func testFreshSwapGrowthBacksOffThenRecovers() {
        var swap = SwapActivity()
        XCTAssertFalse(swap.isBusy(pages: 5_000, pageSize: 16_384, now: 100))
        XCTAssertTrue(swap.isBusy(pages: 7_048, pageSize: 16_384, now: 102))
        XCTAssertTrue(swap.isBusy(pages: 7_048, pageSize: 16_384, now: 108))
        XCTAssertFalse(swap.isBusy(pages: 7_048, pageSize: 16_384, now: 113))
        XCTAssertFalse(swap.isBusy(pages: 15_000, pageSize: 16_384, now: 200))
    }

    func testHostResidencyAndMetalBoundsUseTheSameUnifiedMemory() {
        let gib = 1 << 30
        func room(reclaimable: Int, footprint: Int, active: Int, limit: Int = 0) -> Int {
            ContextHeadroom.admissibleBytes(physical: 32*gib, reclaimable: reclaimable*gib,
                footprint: footprint*gib, active: active*gib, deviceLimit: 25*gib, allocatorLimit: limit*gib)
        }
        XCTAssertEqual(room(reclaimable: 10, footprint: 16, active: 18), 6*gib)
        XCTAssertEqual(room(reclaimable: 10, footprint: 16, active: 18, limit: 20), 2*gib)
        XCTAssertEqual(room(reclaimable: 20, footprint: 25, active: 16), 0)
        XCTAssertEqual(room(reclaimable: 3, footprint: 16, active: 18), 0)
    }

    func testCriticalPressureInterruptsAnActiveStreamAfterItsFirstEvent() async {
        let source = DispatchSource.makeUserDataAddSource(queue: .global())
        let stream: AsyncThrowingStream<Int, Error> = makeCancellableStream { continuation in
            try await withContextPressureGuard(source: source) {
                continuation.yield(1)
                try await Task.sleep(for: .seconds(30))
            }
        }
        var iterator = stream.makeAsyncIterator()
        do {
            let first = try await iterator.next()
            XCTAssertEqual(first, 1)
            source.add(data: 1)
            _ = try await iterator.next()
            XCTFail("a critical interruption must reach the client")
        } catch EngineError.criticalMemoryPressure {
            XCTAssertTrue(Router.streamErrorSSEData(EngineError.criticalMemoryPressure).contains("generation_interrupted"))
        } catch {
            XCTFail("unexpected stream outcome: \(error)")
        }
    }

    func testWarmAdmissionWaitsForKernelAccountingAfterATrim() async throws {
        var samples = 0
        var trims = 0
        let result = try await admitAfterReclaim {
            samples += 1
            guard samples == 4 else { throw EngineError.memoryCapacity(required: 20, available: 9) }
            return 3
        } reclaim: { required, available in
            XCTAssertEqual(required - available, 11)
            trims += 1
            return true
        }
        XCTAssertEqual(result, 3)
        XCTAssertEqual(trims, 1)
        XCTAssertEqual(samples, 4)
    }
}
