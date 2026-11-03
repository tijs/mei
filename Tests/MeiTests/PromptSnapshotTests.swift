import Foundation
import Metal
import MLX
import MLXNN
@testable import MLXLMCommon
import XCTest
@testable import MeiCore
import MLXFast

/// Weightless fixture: the same bounded cache + SDPA helper used by Qwen35
/// after its projections. No tokenizer, model bundle, server or learned weights.
private final class SnapshotProbeAttention: Module, LanguageModel {
    var kvHeads: [Int] { [2] }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [RotatingKVCache(maxSize: parameters?.maxKVSize ?? 65_536, keep: 4)]
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let count = input.text.tokens.size
        let remainder = min(count, windowSize ?? 128)
        let shape = [1, 2, count - remainder, 256]
        _ = cache[0].update(keys: MLXArray.zeros(shape, dtype: .bfloat16),
                            values: MLXArray.ones(shape, dtype: .bfloat16))
        eval(cache)
        return .tokens(.init(tokens: input.text.tokens.reshaped(-1)[(count - remainder)...]))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let shape = [1, 2, inputs.size, 256]
        // Qwen35 builds the prefill mask before attention appends this chunk.
        let mask = cache?.first?.makeMask(n: inputs.size, windowSize: nil,
                                          returnArray: true) ?? .none
        let output = attentionWithCacheUpdate(
            queries: MLXArray.zeros(shape, dtype: .bfloat16),
            keys: MLXArray.full(shape, values: MLXArray(Float(2)), dtype: .bfloat16),
            values: MLXArray.full(shape, values: MLXArray(Float(3)), dtype: .bfloat16),
            cache: cache?.first, scale: 1 / 16, mask: mask)
        let value = sum(output).asType(.float32)
        return stacked([value, value + 1]).reshaped(1, 1, 2)
    }
}

/// Tiny real-cache fixture records prepare calls so a finalization replay
/// cannot hide behind correct cache contents. No learned weights or model I/O.
private final class SnapshotBoundaryModel: Module, LanguageModel {
    enum Topology { case dense, rotating, hybrid, typed }
    let topology: Topology
    var prepares: [Int] = []
    init(_ topology: Topology = .dense) { self.topology = topology }
    var kvHeads: [Int] { topology == .hybrid ? [1, 0] : [1] }
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        switch topology {
        case .dense: [KVCacheSimple()]
        case .rotating: [RotatingKVCache(maxSize: 64, keep: 4)]
        case .hybrid: [RotatingKVCache(maxSize: 64, keep: 4), MambaCache()]
        case .typed: [SnapshotSeedPoolCache()]
        }
    }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        prepares.append(input.text.tokens.size)
        return .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let count = inputs.dim(-1)
        let value = MLXArray.ones([1, 1, count, 4], dtype: .bfloat16)
        for layer in cache ?? [] {
            if let recurrent = layer as? MambaCache {
                recurrent[0] = MLXArray.full([1, 4], values: MLXArray(Float(recurrent.offset + count)))
                recurrent[1] = MLXArray.ones([1, 4])
                recurrent.offset += count
            } else {
                _ = layer.update(keys: value, values: value + 1)
            }
        }
        return MLXArray.zeros([1, count, 2])
    }
}

/// Minimal typed cache exercises the prefill-captured N-1-only contract.
private final class SnapshotSeedPoolCache: HybridPoolCache {
    var rotating = RotatingKVCache(maxSize: 64)
    let compressRatio = 4
    let slidingWindow = 64
    var offset: Int { rotating.offset }
    var maxSize: Int? { rotating.maxSize }
    var isTrimmable: Bool { false }
    var metaState: [String] {
        get { rotating.metaState }
        set { rotating.metaState = newValue }
    }
    var state: [MLXArray] {
        get { rotating.state }
        set { rotating.state = newValue }
    }
    func innerState() -> [MLXArray] { rotating.innerState() }
    func trim(_ n: Int) -> Int { 0 }
    func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        rotating.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        rotating.update(keys: keys, values: values)
    }
    func copy() -> KVCache {
        let result = SnapshotSeedPoolCache()
        result.rotating = rotating.copy() as! RotatingKVCache
        return result
    }
    func hybridPool(branch: HybridPoolBranch) -> MLXArray? { nil }
    func setHybridPool(branch: HybridPoolBranch, value: MLXArray?) {}
    func hybridBuffers(branch: HybridPoolBranch) -> (kv: MLXArray?, gate: MLXArray?) { (nil, nil) }
    func setHybridBuffers(branch: HybridPoolBranch, kv: MLXArray?, gate: MLXArray?) {}
}

final class PromptSnapshotTests: XCTestCase {
    private var roots: [URL] = []
    override func setUpWithError() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Real Metal required") }
    }
    override func tearDown() {
        Stream.gpu.synchronize()
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots = []
    }
    private func coordinator(cap: Int? = 1_048_576, disk: Bool = true,
                             paged: Bool = false, blocks: Int = 32,
                             pressure: Int = 1) -> CacheCoordinator {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        roots.append(root)
        return CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: paged, enableDiskCache: disk, pagedBlockSize: 4,
            maxCacheBlocks: blocks, maxCacheBytes: cap, diskCacheMaxGB: 0.01,
            diskCacheDir: root, enableSSMReDerive: false, modelKey: "prompt-snapshot-test"),
            diskIndexBusyTimeoutMs: 100,
            promptSnapshotStoreAdmission: { cache in
                CacheStoreBudget.canStore(cacheBytes: CacheStoreBudget.cacheBytes(cache),
                    activeBytes: 0, budgetBytes: 1 << 30, hostPressureLevel: pressure)
            })
    }
    private func input(_ count: Int = 8, prefixes: [Int] = [], stable: [Int] = [],
                       intent: LMInput.CachePromptIntent = .generation) -> LMInput {
        LMInput(tokens: MLXArray((0..<count).map { Int32($0 + 2) }).reshaped(1, count),
                cachePrefixTokenCounts: prefixes, cacheStablePrefixTokenCounts: stable,
                cachePromptIntent: intent)
    }
    private var parameters: GenerateParameters {
        var p = GenerateParameters(maxTokens: 2, temperature: 0, prefillStepSize: 128)
        p.enableCompiledDecode = false
        return p
    }
    private func backing(_ array: MLXArray) -> UInt {
        array.asData(access: .noCopy).data.withUnsafeBytes { UInt(bitPattern: $0.baseAddress!) }
    }

    func testCacheOffAndIneffectiveTiersDoNotDoubleRotatingBacking() throws {
        let oldLimit = Memory.cacheLimit
        Memory.cacheLimit = 0
        defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = oldLimit }
        let disabled = coordinator(disk: false)
        let zero = coordinator(cap: 0, disk: false, paged: true)
        let sentinelOnly = coordinator(disk: false, paged: true, blocks: 1)
        let incompatible = coordinator(disk: false, paged: true)
        incompatible.setPagedIncompatible(true)
        let topologyIncompatible = coordinator(disk: false, paged: true)
        for c in [nil, disabled, zero, sentinelOnly, incompatible, topologyIncompatible] as [CacheCoordinator?] {
            Stream.gpu.synchronize()
            Memory.clearCache()
            let baseline = Memory.activeMemory
            let count = 8192
            let cache = RotatingKVCache(maxSize: 65_536, keep: 4)
            let model = SnapshotProbeAttention()
            var p = parameters
            p.maxKVSize = 65_536
            let iterator = try TokenIterator(input: input(count), model: model, cache: [cache],
                                              parameters: p, cacheCoordinator: c)
            eval(iterator.y.tokens)
            eval(iterator.cache)
            Stream.gpu.synchronize()
            let liveBytes = cache.innerState().reduce(0) { $0 + $1.nbytes }
            XCTAssertEqual(liveBytes, count * 2 * 256 * 2 * 2)
            XCTAssertNil(iterator.promptCacheSnapshot)
            XCTAssertEqual(c?.promptSnapshotBytes ?? 0, 0)
            let activeDelta = Memory.activeMemory - baseline
            print("PROMPT_SNAPSHOT_FIXED cacheOff=true live=\(liveBytes) activeDelta=\(activeDelta)")
            XCTAssertLessThan(activeDelta, liveBytes + liveBytes / 2,
                              "An unused prompt copy must not double the evaluated backing")
        }
    }

    func testLegacyInitializersNeverRetainUnusedExactPromptCopies() throws {
        let model = SnapshotBoundaryModel()
        let p = parameters
        let old = try TokenIterator(prompt: input().text.tokens, model: model, parameters: p)
        let direct = try TokenIterator(input: input(), model: model,
            processor: nil, sampler: ArgMaxSampler(), prefillStepSize: 128, maxTokens: 2)
        XCTAssertNil(old.promptCacheSnapshot)
        XCTAssertNil(direct.promptCacheSnapshot)
    }

    func testNoStoreConsumerSkipsBoundaryCopiesAndPreservesLiveCache() throws {
        let c = coordinator()
        c.setHybrid(true, requiresSeparateRecurrentPayload: false)
        let baselineModel = SnapshotBoundaryModel(.hybrid)
        let baseline = try TokenIterator(input: input(12), model: baselineModel,
            parameters: parameters, cacheCoordinator: nil)
        eval(baseline.y.tokens)
        eval(baseline.cache)
        let expectedToken = baseline.y.tokens.asArray(Int.self)
        let expectedOffsets = baseline.cache.map(\.offset)
        let expectedStates = baseline.cache.map { $0.state.map { $0.asArray(Float.self) } }
        XCTAssertEqual(baselineModel.prepares, [12])

        let cases: [(CacheCoordinator?, LMInput.CachePromptIntent)] = [
            (nil, .generation), (c, .auxiliary),
        ]
        for (coordinator, intent) in cases {
            let model = SnapshotBoundaryModel(.hybrid)
            let iterator = try TokenIterator(
                input: input(12, prefixes: [4, 9], stable: [4], intent: intent),
                model: model, parameters: parameters, cacheCoordinator: coordinator)
            eval(iterator.y.tokens)
            eval(iterator.cache)
            XCTAssertEqual(model.prepares, [12], "No consumer needs segmented prefill")
            XCTAssertTrue(iterator.stableBoundarySnapshots.isEmpty)
            XCTAssertNil(iterator.hybridStripSnapshot)
            XCTAssertNil(iterator.diskSeedSnapshot)
            XCTAssertNil(iterator.promptCacheSnapshot)
            XCTAssertEqual(iterator.y.tokens.asArray(Int.self), expectedToken)
            XCTAssertEqual(iterator.cache.map(\.offset), expectedOffsets)
            XCTAssertEqual(iterator.cache.map { $0.state.map { $0.asArray(Float.self) } }, expectedStates)
        }
    }

    func testAuxiliaryRestoresWarmPagedPrefixWithoutCreatingSnapshotsOrStores() throws {
        let c = coordinator(disk: false, paged: true)
        let tokens = Array(2..<10)
        let keys = MLXArray.ones([1, 1, 4, 4], dtype: .bfloat16)
        let auxiliaryInput = input(12, prefixes: [4, 9], stable: [4], intent: .auxiliary)
        let salt = computeCacheSalt(for: auxiliaryInput, parameters: parameters)
        c.pagedCache!.storeTokenSequence(
            tokens: tokens,
            layerData: [[(keys: keys, values: keys + 1)], [(keys: keys, values: keys + 1)]],
            mediaSalt: salt)
        let before = c.pagedCache!.snapshotStats()
        XCTAssertGreaterThan(before.residentBytes, 0)

        let coldModel = SnapshotBoundaryModel()
        let cold = try TokenIterator(input: input(12), model: coldModel,
            parameters: parameters, cacheCoordinator: nil)
        eval(cold.y.tokens)
        eval(cold.cache)

        let warmModel = SnapshotBoundaryModel()
        var warm = try TokenIterator(
            input: auxiliaryInput,
            model: warmModel, parameters: parameters, cacheCoordinator: c)
        eval(warm.y.tokens)
        eval(warm.cache)
        let afterRestore = c.pagedCache!.snapshotStats()
        XCTAssertGreaterThan(afterRestore.cacheHits, before.cacheHits)
        XCTAssertEqual(warmModel.prepares, [4], "Only the suffix should be prefilled")
        XCTAssertEqual(warm.y.tokens.asArray(Int.self), cold.y.tokens.asArray(Int.self))
        XCTAssertEqual(warm.cache.map(\.offset), cold.cache.map(\.offset))
        XCTAssertEqual(warm.cache.map { $0.state.map { $0.asArray(Float.self) } },
                       cold.cache.map { $0.state.map { $0.asArray(Float.self) } })
        XCTAssertTrue(warm.stableBoundarySnapshots.isEmpty)
        XCTAssertNil(warm.hybridStripSnapshot)
        XCTAssertNil(warm.diskSeedSnapshot)
        XCTAssertNil(warm.promptCacheSnapshot)

        warm.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        let afterFinalization = c.pagedCache!.snapshotStats()
        XCTAssertEqual(afterFinalization.allocatedBlocks, afterRestore.allocatedBlocks)
        XCTAssertEqual(afterFinalization.residentBytes, afterRestore.residentBytes)
        XCTAssertEqual(c.promptSnapshotBytes, 0)
    }

    func testEnabledSnapshotIsImmutableSharedOwnedAndVisibleUntilLastIteratorRelease() throws {
        let c = coordinator()
        c.setHybrid(true, requiresSeparateRecurrentPayload: false)
        let model = SnapshotBoundaryModel(.hybrid)
        var iterator: TokenIterator? = try TokenIterator(input: input(), model: model,
            parameters: parameters, cacheCoordinator: c)
        let charge = c.promptSnapshotBytes
        XCTAssertGreaterThan(charge, 0)
        XCTAssertEqual(c.retainedCacheBytes, charge)
        var snapshot = try XCTUnwrap(iterator!.promptCacheSnapshot)
        let expected = snapshot[0].state[0].asArray(Float.self)
        let expectedRecurrent = snapshot[1].state[0].asArray(Float.self)
        XCTAssertNotEqual(backing(snapshot[0].state[0]), backing(iterator!.cache[0].state[0]))
        let append = MLXArray.full([1, 1, 1, 4], values: MLXArray(Float(9)), dtype: .bfloat16)
        _ = iterator!.cache[0].update(keys: append, values: append)
        (iterator!.cache[1] as! MambaCache)[0] = MLXArray.full([1, 4], values: MLXArray(Float(99)))
        eval(iterator!.cache)
        XCTAssertEqual(snapshot[0].offset, 8)
        XCTAssertEqual(snapshot[0].state[0].asArray(Float.self), expected)
        XCTAssertEqual(snapshot[1].state[0].asArray(Float.self), expectedRecurrent)
        let status = try XCTUnwrap(Engine.cacheStatus(c))
        XCTAssertEqual(status.promptSnapshotBytes, charge)
        XCTAssertEqual(status.retainedBytes, charge)
        XCTAssertEqual(status.maxRetainedBytes, 1_048_576)
        var duplicate = iterator
        c.clear()
        XCTAssertEqual(c.promptSnapshotBytes, charge, "Clear cannot revoke live iterator backing")
        iterator = nil
        XCTAssertEqual(c.promptSnapshotBytes, charge, "Iterator value copies share the ARC ticket")
        XCTAssertNotNil(duplicate?.promptCacheSnapshot)
        snapshot.removeAll()
        duplicate = nil
        XCTAssertEqual(c.promptSnapshotBytes, 0)
        XCTAssertEqual(c.retainedCacheBytes, 0)
    }

    func testSnapshotReservationPreventsOtherTierOverAdmissionAndReleasesOnFinalization() throws {
        let c = coordinator(cap: 131_072, disk: false, paged: true)
        let model = SnapshotBoundaryModel()
        var iterator = try TokenIterator(input: input(), model: model,
            parameters: parameters, cacheCoordinator: c)
        XCTAssertGreaterThan(c.promptSnapshotBytes, 0)
        let large = MLXArray.zeros([24_576]) // 96 KiB + buffer slack exceeds remaining allowance.
        c.ssmStateCache.store(ssmStates: [large], tokens: [99], boundary: 1)
        XCTAssertEqual(c.ssmStateCache.snapshotStats().residentBytes, 0)
        XCTAssertLessThanOrEqual(c.retainedCacheBytes, 131_072)
        iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertNil(iterator.promptCacheSnapshot)
        XCTAssertEqual(c.promptSnapshotBytes, 0)
        c.pagedCache!.clear() // Finalization legitimately admitted a paged entry.
        c.ssmStateCache.store(ssmStates: [large], tokens: [99], boundary: 1)
        XCTAssertGreaterThan(c.ssmStateCache.snapshotStats().residentBytes, 0)
        XCTAssertLessThanOrEqual(c.retainedCacheBytes, 131_072)
        c.clear()
        XCTAssertEqual(c.retainedCacheBytes, 0)
    }

    func testZeroOversizedAndPressureDeclinesDoNotReplayMissingHistoryBoundary() throws {
        for (cap, pressure) in [(0, 1), (1, 1), (1_048_576, 2), (1_048_576, 4)] {
            let c = coordinator(cap: cap, pressure: pressure)
            let model = SnapshotBoundaryModel(.hybrid)
            c.setHybrid(true, requiresSeparateRecurrentPayload: false)
            var iterator = try TokenIterator(input: input(12, prefixes: [4, 9], stable: [4]), model: model,
                parameters: parameters, cacheCoordinator: c)
            XCTAssertNil(iterator.promptCacheSnapshot)
            XCTAssertEqual(c.promptSnapshotBytes, 0)
            // Model the missing-capture case (e.g. an already-restored prefix).
            // A deliberate decline must not fall back to live topology and
            // reconstruct either the stable rung or the stripped checkpoint.
            iterator.stableBoundarySnapshots = [:]
            iterator.hybridStripSnapshot = nil
            let before = model.prepares
            iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            XCTAssertEqual(model.prepares, before)
            XCTAssertFalse(c.hasDurableDiskEntry(tokens: iterator.promptTokenIds, mediaSalt: iterator.mediaSalt))
        }
    }

    func testSnapshotAdmissionUsesAllowanceAlreadyOwnedByOtherTiers() throws {
        let c = coordinator(cap: 131_072, disk: false, paged: true)
        let state = MLXArray.zeros([24_576])
        c.ssmStateCache.store(ssmStates: [state], tokens: [99], boundary: 1)
        let retained = c.retainedCacheBytes
        XCTAssertGreaterThan(retained, 0)
        let iterator = try TokenIterator(input: input(), model: SnapshotBoundaryModel(),
            parameters: parameters, cacheCoordinator: c)
        XCTAssertNil(iterator.promptCacheSnapshot)
        XCTAssertEqual(c.promptSnapshotBytes, 0)
        XCTAssertEqual(c.retainedCacheBytes, retained)
        c.clear()
        XCTAssertEqual(c.retainedCacheBytes, 0)
    }

    func testDeclinedExactCopyStillStoresCapturedStripAndStableBoundaries() throws {
        for (cap, pressure) in [(1, 1), (1_048_576, 2)] {
            let c = coordinator(cap: cap, pressure: pressure)
            let model = SnapshotBoundaryModel(.hybrid)
            c.setHybrid(true, requiresSeparateRecurrentPayload: false)
            var iterator = try TokenIterator(input: input(12, prefixes: [4, 9], stable: [4]),
                model: model, parameters: parameters, cacheCoordinator: c)
            XCTAssertNil(iterator.promptCacheSnapshot)
            XCTAssertNotNil(iterator.hybridStripSnapshot)
            XCTAssertNotNil(iterator.stableBoundarySnapshots[3])
            let before = model.prepares
            iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
            XCTAssertEqual(model.prepares, before, "Declined optional copies must not cause another prepare")
            XCTAssertTrue(c.hasDurableDiskEntry(tokens: Array(iterator.promptTokenIds.prefix(9)), mediaSalt: iterator.mediaSalt))
            XCTAssertTrue(c.hasDurableDiskEntry(tokens: Array(iterator.promptTokenIds.prefix(3)), mediaSalt: iterator.mediaSalt))
            XCTAssertFalse(c.hasDurableDiskEntry(tokens: iterator.promptTokenIds, mediaSalt: iterator.mediaSalt))
        }
    }

    func testDeclinedExactCopyStillStoresCapturedRotatingNMinusOneSeed() throws {
        let c = coordinator(cap: 1)
        let model = SnapshotBoundaryModel(.rotating)
        var iterator = try TokenIterator(input: input(), model: model,
            parameters: parameters, cacheCoordinator: c)
        XCTAssertNil(iterator.promptCacheSnapshot)
        XCTAssertNotNil(iterator.diskSeedSnapshot)
        let before = model.prepares
        iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertEqual(model.prepares, before)
        XCTAssertTrue(c.hasDurableDiskEntry(tokens: Array(iterator.promptTokenIds.dropLast()), mediaSalt: iterator.mediaSalt))
        XCTAssertNil(iterator.diskSeedSnapshot)
    }

    func testTypedSeedOnlyPersistenceStillStoresExactCapturedNMinusOneWithoutReplay() throws {
        let c = coordinator()
        c.setPagedIncompatible(true)
        let model = SnapshotBoundaryModel(.typed)
        var iterator = try TokenIterator(input: input(), model: model,
            parameters: parameters, cacheCoordinator: c)
        XCTAssertNil(iterator.promptCacheSnapshot)
        XCTAssertEqual(c.promptSnapshotBytes, 0)
        XCTAssertNotNil(iterator.diskSeedSnapshot)
        let before = model.prepares
        iterator.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertEqual(model.prepares, before)
        XCTAssertTrue(c.hasDurableDiskEntry(tokens: Array(iterator.promptTokenIds.dropLast()), mediaSalt: iterator.mediaSalt))
        XCTAssertFalse(c.hasDurableDiskEntry(tokens: iterator.promptTokenIds, mediaSalt: iterator.mediaSalt))
    }

    func testAuxiliarySkipsAndNilAllowancePreservesEnabledExactSnapshot() throws {
        let c = coordinator(cap: nil)
        let model = SnapshotBoundaryModel()
        let auxiliary = try TokenIterator(input: input(intent: .auxiliary), model: model,
            parameters: parameters, cacheCoordinator: c)
        XCTAssertNil(auxiliary.promptCacheSnapshot)
        var ordinary = try TokenIterator(input: input(), model: model,
            parameters: parameters, cacheCoordinator: c)
        XCTAssertNotNil(ordinary.promptCacheSnapshot)
        XCTAssertGreaterThan(c.promptSnapshotBytes, 0)
        ordinary.storeCacheAfterGeneration(generatedTokenIds: [], includeGeneratedBoundary: false)
        XCTAssertTrue(c.hasDurableDiskEntry(tokens: ordinary.promptTokenIds, mediaSalt: ordinary.mediaSalt))
        XCTAssertEqual(c.promptSnapshotBytes, 0)
    }
}
