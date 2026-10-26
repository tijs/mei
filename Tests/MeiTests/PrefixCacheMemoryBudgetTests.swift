import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import MeiCore

/// Retained KV and rotating state must be bounded independently of the
/// allocator's freed-buffer pool. Run serially with the other MLX tests.
final class PrefixCacheMemoryBudgetTests: XCTestCase {
    private func layer(_ value: Float = 0) -> [(keys: MLXArray, values: MLXArray)] {
        let k = MLXArray.full([1, 1, 4, 64], values: MLXArray(value))
        return [(keys: k, values: k + 1)]
    }

    private func companion(_ value: Float = 0) -> [String: MLXArray] {
        ["ring": MLXArray.full([1, 2, 16, 64], values: MLXArray(value))]
    }

    func testBudgetIncludesCompanionsAndEvictsOldLeafWithoutStaleHit() throws {
        let c = PagedCacheManager(blockSize: 4, maxBlocks: 128, maxBytes: 131_072)
        for i in 0..<40 {
            let tokens = [1, 2, 3, 4, 100 + i, 2, 3, 4]
            c.storeTokenSequence(tokens: tokens, layerData: [layer(), layer(Float(i))],
                                 boundaryCompanionData: companion(Float(i)))
            XCTAssertLessThanOrEqual(c.snapshotStats().residentBytes, 131_072)
            let hit = try XCTUnwrap(c.fetchPrefix(tokens: tokens + [99], requireBoundaryCompanion: true))
            XCTAssertEqual(hit.matchedTokens, tokens.count)
            XCTAssertEqual(hit.blocks.last!.boundaryCompanionData!["ring"]!.asArray(Float.self).first, Float(i))
            for b in hit.blocks { c.freeBlock(b) }
            if i > 0 {
                let old = c.fetchPrefix(tokens: [1, 2, 3, 4, 99 + i, 2, 3, 4, 99], requireBoundaryCompanion: true)
                XCTAssertNil(old, "eviction must remove old companion boundary indexes")
                if let old { for b in old.blocks { c.freeBlock(b) } }
            }
        }
        XCTAssertGreaterThan(c.snapshotStats().evictions, 0)
        XCTAssertGreaterThan(c.snapshotStats().companionBytes, 0)
        c.clear()
        XCTAssertEqual(c.snapshotStats().residentBytes, 0)
        XCTAssertEqual(c.snapshotStats().companionBytes, 0)
        XCTAssertEqual(c.snapshotStats().maxResidentBytes, 131_072)
    }

    func testPinnedFetchCannotBeRecycledToFitNewStore() throws {
        let c = PagedCacheManager(blockSize: 4, maxBlocks: 128, maxBytes: 131_072)
        let tokens = Array(0..<8)
        c.storeTokenSequence(tokens: tokens, layerData: [layer(10), layer(20)], boundaryCompanionData: companion(30))
        let hit = try XCTUnwrap(c.fetchPrefix(tokens: tokens + [9], requireBoundaryCompanion: true))
        c.storeTokenSequence(tokens: Array(100..<108), layerData: [layer(40), layer(50)], boundaryCompanionData: companion(60))
        XCTAssertEqual(hit.blocks[0].cacheData![0]!.keys.asArray(Float.self).first, 10)
        XCTAssertEqual(hit.blocks[1].boundaryCompanionData!["ring"]!.asArray(Float.self).first, 30)
        XCTAssertLessThanOrEqual(c.snapshotStats().residentBytes, 131_072)
        for b in hit.blocks { c.freeBlock(b) }
        c.storeTokenSequence(tokens: Array(100..<108), layerData: [layer(40), layer(50)], boundaryCompanionData: companion(60))
        let replacement = try XCTUnwrap(c.fetchPrefix(tokens: Array(100..<109), requireBoundaryCompanion: true))
        XCTAssertEqual(replacement.matchedTokens, 8)
        for b in replacement.blocks { c.freeBlock(b) }
    }

    func testOversizedCompanionIsNotAdvertisedAsCompleteBoundary() {
        let c = PagedCacheManager(blockSize: 4, maxBlocks: 32, maxBytes: 65_536)
        c.storeTokenSequence(tokens: [1,2,3,4], layerData: [layer()],
                            boundaryCompanionData: ["ring": MLXArray.zeros([100_000])])
        XCTAssertNil(c.fetchPrefix(tokens: [1,2,3,4,5], requireBoundaryCompanion: true))
        XCTAssertEqual(c.snapshotStats().companionBytes, 0)
        XCTAssertLessThanOrEqual(c.snapshotStats().residentBytes, 65_536)
    }

    func testReplacingSameLeafDoesNotAccumulateCharge() {
        let c = PagedCacheManager(blockSize: 4, maxBlocks: 32, maxBytes: 131_072)
        var expected: Int?
        for i in 0..<30 {
            c.storeTokenSequence(tokens: [1,2,3,4], layerData: [layer()], boundaryCompanionData: companion(Float(i)))
            let bytes = c.snapshotStats().residentBytes
            if let expected { XCTAssertEqual(bytes, expected) } else { expected = bytes }
        }
        XCTAssertEqual(c.snapshotStats().evictions, 0)
    }

    func testCompactSlicesReleaseLargeSourceAllocationAndPreserveBits() throws {
        let oldLimit = Memory.cacheLimit
        Memory.cacheLimit = 0
        defer { Memory.cacheLimit = oldLimit }
        StreamOrDevice.default.stream.synchronize()
        let baseline = Memory.activeMemory
        let c = PagedCacheManager(blockSize: 4, maxBlocks: 32, maxBytes: 1_048_576)
        func storeViews() {
            let source = MLXArray.zeros([1, 1, 4096, 1024])
            eval(source)
            let view = source[.ellipsis, 0..<4, 0...]
            c.storeTokenSequence(tokens: [1,2,3,4], layerData: [[(keys: view, values: view)]])
        }
        storeViews()
        StreamOrDevice.default.stream.synchronize()
        XCTAssertLessThan(Memory.activeMemory - baseline, 2_097_152,
                          "tiny retained slices must not pin their 16 MiB source allocation")
        let bits: [UInt32] = [0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x3f800000]
        let input = MLXArray(bits).view(dtype: .float32).reshaped(1, 1, 1, bits.count)
        c.storeTokenSequence(tokens: [5], layerData: [[(keys: input, values: input)]])
        let hit = try XCTUnwrap(c.fetchPrefix(tokens: [5,6]))
        XCTAssertEqual(hit.blocks[0].cacheData![0]!.keys.view(dtype: .uint32).asArray(UInt32.self), bits)
        for b in hit.blocks { c.freeBlock(b) }
    }

    func testRotatingSafeAndMaskedPartialBoundariesSurviveBudgetEviction() throws {
        let c = CacheCoordinator(config: CacheCoordinatorConfig(usePagedCache: true,
            enableDiskCache: false, pagedBlockSize: 4, maxCacheBlocks: 128,
            maxCacheBytes: 1_048_576, modelKey: "bounded-rings"))
        c.setPagedBoundaryCompanionRequired(true)
        func store(_ tokens: [Int]) {
            let data = MLXArray(tokens.map(Float.init)).reshaped(1,1,tokens.count,1)
            let full = KVCacheSimple(); _ = full.update(keys: data, values: data + 100)
            let ring = RotatingKVCache(maxSize: 5, keep: 0)
            for i in tokens.indices {
                _ = ring.update(keys: data[.ellipsis, i..<(i+1), 0...], values: (data + 100)[.ellipsis, i..<(i+1), 0...])
                eval(ring)
            }
            let caches: [KVCache] = [full, ring]; eval(caches)
            c.storeAfterGeneration(promptTokens: tokens, perLayerData: extractLayerData(from: caches), ssmStates: nil, cache: caches)
        }
        for group in 0..<12 {
            let tokens = Array((group * 100)..<(group * 100 + 14))
            let seed = Array(tokens.prefix(11))
            store(seed); store(tokens)
            guard case .hit(let matched, let remaining, _, let blocks, _, _) = c.fetch(tokens: Array(tokens.prefix(13)) + [999], skipExactDiskBoundary: true) else {
                return XCTFail("complete shorter partial companion must remain reachable under byte pressure")
            }
            XCTAssertEqual(matched, 11); XCTAssertEqual(remaining, Array(tokens[11..<13]) + [999])
            let restored: [KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: 5, keep: 0)]
            XCTAssertEqual(restoreLayerData(from: blocks, into: restored), 11)
            XCTAssertTrue(restored.allSatisfy { $0.offset == 11 })
            let ring = try XCTUnwrap(restored[1] as? RotatingKVCache)
            XCTAssertEqual(ring.temporallyOrderedKV()!.keys.asArray(Float.self), Array(seed.suffix(5)).map(Float.init))
            let next = MLXArray([Float(999)]).reshaped(1,1,1,1)
            for layer in restored { _ = layer.update(keys: next, values: next + 100) }
            eval(restored)
            c.release(blocks: blocks)
            // A consumer may mutate its restored slot. The retained boundary
            // must still represent the original tokens on the next fetch.
            guard case .hit(_, _, _, let secondBlocks, _, _) = c.fetch(tokens: Array(tokens.prefix(13)) + [999], skipExactDiskBoundary: true) else {
                return XCTFail("stored boundary disappeared after consumer advanced")
            }
            let second: [KVCache] = [KVCacheSimple(), RotatingKVCache(maxSize: 5, keep: 0)]
            XCTAssertEqual(restoreLayerData(from: secondBlocks, into: second), 11)
            let secondRing = try XCTUnwrap(second[1] as? RotatingKVCache)
            XCTAssertEqual(secondRing.temporallyOrderedKV()!.keys.asArray(Float.self), Array(seed.suffix(5)).map(Float.init),
                           "advancing a restored ring must not mutate its cached companion")
            c.release(blocks: secondBlocks)
            XCTAssertLessThanOrEqual(c.pagedCache!.snapshotStats().residentBytes, 1_048_576)
        }
        XCTAssertGreaterThan(c.pagedCache!.snapshotStats().evictions, 0)
    }

    func testPrefixBudgetCLIIsExplicitAndIndependentOfAllocatorPool() throws {
        let base = ["--model-dir", "/tmp/model", "--served-model-id", "test"]
        let defaults = try ServerConfig.parse(arguments: base, recommendedWorkingSetBytes: nil)
        XCTAssertEqual(defaults.prefixCacheLimitBytes, 536_870_912)
        let custom = try ServerConfig.parse(arguments: base + ["--prefix-cache-limit-bytes", "1048576", "--cache-limit-bytes", "2097152"], recommendedWorkingSetBytes: nil)
        XCTAssertEqual(custom.prefixCacheLimitBytes, 1_048_576)
        XCTAssertEqual(custom.cacheLimitBytes, 2_097_152)
        for value in ["0", "-1", "not-an-integer"] {
            XCTAssertThrowsError(try ServerConfig.parse(arguments: base + ["--prefix-cache-limit-bytes", value], recommendedWorkingSetBytes: nil))
        }
    }
}
