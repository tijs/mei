import MLX
import MLXLMCommon
import XCTest

final class RotatingRestoreIsolationTests: XCTestCase {
    private func append(_ value: Float, to ring: RotatingKVCache) {
        let keys = MLXArray([value]).reshaped(1, 1, 1, 1)
        _ = ring.update(keys: keys, values: keys + 100)
        eval(ring)
    }

    func testRestoredRequestsCannotMutateRetainedRingOrEachOther() throws {
        for nested in [false] {
            for keep in [0, 1] {
                let source = RotatingKVCache(maxSize: 5, keep: keep, step: 1)
                for token in 1...12 { append(Float(token), to: source) }
                let retained: [any KVCache] = nested ? [CacheList(source)] : [source]
                let payload = TQDiskSerializer.serialize(cache: retained)
                eval(Array(payload.values))
                let expected = source.state.map { $0.asArray(Float.self) }
                let expectedMeta = source.metaState
                let key = nested ? "rot_0_sub_0_keys" : "rot_0_keys"
                let valueKey = nested ? "rot_0_sub_0_values" : "rot_0_values"
                let first = RotatingKVCache(maxSize: 5, keep: keep, step: 1)
                let second = RotatingKVCache(maxSize: 5, keep: keep, step: 1)
                var firstTarget: [any KVCache] = nested ? [CacheList(first)] : [first]
                var secondTarget: [any KVCache] = nested ? [CacheList(second)] : [second]
                XCTAssertEqual(restoreOriginal(payload, into: &firstTarget), 12)
                XCTAssertEqual(restoreOriginal(payload, into: &secondTarget), 12)
                for token in 13...24 { append(Float(token), to: first) }
                XCTAssertEqual(first.offset, 24)
                XCTAssertNotEqual(first.state.map { $0.asArray(Float.self) }, expected)
                XCTAssertEqual(second.state.map { $0.asArray(Float.self) }, expected)
                XCTAssertEqual(second.metaState, expectedMeta)
                XCTAssertEqual(source.state.map { $0.asArray(Float.self) }, expected)
                XCTAssertEqual(source.metaState, expectedMeta)
                XCTAssertEqual(try XCTUnwrap(payload[key]).asArray(Float.self), expected[0])
                XCTAssertEqual(try XCTUnwrap(payload[valueKey]).asArray(Float.self), expected[1])
                let firstState = first.state.map { $0.asArray(Float.self) }
                append(101, to: second)
                XCTAssertEqual(second.offset, 13)
                XCTAssertEqual(first.state.map { $0.asArray(Float.self) }, firstState)
                XCTAssertEqual(try XCTUnwrap(payload[key]).asArray(Float.self), expected[0])
            }
        }
    }
}

private func restoreOriginal(_ payload: [String: MLXArray], into target: inout [any KVCache]) -> Int {
    guard case let .rotating(comp)? = TQDiskSerializer.deserialize(payload).first else { return 0 }
    restoreRotatingLayer(comp, into: target[0])
    return comp.offset
}

private func restoreRotatingLayer(
    _ comp: TQDiskSerializer.RotatingLayerComponents,
    into layer: any KVCache
) {
    func apply(_ rot: RotatingKVCache) {
        rot.state = [comp.keys, comp.values]
        rot.metaState = [
            String(comp.keep),
            String(comp.maxSize),
            String(comp.step),
            String(comp.offset),
            String(comp.idx),
        ]
    }
    if let rot = layer as? RotatingKVCache {
        apply(rot)
        return
    }
    // Composite cache that wraps a RotatingKVCache (e.g. DeepseekV4Cache).
    // Restores the inner rotating state; the wrapper's ephemeral buffer
    // state is cleared and will be repopulated on the next prefill.
    if let wrapper = layer as? RotatingKVCacheWrapper {
        apply(wrapper.rotating)
        return
    }
    if let cacheList = layer as? CacheList {
        for i in 0..<cacheList.count {
            if let rot = cacheList[i] as? RotatingKVCache {
                apply(rot)
                return
            }
            if let wrapper = cacheList[i] as? RotatingKVCacheWrapper {
                apply(wrapper.rotating)
                return
            }
        }
    }
}
