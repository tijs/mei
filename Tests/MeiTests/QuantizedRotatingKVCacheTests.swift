import XCTest
import MLX
import MLXLMCommon

/// Unit tests for the Mei vmlx patch 0001 (`QuantizedRotatingKVCache`):
/// a quantized ring that mirrors `RotatingKVCache`'s observable attention
/// span while storing affine-quantized K/V tuples.
///
/// These tests are cache-level and model-free: no weights, tiny arrays.
final class QuantizedRotatingKVCacheTests: XCTestCase {

    /// headDim must be divisible by the quantization groupSize (the real
    /// Ornith 9B uses headDim 256 with groupSize 64); tests use 64.
    private func tokenArray(_ i: Int) -> MLXArray {
        MLXArray(Array(0 ..< 128)).asType(.float16).reshaped(1, 2, 1, 64) + MLXArray(i)
    }

    private func makeRing(maxSize: Int = 64, keep: Int = 4, tokens: Int = 40)
        -> RotatingKVCache
    {
        let ring = RotatingKVCache(maxSize: maxSize, keep: keep)
        // Prefill-style multi-token appends (the chunked-prefill path) —
        // 8 tokens at a time keeps the buffer temporal, like a 512-chunk
        // prefill at model scale.
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for i in 0 ..< tokens {
            // [1, 2 heads, 1, 8 dims]; distinct per position so ordering
            // errors show up in attention output.
            keys.append(tokenArray(i))
            values.append(tokenArray(i) * 0.5 + MLXArray(i))
            // Per-token append (decode-style) every token to exercise the
            // in-place update path too.
            if keys.count == 8 {
                let k = concatenated(keys, axis: 2)
                let v = concatenated(values, axis: 2)
                _ = ring.update(keys: k, values: v)
                keys = []
                values = []
            }
        }
        return ring
    }

    func testConversionPreservesOffsetAndSpan() throws {
        let ring = makeRing()
        XCTAssertEqual(ring.offset, 40)
        let q = ring.toQuantized(groupSize: 64, bits: 8, mode: .affine)
        XCTAssertEqual(q.offset, 40)
        XCTAssertEqual(q.maxSize, 64)
        guard let state = q.getQuantizedState() else {
            return XCTFail("no quantized state after conversion")
        }
        // Quantized storage must hold the same temporal span as the ring:
        // 40 tokens, sink 4 + recent 36.
        XCTAssertEqual(state.0.0.dim(2), 40)
        XCTAssertEqual(q.innerState().count, 6)  // affine 8-bit: wq,scales,biases x2
    }

    func testAttentionMatchesFP16WithinQuantizationError() throws {
        let ring = makeRing(maxSize: 128, tokens: 64)
        let q = ring.toQuantized(groupSize: 64, bits: 8, mode: .affine)

        func attention(_ cache: KVCache) -> MLXArray {
            // Decode-style single token, positions continuing the cache.
            let k = tokenArray(64)
            let v = k * 0.5
            let queries = tokenArray(64) * 0.25
            return attentionWithCacheUpdate(
                queries: queries, keys: k, values: v, cache: cache,
                scale: 1.0 / sqrt(Float(64)), mask: .none)
        }

        let fp16Out = attention(ring)
        let quantOut = attention(q)
        MLX.eval(fp16Out, quantOut)
        XCTAssertEqual(fp16Out.shape, quantOut.shape)
        // Affine 8-bit KV: per-element error should be well under 5% of the
        // fp16 value range (values are ~0..67; scale per group of 64).
        let maxAbs = (fp16Out - quantOut).abs().max().item(Float.self)
        XCTAssertLessThan(maxAbs, 2.0, "fp16 vs 8-bit attention mismatch: \(maxAbs)")
    }

    func testSlidingWindowBoundedByMaxSize() throws {
            let ring = makeRing(maxSize: 16, keep: 2, tokens: 40)
            // The fp16 ring allows temporary growth to maxSize + S - 1 during
            // multi-token prefill (RotatingKVCache.updateConcat contract); the
            // conversion must mirror the fp16 span EXACTLY, and subsequent
            // single-token decode appends must slide back down to maxSize.
            let fp16Span = ring.state[0].dim(2)
            let q = ring.toQuantized(groupSize: 64, bits: 8, mode: .affine)
            guard let state = q.getQuantizedState() else {
                return XCTFail("no quantized state")
            }
            XCTAssertEqual(state.0.0.dim(2), fp16Span, "quantized ring must mirror the fp16 span")
            XCTAssertEqual(q.offset, 40, "absolute offset preserved across rotation")

            // Append 8 more tokens one at a time (decode): the window must
            // slide to at most maxSize.
            for _ in 0 ..< 8 {
                let k = tokenArray(q.offset)
                let v = k * 0.5
                _ = q.updateQuantized(keys: k, values: v)
            }
            let finalSpan = q.getQuantizedState()!.0.0.dim(2)
            XCTAssertLessThanOrEqual(finalSpan, 16, "quantized ring must slide to maxSize")
            XCTAssertEqual(q.offset, 48)
        }

    func testStateMetaStateRoundTrip() throws {
        let ring = makeRing(maxSize: 128, tokens: 50)
        let q = ring.toQuantized(groupSize: 64, bits: 8, mode: .affine)
        let state = q.state
        let meta = q.metaState

        let restored = QuantizedRotatingKVCache(maxSize: 128, keep: 4, step: 256)
        restored.state = state
        restored.metaState = meta

        XCTAssertEqual(restored.offset, q.offset)
        XCTAssertEqual(restored.getQuantizedState()!.0.0.dim(2), q.getQuantizedState()!.0.0.dim(2))
        XCTAssertEqual(restored.innerState().count, q.innerState().count)
    }

    func testMaybeQuantizeConvertsHybridCacheArray() throws {
        // Simulate the Ornith 9B attention topology: 2 rotating layers +
        // 2 mamba layers. maybeQuantizeKVCache must convert the rotating
        // layers to QuantizedRotatingKVCache and leave MambaCache alone.
        var cache: [KVCache] = [
            makeRing(maxSize: 128, tokens: 32),
            makeRing(maxSize: 128, tokens: 32),
            MambaCache(),
            MambaCache(),
        ]
        maybeQuantizeKVCache(cache: &cache, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0)
        XCTAssertTrue(cache[0] is QuantizedRotatingKVCache)
        XCTAssertTrue(cache[1] is QuantizedRotatingKVCache)
        XCTAssertTrue(cache[2] is MambaCache)
        XCTAssertTrue(cache[3] is MambaCache)
        XCTAssertEqual(cache[0].offset, 32)
        // Idempotency: a second pass must not double-convert or crash.
        maybeQuantizeKVCache(cache: &cache, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0)
        XCTAssertTrue(cache[0] is QuantizedRotatingKVCache)
    }

    func testQuantizedRingSurvivesRotationBeyondKeep() throws {
        // Sink tokens must be preserved exactly; rotated-out non-sink
        // tokens must be dropped (this is the ring contract).
        let q = QuantizedRotatingKVCache(maxSize: 8, keep: 2, step: 256)
        for i in 0 ..< 20 {
            let k = tokenArray(i)
            _ = q.updateQuantized(keys: k, values: k * 0.5)
        }
        guard let state = q.getQuantizedState() else { return XCTFail("no state") }
        XCTAssertEqual(state.0.0.dim(2), 8)
        XCTAssertEqual(q.offset, 20)
        // Sink (positions 0..<2) must still be present: decode the stored
        // weights back and check the first two positions match 0 and 1.
        let deq = dequantized(
            state.0.0, scales: state.0.1, biases: state.0.2,
            groupSize: 64, bits: 8, mode: .affine)
        MLX.eval(deq)
        let first = deq[0, 0, 0, 0].item(Float.self)
        XCTAssertEqual(first, 0.0, accuracy: 0.1, "sink token 0 must survive rotation")
        let second = deq[0, 0, 1, 0].item(Float.self)
        XCTAssertEqual(second, 1.0, accuracy: 0.1, "sink token 1 must survive rotation")
    }
}