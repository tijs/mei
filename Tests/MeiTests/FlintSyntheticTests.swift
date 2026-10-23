import Foundation
import MLX
import MLXNN
import MLXLMCommon
import XCTest
@testable import MeiCore

/// Model-free MLX tests with tiny generated arrays, never downloaded weights.
final class FlintSyntheticTests: XCTestCase {
    func testPublicRegistryActuallyCreatesFlintAndPreservesDtypes() async throws {
        let dir = try FlintFixture.directory(FlintFixture.metadata(tiny: true))
        defer { try? FileManager.default.removeItem(at: dir) }
        let registry = ModelTypeRegistry()
        _ = try await FlintSupport.prepare(directory: dir, enabled: true, registry: registry)
        let model = try await registry.createModel(configuration: FlintFixture.data(FlintFixture.metadata(tiny: true)), modelType: "flint")
        XCTAssertTrue(model is FlintModel)
        XCTAssertTrue(model.preservesCheckpointParameterDTypes)
    }

    func testPinnedAffineLoaderBindsMixedThreeFourEightBitModulesAndUnquantizedRouter() throws {
        let metadata = FlintFixture.metadata(tiny: true)
        let c = try FlintConfiguration(data: FlintFixture.data(metadata))
        let dir = try FlintFixture.directory(metadata)
        defer { try? FileManager.default.removeItem(at: dir) }
        let original = FlintModel(c)
        var tensors = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
        // Quantize only the same module classes the pinned loader replaces,
        // deliberately leaving the router float because its checkpoint has no scales.
        for (path, module) in original.leafModules().flattened() {
            guard (module is Linear && !path.hasSuffix(".mlp.gate")) || module is SwitchLinear || module is Embedding else { continue }
            let q = try XCTUnwrap(c.quantization?.quantization(layer: path))
            let weight = try XCTUnwrap(tensors[path + ".weight"])
            let (packed, scales, biases) = MLX.quantized(weight, groupSize: q.groupSize, bits: q.bits)
            tensors[path + ".weight"] = packed
            tensors[path + ".scales"] = scales
            tensors[path + ".biases"] = biases
        }
        tensors.removeValue(forKey: "model.layers.2.mlp.expert_bias")
        let weightURL = dir.appendingPathComponent("synthetic.safetensors")
        try MLX.save(arrays: tensors, metadata: ["format": "mlx"], url: weightURL)
        let loaded = FlintModel(c)
        try loadWeights(modelDirectory: dir, model: loaded, perLayerQuantization: c.quantization)
        let modules = Dictionary(uniqueKeysWithValues: loaded.leafModules().flattened())
        XCTAssertEqual((modules["model.embed_tokens"] as? QuantizedEmbedding)?.bits, 8)
        XCTAssertEqual((modules["model.layers.0.mlp.gate_proj"] as? QuantizedLinear)?.bits, 4)
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let module = try XCTUnwrap(modules["model.layers.2.mlp.switch_mlp.\(projection)"] as? QuantizedSwitchLinear)
            XCTAssertEqual(module.bits, 3)
            XCTAssertEqual(module.groupSize, 64)
        }
        XCTAssertTrue(modules["model.layers.2.mlp.gate"] is Linear)
        XCTAssertFalse(modules["model.layers.2.mlp.gate"] is QuantizedLinear)
        let rebound = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
        for (key, value) in tensors {
            let actual = try XCTUnwrap(rebound[key], key)
            XCTAssertEqual(actual.shape, value.shape, key)
            XCTAssertEqual(actual.dtype, value.dtype, key)
            XCTAssertTrue((actual .== value).all().item(Bool.self), key)
        }
        let logits = loaded(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: loaded.newCache(parameters: nil))
        XCTAssertEqual(logits.dtype, .float32)
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))

        // Unknown source tensors remain a hard error through the REAL loader,
        // not just a string helper. No relaxation of .noUnusedKeys is allowed.
        tensors["model.layers.2.mlp.unapproved.weight"] = MLXArray([Float(1)])
        try MLX.save(arrays: tensors, metadata: ["format": "mlx"], url: weightURL)
        XCTAssertThrowsError(try loadWeights(modelDirectory: dir, model: FlintModel(c), perLayerQuantization: c.quantization))
    }

    func testQKRMSNormOptionalRoPESinksMasksAndChunkedRotatingCacheMatchScalarOracle() throws {
        var metadata = FlintFixture.metadata(tiny: true)
        metadata["hidden_size"] = 2; metadata["head_dim"] = 2
        metadata["num_attention_heads"] = 1; metadata["num_key_value_heads"] = 1
        metadata["sliding_window"] = 3
        let c = try FlintConfiguration(data: FlintFixture.data(metadata))
        let input: [[Double]] = [[1, 2], [-2, 1], [3, -1], [2, 4], [-1, -3]]
        let x = MLXArray(input.flatMap { $0.map(Float.init) }).reshaped(1, input.count, 2)
        let identity = MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2)
        for layer in [0, 2] {
            let attention = FlintAttention(c, layer: layer)
            try attention.update(parameters: .unflattened([
                "q_proj.weight": identity, "k_proj.weight": identity,
                "v_proj.weight": identity, "o_proj.weight": identity,
                "q_norm.weight": MLXArray([Float(2), 0.5]),
                "k_norm.weight": MLXArray([Float(1.25), 0.75]),
                "softmax_offset": MLXArray([Float(0.7)]),
            ]), verify: [.noUnusedKeys])
            let usesRoPE = layer == 2
            let window: Int? = usesRoPE ? 3 : nil
            func qk(_ row: [Double], _ position: Int, _ norm: [Double]) -> [Double] {
                let denominator = sqrt(row.map { $0 * $0 }.reduce(0, +) / 2 + Double(c.rmsNormEps))
                let v = zip(row, norm).map { $0.0 * $0.1 / denominator }
                guard usesRoPE else { return v }
                let angle = Double(position) // head_dim=2: non-traditional RoPE frequency is 1.
                return [v[0] * cos(angle) - v[1] * sin(angle), v[0] * sin(angle) + v[1] * cos(angle)]
            }
            var expected: [Float] = []
            for i in input.indices {
                let q = qk(input[i], i, [2, 0.5])
                let start = window.map { max(0, i - $0 + 1) } ?? 0
                let positions = Array(start...i)
                let expScores = positions.map { j -> Double in
                    let k = qk(input[j], j, [1.25, 0.75])
                    return exp(zip(q, k).map { $0.0 * $0.1 }.reduce(0, +) / sqrt(2))
                }
                let denominator = expScores.reduce(exp(0.7), +) // learned sink has a zero value vector.
                for d in 0..<2 {
                    expected.append(Float(zip(positions, expScores).map { input[$0.0][d] * $0.1 }.reduce(0, +) / denominator))
                }
            }
            let mask = createAttentionMask(h: x, cache: nil, windowSize: window)
            let full = attention(x, mask: mask, cache: nil).asArray(Float.self)
            for (actual, reference) in zip(full, expected) { XCTAssertEqual(actual, reference, accuracy: 2e-6) }
            let cache: KVCache
            if let window { cache = RotatingKVCache(maxSize: window, keep: 0) }
            else { cache = KVCacheSimple() }
            var chunks: [MLXArray] = []
            for range in [0..<2, 2..<4, 4..<5] {
                let chunk = x[0..., range, 0...]
                let mask = createAttentionMask(h: chunk, cache: cache, windowSize: window)
                let out = attention(chunk, mask: mask, cache: cache)
                MLX.eval(out)
                chunks.append(out)
            }
            let cached = concatenated(chunks, axis: 1).asArray(Float.self)
            XCTAssertEqual(cache.offset, 5)
            for (actual, reference) in zip(cached, expected) { XCTAssertEqual(actual, reference, accuracy: 2e-6) }
        }
    }
}
