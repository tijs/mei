import Foundation
import XCTest
import MLX
import MLXLMCommon
@testable import MeiCore

/// In-memory metadata and tiny synthetic tensors only: never downloads a checkpoint.
enum FlintFixture {
    static func metadata(tiny: Bool = false) -> [String: Any] {
        let count = tiny ? 3 : 49
        let full = tiny ? [0, 1] : [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
        var quant: [String: Any] = ["group_size": 64, "bits": 4,
            "model.embed_tokens": ["group_size": 64, "bits": 8]]
        for layer in 2..<count {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                quant["model.layers.\(layer).mlp.switch_mlp.\(projection)"] = ["group_size": 64, "bits": 3]
            }
        }
        return [
            "model_type": "flint", "architectures": ["FlintForCausalLM"],
            "hidden_size": tiny ? 64 : 2048, "intermediate_size": tiny ? 64 : 6144,
            "num_hidden_layers": count, "num_attention_heads": tiny ? 4 : 32,
            "num_key_value_heads": tiny ? 2 : 4, "head_dim": tiny ? 16 : 128,
            "vocab_size": tiny ? 64 : 150144, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144, "rope_theta": 10000.0,
            "rope_parameters": ["rope_type": "default", "rope_theta": 10000.0],
            "layer_types": (0..<count).map { full.contains($0) ? "full_attention" : "sliding_attention" },
            "no_rope_layer_indices": full, "use_sliding_window": true, "sliding_window": 513,
            "softmax_type": "learnable", "attention_bias": false, "hidden_act": "silu",
            "mlp_only_layers": [0, 1], "decoder_sparse_step": 1,
            "num_experts": tiny ? 8 : 256, "num_experts_per_tok": 6,
            "moe_intermediate_size": tiny ? 64 : 512,
            "shared_expert_intermediate_size": tiny ? 64 : 1024,
            "router_score_function": "sigmoid", "norm_topk_prob": true,
            "tie_word_embeddings": false, "quantization": quant, "quantization_config": quant,
        ]
    }

    static func data(_ metadata: [String: Any] = metadata()) throws -> Data {
        try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys])
    }

    static func directory(_ metadata: [String: Any] = metadata()) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("mei-flint-test-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try data(metadata).write(to: directory.appendingPathComponent("config.json"))
        return directory
    }
}

final class FlintConfigurationTests: XCTestCase {
    func testSelectedArchitectureContract() throws {
        let c = try FlintConfiguration(data: FlintFixture.data())
        XCTAssertEqual(c.hiddenLayers, 49)
        XCTAssertEqual(c.headDim, 128)
        XCTAssertEqual(c.attentionHeads, 32)
        XCTAssertEqual(c.kvHeads, 4)
        XCTAssertEqual(c.numExperts, 256)
        XCTAssertEqual(c.topK, 6)
        XCTAssertEqual(c.mlpOnlyLayers, [0, 1])
        XCTAssertNil(c.windowSize(layer: 0))
        XCTAssertNil(c.windowSize(layer: 48))
        XCTAssertEqual(c.windowSize(layer: 2), 513)
        XCTAssertFalse(c.usesRoPE(layer: 5))
        XCTAssertTrue(c.usesRoPE(layer: 2))
        XCTAssertEqual(c.ropeTheta, 10000)
    }

    func testInvalidOrUnimplementedArchitectureChoicesFailClosed() throws {
        let invalid: [(String, Any)] = [
            ("model_type", "llama"), ("architectures", ["LlamaForCausalLM"]),
            ("hidden_size", 0), ("num_hidden_layers", 0), ("num_attention_heads", 31),
            ("num_key_value_heads", 0), ("head_dim", 127), ("vocab_size", 0),
            ("rms_norm_eps", 0), ("layer_types", ["full_attention"]),
            ("layer_types", Array(repeating: "linear_attention", count: 49)),
            ("sliding_window", 0), ("mlp_only_layers", [49]), ("no_rope_layer_indices", [-1]),
            ("hidden_act", "gelu"), ("router_score_function", "softmax"),
            ("softmax_type", "vanilla"), ("num_experts_per_tok", 257),
            ("num_experts_per_tok", 0), ("norm_topk_prob", false),
            ("decoder_sparse_step", 2), ("tie_word_embeddings", true),
            ("routed_scaling_factor", 2.0),
            ("rope_parameters", ["rope_type": "yarn", "factor": 4]),
        ]
        for (key, value) in invalid {
            var data = FlintFixture.metadata()
            data[key] = value
            XCTAssertThrowsError(try FlintConfiguration(data: FlintFixture.data(data)), key)
        }
    }

    func testQuantizationMetadataPreservesAllMixedPrecisionOverrides() throws {
        let c = try FlintConfiguration(data: FlintFixture.data())
        let q = try XCTUnwrap(c.quantization)
        XCTAssertEqual(q.quantization?.bits, 4)
        XCTAssertEqual(q.quantization?.groupSize, 64)
        XCTAssertEqual(q.perLayerQuantization.count, 142)
        for layer in 2..<49 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                XCTAssertEqual(q.quantization(layer: "model.layers.\(layer).mlp.switch_mlp.\(projection)")?.bits, 3)
            }
        }
        XCTAssertEqual(q.quantization(layer: "model.embed_tokens")?.bits, 8)
        XCTAssertEqual(q.quantization(layer: "model.layers.0.mlp.gate_proj")?.bits, 4)
    }

    func testCustomPacksAndConflictingQuantizationAreRejected() throws {
        for (key, value) in [("weight_format", "mxtq"), ("weight_format", "jangtq2")] {
            var d = FlintFixture.metadata(); d[key] = value
            XCTAssertThrowsError(try FlintConfiguration(data: FlintFixture.data(d)))
        }
        for q: [String: Any] in [
            ["bits": 2, "group_size": 64, "mode": "jangtq2"],
            ["bits": 4, "group_size": 64, "mode": "mxfp4"],
            ["bits": 4, "group_size": 0],
        ] {
            var d = FlintFixture.metadata(); d["quantization"] = q; d["quantization_config"] = q
            XCTAssertThrowsError(try FlintConfiguration(data: FlintFixture.data(d)))
        }
        var d = FlintFixture.metadata()
        d["quantization_config"] = ["bits": 8, "group_size": 64]
        XCTAssertThrowsError(try FlintConfiguration(data: FlintFixture.data(d)))
    }

    func testSanitizerPreservesStackedKeysQuantSidecarsSinksAndUnknowns() throws {
        let c = try FlintConfiguration(data: FlintFixture.data())
        let prefix = "model.layers.2.mlp."
        let input = [prefix + "switch_mlp.gate_proj.weight": 1,
                     prefix + "switch_mlp.gate_proj.scales": 2,
                     prefix + "switch_mlp.gate_proj.biases": 3,
                     prefix + "gate.weight": 4,
                     "model.layers.2.self_attn.softmax_offset": 5,
                     "unknown.weight": 6]
        let output = c.sanitize(input, zeroExpertBias: { 0 })
        for (key, value) in input { XCTAssertEqual(output[key], value, key) }
        XCTAssertEqual(output[prefix + "expert_bias"], 0)
        XCTAssertNil(output[prefix + "gate.scales"], "an unquantized router stays unquantized")
        XCTAssertEqual(output, c.sanitize(output, zeroExpertBias: { 0 }))
    }

    func testCustomQuantizationMetadataCannotMasqueradeAsAffine() throws {
        let additions: [[String: Any]] = [
            ["mxtq_bits": 2], ["quant_method": "jangh2"],
            ["model.layers.2.mlp.switch_mlp.gate_proj": ["bits": 3, "packing": "jangh2"]],
            ["per_tensor": ["model.layers.2.mlp.switch_mlp.gate_proj": ["bits": 3, "mode": "jangtq2"]]],
        ]
        for extra in additions {
            var d = FlintFixture.metadata()
            var q = try XCTUnwrap(d["quantization"] as? [String: Any])
            q.merge(extra) { _, new in new }
            d["quantization"] = q; d["quantization_config"] = q
            XCTAssertThrowsError(try FlintConfiguration(data: FlintFixture.data(d)), "\(extra)")
        }
    }

    func testMalformedOrConflictingRoPEFieldsAreNotSilentlyIgnored() throws {
        for rope: [String: Any] in [
            ["rope_type": "default", "rope_theta": "invalid"],
            ["rope_type": "default", "type": "linear"],
        ] {
            var d = FlintFixture.metadata(); d["rope_parameters"] = rope
            XCTAssertThrowsError(try FlintConfiguration(data: FlintFixture.data(d)))
        }
    }

    func testSanitizerMapsOnlyKnownAliasesAndNeverDiscardsCollisions() throws {
        let c = try FlintConfiguration(data: FlintFixture.data())
        for alias in ["expert_bias", "e_score_correction_bias"] {
            let input = ["language_model.model.layers.2.mlp.gate.\(alias)": 7,
                         "language_model.model.embed_tokens.scales": 8,
                         "language_modelish.model.norm.weight": 9]
            let out = c.sanitize(input, zeroExpertBias: { 0 })
            XCTAssertEqual(out["model.layers.2.mlp.expert_bias"], 7)
            XCTAssertEqual(out["model.embed_tokens.scales"], 8)
            XCTAssertEqual(out["language_modelish.model.norm.weight"], 9)
            XCTAssertEqual(out.count, input.count)
        }
        let input = ["model.layers.2.mlp.expert_bias": 1,
                     "model.layers.2.mlp.gate.expert_bias": 2,
                     "model.norm.weight": 3, "language_model.model.norm.weight": 4]
        XCTAssertEqual(c.sanitize(input, zeroExpertBias: { 0 }).count, input.count,
                       "colliding source keys must survive for strict unused-key failure")
    }

    func testCustomPackedTensorNamesAreRefusedWithoutDroppingThem() throws {
        for suffix in ["tq2_packed", "tq2_scales", "tq_packed", "tq_norms"] {
            XCTAssertThrowsError(try FlintSupport.validateWeightKeys([
                "model.layers.2.mlp.switch_mlp.gate_proj.\(suffix)"]))
        }
        XCTAssertNoThrow(try FlintSupport.validateWeightKeys(["unknown.weight", "model.layers.2.mlp.gate.weight"]))
    }
}

final class FlintFeatureGateTests: XCTestCase {
    func testCLIFlagIsDefaultOffAndStrictlyParsed() throws {
        let args = ["--model-dir", "/nonexistent-flint", "--served-model-id", "test"]
        XCTAssertFalse(try ServerConfig.parse(arguments: args, recommendedWorkingSetBytes: nil).enableFlint)
        XCTAssertTrue(try ServerConfig.parse(arguments: args + ["--enable-flint", "true"], recommendedWorkingSetBytes: nil).enableFlint)
        XCTAssertFalse(try ServerConfig.parse(arguments: args + ["--enable-flint", "false"], recommendedWorkingSetBytes: nil).enableFlint)
        XCTAssertThrowsError(try ServerConfig.parse(arguments: args + ["--enable-flint", "maybe"], recommendedWorkingSetBytes: nil))
        XCTAssertTrue(ServerConfig.usage.contains("--enable-flint"))
    }

    func testDisabledEngineLoadRefusesBeforeTokenizerWeightsOrGPU() async throws {
        let dir = try FlintFixture.directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try await Engine.load(config: ServerConfig(modelDirectory: dir.path, servedModelID: "flint"))
            XCTFail("Flint must not reach weight loading by default")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("--enable-flint true"), error.localizedDescription)
        }
    }

    func testDisabledPrepareDoesNotRegisterAndNonFlintIsUnchanged() async throws {
        let registry = ModelTypeRegistry()
        let dir = try FlintFixture.directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try await FlintSupport.prepare(directory: dir, enabled: false, registry: registry)
            XCTFail("disabled Flint accepted")
        } catch { XCTAssertTrue(error.localizedDescription.contains("--enable-flint true")) }
        let types = await registry.registeredModelTypes
        XCTAssertFalse(types.contains("flint"))
        let other = try FlintFixture.directory(["model_type": "qwen3"])
        defer { try? FileManager.default.removeItem(at: other) }
        let config = try await FlintSupport.prepare(directory: other, enabled: false, registry: registry)
        XCTAssertNil(config.toolCallFormat)
        XCTAssertNil(config.reasoningParserName)
        XCTAssertNil(config.generationDefaults)
        do {
            _ = try await registry.createModel(configuration: Data("{}".utf8), modelType: "unapproved")
            XCTFail("unapproved model type accepted")
        } catch { XCTAssertTrue(error.localizedDescription.contains("unapproved")) }
    }

    func testNonFlintJSON5MetadataKeepsTheExistingLoaderContract() async throws {
        let dir = try FlintFixture.directory(["model_type": "qwen3"])
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{model_type: 'qwen3', // provider supports JSON5\n}".utf8)
            .write(to: dir.appendingPathComponent("config.json"))
        for enabled in [false, true] {
            let prepared = try await FlintSupport.prepare(directory: dir, enabled: enabled, registry: ModelTypeRegistry())
            XCTAssertNil(prepared.toolCallFormat)
            let config = try ServerConfig.parse(arguments: ["--model-dir", dir.path, "--enable-flint", String(enabled)],
                                                recommendedWorkingSetBytes: nil)
            XCTAssertEqual(config.temperature, 0.6)
        }
    }

    func testEnabledRegistrationValidatesAndStampsOnlyParsers() async throws {
        let registry = ModelTypeRegistry()
        let dir = try FlintFixture.directory(FlintFixture.metadata(tiny: true))
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try await FlintSupport.prepare(directory: dir, enabled: true, registry: registry)
        let types = await registry.registeredModelTypes
        XCTAssertTrue(types.contains("flint"))
        XCTAssertEqual(config.toolCallFormat, .glm4)
        XCTAssertEqual(config.reasoningParserName, "think_xml")
        XCTAssertNil(config.generationDefaults, "factory must still read the bundle's generation_config.json")
        XCTAssertTrue(config.extraEOSTokens.isEmpty)
        do {
            _ = try await registry.createModel(configuration: Data("{\"model_type\":\"llama\"}".utf8), modelType: "flint")
            XCTFail("registration must validate, not reinterpret another architecture")
        } catch { /* validation before constructing a model */ }
        do {
            _ = try await FlintSupport.prepare(directory: dir, enabled: false, registry: registry)
            XCTFail("prior registration must not bypass the next load's gate")
        } catch { XCTAssertTrue(error.localizedDescription.contains("--enable-flint true")) }
    }

    func testGenerationDefaultsAndExplicitSamplerOverrides() throws {
        let dir = try FlintFixture.directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"do_sample\":true}".utf8)
            .write(to: dir.appendingPathComponent("generation_config.json"))
        let args = ["--model-dir", dir.path, "--enable-flint", "true"]
        let config = try ServerConfig.parse(arguments: args, recommendedWorkingSetBytes: nil)
        XCTAssertEqual(config.temperature, 1.0)
        XCTAssertEqual(config.topP, 0.95)
        XCTAssertEqual(config.topK, 20)
        XCTAssertNil(config.enableThinking)
        XCTAssertNil(config.reasoningEffort)
        XCTAssertNil(config.repetitionPenalty)
        let explicit = try ServerConfig.parse(arguments: args + ["--temperature", "0.25", "--top-p", "0.8", "--top-k", "4"], recommendedWorkingSetBytes: nil)
        XCTAssertEqual(explicit.temperature, 0.25)
        XCTAssertEqual(explicit.topP, 0.8)
        XCTAssertEqual(explicit.topK, 4)
    }

    func testProgrammaticAndPartialSamplerOverridesSurviveRepeatedResolution() throws {
        let dir = try FlintFixture.directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{\"temperature\":1.0,\"top_p\":0.8,\"top_k\":6,\"do_sample\":true}".utf8)
            .write(to: dir.appendingPathComponent("generation_config.json"))
        var config = ServerConfig(modelDirectory: dir.path, servedModelID: "flint")
        config.enableFlint = true
        config.temperature = 0.6 // Even an assignment equal to Mei's default is explicit.
        config.topK = 0
        for _ in 0..<2 { try config.applyFlintGenerationDefaults() }
        XCTAssertEqual(config.temperature, 0.6)
        XCTAssertEqual(config.topP, 0.8)
        XCTAssertEqual(config.topK, 0)
        var parsed = try ServerConfig.parse(arguments: ["--model-dir", dir.path,
            "--enable-flint", "true", "--top-p", "0.7"], recommendedWorkingSetBytes: nil)
        try parsed.applyFlintGenerationDefaults()
        XCTAssertEqual(parsed.temperature, 1)
        XCTAssertEqual(parsed.topP, 0.7)
        XCTAssertEqual(parsed.topK, 6)
    }

    func testQuantizedKVIsRefusedBeforeFlintCanReachUnsupportedSinkDispatch() throws {
        let dir = try FlintFixture.directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"do_sample\":true}".utf8)
            .write(to: dir.appendingPathComponent("generation_config.json"))
        let args = ["--model-dir", dir.path, "--enable-flint", "true", "--kv-bits", "4"]
        XCTAssertThrowsError(try ServerConfig.parse(arguments: args, recommendedWorkingSetBytes: nil)) {
            XCTAssertTrue($0.localizedDescription.contains("--kv-bits"), $0.localizedDescription)
        }
        var direct = ServerConfig(modelDirectory: dir.path, servedModelID: "flint")
        direct.enableFlint = true; direct.kvBits = 8
        XCTAssertThrowsError(try direct.applyFlintGenerationDefaults())
        direct.enableFlint = false
        XCTAssertNoThrow(try direct.applyFlintGenerationDefaults(), "gate-off behavior is unchanged")
    }

    func testCustomSidecarsAndIndexedPacksRefuseBeforeRegistration() async throws {
        let registry = ModelTypeRegistry()
        for sidecar in jangConfigFileNames + ["jangtq_runtime.safetensors"] {
            let dir = try FlintFixture.directory()
            defer { try? FileManager.default.removeItem(at: dir) }
            try Data().write(to: dir.appendingPathComponent(sidecar))
            do {
                _ = try await FlintSupport.prepare(directory: dir, enabled: true, registry: registry)
                XCTFail("accepted custom sidecar \(sidecar)")
            } catch { XCTAssertTrue(error.localizedDescription.contains("not an affine checkpoint")) }
        }
        for key in ["model.layers.2.mlp.switch_mlp.gate_proj.tq2_packed", "model.layers.2.mlp.experts.0.gate_proj.weight"] {
            let dir = try FlintFixture.directory()
            defer { try? FileManager.default.removeItem(at: dir) }
            try JSONSerialization.data(withJSONObject: ["weight_map": [key: "unused.safetensors"]])
                .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
            do {
                _ = try await FlintSupport.prepare(directory: dir, enabled: true, registry: registry)
                XCTFail("accepted custom indexed tensor \(key)")
            } catch { XCTAssertTrue(error.localizedDescription.contains("Unsupported Flint bundle")) }
        }
        let types = await registry.registeredModelTypes
        XCTAssertFalse(types.contains("flint"))
    }

    func testReasoningAndToolParsersMatchSelfEmittedFlintMarkers() throws {
        for text in ["plain answer", "<think>private</think>answer"] {
            var parser = try XCTUnwrap(ReasoningParser.forPrompt(stampName: "think_xml", promptTail: "<|assistant|>"))
            var segments = parser.feed(text)
            segments += parser.flush()
            let content = segments.compactMap { if case .content(let s) = $0 { s } else { nil } }.joined()
            let reasoning = segments.compactMap { if case .reasoning(let s) = $0 { s } else { nil } }.joined()
            XCTAssertEqual(content, text == "plain answer" ? text : "answer")
            XCTAssertEqual(reasoning, text == "plain answer" ? "" : "private")
        }
        let processor = ToolCallProcessor(format: .glm4)
        for chunk in ["<tool_", "call>sum<arg_key>x</arg_key><arg_value>15</arg_value>", "</tool_call>"] {
            _ = processor.processChunk(chunk)
        }
        processor.processEOS()
        XCTAssertEqual(processor.toolCalls.count, 1)
        XCTAssertEqual(processor.toolCalls.first?.function.name, "sum")
        // The pinned GLM scalar parser preserves a bare numeric fragment as
        // text; Mei's existing schema-aware API normalizer supplies JSON typing.
        XCTAssertEqual(processor.toolCalls.first?.function.arguments["x"]?.anyValue as? String, "15")
        let normalized = ToolArgumentNormalizer.normalize(arguments: .object(["x": .string("15")]),
            parametersSchema: .object(["type": .string("object"), "properties": .object([
                "x": .object(["type": .string("integer")])])]))
        XCTAssertEqual(normalized, .object(["x": .number(15)]))
    }
}

/// Tiny synthetic MLX computation, not model admission, real-checkpoint parity or performance evidence.
final class FlintNumericsTests: XCTestCase {
    func testSigmoidRouterBiasSelectsButDoesNotWeightAndRenormalizesTopSix() {
        let logits = MLXArray([Float(-2), -1, 0, 1, 2, 3, 4, 5]).reshaped(1, 1, 8)
        let bias = MLXArray([Float(100), 100, 0, 0, 0, 0, 0, 0])
        let (indices, weights) = FlintRouting.route(logits: logits, expertBias: bias, topK: 6)
        let ids = indices.asArray(Int32.self).map(Int.init)
        let values = weights.asArray(Float.self)
        XCTAssertEqual(Set(ids), Set([0, 1, 4, 5, 6, 7]))
        let scores = ids.map { Float(1 / (1 + exp(-Double($0 - 2)))) }
        let total = scores.reduce(0, +)
        for (value, score) in zip(values, scores) { XCTAssertEqual(value, score / total, accuracy: 1e-6) }
        XCTAssertEqual(values.reduce(0, +), 1, accuracy: 1e-6)
        XCTAssertEqual(weights.dtype, .float32)
    }

    func testRouterProjectsThenPromotesToFP32AndTopOneStillNormalizes() {
        let logits = MLXArray([Float(1), 2, 3, 4]).asType(.bfloat16)
        let (_, weights) = FlintRouting.route(logits: logits, expertBias: zeros([4]), topK: 1)
        XCTAssertEqual(weights.dtype, .float32)
        XCTAssertEqual(weights.item(Float.self), 1, accuracy: 1e-6)
        let (_, extreme) = FlintRouting.route(logits: MLXArray([Float(-1000), -1000]), expertBias: zeros([2]), topK: 2)
        XCTAssertTrue(extreme.asArray(Float.self).allSatisfy { $0.isFinite && $0 == 0 })
    }

    func testRoutedReductionUsesFP32AndAlwaysAddsUnweightedSharedExpert() {
        let routed = MLXArray([Float(2), 4, 6, 8]).reshaped(1, 1, 2, 2).asType(.bfloat16)
        let weights = MLXArray([Float(0.25), 0.75]).reshaped(1, 1, 2)
        let shared = MLXArray([Float(10), 20]).reshaped(1, 1, 2).asType(.bfloat16)
        let out = FlintRouting.combine(routed: routed, weights: weights, shared: shared, dtype: .bfloat16)
        XCTAssertEqual(out.dtype, .bfloat16)
        XCTAssertEqual(out.asArray(Float.self), [15, 27])
    }

    func testTinyModelModuleTreeAndCacheTypesAreCheckpointExact() throws {
        let c = try FlintConfiguration(data: FlintFixture.data(FlintFixture.metadata(tiny: true)))
        let model = FlintModel(c)
        let keys = Set(model.parameters().flattened().map(\.0))
        for name in ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
                     "model.layers.0.mlp.gate_proj.weight", "model.layers.1.mlp.up_proj.weight",
                     "model.layers.2.mlp.gate.weight", "model.layers.2.mlp.expert_bias",
                     "model.layers.2.mlp.shared_expert.down_proj.weight",
                     "model.layers.2.mlp.switch_mlp.gate_proj.weight",
                     "model.layers.2.self_attn.softmax_offset", "model.layers.2.self_attn.q_norm.weight"] {
            XCTAssertTrue(keys.contains(name), name)
        }
        let caches = model.newCache(parameters: GenerateParameters(maxKVSize: 999))
        XCTAssertEqual(caches.count, 3)
        XCTAssertTrue(caches[0] is KVCacheSimple)
        XCTAssertTrue(caches[1] is KVCacheSimple)
        XCTAssertTrue(caches[2] is RotatingKVCache)
        XCTAssertEqual(caches[2].maxSize, 513)
        let logits = model(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: caches)
        XCTAssertEqual(logits.shape, [1, 2, 64])
        XCTAssertEqual(logits.dtype, .float32)
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertThrowsError(try model.update(parameters: .unflattened(["unknown.weight": MLXArray([Float(1)])]), verify: [.noUnusedKeys]))
    }
}
