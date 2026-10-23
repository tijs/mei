import Foundation
import CoreFoundation
import MLXLMCommon

/// Narrow, validated contract for Flint's pre-stacked standard MLX affine export.
/// Architecture reference: Blaizzy/mlx-vlm v0.7.10, commit
/// 8c061ef516e7f8b584c0d5b64f26b50d5806303b (MIT; see THIRD-PARTY-NOTICES.md).
struct FlintConfiguration: Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let vocabularySize: Int
    let maxPositionEmbeddings: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let layerTypes: [String]
    let noRoPELayers: Set<Int>
    let slidingWindow: Int?
    let attentionBias: Bool
    let mlpOnlyLayers: Set<Int>
    let numExperts: Int
    let topK: Int
    let moeIntermediateSize: Int
    let sharedExpertIntermediateSize: Int
    let quantization: BaseConfiguration.PerLayerQuantization?

    init(data: Data) throws {
        let f = try JSONDecoder().decode(Fields.self, from: data)
        let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        func require(_ condition: Bool, _ message: String) throws {
            if !condition { throw FlintError.unsupported(message) }
        }
        try require(f.modelType == "flint", "model_type must be flint (no architecture fallback)")
        try require(f.architectures == nil || f.architectures == ["FlintForCausalLM"], "architectures must be FlintForCausalLM")
        for (name, value) in [
            ("hidden_size", f.hiddenSize), ("intermediate_size", f.intermediateSize),
            ("num_hidden_layers", f.hiddenLayers), ("num_attention_heads", f.attentionHeads),
            ("num_key_value_heads", f.kvHeads), ("head_dim", f.headDim),
            ("vocab_size", f.vocabularySize), ("num_experts", f.numExperts),
            ("max_position_embeddings", f.maxPositionEmbeddings),
            ("moe_intermediate_size", f.moeIntermediateSize),
        ] { try require(value > 0, "\(name) must be positive") }
        try require(f.sharedExpertIntermediateSize >= 0, "shared_expert_intermediate_size must be nonnegative")
        try require(f.attentionHeads % f.kvHeads == 0, "query heads must be divisible by KV heads")
        try require(f.headDim % 2 == 0, "head_dim must be even")
        try require(!f.attentionHeads.multipliedReportingOverflow(by: f.headDim).overflow, "attention dimensions overflow")
        try require(f.rmsNormEps.isFinite && f.rmsNormEps > 0, "rms_norm_eps must be positive and finite")
        try require(f.layerTypes.count == f.hiddenLayers, "layer_types must match num_hidden_layers")
        try require(Set(f.layerTypes).isSubset(of: ["full_attention", "sliding_attention"]), "unsupported attention layer type")
        for (name, indices) in [("mlp_only_layers", f.mlpOnlyLayers), ("no_rope_layer_indices", f.noRoPELayers)] {
            try require(indices.allSatisfy { (0..<f.hiddenLayers).contains($0) } && Set(indices).count == indices.count,
                        "\(name) must contain unique, in-range indices")
        }
        try require(f.slidingWindow == nil || f.slidingWindow! > 0, "sliding_window must be positive")
        try require(!f.useSlidingWindow || !f.layerTypes.contains("sliding_attention") || f.slidingWindow != nil,
                    "sliding attention requires sliding_window")
        try require(f.hiddenAct == "silu" && f.routerScoreFunction == "sigmoid", "requires SiLU experts and sigmoid routing")
        try require(f.softmaxType == "learnable" && f.attentionSink != false, "requires learned attention sinks")
        try require((1...f.numExperts).contains(f.topK), "num_experts_per_tok must be between 1 and num_experts")
        try require(f.normTopKProb, "only renormalized top-k routing is supported")
        try require(f.decoderSparseStep == 1, "only decoder_sparse_step=1 is supported")
        try require(!f.tieWordEmbeddings, "this affine path requires an untied lm_head")
        try require(f.scoringFunc == nil || f.scoringFunc == "sigmoid", "scoring_func conflicts with sigmoid routing")
        try require(f.routedScalingFactor == nil || f.routedScalingFactor == 1, "Flint has no routed scaling factor")

        // Selected checkpoint uses default, non-traditional RoPE. Other scaling
        // families need their own reference/parity gate, never a silent fallback.
        var theta = f.ropeTheta
        for key in ["rope_scaling", "rope_parameters"] {
            if let value = raw[key], !(value is NSNull) {
                guard let rope = value as? [String: Any] else { throw FlintError.unsupported("invalid \(key)") }
                try require(["rope_type", "type"].allSatisfy { rope[$0] == nil || rope[$0] as? String == "default" }
                            && Set(rope.keys).isSubset(of: ["rope_type", "type", "rope_theta"]),
                            "only default RoPE is supported")
                if let value = rope["rope_theta"] {
                    guard let base = value as? NSNumber, CFGetTypeID(base) != CFBooleanGetTypeID() else {
                        throw FlintError.unsupported("\(key).rope_theta must be a number")
                    }
                    theta = base.floatValue
                }
            }
        }
        try require(theta.isFinite && theta > 0, "rope_theta must be positive and finite")

        try require(raw["weight_format"] == nil || ["affine", "mlx"].contains(raw["weight_format"] as? String ?? ""),
                    "only standard MLX affine weights are supported, not JANGH2/JANGTQ")
        for key in ["mxtq_bits", "mxtq_seed", "jangtq", "jangtq2", "tq2_packed", "tq2_scales"] {
            try require(raw[key] == nil, "unsupported custom pack metadata: \(key)")
        }
        // The general-purpose dependency tolerates custom pack policy metadata
        // and even falls back to affine for some missing-group-size overrides.
        // This lane must validate the raw declarations BEFORE that decoder so
        // unsupported packs can never masquerade as an ordinary affine plan.
        func validateQuantFields(_ fields: [String: Any]) throws {
            try require(Set(fields.keys).isSubset(of: ["bits", "group_size", "mode"]),
                        "unsupported affine quantization fields")
            try require(fields["mode"] == nil || fields["mode"] as? String == "affine",
                        "only affine quantization mode is supported")
        }
        func validateLayerDeclaration(_ value: Any) throws {
            if let fields = value as? [String: Any] { try validateQuantFields(fields) }
            else {
                try require((value as? NSNumber).map {
                    CFGetTypeID($0) == CFBooleanGetTypeID() && !$0.boolValue
                } == true, "module quantization must be an affine declaration or false")
            }
        }
        for key in ["quantization", "quantization_config"] {
            if let map = raw[key] as? [String: Any] {
                try validateQuantFields(map.filter { ["bits", "group_size", "mode"].contains($0.key) })
                for (name, value) in map where !["bits", "group_size", "mode"].contains(name) {
                    if name == "per_tensor", let entries = value as? [String: Any] {
                        for entry in entries.values { try validateLayerDeclaration(entry) }
                    } else { try validateLayerDeclaration(value) }
                }
            }
        }
        // BaseConfiguration rejects conflicting quantization/quantization_config
        // maps and retains all per-module overrides. Do not flatten them to 4-bit.
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        guard let quant = base.perLayerQuantization, let defaultQuant = quant.quantization else {
            throw FlintError.unsupported("requires standard MLX affine quantization metadata")
        }
        func validateQuant(_ q: BaseConfiguration.Quantization) throws {
            try require(q.mode == .affine && [3, 4, 8].contains(q.bits) && [32, 64, 128].contains(q.groupSize),
                        "affine quantization requires 3/4/8 bits and group_size 32/64/128")
        }
        try validateQuant(defaultQuant)
        for option in quant.perLayerQuantization.values {
            if case .quantize(let q) = option { try validateQuant(q) }
        }
        hiddenSize = f.hiddenSize
        intermediateSize = f.intermediateSize
        hiddenLayers = f.hiddenLayers
        attentionHeads = f.attentionHeads
        kvHeads = f.kvHeads
        headDim = f.headDim
        vocabularySize = f.vocabularySize
        maxPositionEmbeddings = f.maxPositionEmbeddings
        rmsNormEps = f.rmsNormEps
        ropeTheta = theta
        layerTypes = f.layerTypes
        noRoPELayers = Set(f.noRoPELayers)
        slidingWindow = f.useSlidingWindow ? f.slidingWindow : nil
        attentionBias = f.attentionBias
        mlpOnlyLayers = Set(f.mlpOnlyLayers)
        numExperts = f.numExperts
        topK = f.topK
        moeIntermediateSize = f.moeIntermediateSize
        sharedExpertIntermediateSize = f.sharedExpertIntermediateSize
        quantization = quant
    }

    func windowSize(layer: Int) -> Int? { layerTypes[layer] == "sliding_attention" ? slidingWindow : nil }
    func usesRoPE(layer: Int) -> Bool { !noRoPELayers.contains(layer) }

    /// Tensor-free production seam. Preserve unknown keys and colliding aliases
    /// so the dependency's strict .noUnusedKeys check can still refuse them.
    func sanitize<T>(_ weights: [String: T], zeroExpertBias: () -> T) -> [String: T] {
        var result = weights
        for key in weights.keys.sorted() {
            var target = key.hasPrefix("language_model.") ? String(key.dropFirst("language_model.".count)) : key
            for layer in 0..<hiddenLayers where !mlpOnlyLayers.contains(layer) {
                let prefix = "model.layers.\(layer).mlp."
                if target == prefix + "gate.e_score_correction_bias" || target == prefix + "gate.expert_bias" {
                    target = prefix + "expert_bias"
                    break
                }
            }
            if target != key, result[target] == nil { result[target] = result.removeValue(forKey: key) }
        }
        for layer in 0..<hiddenLayers where !mlpOnlyLayers.contains(layer) {
            let prefix = "model.layers.\(layer).mlp."
            if result[prefix + "gate.weight"] != nil, result[prefix + "expert_bias"] == nil {
                result[prefix + "expert_bias"] = zeroExpertBias()
            }
        }
        return result
    }

    private struct Fields: Decodable {
        let modelType: String
        let architectures: [String]?
        let hiddenSize: Int, intermediateSize: Int, hiddenLayers: Int
        let attentionHeads: Int, kvHeads: Int, headDim: Int, vocabularySize: Int
        let maxPositionEmbeddings: Int
        let rmsNormEps: Float, ropeTheta: Float
        let layerTypes: [String], noRoPELayers: [Int], mlpOnlyLayers: [Int]
        let useSlidingWindow: Bool, slidingWindow: Int?
        let attentionBias: Bool, attentionSink: Bool?
        let hiddenAct: String, routerScoreFunction: String, softmaxType: String
        let numExperts: Int, topK: Int, moeIntermediateSize: Int, sharedExpertIntermediateSize: Int
        let normTopKProb: Bool, decoderSparseStep: Int, tieWordEmbeddings: Bool
        let scoringFunc: String?, routedScalingFactor: Float?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type", architectures
            case hiddenSize = "hidden_size", intermediateSize = "intermediate_size", hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads", kvHeads = "num_key_value_heads", headDim = "head_dim"
            case vocabularySize = "vocab_size", rmsNormEps = "rms_norm_eps", ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case layerTypes = "layer_types", noRoPELayers = "no_rope_layer_indices", mlpOnlyLayers = "mlp_only_layers"
            case useSlidingWindow = "use_sliding_window", slidingWindow = "sliding_window"
            case attentionBias = "attention_bias", attentionSink = "attention_sink"
            case hiddenAct = "hidden_act", routerScoreFunction = "router_score_function", softmaxType = "softmax_type"
            case numExperts = "num_experts", topK = "num_experts_per_tok", moeIntermediateSize = "moe_intermediate_size"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size", normTopKProb = "norm_topk_prob"
            case decoderSparseStep = "decoder_sparse_step", tieWordEmbeddings = "tie_word_embeddings"
            case scoringFunc = "scoring_func", routedScalingFactor = "routed_scaling_factor"
        }
    }
}

/// Errors are raised before the loader touches weights; no alternative architecture is tried.
enum FlintError: LocalizedError {
    case disabled
    case unsupported(String)

    var errorDescription: String? {
        switch self {
        case .disabled: "Flint is experimental and disabled. Explicitly pass --enable-flint true (launcher: MEI_ENABLE_FLINT=true) to try the standard MLX affine path."
        case .unsupported(let detail): "Unsupported Flint bundle: \(detail). Only the pre-stacked standard MLX affine export is implemented; JANGH2/JANGTQ is not supported."
        }
    }
}
