// Flint architecture adapted from Blaizzy/mlx-vlm v0.7.10 (MIT), commit
// 8c061ef516e7f8b584c0d5b64f26b50d5806303b, mlx_vlm/models/flint/language.py.
// Copyright © 2025 Prince Canuma. Full notice: THIRD-PARTY-NOTICES.md.
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM

/// Keep Flint routing separate from other MoE families: no softmax, group
/// selection, routed scaling or shared-expert gate belongs in this graph.
enum FlintRouting {
    static func route(logits: MLXArray, expertBias: MLXArray, topK: Int) -> (MLXArray, MLXArray) {
        // Projection has ALREADY happened in activation dtype. Only sigmoid,
        // selection bias and renormalization are promoted to FP32.
        let scores = sigmoid(logits.asType(.float32))
        let choice = scores + expertBias.asType(.float32)
        let start = choice.dim(-1) - topK
        let indices = stopGradient(argPartition(choice, kth: start, axis: -1)[.ellipsis, start...])
        let unbiased = takeAlong(scores, indices, axis: -1)
        let weights = unbiased / (unbiased.sum(axis: -1, keepDims: true) + Float(1e-20))
        return (indices, weights)
    }

    static func combine(routed: MLXArray, weights: MLXArray, shared: MLXArray?, dtype: DType) -> MLXArray {
        let reduced = (routed.asType(.float32) * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(dtype)
        return shared.map { reduced + $0 } ?? reduced
    }
}

final class FlintDenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

final class FlintSparseMLP: Module, UnaryLayer {
    let topK: Int
    let gate: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: FlintDenseMLP?

    init(_ c: FlintConfiguration) {
        topK = c.topK
        gate = Linear(c.hiddenSize, c.numExperts, bias: false)
        _expertBias.wrappedValue = zeros([c.numExperts])
        // Existing public switch projections preserve checkpoint module names
        // and per-module 3-bit affine loading. Do not duplicate the expert bank
        // into a fused weight cache just to implement the reference SwiGLU.
        _switchMLP.wrappedValue = SwitchGLU(inputDims: c.hiddenSize, hiddenDims: c.moeIntermediateSize,
                                           numExperts: c.numExperts, allowFusedGateUpCache: false)
        _sharedExpert.wrappedValue = c.sharedExpertIntermediateSize > 0
            ? FlintDenseMLP(hiddenSize: c.hiddenSize, intermediateSize: c.sharedExpertIntermediateSize) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = FlintRouting.route(logits: gate(x), expertBias: expertBias, topK: topK)
        return FlintRouting.combine(routed: switchMLP(x, indices), weights: weights,
                                    shared: sharedExpert?(x), dtype: x.dtype)
    }
}

final class FlintAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int
    let scale: Float
    let rope: RoPELayer?
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var softmaxOffset: MLXArray

    init(_ c: FlintConfiguration, layer: Int) {
        heads = c.attentionHeads; kvHeads = c.kvHeads; headDim = c.headDim
        scale = pow(Float(c.headDim), -0.5)
        _qProj.wrappedValue = Linear(c.hiddenSize, c.attentionHeads * c.headDim, bias: c.attentionBias)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: c.attentionBias)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: c.attentionBias)
        _oProj.wrappedValue = Linear(c.attentionHeads * c.headDim, c.hiddenSize, bias: c.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _softmaxOffset.wrappedValue = zeros([c.attentionHeads])
        rope = c.usesRoPE(layer: layer)
            ? initializeRope(dims: c.headDim, base: c.ropeTheta, traditional: false,
                             scalingConfig: nil, maxPositionEmbeddings: c.maxPositionEmbeddings) : nil
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(batch, length, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(batch, length, kvHeads, headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        if let rope {
            q = applyRotaryPosition(rope, to: q, cache: cache)
            k = applyRotaryPosition(rope, to: k, cache: cache)
        }
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale,
                                                   mask: mask, sinks: softmaxOffset.asType(q.dtype))
        return oProj(out.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class FlintDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: FlintAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ c: FlintConfiguration, layer: Int) {
        _attention.wrappedValue = FlintAttention(c, layer: layer)
        mlp = c.mlpOnlyLayers.contains(layer)
            ? FlintDenseMLP(hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize) : FlintSparseMLP(c)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

final class FlintModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [FlintDecoderLayer]
    let norm: RMSNorm

    init(_ c: FlintConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = (0..<c.hiddenLayers).map { FlintDecoderLayer(c, layer: $0) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
}

/// Registered by Mei, not by a changed dependency. No downloaded model is
/// constructed by the model-free metadata/gate tests.
final class FlintModel: Module, LLMModel {
    let configuration: FlintConfiguration
    let model: FlintModelInner
    let vocabularySize: Int
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    // Match reference dtype boundaries, including the explicit FP32 output
    // projection; do not let the loader blanket-cast checkpoint floats to BF16.
    var preservesCheckpointParameterDTypes: Bool { true }
    var loraLayers: [Module] { model.layers }

    init(_ c: FlintConfiguration) {
        configuration = c
        vocabularySize = c.vocabularySize
        model = FlintModelInner(c)
        _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = model.embedTokens(inputs)
        for (i, layer) in model.layers.enumerated() {
            let c = cache?[i]
            let mask = createAttentionMask(h: h, cache: c, windowSize: configuration.windowSize(layer: i))
            h = layer(h, mask: mask, cache: c)
        }
        return lmHead(model.norm(h).asType(.float32))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Full attention must remain full even when Mei passes maxKVSize.
        // Sliding caches have zero retained prefix tokens, NOT generic keep=4.
        (0..<configuration.hiddenLayers).map {
            if let window = configuration.windowSize(layer: $0) {
                RotatingKVCache(maxSize: window, keep: 0)
            } else { KVCacheSimple() }
        }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        configuration.sanitize(weights, zeroExpertBias: { zeros([configuration.numExperts]) })
    }
}
