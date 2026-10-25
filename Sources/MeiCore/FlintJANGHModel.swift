import MLX
import MLXNN

final class FlintJANGHProjection: Module {
  @ParameterInfo(key: "tq2_packed") var packed: MLXArray
  @ParameterInfo(key: "tq2_scales") var scales: MLXArray
  private let kernel: JANGHProjectionKernel
  init(contract: JANGHFormatContract, module: String) {
    // All metadata and real headers were admitted before model creation.
    kernel = try! JANGHProjectionKernel(contract: contract, module: module)
    _packed.wrappedValue = zeros([1, 1, 1], type: UInt32.self)
    _scales.wrappedValue = zeros([1, 1], type: Float16.self)
  }
  func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
    // Shapes and immutable parameter dtypes have already passed strict load.
    try! kernel.project(x, packed: packed, scales: scales, indices: indices.asType(.uint32))
  }
}
final class FlintJANGHBank: Module {
  @ModuleInfo(key: "gate_proj") var gateProj: FlintJANGHProjection
  @ModuleInfo(key: "up_proj") var upProj: FlintJANGHProjection
  @ModuleInfo(key: "down_proj") var downProj: FlintJANGHProjection
  init(contract: JANGHFormatContract, parent: String) {
    _gateProj.wrappedValue = FlintJANGHProjection(contract: contract, module: parent + ".gate_proj")
    _upProj.wrappedValue = FlintJANGHProjection(contract: contract, module: parent + ".up_proj")
    _downProj.wrappedValue = FlintJANGHProjection(contract: contract, module: parent + ".down_proj")
  }
  func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
    let flat = x.asType(.float32).reshaped([-1, x.dim(-1)])
    let routes = indices.size / flat.dim(0)
    // Publisher decode rotates/accumulates gate/up and SwiGLU in FP32.
    // Retain FP32 through down projection and routed reduction; the graph
    // casts the completed routed sum back to the BF16 residual dtype.
    let gate = gateProj(flat, indices)
    let up = upProj(flat, indices)
    let hidden = silu(gate) * up
    let down = downProj(hidden, indices)
    return down.reshaped(Array(x.shape.dropLast()) + [routes, down.dim(-1)])
  }
}
final class FlintJANGHSparseMLP: Module, UnaryLayer {
  let topK: Int
  let gate: Linear
  @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
  @ModuleInfo(key: "switch_mlp") var bank: FlintJANGHBank
  @ModuleInfo(key: "shared_expert") var shared: FlintDenseMLP?
  init(_ c: FlintConfiguration, layer: Int, contract: JANGHFormatContract) {
    topK = c.topK
    gate = Linear(c.hiddenSize, c.numExperts, bias: false)
    _expertBias.wrappedValue = zeros([c.numExperts])
    _bank.wrappedValue = FlintJANGHBank(
      contract: contract, parent: "model.layers.\(layer).mlp.switch_mlp")
    _shared.wrappedValue =
      c.sharedExpertIntermediateSize > 0
      ? FlintDenseMLP(hiddenSize: c.hiddenSize, intermediateSize: c.sharedExpertIntermediateSize)
      : nil
  }
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (indices, weights) = FlintRouting.route(logits: gate(x), expertBias: expertBias, topK: topK)
    return FlintRouting.combine(
      routed: bank(x, indices), weights: weights, shared: shared?(x), dtype: x.dtype)
  }
}
