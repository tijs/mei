import Foundation

/// Narrow JANGH v2 contract for Flint, independent from historical JANGTQ.
/// Semantics follow the MIT reference pinned in THIRD-PARTY-NOTICES.md.
struct JANGHFormatContract: Sendable {
  enum ValidationError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? {
      if case .invalid(let text) = self { return "Flint JANGH: " + text }
      return nil
    }
  }
  enum Rotation: String, Sendable { case hadamard32 }
  struct Codebook: Sendable {
    let alpha: Double
    let beta: Double
    let levels: [Double]
  }
  struct Projection: Sendable {
    let bits: Int
    let rotation: Rotation
  }
  let codebooks: [Int: Codebook]
  let projections: [String: Projection]
  let ordinaryConfiguration: Data

  static func require(_ ok: Bool, _ message: String) throws {
    if !ok { throw ValidationError.invalid(message) }
  }
  init(configuration data: Data) throws {
    guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let q = root["quantization"] as? [String: Any],
      let other = root["quantization_config"] as? [String: Any],
      let h = root["jangtq"] as? [String: Any],
      let rawBooks = h["codebooks"] as? [String: [String: Any]],
      let layers = root["num_hidden_layers"] as? Int,
      let dense = root["mlp_only_layers"] as? [Int]
    else { throw ValidationError.invalid("missing contract or quantization maps") }
    try Self.require(
      root["model_type"] as? String == "flint" && NSDictionary(dictionary: q).isEqual(to: other),
      "conflicting quantization maps or model type")
    try Self.require(
      h["version"] as? Int == 2 && h["packing"] as? String == "lsb-bitstream"
        && h["scale_dtype"] as? String == "float16"
        && h["codebook_family"] as? String == "odd-cubic"
        && h["rotation"] as? String == "hadamard32", "unsupported format/packing/rotation")
    try Self.require(
      q["bits"] as? Int == 8 && q["group_size"] as? Int == 64 && q["mode"] as? String == "affine",
      "ordinary modules require affine8/group64")
    var books: [Int: Codebook] = [:]
    for bits in [2, 3, 4] {
      guard let raw = rawBooks[String(bits)], let alpha = raw["alpha"] as? Double,
        let beta = raw["beta"] as? Double, let levels = raw["levels"] as? [Double]
      else { throw ValidationError.invalid("missing codebook \(bits)") }
      try Self.require(
        alpha.isFinite && beta.isFinite && Float(alpha).isFinite && Float(beta).isFinite
          && levels.count == 1 << bits, "invalid codebook coefficients")
      for (i, level) in levels.enumerated() {
        let u = Double(i) - Double((1 << bits) - 1) / 2
        let expected = u * (alpha + beta * u * u)
        try Self.require(
          level.isFinite && Float(level).isFinite
            && abs(level - expected) <= max(1, abs(expected)) * 2e-6
            && (i == 0 || level > levels[i - 1]), "codebook levels disagree with coefficients")
      }
      books[bits] = Codebook(alpha: alpha, beta: beta, levels: levels)
    }
    try Self.require(
      layers > 0 && layers < 4096 && dense.allSatisfy { (0..<layers).contains($0) },
      "invalid sparse layer schedule")
    var expected = Set<String>()
    for layer in 0..<layers where !dense.contains(layer) {
      for role in ["gate_proj", "up_proj", "down_proj"] {
        expected.insert("model.layers.\(layer).mlp.switch_mlp.\(role)")
      }
    }
    var modules: [String: Projection] = [:]
    var ordinary = q
    for (name, raw) in q where !["bits", "group_size", "mode"].contains(name) {
      guard let value = raw as? [String: Any] else {
        throw ValidationError.invalid("invalid module declaration \(name)")
      }
      if expected.contains(name) {
        guard let bits = value["bits"] as? Int, [2, 3, 4].contains(bits),
          value["mode"] as? String == "jangtq2",
          value["rotation"] as? String == "hadamard32",
          Set(value.keys) == ["mode", "bits", "rotation"]
        else { throw ValidationError.invalid("invalid projection \(name)") }
        modules[name] = Projection(bits: bits, rotation: .hadamard32)
        ordinary[name] = false
      } else {
        try Self.require(
          name == "model.embed_tokens" && value["mode"] as? String == "affine"
            && value["bits"] as? Int == 8 && value["group_size"] as? Int == 64
            && Set(value.keys) == ["mode", "bits", "group_size"],
          "unexpected ordinary override \(name)")
      }
    }
    try Self.require(
      Set(modules.keys) == expected && !modules.isEmpty, "incomplete routed projection map")
    root.removeValue(forKey: "jangtq")
    root["quantization"] = ordinary
    root["quantization_config"] = ordinary
    root["mei_jangh_adapter_version"] = 1
    ordinaryConfiguration = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    // Architecture must pass the unchanged affine lane's geometry/attention checks.
    _ = try FlintConfiguration(data: ordinaryConfiguration)
    codebooks = books
    projections = modules
  }
}
