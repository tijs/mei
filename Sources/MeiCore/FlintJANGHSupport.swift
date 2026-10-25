import CryptoKit
import Foundation
import MLX
import MLXLMCommon

/// Pre-weight admission for the explicit metadata adapter. Does not reinterpret
/// tq2 buffers as affine weights and never fabricates a legacy runtime sidecar.
enum FlintJANGHSupport {
  static func validate(directory: URL, ordinary: Data) throws -> JANGHFormatContract {
    let raw = try Data(contentsOf: directory.appendingPathComponent("mei_jangh_config.json"))
    let contract = try JANGHFormatContract(configuration: raw)
    guard let a = try JSONSerialization.jsonObject(with: ordinary) as? [String: Any],
      let b = try JSONSerialization.jsonObject(with: contract.ordinaryConfiguration)
        as? [String: Any]
    else {
      throw JANGHFormatContract.ValidationError.invalid(
        "adapter configuration must be a JSON object")
    }
    try JANGHFormatContract.require(
      NSDictionary(dictionary: a).isEqual(to: b),
      "adapter metadata differs from complete original contract")
    let manifest =
      try JSONSerialization.jsonObject(
        with: Data(contentsOf: directory.appendingPathComponent("mei_jangh_manifest.json")))
      as? [String: Any]
    let digest = SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
    try JANGHFormatContract.require(
      manifest?["version"] as? Int == 1 && manifest?["config_sha256"] as? String == digest
        && manifest?["tensor_payloads_unchanged"] as? Bool == true, "adapter provenance mismatch")
    let bundleData = try Data(contentsOf: directory.appendingPathComponent("mei_jangh_bundle.json"))
    let bundle = try JSONSerialization.jsonObject(with: bundleData) as? [String: Any]
    let bundleDigest = SHA256.hash(data: bundleData).map { String(format: "%02x", $0) }.joined()
    try JANGHFormatContract.require(
      bundle?["format"] as? String == "jangtq2" && bundle?["format_version"] as? Int == 2
        && bundle?["profile"] as? String == "JANGH2"
        && manifest?["bundle_sha256"] as? String == bundleDigest,
      "bundle sidecar version/profile/digest mismatch")
    for filename in jangConfigFileNames + [
      "jangtq_runtime.safetensors", "jangtq_stacked.safetensors",
    ] {
      try JANGHFormatContract.require(
        !FileManager.default.fileExists(atPath: directory.appendingPathComponent(filename).path),
        "legacy/custom executable sidecar exposed to ordinary loader")
    }
    try validateHeaders(directory: directory, contract: contract)
    return contract
  }

  static func validateHeaders(directory: URL, contract: JANGHFormatContract) throws {
    struct Index: Decodable { let weight_map: [String: String] }
    let index = try JSONDecoder().decode(
      Index.self,
      from: Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
    ).weight_map
    let c = try FlintConfiguration(data: contract.ordinaryConfiguration)
    func packedWords(_ input: Int, _ bits: Int) throws -> Int {
      let product = input.multipliedReportingOverflow(by: bits)
      try JANGHFormatContract.require(
        !product.overflow && input % 32 == 0,
        "packed input geometry overflow or alignment mismatch")
      return product.partialValue / 32
    }
    // noUnusedKeys alone does not reject missing ordinary parameters.
    // Enforce the entire architecture key/shape/dtype set before weight IO.
    var expectedHeaders: [String: (shape: [Int], dtype: String)] = [:]
    func add(_ key: String, _ shape: [Int], _ dtype: String) {
      expectedHeaders[key] = (shape, dtype)
    }
    func affine(_ name: String, _ output: Int, _ input: Int) throws {
      try JANGHFormatContract.require(input % 64 == 0, "affine group geometry mismatch")
      add(name + ".weight", [output, input / 4], "U32")
      add(name + ".scales", [output, input / 64], "BF16")
      add(name + ".biases", [output, input / 64], "BF16")
    }
    try affine("lm_head", c.vocabularySize, c.hiddenSize)
    try affine("model.embed_tokens", c.vocabularySize, c.hiddenSize)
    add("model.norm.weight", [c.hiddenSize], "BF16")
    for layer in 0..<c.hiddenLayers {
      let p = "model.layers.\(layer)"
      for norm in ["input_layernorm", "post_attention_layernorm"] {
        add(p + "." + norm + ".weight", [c.hiddenSize], "BF16")
      }
      for (role, output, input) in [
        ("q_proj", c.attentionHeads * c.headDim, c.hiddenSize),
        ("k_proj", c.kvHeads * c.headDim, c.hiddenSize),
        ("v_proj", c.kvHeads * c.headDim, c.hiddenSize),
        ("o_proj", c.hiddenSize, c.attentionHeads * c.headDim),
      ] { try affine(p + ".self_attn." + role, output, input) }
      for norm in ["q_norm", "k_norm"] {
        add(p + ".self_attn." + norm + ".weight", [c.headDim], "BF16")
      }
      add(p + ".self_attn.softmax_offset", [c.attentionHeads], "BF16")
      let dense = c.mlpOnlyLayers.contains(layer)
      let middle = dense ? c.intermediateSize : c.sharedExpertIntermediateSize
      let parent = p + (dense ? ".mlp" : ".mlp.shared_expert")
      for (role, output, input) in [
        ("gate_proj", middle, c.hiddenSize), ("up_proj", middle, c.hiddenSize),
        ("down_proj", c.hiddenSize, middle),
      ] { try affine(parent + "." + role, output, input) }
      if !dense {
        add(p + ".mlp.gate.weight", [c.numExperts, c.hiddenSize], "BF16")
        for role in ["gate_proj", "up_proj", "down_proj"] {
          let name = p + ".mlp.switch_mlp." + role
          let bits = contract.projections[name]!.bits
          let output = role == "down_proj" ? c.hiddenSize : c.moeIntermediateSize
          let input = role == "down_proj" ? c.moeIntermediateSize : c.hiddenSize
          add(name + ".tq2_packed", [c.numExperts, output, try packedWords(input, bits)], "U32")
          add(name + ".tq2_scales", [c.numExperts, output], "F16")
        }
      }
    }
    try JANGHFormatContract.require(
      Set(index.keys) == Set(expectedHeaders.keys), "complete architecture key set mismatch")
    var seen = Set<String>()
    var packedSeen = Set<String>()
    for file in Set(index.values) {
      try JANGHFormatContract.require(
        URL(fileURLWithPath: file).lastPathComponent == file && file.hasSuffix(".safetensors"),
        "unsafe shard path")
      let url = directory.appendingPathComponent(file)
      let handle = try FileHandle(forReadingFrom: url)
      defer { try? handle.close() }
      let first = try handle.read(upToCount: 8) ?? Data()
      try JANGHFormatContract.require(first.count == 8, "truncated header")
      let rawLength = first.withUnsafeBytes {
        UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
      }
      try JANGHFormatContract.require(
        rawLength > 0 && rawLength < 4_000_000, "invalid header length")
      let length = Int(rawLength)
      let data = try handle.read(upToCount: length) ?? Data()
      try JANGHFormatContract.require(data.count == length, "truncated header")
      guard let header = try JSONSerialization.jsonObject(with: data) as? [String: Any]
      else {
        throw JANGHFormatContract.ValidationError.invalid(
          "safetensors header must be a JSON object")
      }
      let targetSize = try handle.seekToEnd()
      try JANGHFormatContract.require(targetSize <= UInt64(Int.max), "shard size overflow")
      let size = Int(targetSize)
      var spans: [(Int, Int)] = []
      for (key, raw) in header where key != "__metadata__" {
        guard let v = raw as? [String: Any], let shape = v["shape"] as? [Int],
          let dtype = v["dtype"] as? String,
          let offsets = v["data_offsets"] as? [Int], offsets.count == 2,
          let width = ["U32": 4, "F16": 2, "BF16": 2, "F32": 4][dtype]
        else { throw JANGHFormatContract.ValidationError.invalid("invalid tensor header \(key)") }
        try JANGHFormatContract.require(
          index[key] == file && seen.insert(key).inserted && shape.allSatisfy { $0 > 0 },
          "duplicate/unindexed tensor")
        var count = 1
        for dim in shape {
          let product = count.multipliedReportingOverflow(by: dim)
          try JANGHFormatContract.require(!product.overflow, "tensor shape overflow")
          count = product.partialValue
        }
        let bytes = count.multipliedReportingOverflow(by: width)
        try JANGHFormatContract.require(
          !bytes.overflow && offsets[0] >= 0 && offsets[1] >= offsets[0]
            && offsets[1] <= size - 8 - length && offsets[1] - offsets[0] == bytes.partialValue
            && (8 + length + offsets[0]) % width == 0, "tensor span/alignment mismatch \(key)")
        let expectedHeader = expectedHeaders[key]!
        try JANGHFormatContract.require(
          shape == expectedHeader.shape && dtype == expectedHeader.dtype,
          "ordinary/packed tensor shape or dtype mismatch \(key)")
        spans.append((offsets[0], offsets[1]))
        if key.hasSuffix(".tq2_packed") || key.hasSuffix(".tq2_scales") {
          let scale = key.hasSuffix(".tq2_scales")
          let module = String(key.dropLast(scale ? ".tq2_scales".count : ".tq2_packed".count))
          guard let projection = contract.projections[module] else {
            throw JANGHFormatContract.ValidationError.invalid("unexpected packed key \(key)")
          }
          let down = module.hasSuffix("down_proj")
          let output = down ? c.hiddenSize : c.moeIntermediateSize
          let input = down ? c.moeIntermediateSize : c.hiddenSize
          let expected =
            scale
            ? [c.numExperts, output]
            : [c.numExperts, output, try packedWords(input, projection.bits)]
          try JANGHFormatContract.require(
            shape == expected && dtype == (scale ? "F16" : "U32") && input % 32 == 0,
            "packed projection geometry/dtype mismatch \(key)")
          if scale {
            try handle.seek(toOffset: UInt64(8 + length + offsets[0]))
            let values = try handle.read(upToCount: bytes.partialValue) ?? Data()
            try JANGHFormatContract.require(
              values.count == bytes.partialValue, "truncated row scales")
            let valid = values.withUnsafeBytes { buffer in
              for i in stride(from: 0, to: buffer.count, by: 2) {
                let bits = UInt16(
                  littleEndian: buffer.loadUnaligned(fromByteOffset: i, as: UInt16.self))
                let value = Float16(bitPattern: bits)
                if !value.isFinite || value < 0 { return false }
              }
              return true
            }
            try JANGHFormatContract.require(valid, "nonfinite/negative row scales \(key)")
          }
          packedSeen.insert(key)
        } else {
          try JANGHFormatContract.require(
            !key.contains(".tq_") && !key.contains(".mlp.experts."), "legacy/unstacked packed key")
        }
      }
      spans.sort { $0.0 < $1.0 }
      for pair in zip(spans, spans.dropFirst()) {
        try JANGHFormatContract.require(pair.0.1 <= pair.1.0, "overlapping tensors")
      }
    }
    let expected = Set(
      contract.projections.keys.flatMap { [$0 + ".tq2_packed", $0 + ".tq2_scales"] })
    try JANGHFormatContract.require(
      seen == Set(index.keys) && packedSeen == expected, "missing indexed/packed tensors")
    // The generic loader subsequently enforces the complete ordinary model
    // key set and geometry through its unchanged strict .noUnusedKeys update.
  }
}
