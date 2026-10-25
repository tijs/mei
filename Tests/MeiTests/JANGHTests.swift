import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MeiCore

final class JANGHContractTests: XCTestCase {
  func testExplicitGateDefaultsOff() throws {
    let args = ["--model-dir", "/not-a-model", "--served-model-id", "test"]
    XCTAssertFalse(
      try ServerConfig.parse(arguments: args, recommendedWorkingSetBytes: nil).enableFlintJANGH)
    XCTAssertTrue(
      try ServerConfig.parse(
        arguments: args + ["--enable-flint", "true", "--enable-flint-jangh", "true"],
        recommendedWorkingSetBytes: nil
      ).enableFlintJANGH)
    XCTAssertThrowsError(
      try ServerConfig.parse(
        arguments: args + ["--enable-flint-jangh", "maybe"], recommendedWorkingSetBytes: nil))
  }

  func testOddCubicReferenceAndMalformedContract() throws {
    let raw = try fixture()
    let contract = try JANGHFormatContract(configuration: raw)
    XCTAssertEqual(contract.projections.count, 3)
    XCTAssertEqual(contract.projections["model.layers.2.mlp.switch_mlp.down_proj"]?.bits, 3)
    var root = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    var header = root["jangtq"] as! [String: Any]
    header["packing"] = "msb-bitstream"
    root["jangtq"] = header
    XCTAssertThrowsError(
      try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: root)))
  }

  func testProjectionMapMustBeCompleteAndAgree() throws {
    var root = try JSONSerialization.jsonObject(with: fixture()) as! [String: Any]
    var q = root["quantization"] as! [String: Any]
    q.removeValue(forKey: "model.layers.2.mlp.switch_mlp.up_proj")
    root["quantization"] = q
    XCTAssertThrowsError(
      try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: root)))
    root["quantization_config"] = q
    XCTAssertThrowsError(
      try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: root)))
  }

  func fixture() throws -> Data {
    var root = FlintFixture.metadata(tiny: true)
    var books: [String: Any] = [:]
    for (bits, a, b) in [(2, 0.893, 0.05065), (3, 0.47125, 0.0116), (4, 0.2405, 0.0021)] {
      let center = Double((1 << bits) - 1) / 2
      books[String(bits)] = [
        "alpha": a, "beta": b,
        "levels": (0..<(1 << bits)).map { i in
          let u = Double(i) - center
          return Double(Float(u * (a + b * u * u)))
        },
      ]
    }
    root["jangtq"] = [
      "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
      "codebook_family": "odd-cubic", "rotation": "hadamard32", "codebooks": books,
    ]
    var q: [String: Any] = ["bits": 8, "group_size": 64, "mode": "affine"]
    for role in ["gate_proj", "up_proj", "down_proj"] {
      q["model.layers.2.mlp.switch_mlp.\(role)"] = [
        "mode": "jangtq2", "bits": role == "down_proj" ? 3 : 2, "rotation": "hadamard32",
      ]
    }
    root["quantization"] = q
    root["quantization_config"] = q
    return try JSONSerialization.data(withJSONObject: root)
  }
}

final class JANGHNumericalTests: XCTestCase {
  func testIndependentOddCubicProjectionAllWidthsAndCrossWordPacking() throws {
    let raw = try JANGHContractTests().fixture()
    let width = 64
    let outputs = 8
    let experts = 8
    for bits in [2, 3, 4] {
      var root = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
      var q = root["quantization"] as! [String: Any]
      let module = "model.layers.2.mlp.switch_mlp.gate_proj"
      q[module] = ["bits": bits, "mode": "jangtq2", "rotation": "hadamard32"]
      root["quantization"] = q
      root["quantization_config"] = q
      let contract = try JANGHFormatContract(
        configuration: JSONSerialization.data(withJSONObject: root))
      let kernel = try JANGHProjectionKernel(contract: contract, module: module)
      let count = experts * outputs * width
      let codes = (0..<count).map { UInt32(($0 * 13 + $0 / width * 7) % (1 << bits)) }
      var words = [UInt32](repeating: 0, count: count * bits / 32)
      // Independent, intentionally slow bit writer: no shared word-unpack math.
      for (index, code) in codes.enumerated() {
        for bit in 0..<bits where (code & (1 << bit)) != 0 {
          let offset = index * bits + bit
          words[offset / 32] |= 1 << (offset % 32)
        }
      }
      let scales = (0..<experts * outputs).map { Float16(Float(($0 % 5) + 1) / 37) }
      let x = (0..<2 * width).map { Float(($0 * 7) % 17 - 8) / 9 }
      let routes = [7, 1, 3, 0]
      let result = try kernel.project(
        MLXArray(x, [2, width]), packed: MLXArray(words, [experts, outputs, width * bits / 32]),
        scales: MLXArray(scales, [experts, outputs]), indices: MLXArray(routes.map(UInt32.init))
      ).asArray(Float.self)
      let book = contract.codebooks[bits]!
      for dispatch in 0..<routes.count {
        let input = Array(x[(dispatch / 2 * width)..<(dispatch / 2 * width + width)])
        let rotated = hadamard(input)
        for row in 0..<outputs {
          let expert = routes[dispatch]
          var expected = 0.0
          for column in 0..<width {
            let code = Int(codes[(expert * outputs + row) * width + column])
            expected += Double(rotated[column]) * book.levels[code]
          }
          expected *= Double(scales[expert * outputs + row])
          XCTAssertEqual(
            result[dispatch * outputs + row], Float(expected), accuracy: 0.00005,
            "bits=\(bits) expert=\(expert) row=\(row)")
        }
      }
      let first = try kernel.hadamard32(MLXArray(x, [2, width]))
      let roundTrip = try kernel.hadamard32(first).asArray(Float.self)
      for (a, b) in zip(x, roundTrip) { XCTAssertEqual(a, b, accuracy: 0.000002) }
      XCTAssertThrowsError(
        try kernel.project(
          MLXArray(x, [2, width]), packed: MLXArray(words, [experts, outputs, width * bits / 32]),
          scales: MLXArray(scales, [experts, outputs]).asType(.bfloat16),
          indices: MLXArray(routes.map(UInt32.init))))
    }
  }
  func testTwoBitMidLevelsUseExactPublishedFloatBits() throws {
    let contract = try JANGHFormatContract(configuration: JANGHContractTests().fixture())
    let kernel = try JANGHProjectionKernel(
      contract: contract, module: "model.layers.2.mlp.switch_mlp.gate_proj")
    let x = MLXArray([Float](repeating: 0.25, count: 32), [1, 32])
    let rotated = try kernel.hadamard32(x).asArray(Float.self)
    for (code, word) in [(1, UInt32(0x5555_5555)), (2, UInt32(0xaaaa_aaaa))] {
      let actual = try kernel.project(
        x, packed: MLXArray([word, word], [1, 1, 2]), scales: MLXArray([Float16(1)], [1, 1]),
        indices: MLXArray([UInt32(0)])
      ).item(Float.self)
      let expected = rotated[0] * Float(contract.codebooks[2]!.levels[code])
      XCTAssertEqual(
        actual.bitPattern, expected.bitPattern,
        "2-bit level \(code) must retain the published F32 bits")
    }
  }
  private func hadamard(_ values: [Float]) -> [Float] {
    var out = values.map(Double.init)
    for base in stride(from: 0, to: values.count, by: 32) {
      var step = 1
      while step < 32 {
        for start in stride(from: base, to: base + 32, by: 2 * step) {
          for i in 0..<step {
            let a = out[start + i]
            let b = out[start + i + step]
            out[start + i] = a + b
            out[start + i + step] = a - b
          }
        }
        step *= 2
      }
      for i in base..<base + 32 { out[i] /= sqrt(32) }
    }
    return out.map(Float.init)
  }
}

final class JANGHAdmissionTests: XCTestCase {
  func testHugePackedGeometryRefusesBeforeWeightIO() throws {
    for field in ["hidden_size", "moe_intermediate_size"] {
      var object =
        try JSONSerialization.jsonObject(with: JANGHContractTests().fixture()) as! [String: Any]
      object[field] = Int.max - 63
      let raw = try JSONSerialization.data(withJSONObject: object)
      let contract = try JANGHFormatContract(configuration: raw)
      let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: dir) }
      try Data("{\"weight_map\":{}}".utf8).write(
        to: dir.appendingPathComponent("model.safetensors.index.json"))
      XCTAssertThrowsError(
        try FlintJANGHSupport.validateHeaders(directory: dir, contract: contract)
      ) {
        XCTAssertTrue($0.localizedDescription.contains("overflow"))
      }
    }
  }

  func testAdapterRequiresSeparateFlagBeforeReadingWeights() async throws {
    let raw = try JANGHContractTests().fixture()
    let contract = try JANGHFormatContract(configuration: raw)
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try contract.ordinaryConfiguration.write(to: dir.appendingPathComponent("config.json"))
    do {
      _ = try await FlintSupport.prepare(directory: dir, enabled: true)
      XCTFail("must reject separately disabled JANGH")
    } catch { XCTAssertTrue(error.localizedDescription.contains("--enable-flint-jangh")) }
    do {
      _ = try await FlintSupport.prepare(directory: dir, enabled: false, janghEnabled: true)
      XCTFail("outer Flint gate remains required")
    } catch { XCTAssertTrue(error.localizedDescription.contains("--enable-flint true")) }
  }
  func testAdapterCannotHideAlteredOrdinaryConfigurationOrMissingProvenance() throws {
    let raw = try JANGHContractTests().fixture()
    let contract = try JANGHFormatContract(configuration: raw)
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try raw.write(to: dir.appendingPathComponent("mei_jangh_config.json"))
    XCTAssertThrowsError(try FlintJANGHSupport.validate(directory: dir, ordinary: Data("[]".utf8)))
    var ordinary =
      try JSONSerialization.jsonObject(with: contract.ordinaryConfiguration) as! [String: Any]
    ordinary["hidden_size"] = 128
    XCTAssertThrowsError(
      try FlintJANGHSupport.validate(
        directory: dir, ordinary: JSONSerialization.data(withJSONObject: ordinary)))
    XCTAssertThrowsError(
      try FlintJANGHSupport.validate(directory: dir, ordinary: contract.ordinaryConfiguration))
  }
  func testBadCodebookWidthUnknownRotationAndExtremeCoefficientAreRejected() throws {
    let raw = try JANGHContractTests().fixture()
    for (key, value) in [("bits", 5 as Any), ("rotation", "none"), ("mode", "jangtq")] {
      var root = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
      var q = root["quantization"] as! [String: Any]
      var entry: [String: Any] = ["bits": 2, "mode": "jangtq2", "rotation": "hadamard32"]
      entry[key] = value
      q["model.layers.2.mlp.switch_mlp.gate_proj"] = entry
      root["quantization"] = q
      root["quantization_config"] = q
      XCTAssertThrowsError(
        try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: root)))
    }
    var root = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    var h = root["jangtq"] as! [String: Any]
    var books = h["codebooks"] as! [String: [String: Any]]
    books["2"]?["alpha"] = Double.greatestFiniteMagnitude
    h["codebooks"] = books
    root["jangtq"] = h
    XCTAssertThrowsError(
      try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: root)))
  }
}

final class JANGHLoaderTests: XCTestCase {
  func testRealStrictLoaderConsumesMixedBanksAndRejectsMissingOrdinaryKeys() throws {
    let raw = try JANGHContractTests().fixture()
    let contract = try JANGHFormatContract(configuration: raw)
    let c = try FlintConfiguration(data: contract.ordinaryConfiguration)
    let original = FlintModel(c, jangh: contract)
    var tensors = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
    for (key, value) in tensors where value.dtype != .uint32 {
      tensors[key] = value.asType(.bfloat16)
    }
    for (name, projection) in contract.projections {
      let input = name.hasSuffix("down_proj") ? c.moeIntermediateSize : c.hiddenSize
      let output = name.hasSuffix("down_proj") ? c.hiddenSize : c.moeIntermediateSize
      tensors[name + ".tq2_packed"] = zeros(
        [c.numExperts, output, input * projection.bits / 32], type: UInt32.self)
      tensors[name + ".tq2_scales"] = MLXArray.full(
        [c.numExperts, output], values: MLXArray(Float16(0.005)))
    }
    for (path, module) in original.leafModules().flattened() {
      guard (module is MLXNN.Linear && !path.hasSuffix(".mlp.gate")) || module is MLXNN.Embedding
      else { continue }
      let weight = try XCTUnwrap(tensors[path + ".weight"])
      let (packed, scales, biases) = MLX.quantized(weight, groupSize: 64, bits: 8)
      tensors[path + ".weight"] = packed
      tensors[path + ".scales"] = scales
      tensors[path + ".biases"] = biases
    }
    tensors.removeValue(forKey: "model.layers.2.mlp.expert_bias")
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try contract.ordinaryConfiguration.write(to: dir.appendingPathComponent("config.json"))
    try MLX.save(
      arrays: tensors, metadata: ["format": "mlx"],
      url: dir.appendingPathComponent("model.safetensors"))
    // MLX.save may emit an unpadded JSON header. Keep this generated
    // fixture naturally aligned, just like the immutable published pack.
    let savedURL = dir.appendingPathComponent("model.safetensors")
    let saved = try Data(contentsOf: savedURL)
    let oldLength = saved.withUnsafeBytes {
      Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)))
    }
    let padding = (8 - (8 + oldLength) % 8) % 8
    if padding > 0 {
      var length = UInt64(oldLength + padding).littleEndian
      var aligned = withUnsafeBytes(of: &length) { Data($0) }
      aligned.append(saved.subdata(in: 8..<8 + oldLength))
      aligned.append(Data(repeating: 32, count: padding))
      aligned.append(saved.subdata(in: 8 + oldLength..<saved.count))
      try aligned.write(to: savedURL)
    }
    func writeIndex(_ keys: [String]) throws {
      let map = Dictionary(uniqueKeysWithValues: keys.map { ($0, "model.safetensors") })
      try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(
        to: dir.appendingPathComponent("model.safetensors.index.json"))
    }
    try writeIndex(Array(tensors.keys))
    try FlintJANGHSupport.validateHeaders(directory: dir, contract: contract)
    // Real adapters link preserved shards. Their size is the opened file's
    // size, never the byte length of the symlink destination string.
    let linked = dir.appendingPathComponent("model.safetensors")
    let backing = dir.appendingPathComponent("source.payload")
    try FileManager.default.moveItem(at: linked, to: backing)
    try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: backing)
    XCTAssertNoThrow(try FlintJANGHSupport.validateHeaders(directory: dir, contract: contract))
    let file = dir.appendingPathComponent("model.safetensors")
    let validFile = try Data(contentsOf: file)
    let hlen = validFile.withUnsafeBytes {
      Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)))
    }
    // A syntactically valid non-object header must refuse, not trap.
    var arrayLength = UInt64(2).littleEndian
    var arrayHeader = withUnsafeBytes(of: &arrayLength) { Data($0) }
    arrayHeader.append(Data("[]".utf8))
    try arrayHeader.write(to: file)
    XCTAssertThrowsError(try FlintJANGHSupport.validateHeaders(directory: dir, contract: contract))
    try validFile.write(to: file)
    let header =
      try JSONSerialization.jsonObject(with: validFile.subdata(in: 8..<8 + hlen)) as! [String: Any]
    let scaleHeader = header["model.layers.2.mlp.switch_mlp.gate_proj.tq2_scales"] as! [String: Any]
    let offsets = scaleHeader["data_offsets"] as! [Int]
    var corrupt = validFile
    corrupt[8 + hlen + offsets[0]] = 0
    corrupt[8 + hlen + offsets[0] + 1] = 0x7e
    try corrupt.write(to: file)
    XCTAssertThrowsError(try FlintJANGHSupport.validateHeaders(directory: dir, contract: contract))
    try validFile.write(to: file)
    let loaded = FlintModel(c, jangh: contract)
    try loadWeights(modelDirectory: dir, model: loaded, perLayerQuantization: c.quantization)
    let actual = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
    for (key, value) in tensors {
      let got = try XCTUnwrap(actual[key], key)
      XCTAssertEqual(got.shape, value.shape, key)
      XCTAssertEqual(got.dtype, value.dtype, key)
      XCTAssertTrue((got .== value).all().item(Bool.self), key)
    }
    let logits = loaded(
      MLXArray([Int32(1), 2]).reshaped(1, 2), cache: loaded.newCache(parameters: nil))
    XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
    var keys = Array(tensors.keys)
    keys.removeAll { $0 == "model.norm.weight" }
    try writeIndex(keys)
    XCTAssertThrowsError(try FlintJANGHSupport.validateHeaders(directory: dir, contract: contract))
    tensors["unapproved.weight"] = MLXArray([Float(1)])
    try MLX.save(
      arrays: tensors, metadata: ["format": "mlx"],
      url: dir.appendingPathComponent("model.safetensors"))
    XCTAssertThrowsError(
      try loadWeights(
        modelDirectory: dir, model: FlintModel(c, jangh: contract),
        perLayerQuantization: c.quantization))
  }
}

/// Opt-in actual-pack oracle; ordinary test runs never load a staged model.
final class JANGHRealPackParityTests: XCTestCase {
  func testExactPackedArtifactMatchesIndependentFullForward() throws {
    let env = ProcessInfo.processInfo.environment
    guard let path = env["MEI_JANGH_PARITY_MODEL"], let oraclePath = env["MEI_JANGH_PARITY_ORACLE"]
    else { throw XCTSkip("Actual packed-model oracle requires explicit model/oracle paths") }
    let dir = URL(fileURLWithPath: path)
    let ordinary = try Data(contentsOf: dir.appendingPathComponent("config.json"))
    let contract = try FlintJANGHSupport.validate(directory: dir, ordinary: ordinary)
    let config = try FlintConfiguration(data: ordinary)
    let oracle =
      try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: oraclePath)))
      as! [String: Any]
    let ids = oracle["tokens"] as! [Int]
    let expected = (oracle["logits"] as! [Double]).map(Float.init)
    Memory.cacheLimit = 512 * 1024 * 1024
    let model = FlintModel(config, jangh: contract)
    try loadWeights(modelDirectory: dir, model: model, perLayerQuantization: config.quantization)
    let start = Date()
    let tokens = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
    let output: MLXArray
    if let tracePath = env["MEI_JANGH_PARITY_TRACE"] {
      var h = model.model.embedTokens(tokens)
      var trace: [[String: Any]] = []
      func capture(_ layer: Int, _ value: MLXArray) {
        trace.append([
          "layer": layer, "dtype": String(describing: value.dtype),
          "values": value.asType(.float32).asArray(Float.self).map(Double.init),
        ])
      }
      capture(-1, h)
      let cache: [KVCache]? = nil
      for (index, layer) in model.model.layers.enumerated() {
        let mask = createAttentionMask(
          h: h, cache: cache?[index], windowSize: config.windowSize(layer: index))
        h = layer(h, mask: mask, cache: cache?[index])
        capture(index, h)
      }
      output = model.lmHead(model.model.norm(h).asType(.float32))
      try JSONSerialization.data(withJSONObject: trace).write(to: URL(fileURLWithPath: tracePath))
    } else {
      output = model(tokens, cache: nil)
    }
    let actual = output[0, ids.count - 1].asArray(Float.self)
    XCTAssertEqual(actual.count, expected.count)
    let maxDiff = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
    let actualTop = actual.indices.max { actual[$0] < actual[$1] }!
    let expectedTop = expected.indices.max { expected[$0] < expected[$1] }!
    let report: [String: Any] = [
      "max_absolute_logit_difference": Double(maxDiff), "actual_top_token": actualTop,
      "reference_top_token": expectedTop, "logit_count": actual.count,
      "forward_seconds": Date().timeIntervalSince(start), "tolerance": 0.05,
      "finite": actual.allSatisfy(\.isFinite),
    ]
    if let destination = env["MEI_JANGH_PARITY_REPORT"] {
      try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: destination))
    }
    XCTAssertTrue(actual.allSatisfy(\.isFinite))
    XCTAssertEqual(actualTop, expectedTop)
    XCTAssertLessThanOrEqual(maxDiff, 0.05)
  }
}

final class JANGHMemoryPolicyTests: XCTestCase {
  func testExplicitBufferCacheLimitAppliesWithoutOverridingAutomaticMemoryLimit() {
    let previousMemory = Memory.memoryLimit
    let previousCache = Memory.cacheLimit
    defer {
      Memory.memoryLimit = previousMemory
      Memory.cacheLimit = previousCache
    }
    var config = ServerConfig(modelDirectory: "/unused", servedModelID: "test")
    config.cacheLimitBytes = 512 * 1024 * 1024
    XCTAssertEqual(config.memoryLimitBytes, 0)
    Memory.cacheLimit = 1024 * 1024 * 1024
    Engine.applyExplicitMemoryLimits(config)
    XCTAssertEqual(Memory.cacheLimit, config.cacheLimitBytes)
    XCTAssertEqual(Memory.memoryLimit, previousMemory)
  }
}
