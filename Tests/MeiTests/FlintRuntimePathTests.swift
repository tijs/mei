import Foundation
import CryptoKit
import MLX
import MLXNN
import MLXLMCommon
import MLXHuggingFace
import VMLXTokenizers
import XCTest
@testable import MeiCore

private final class ForcedLogits {
    let tokens: [Int]
    var rows: [[Float]] = []
    init(_ tokens: [Int]) { self.tokens = tokens }
}

private struct CaptureAndForce: LogitProcessor {
    let record: ForcedLogits
    mutating func prompt(_ prompt: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray {
        let index = record.rows.count
        record.rows.append(logits.reshaped(-1).asArray(Float.self))
        var mask = [Float](repeating: -.infinity, count: logits.size)
        mask[record.tokens[min(index, record.tokens.count - 1)]] = 0
        return MLXArray(mask).reshaped(logits.shape)
    }
    mutating func didSample(token: MLXArray) {}
}

private final class RestoredCount: @unchecked Sendable {
    let lock = NSLock()
    var value = 0
    func observe(_ p: PrefillProgress) {
        lock.lock(); defer { lock.unlock() }
        if p.stage == .cacheRestore { value = p.completedUnitCount }
    }
}

final class FlintRuntimePathTests: XCTestCase {
    @MainActor
    func testTeacherForcedRuntimePaths() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["FLINT_PATH_MODEL"], let requestPath = env["FLINT_PATH_REQUESTS"],
              let outputPath = env["FLINT_PATH_OUTPUT"], let textPath = env["FLINT_PATH_FORCED_TEXT"] else {
            throw XCTSkip("Explicit actual-pack model, requests and output paths required")
        }
        let dir = URL(fileURLWithPath: modelPath)
        let requests = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: requestPath))) as! [String: Any]
        let requestData = try JSONSerialization.data(withJSONObject: requests["capture-multipart"]!)
        let request = try ChatRequest(json: requestData)
        let loader = #huggingFaceTokenizerLoader()
        let tokenizer = try await loader.load(from: dir)
        let tools = MessageMapping.templateTools(request.tools)
        let ids = try tokenizer.applyChatTemplate(
            messages: request.messages.map { MessageMapping.templateDictionary(from: $0) },
            tools: tools, additionalContext: nil)
        XCTAssertEqual(ids.count, 1255, "Must use the native failing HTTP request render")
        let forcedText = try String(contentsOfFile: textPath, encoding: .utf8)
        let forced = Array(tokenizer.encode(text: "<think>" + forcedText, addSpecialTokens: false).prefix(64))
        let ordinary = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let contract = try FlintJANGHSupport.validate(directory: dir, ordinary: ordinary)
        let config = try FlintConfiguration(data: ordinary)
        Memory.cacheLimit = 536870912
        let model = FlintModel(config, jangh: contract)
        try loadWeights(modelDirectory: dir, model: model, perLayerQuantization: config.quantization)
        var parameters = GenerateParameters(maxTokens: forced.count, maxKVSize: 65536, temperature: 1, topP: 0.95, topK: 20, randomSeed: 173, prefillStepSize: 64)
        parameters.enableCompiledDecode = false
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true, enableDiskCache: false, pagedBlockSize: 16,
            maxCacheBlocks: 8192, maxCacheBytes: 536870912, modelKey: "flint-path-diagnostic"))
        if let gp = tokenizer as? any GenerationPromptControllableTokenizer {
            let dummy: [[String: any Sendable]] = [["role": "user", "content": "x"]]
            let withGen = try gp.applyChatTemplate(messages: dummy, tools: nil, additionalContext: nil, addGenerationPrompt: true)
            let withoutGen = try gp.applyChatTemplate(messages: dummy, tools: nil, additionalContext: nil, addGenerationPrompt: false)
            var common = 0
            while common < min(withGen.count, withoutGen.count), withGen[common] == withoutGen[common] { common += 1 }
            coordinator.setGenPromptSuffixTokens(Array(withGen[common...]))
        }
        let output = URL(fileURLWithPath: outputPath)
        try JSONSerialization.data(withJSONObject: ["tokens": ids, "forced_tokens": forced], options: [.prettyPrinted]).write(to: output.appendingPathExtension("input.json"))
        var matrices: [String: [[Float]]] = [:]
        var restores: [String: Int] = [:]
        func run(_ name: String, coord: CacheCoordinator?, split: Bool = false) throws {
            let record = ForcedLogits(forced)
            let cache = model.newCache(parameters: parameters)
            var inputIDs = ids
            if split {
                let head = LMInput(tokens: MLXArray(ids.dropLast().map(Int32.init)).reshaped(1, -1))
                switch try model.prepare(head, cache: cache, windowSize: 64) {
                case .tokens(let rest): eval(model(rest.tokens.reshaped(1, -1), cache: cache))
                case .logits: XCTFail("Unexpected prepared logits")
                }
                inputIDs = [ids.last!]
            }
            let progress = RestoredCount()
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(inputIDs.map(Int32.init)).reshaped(1, -1), tokenIds: inputIDs),
                model: model, cache: cache, parameters: parameters, cacheCoordinator: coord,
                prefillProgressHandler: { progress.observe($0) }, additionalProcessor: CaptureAndForce(record: record))
            var generated: [Int] = []
            while let token = iterator.next() { generated.append(token) }
            XCTAssertEqual(generated, forced)
            matrices[name] = Array(record.rows.prefix(forced.count))
            restores[name] = progress.value
            if coord != nil { iterator.storeCacheAfterGeneration(generatedTokenIds: generated, includeGeneratedBoundary: false) }
            print("flint-path: \(name) rows=\(record.rows.count) restored=\(progress.value)")
        }
        try run("cache_off", coord: nil)
        try run("cold_coordinator", coord: coordinator)
        try run("warm_restore", coord: coordinator)
        try run("cache_off_matching_split", coord: nil, split: true)
        XCTAssertGreaterThan(restores["warm_restore"]!, 500)
        // Full causal forward supplies all teacher-forced next-token rows in one
        // pass. Project only the compared positions to avoid a huge vocab tensor.
        let all = ids + forced.dropLast()
        var h = model.model.embedTokens(MLXArray(all.map(Int32.init)).reshaped(1, -1))
        for (i, layer) in model.model.layers.enumerated() {
            h = layer(h, mask: createAttentionMask(h: h, cache: nil, windowSize: config.windowSize(layer: i)), cache: nil)
            eval(h)
        }
        let logits = model.lmHead(model.model.norm(h)[0, (ids.count - 1)...].asType(.float32))
        matrices["full_forward"] = (0..<forced.count).map { logits[$0].asArray(Float.self) }
        func digest(_ values: [Float]) -> String {
            values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
        }
        var comparisons: [String: Any] = [:]
        for (a, b) in [("cache_off", "cold_coordinator"), ("cold_coordinator", "warm_restore"), ("cold_coordinator", "cache_off_matching_split"), ("full_forward", "cache_off"), ("full_forward", "cold_coordinator")] {
            var steps: [[String: Any]] = []
            for (index, pair) in zip(matrices[a]!, matrices[b]!).enumerated() {
                let (x, y) = pair
                let maxDiff = zip(x, y).map { abs($0 - $1) }.max()!
                let topX = x.indices.max { x[$0] < x[$1] }!
                let topY = y.indices.max { y[$0] < y[$1] }!
                steps.append(["step": index, "max_abs": Double(maxDiff), "top_a": topX, "top_b": topY, "finite": y.allSatisfy(\.isFinite), "exact": x == y])
            }
            comparisons[a + "__" + b] = steps
        }
        let hashes = matrices.mapValues { $0.map(digest) }
        if let rawPath = env["FLINT_PATH_RAW_DIR"] {
            let rawDir = URL(fileURLWithPath: rawPath)
            try FileManager.default.createDirectory(at: rawDir, withIntermediateDirectories: true)
            for (name, rows) in matrices {
                var bytes = Data()
                for row in rows { row.withUnsafeBytes { bytes.append(contentsOf: $0) } }
                try bytes.write(to: rawDir.appendingPathComponent(name + ".f32"))
            }
        }
        let report: [String: Any] = ["prompt_tokens": ids.count, "teacher_tokens": forced.count, "restored": restores, "comparisons": comparisons, "row_sha256": hashes]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: output)
        XCTAssertTrue(matrices.values.flatMap { $0 }.allSatisfy { $0.allSatisfy(\.isFinite) })
    }
}
