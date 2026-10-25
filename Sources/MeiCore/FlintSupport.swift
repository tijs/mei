import Foundation
import MLXLMCommon
import MLXLLM

/// App-owned registration and early admission, before allocator limits,
/// tokenizer loading, weight IO or factory fallback. Each Engine.load must
/// pass this gate even if a previous enabled load registered the global type.
enum FlintSupport {
    /// Match the existing factory's JSON5 sniffing for every architecture.
    /// A default-off extension must not narrow non-Flint loader inputs.
    static func modelType(data: Data) throws -> String {
        struct TypeOnly: Decodable { let model_type: String }
        return try JSONDecoder.json5().decode(TypeOnly.self, from: data).model_type
    }

    static func prepare(
        directory: URL, enabled: Bool, janghEnabled: Bool = false, registry: ModelTypeRegistry = LLMTypeRegistry.shared
    ) async throws -> ModelConfiguration {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let type = try modelType(data: data)
        guard type == "flint" else { return ModelConfiguration(directory: directory) }
        guard enabled else { throw FlintError.disabled }
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let adapted = object?["mei_jangh_adapter_version"] != nil
        if adapted && !janghEnabled { throw FlintError.unsupported("JANGH requires --enable-flint-jangh true") }
        let jangh = adapted ? try FlintJANGHSupport.validate(directory: directory, ordinary: data) : nil
        _ = try FlintConfiguration(data: data)
        for name in jangConfigFileNames + ["jangtq_runtime.safetensors"] {
            if FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path) {
                throw FlintError.unsupported("custom runtime sidecar \(name) is not an affine checkpoint")
            }
        }
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: index.path) {
            struct Index: Decodable { let weight_map: [String: String] }
            let keys = try JSONDecoder().decode(Index.self, from: Data(contentsOf: index)).weight_map.keys
            if jangh == nil { try validateWeightKeys(Array(keys)) }
        }
        await registry.registerModelType("flint", creator: { (data: Data) in
            FlintModel(try FlintConfiguration(data: data), jangh: jangh)
        })
        // The pinned Evaluate.generateTask uses ReasoningParser.forPrompt,
        // not the stamp-only default: a template tail without <think> starts
        // in content, then self-emitted <think>...</think> is parsed normally.
        // No sampler, EOS, template or enable_thinking override is installed.
        return ModelConfiguration(directory: directory, toolCallFormat: .glm4, reasoningParserName: "think_xml")
    }

    static func validateWeightKeys(_ keys: [String]) throws {
        let packed = Set(["tq2_packed", "tq2_scales", "tq_packed", "tq_norms"])
        for key in keys {
            if packed.contains(key.split(separator: ".").last.map(String.init) ?? "") {
                throw FlintError.unsupported("custom packed tensor \(key)")
            }
            if key.contains(".mlp.experts.") {
                throw FlintError.unsupported("individual experts must be exported as pre-stacked mlp.switch_mlp tensors: \(key)")
            }
        }
        // All other keys survive sanitization and the loader's strict
        // .noUnusedKeys check. Never discard an unfamiliar key to force a load.
    }
}
