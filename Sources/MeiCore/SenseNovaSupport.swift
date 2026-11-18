import Foundation

/// Explicit admission for the first experimental text-only Loop envelope.
/// Numerical/quality evidence and remaining gates live in Kiem proj/mei.
public enum SenseNovaSupport {
    public static let releaseRevision = "329a65f701ba8c48b73c30d0cbaa6f0449733673"

    private struct Architecture: Decodable {
        let model_type: String
    }

    private struct Export: Decodable {
        struct Quantization: Decodable {
            let bits: Int
            let group_size: Int
        }

        let source_revision: String
        let mei_text_only_export: Bool
        let activation_dtype: String
        let quantization: Quantization
    }

    public static func isModel(directory: String) -> Bool {
        guard let data = metadata(directory) else { return false }
        return isModel(data)
    }

    private static func metadata(_ directory: String) -> Data? {
        let url = URL(fileURLWithPath: directory).appendingPathComponent("config.json")
        return try? Data(contentsOf: url)
    }

    private static func isModel(_ data: Data) -> Bool {
        (try? decode(Architecture.self, from: data))?.model_type == "sensenova_loop"
    }

    private static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let decoder = JSONDecoder()
        decoder.allowsJSON5 = true
        return try decoder.decode(type, from: data)
    }

    static func validateLoad(_ config: ServerConfig) throws -> Bool {
        guard let data = metadata(config.modelDirectory), isModel(data) else {
            if config.enableSenseNovaLoop || config.optimizationProfile == .senseNovaLoop
                || config.requestedOptimizationProfile == .senseNovaLoop {
                throw ConfigError.invalidValue("--enable-sensenova-loop requires a sensenova_loop text export")
            }
            return false
        }
        guard config.enableSenseNovaLoop else {
            throw ConfigError.invalidValue("SenseNova Loop is experimental; pass --enable-sensenova-loop true")
        }
        guard let export = try? decode(Export.self, from: data),
              export.source_revision == releaseRevision,
              export.mei_text_only_export,
              export.activation_dtype == "float32",
              [6, 8].contains(export.quantization.bits),
              export.quantization.group_size == 64 else {
            throw ConfigError.invalidValue("SenseNova experiment requires the pinned release text export with 6-bit or 8-bit group-64 weights and activation_dtype=float32")
        }
        guard config.contextCap > 0, config.contextCap <= 4096,
              config.prefillStepSize > 0, config.prefillStepSize <= 64,
              config.maxActiveGenerations == 1,
              !config.enableCompiledDecode, config.maxKVWindowSize == 0,
              !config.enableFlint, !config.enableFlintJANGH, config.modelTuning == nil else {
            throw ConfigError.invalidValue("SenseNova experiment supports at most 4K total context, prefill chunks <=64, one active generation, uncompiled decode and no sliding KV window")
        }
        guard config.kvBits == nil else {
            throw ConfigError.invalidValue("SenseNova experiment requires unquantized attention KV")
        }
        return true
    }

    public enum InputError: LocalizedError, Sendable {
        case textOnly
        public var errorDescription: String? {
            "SenseNova Loop currently supports text content only; image, video and other content parts are unsupported"
        }
    }

    public static func validateInput(_ request: ChatRequest, textOnly: Bool) throws {
        if textOnly, request.containsUnsupportedContent { throw InputError.textOnly }
    }

    static func validateCapacity(promptTokens: Int, config: ServerConfig) throws {
        if config.isSenseNovaLoop, promptTokens >= config.contextCap {
            throw EngineError.overContextCap(promptTokens: promptTokens, cap: config.contextCap - 1)
        }
    }
}
