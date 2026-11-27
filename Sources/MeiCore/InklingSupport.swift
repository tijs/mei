import CryptoKit
import Foundation

/// First Inkling envelope: native text AR with visit-local companion state.
/// Unsupported cache/speculation/media paths fail before loading or sampling.
public enum InklingSupport {
    static let maximumContext = 1_048_576

    enum Profile {
        case original, publishedOriginal, vocabularyQ4

        func usesBoundedLocalCache(contextCap: Int) -> Bool {
            self == .vocabularyQ4 || contextCap > 8192
        }

        func defaultContext(physicalMemory: UInt64) -> Int {
            if physicalMemory >= 32 * 1024 * 1024 * 1024 {
                return ServerConfig.defaultContextCap
            }
            return 8192
        }
    }

    static var cpuBrand: String {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else { return "" }
        return bytes.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
    }

    static func greedyCompletionStep(profile: Profile, cpuBrand: String, memory: UInt64, cap: Int) -> Int? {
        supportsQualifiedQ4(profile: profile, cpuBrand: cpuBrand, memory: memory, cap: cap) ? 256 : nil
    }

    private static let indexedHistoryDisabled = ProcessInfo.processInfo.environment["VMLX_INKLING_INDEXED_KV"] == "0"

    static func useIndexedHistory(promptTokens: Int, outputBudget: Int) -> Bool {
        !indexedHistoryDisabled && promptTokens >= 7936 && outputBudget >= 2048
    }

    static func supportsQualifiedQ4(profile: Profile, cpuBrand: String, memory: UInt64, cap: Int) -> Bool {
        profile == .vocabularyQ4 && supportsMeasuredHardware(cpuBrand: cpuBrand, memory: memory, cap: cap)
    }

    private static func supportsMeasuredHardware(cpuBrand: String, memory: UInt64, cap: Int) -> Bool {
        cpuBrand == "Apple M1 Max" && memory >= 32 << 30 && (1...ServerConfig.defaultContextCap).contains(cap)
    }

    static func profile(directory: String) -> Profile? {
        struct Quant: Decodable {
            let bits: Int
            let group_size: Int
            let mode: String
            var matchesAffineQ4: Bool { bits == 4 && group_size == 64 && mode == "affine" }
        }
        struct Vocabulary: Decodable {
            let embedding: Quant?
            let head: Quant?
            enum CodingKeys: String, CodingKey {
                case embedding = "language_model.model.embed_tokens"
                case head = "language_model.lm_head"
            }
        }
        struct Architecture: Decodable { let model_type: String }
        struct Precision: Decodable {
            let quantization: Vocabulary?
            let mei_affine_vocabulary_range: [String: Int]?
        }
        let url = URL(fileURLWithPath: directory).appendingPathComponent("config.json")
        let decoder = JSONDecoder()
        decoder.allowsJSON5 = true
        guard let data = try? Data(contentsOf: url),
              let config = try? decoder.decode(Architecture.self, from: data),
              config.model_type == "inkling2_loop" else { return nil }
        // A malformed quant table still belongs to the Inkling loader and
        // its strict validation. Only a recognized pair selects this default.
        let precision = try? decoder.decode(Precision.self, from: data)
        if precision?.mei_affine_vocabulary_range == nil,
           precision?.quantization?.embedding?.matchesAffineQ4 == true,
           precision?.quantization?.head?.matchesAffineQ4 == true {
            return .vocabularyQ4
        }
        // The publisher build and unrecognized Inkling variants share the original
        // cache policy. Only the tested publisher configuration enables the mask.
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        if digest == "b7be690b1c064bb26aa13ce96dde6189aec8b9d434d20de89d115b0011d01487" {
            return .publishedOriginal
        }
        return .original
    }

    public static func isModel(directory: String) -> Bool {
        profile(directory: directory) != nil
    }

    static func validateLoad(_ config: ServerConfig) throws -> Bool {
        guard let profile = profile(directory: config.modelDirectory) else {
            if config.enableInklingMini { throw ConfigError.invalidValue("--enable-inkling-mini requires an inkling2_loop bundle") }
            return false
        }
        guard config.enableInklingMini else {
            throw ConfigError.invalidValue("Inkling Mini support is disabled")
        }
        guard (1...maximumContext).contains(config.contextCap), (1...128).contains(config.prefillStepSize),
              config.maxActiveGenerations == 1, !config.enableCompiledDecode,
              config.maxKVWindowSize == 0, config.kvBits == nil, config.modelTuning == nil,
              !config.enableFlint, !config.enableFlintJANGH, !config.enableSenseNovaLoop else {
            throw ConfigError.invalidValue("Inkling Mini requires context within 1...\(maximumContext), prefill <=128, one generation, native unquantized KV, no compile/window override or unrelated model tuning")
        }
        let alignedMemory = config.cacheMode == .memory
            && supportsQualifiedQ4(profile: profile, cpuBrand: cpuBrand,
                                   memory: ProcessInfo.processInfo.physicalMemory, cap: config.contextCap)
        guard config.cacheMode == .off || alignedMemory, config.kvCacheDir.isEmpty,
              !config.kvCacheDirExplicit, alignedMemory || config.cacheReuseRequested != true,
              config.ssmAnchorBoundaryCount == 0 else {
            throw ConfigError.invalidValue("Inkling supports cache mode off, or memory for qualified M1 Max vocabulary-Q4 models, with no cache directory or anchors")
        }
        return true
    }

    /// Apply before the CLI initializes Metal. These are defaults for the
    /// single-model process, with explicit operator environment taking precedence.
    public static func applyRuntimeEnvironment(modelDirectory: String? = nil, contextCap: Int = ServerConfig.defaultContextCap) {
        if let modelDirectory, let profile = profile(directory: modelDirectory) {
            if supportsMeasuredHardware(cpuBrand: cpuBrand, memory: ProcessInfo.processInfo.physicalMemory,
                                        cap: contextCap) {
                if profile == .publishedOriginal || profile == .vocabularyQ4 {
                    setenv("VMLX_INKLING_FUSED_MASK", "1", 0)
                }
                if profile == .vocabularyQ4 {
                    setenv("VMLX_INKLING_ATTENTION_BITS", "8", 0)
                    setenv("VMLX_INKLING_DEPTH_MIX", "step256", 0)
                }
            }
            // The released original profile was scored using full K/V storage at 8K.
            setenv("VMLX_INKLING_BOUNDED_LOCAL_KV", profile.usesBoundedLocalCache(contextCap: contextCap) ? "1" : "0", 0)
        }
        let defaults = [
            "VMLX_INKLING_FUSED_SHORT_CONV": "1",
            "VMLX_INKLING_SORT_EXPERT_PREFILL": "1",
            "VMLX_INKLING_COMPILE_PRIMITIVES": "1",
            "VMLX_INKLING_MIXED_HEAD": "1",
            "MLX_MAX_MB_PER_BUFFER": "1024",
            "MLX_MAX_OPS_PER_BUFFER": "500",
            "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS": "500",
        ]
        for (name, value) in defaults { setenv(name, value, 0) }
    }

    public enum InputError: LocalizedError, Sendable {
        case textOnly, structuredUnsupported
        public var errorDescription: String? {
            switch self {
            case .textOnly: "Inkling Mini currently supports text input only"
            case .structuredUnsupported: "Inkling Mini structured response constraints require a channel-aware grammar and are not yet supported"
            }
        }
    }
    public static func validateInput(_ request: ChatRequest, enabled: Bool) throws {
        guard enabled else { return }
        if request.containsUnsupportedContent { throw InputError.textOnly }
        if request.responseFormat != .text { throw InputError.structuredUnsupported }
    }

}
