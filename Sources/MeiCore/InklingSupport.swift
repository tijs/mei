import Foundation

/// First Inkling envelope: native text AR with visit-local companion state.
/// Unsupported cache/speculation/media paths fail before loading or sampling.
public enum InklingSupport {
    public static func isModel(directory: String) -> Bool {
        struct Architecture: Decodable { let model_type: String }
        let decoder = JSONDecoder(); decoder.allowsJSON5 = true
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: directory).appendingPathComponent("config.json")),
              let object = try? decoder.decode(Architecture.self, from: data) else { return false }
        return object.model_type == "inkling2_loop"
    }

    static func validateLoad(_ config: ServerConfig) throws -> Bool {
        guard isModel(directory: config.modelDirectory) else {
            if config.enableInklingMini { throw ConfigError.invalidValue("--enable-inkling-mini requires an inkling2_loop bundle") }
            return false
        }
        guard config.enableInklingMini else {
            throw ConfigError.invalidValue("Inkling Mini support is disabled")
        }
        guard (1...8192).contains(config.contextCap), (1...128).contains(config.prefillStepSize),
              config.maxActiveGenerations == 1, !config.enableCompiledDecode,
              config.maxKVWindowSize == 0, config.kvBits == nil, config.modelTuning == nil,
              !config.enableFlint, !config.enableFlintJANGH, !config.enableSenseNovaLoop else {
            throw ConfigError.invalidValue("Inkling Mini requires <=8K context, prefill <=128, one generation, native unquantized KV, no compile/window override or unrelated model tuning")
        }
        guard config.cacheMode == .off, config.kvCacheDir.isEmpty,
              !config.kvCacheDirExplicit, config.cacheReuseRequested != true,
              config.ssmAnchorBoundaryCount == 0 else {
            throw ConfigError.invalidValue("Inkling visit/convolution cache persistence is unqualified; use --cache-mode off with no cache directory/anchors")
        }
        return true
    }

    /// Apply before the CLI initializes Metal. These are defaults for the
    /// single-model process, with explicit operator environment taking precedence.
    public static func applyRuntimeEnvironment() {
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
    static func validateCapacity(promptTokens: Int, config: ServerConfig) throws {
        if config.isInklingMini, promptTokens >= config.contextCap {
            throw EngineError.overContextCap(promptTokens: promptTokens, cap: config.contextCap - 1)
        }
    }
}
