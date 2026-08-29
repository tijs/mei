import Foundation

/// Immutable server configuration, resolved from CLI flags with the same
/// shape the local-model-bench start scripts use for the other engines.
public struct ServerConfig: Sendable {
    public var modelDirectory: String
    public var servedModelID: String
    public var host: String = "127.0.0.1"
    public var port: Int = 8024
    public var contextCap: Int = 65_536
    public var maxTokensDefault: Int = 32_768
    public var prefillStepSize: Int = 512
    /// KV cache quantization: nil = fp16, else bits (4 or 8).
    public var kvBits: Int? = nil
    public var kvGroupSize: Int = 64
    public var quantizedKVStart: Int = 0
    public var temperature: Float = 0.6
    public var topP: Float = 0.95
    public var topK: Int = 20
    public var minP: Float = 0.0
    public var repetitionPenalty: Float? = nil
    public var presencePenalty: Float? = nil
    public var frequencyPenalty: Float? = nil
    /// Emit `reasoning_content` deltas/fields for reasoning models (Ornith is
    /// Qwen3.5-lineage and thinks by default). Hermes ignores unknown fields,
    /// so exposing thought text is safe and useful for transcripts.
    public var emitReasoning: Bool = true
    /// nil = template default (Qwen3.5 templates default to thinking ON).
    public var enableThinking: Bool? = nil
    public var reasoningEffort: String? = nil
    public var maxCacheSlotTokens: Int = 131_072
    public var cacheReuse: Bool = true
    public var logRequests: Bool = false

    public init(modelDirectory: String, servedModelID: String) {
        self.modelDirectory = modelDirectory
        self.servedModelID = servedModelID
    }

    /// KV capacity bound: the cache must hold the full context plus a
    /// generation headroom window.
    public var maxKVSize: Int { contextCap + 4096 }
}

public enum ConfigError: LocalizedError, CustomStringConvertible {
    case missingRequired(String)
    case invalidValue(String)

    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .missingRequired(let field):
            "missing required flag: \(field)"
        case .invalidValue(let message):
            "invalid flag value: \(message)"
        }
    }
}

public extension ServerConfig {
    static func parse(arguments: [String] = Array(CommandLine.arguments.dropFirst())) throws -> ServerConfig {
        var modelDirectory: String?
        var servedModelID: String?
        var config = ServerConfig(modelDirectory: "", servedModelID: "")

        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            func value() throws -> String {
                index += 1
                guard index < arguments.count else {
                    throw ConfigError.invalidValue("\(flag) requires a value")
                }
                return arguments[index]
            }
            switch flag {
            case "--model-dir": modelDirectory = try value()
            case "--served-model-id": servedModelID = try value()
            case "--host": config.host = try value()
            case "--port":
                config.port = try parseInt(flag, value())
            case "--context-cap":
                config.contextCap = try parseInt(flag, value())
            case "--max-tokens":
                config.maxTokensDefault = try parseInt(flag, value())
            case "--prefill-step-size":
                config.prefillStepSize = try parseInt(flag, value())
            case "--kv-bits":
                config.kvBits = try parseInt(flag, value())
            case "--kv-group-size":
                config.kvGroupSize = try parseInt(flag, value())
            case "--quantized-kv-start":
                config.quantizedKVStart = try parseInt(flag, value())
            case "--temperature":
                config.temperature = try parseFloat(flag, value())
            case "--top-p":
                config.topP = try parseFloat(flag, value())
            case "--top-k":
                config.topK = try parseInt(flag, value())
            case "--min-p":
                config.minP = try parseFloat(flag, value())
            case "--repetition-penalty":
                config.repetitionPenalty = try parseFloat(flag, value())
            case "--presence-penalty":
                config.presencePenalty = try parseFloat(flag, value())
            case "--frequency-penalty":
                config.frequencyPenalty = try parseFloat(flag, value())
            case "--emit-reasoning":
                config.emitReasoning = try parseBool(flag, value())
            case "--enable-thinking":
                config.enableThinking = try parseBool(flag, value())
            case "--reasoning-effort":
                config.reasoningEffort = try value()
            case "--max-cache-slot-tokens":
                config.maxCacheSlotTokens = try parseInt(flag, value())
            case "--cache-reuse":
                config.cacheReuse = try parseBool(flag, value())
            case "--log-requests":
                config.logRequests = try parseBool(flag, value())
            case "-h", "--help":
                print(usage)
                exit(0)
            default:
                throw ConfigError.invalidValue("unknown option \(flag)")
            }
            index += 1
        }

        guard let modelDirectory else {
            throw ConfigError.missingRequired("--model-dir")
        }
        guard let servedModelID else {
            throw ConfigError.missingRequired("--served-model-id")
        }
        config.modelDirectory = modelDirectory
        config.servedModelID = servedModelID

        guard config.prefillStepSize > 0 else {
            throw ConfigError.invalidValue("--prefill-step-size must be > 0")
        }
        guard config.contextCap > 0 else {
            throw ConfigError.invalidValue("--context-cap must be > 0")
        }
        return config
    }

    static let usage = """
    mei — native Swift/MLX OpenAI-compatible inference server

    Usage: mei --model-dir DIR --served-model-id ID [options]

    Required:
      --model-dir DIR        Local directory with model config + safetensors
      --served-model-id ID   Exact model ID served by GET /v1/models

    Network:
      --host HOST            (default 127.0.0.1)
      --port PORT            (default 8024)

    Context / capacity:
      --context-cap TOKENS   Reject prompts beyond this many tokens (default 65536)
      --max-tokens TOKENS    Server-side per-request generation cap (default 32768)
      --prefill-step-size N  Chunked prefill window (default 512; the key
                             long-context safeguard for hybrid architectures)
      --kv-bits N            Quantized KV cache bits (4 or 8; nil = fp16)
      --kv-group-size N      KV quantization group size (default 64)
      --quantized-kv-start N First layer index to quantize (default 0)

    Sampling defaults (per-request override wins):
      --temperature F --top-p F --top-k N --min-p F
      --repetition-penalty F --presence-penalty F --frequency-penalty F

    Reasoning:
      --emit-reasoning BOOL     Expose reasoning_content (default true)
      --enable-thinking BOOL    Force template enable_thinking (nil = template default)
      --reasoning-effort LEVEL  low|medium|high (nil = template default)

    Caching:
      --cache-reuse BOOL        In-process KV/prefix reuse across turns (default true)
      --max-cache-slot-tokens N Reset the reused slot above this size (default 131072)

    Misc:
      --log-requests BOOL  Log each request's token counts (default false)
      -h, --help
    """

    private static func parseInt(_ flag: String, _ raw: String) throws -> Int {
        guard let value = Int(raw) else {
            throw ConfigError.invalidValue("\(flag) expects an integer, got '\(raw)'")
        }
        return value
    }

    private static func parseFloat(_ flag: String, _ raw: String) throws -> Float {
        guard let value = Float(raw) else {
            throw ConfigError.invalidValue("\(flag) expects a number, got '\(raw)'")
        }
        return value
    }

    private static func parseBool(_ flag: String, _ raw: String) throws -> Bool {
        switch raw.lowercased() {
        case "1", "true", "yes", "on": return true
        case "0", "false", "no", "off": return false
        default:
            throw ConfigError.invalidValue("\(flag) expects true/false, got '\(raw)'")
        }
    }
}