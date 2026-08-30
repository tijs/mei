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
    /// Explicit MLX allocator limits in bytes (0 = MLX default: 1.5x the
    /// Metal recommended working set). A default limit below the model's
    /// working set makes MLX malloc calls WAIT on scheduled tasks — the
    /// hang failure mode — so benchmark configs set these explicitly.
    public var memoryLimitBytes: Int = 0
    public var cacheLimitBytes: Int = 0
    /// Directory for the coordinator's on-disk L2 KV tier (hybrid models
    /// like Ornith's mamba/GatedDelta layers are disk-backed-restore only,
    /// so in-process prefix reuse for them needs this tier; paged-in-memory
    /// alone covers non-hybrid topologies). "" = disabled.
    public var kvCacheDir: String = ""
    /// After each chat generation the coordinator runs one extra prompt-only
    /// pass to re-derive the hybrid SSM boundary state for reuse (upstream
    /// default on; costs ~1x prefill at turn end). Off when a benchmark row
    /// wants to A/B the fallback full-prefill path.
    public var enableSSMReDerive: Bool = true
    /// vmlx-swift compiled (graph-traced + replayed) decode. Trace once,
    /// replay per token; an evidence-based decode lever for long-context
    /// workloads. Ornith/qwen3_5 is not on the upstream deny list.
    public var enableCompiledDecode: Bool = false
    /// Skip the compiled promote+trace setup when the prefill offset already
    /// exceeds this many tokens (nil = no threshold; the upstream default
    /// traces a buffer sized promptOffset + maxTokens, a multi-minute
    /// prefill tax at 45K on hybrid models). 0 = never compile.
    public var compiledDecodeMaxPromptOffset: Int? = nil
    /// EXPERIMENTAL bounded-window probe: caps the rotating-KV ring size;
    /// decode attention scans at most the ring (sink + recent window).
    /// Correctness-bounded only (a full-attention model loses context
    /// beyond the window). 0/nil = ring sized to maxKVSize (default).
    public var maxKVWindowSize: Int = 0
    /// Mei patch 0005 (default OFF): store SSM companion anchors at the
    /// first K chat role-turn boundaries (early structural offsets), so a
    /// mid-transcript diverging agentic edit can restore from a retained
    /// boundary instead of full-prefilling. 0 = upstream behavior exactly.
    /// TTFT/latency lever only; never a decode tok/s lever.
    public var ssmAnchorBoundaryCount: Int = 0
    /// Use the mmap-backed safetensors loader (upstream default true). The
    /// mmap loader realigns unaligned tensors into copies (the 35B gained
    /// ~5GB active this way); stock file-backed loading uses less resident
    /// memory at the cost of slower first touch.
    public var useMmapSafetensors: Bool = true

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
            case "--memory-limit-bytes":
                config.memoryLimitBytes = try parseInt(flag, value())
            case "--cache-limit-bytes":
                config.cacheLimitBytes = try parseInt(flag, value())
            case "--kv-cache-dir":
                config.kvCacheDir = try value()
            case "--ssm-rederive":
                config.enableSSMReDerive = try parseBool(flag, value())
            case "--compiled-decode":
                config.enableCompiledDecode = try parseBool(flag, value())
            case "--compiled-decode-threshold":
                config.compiledDecodeMaxPromptOffset = try parseInt(flag, value())
            case "--max-kv-window":
                config.maxKVWindowSize = try parseInt(flag, value())
            case "--ssm-anchor-boundaries":
                config.ssmAnchorBoundaryCount = try parseInt(flag, value())
            case "--load-mmap":
                config.useMmapSafetensors = try parseBool(flag, value())
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

    Memory:
      --memory-limit-bytes N    Explicit MLX allocator limit (default 0 = MLX
                                default of 1.5x Metal recommended working set;
                                set this when the default limit is below the
                                model working set, which otherwise hangs)
      --cache-limit-bytes N     MLX buffer-pool cache limit (default 0 = limit)
      --kv-cache-dir DIR        On-disk KV cache dir for the prefix coordinator
                                (hybrid models need the disk tier; default off)

    Misc:
      --log-requests BOOL  Log each request's token counts (default false)
      --compiled-decode BOOL  Graph-traced compiled decode (default false)
      --compiled-decode-threshold N  Skip compiled promote+trace when the
                              prefill offset exceeds N tokens (nil = no
                              threshold; 0 = never compile). The upstream
                              default traces a promptOffset-sized buffer,
                              which is a multi-minute prefill tax at 45K.
      --max-kv-window N          EXPERIMENTAL: cap the rotating-KV ring at
                              N tokens (attention scans at most the ring).
                              Correctness-bounded only; 0 = default ring.
      --ssm-anchor-boundaries K  EXPERIMENTAL (patch 0005, default off):
                              store SSM companion anchors at the first K
                              chat role-turn boundaries so a diverging
                              agentic edit restores from a boundary instead
                              of full-prefilling (TTFT lever; 0 = upstream).
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