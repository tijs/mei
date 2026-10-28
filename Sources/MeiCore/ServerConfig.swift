import Foundation
import MLX
import MLXLMCommon

/// Immutable server configuration, resolved from CLI flags with the same
/// shape the local-model-bench start scripts use for the other engines.
public struct ServerConfig: Sendable {
    public static let version = "0.10.0-rc.1"
    public var modelDirectory: String
    public var servedModelID: String
    /// Experimental app-owned Flint affine architecture. Never auto-enabled.
    public var enableFlint: Bool = false
    /// Separately gated, payload-preserving JANGH2 adapter.
    public var enableFlintJANGH: Bool = false
    /// Set when --served-model-id was omitted and defaulted from the
    /// bundle directory name, so startup can report the chosen ID.
    public var servedModelIDWasDefaulted: Bool = false
    /// Requested profile from the operator. `auto` is resolved from config.json.
    public var requestedOptimizationProfile: ModelOptimizationProfile = .auto
    /// Effective profile used by this process after model metadata detection.
    public private(set) var optimizationProfile: ModelOptimizationProfile = .generic
    public var host: String = "127.0.0.1"
    public var port: Int = 8024
    public var contextCap: Int = 65_536
    public var maxTokensDefault: Int = 32_768
    /// Set when --max-tokens was given, so a profile default never
    /// overrides an explicit operator choice.
    public var maxTokensExplicit: Bool = false

    /// The named model tuning selected with `--model-profile`, if any.
    /// Nil means no per-model tuning was requested: architecture detection
    /// still applies, but none of the measured per-model numbers do.
    public private(set) var modelTuning: ModelTuning?
    public var prefillStepSize: Int = 64
    /// Set when a named profile's prefill step did not fit this device's
    /// recommended working set and was reduced. Carries the ORIGINAL value so
    /// startup can name what was given up.
    public var prefillStepClampedFrom: Int?
    /// KV cache quantization: nil = fp16, else bits (4 or 8).
    public var kvBits: Int? = nil
    public var kvGroupSize: Int = 64
    public var quantizedKVStart: Int = 0
    public var temperature: Float = 0.6 { didSet { temperatureExplicit = true } }
    public var topP: Float = 0.95 { didSet { topPExplicit = true } }
    public var topK: Int = 20 { didSet { topKExplicit = true } }
    private var temperatureExplicit = false
    private var topPExplicit = false
    private var topKExplicit = false
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
    /// Legacy caller override is distinct from the resolver's effective value.
    public var cacheReuse: Bool {
        get { cacheReuseValue }
        set { cacheReuseRequested = newValue; cacheReuseValue = newValue }
    }
    var cacheReuseRequested: Bool?
    var cacheReuseValue = true
    public var cacheMode: CacheMode = .auto
    public internal(set) var cacheDecision = "unresolved"
    public internal(set) var disposableKVCache = false
    public var mlxPressPolicy: WeightAdvicePolicy = .off {
        didSet { mlxPressPolicySource = "programmatic" }
    }
    public internal(set) var mlxPressPolicySource = "default"
    public var logRequests: Bool = false
    /// Path for the per-request JSONL log (`--request-log`). Empty disables it.
    public var requestLogPath: String = ""
    /// MLX allocation budget: 0/omitted selects the backend default. This
    /// is not a process RAM ceiling; an undersized budget can stall allocation.
    public var memoryLimitBytes: Int = 0
    public var cacheLimitBytes: Int = 0 { didSet { cacheLimitExplicit = true } }
    public private(set) var cacheLimitExplicit = false
    /// Retained prefix KV/rotating-companion budget, distinct from freed
    /// allocator buffers. No prompt or inference-memory cap is implied.
    public var prefixCacheLimitBytes: Int = 536_870_912
    /// Directory for the coordinator's on-disk L2 KV tier (hybrid models
    /// like Ornith's mamba/GatedDelta layers are disk-backed-restore only,
    /// so in-process prefix reuse for them needs this tier; paged-in-memory
    /// alone covers non-hybrid topologies). Empty means no directory has
    /// been selected yet. cacheMode resolves whether a disk tier is needed.
    public var kvCacheDir: String = "" { didSet { disposableKVCache = false } }
    /// Set when `--ssm-anchor-boundaries` caused a disposable KV directory to
    /// be created that the operator did not ask for, so startup can say so.
    public var autoEnabledAnchorKVCacheDir: Bool = false
    /// True when the operator passed `--kv-cache-dir` (even an empty value):
    /// empty explicit paths fail if the model requires disk-backed reuse.
    public private(set) var kvCacheDirExplicit = false
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

    /// Bounded admission queue length: how many requests may wait for an
    /// execution slot (0 = no queue: reject whenever every slot is busy).
    /// Over-limit arrivals get HTTP 429 `queue_full`. Provisional default;
    /// the plan's stage-7 benchmark sets the final values (see
    /// docs/INFERENCE-SCHEDULING.md).
    public var maxQueueDepth: Int = 32
    /// Queue wait budget in milliseconds before a waiting request is refused
    /// with HTTP 503 `queue_timeout` (0 = wait forever). Provisional default
    /// like `maxQueueDepth`.
    public var queueWaitTimeoutMilliseconds: Int = 600_000
    /// Maximum number of generations that may run at once. **1 is the
    /// default strict single-flight mode** — byte-compatible with 0.8.0,
    /// including when this flag is absent. 2 is shipped as the explicit
    /// opt-in bounded-overlap mode (stage 5+ of the plan), for operators
    /// who accept the measured trade-offs; values outside 1...2 are
    /// rejected at parse time. The overlap mode never changes the default
    /// behavior of a process started without this flag.
    public var maxActiveGenerations: Int = 1

    public init(modelDirectory: String, servedModelID: String) {
        self.modelDirectory = modelDirectory
        self.servedModelID = servedModelID
    }

    /// KV capacity bound: the cache must hold the full context plus a
    /// generation headroom window.
    public var maxKVSize: Int { contextCap + 4096 }

    /// Flint's selected bundle declares its sampler. Adopt only its authored
    /// values when the operator did not set a CLI/programmatic override. This
    /// is also called by Engine.load for library callers that bypass parsing.
    mutating func applyFlintGenerationDefaults() throws {
        guard enableFlint else { return }
        let directory = URL(fileURLWithPath: modelDirectory)
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              try FlintSupport.modelType(data: data) == "flint" else { return }
        // The pinned quantized-cache update traps through the ordinary KV API,
        // and its public attention helper cannot carry Flint's learned sinks.
        // Refuse before load rather than dropping sinks or ignoring the flag.
        guard kvBits == nil else {
            throw FlintError.unsupported("--kv-bits is not supported with Flint's learned attention sinks; omit it")
        }
        let defaults = try JSONDecoder.json5().decode(GenerationConfigFile.self,
            from: Data(contentsOf: directory.appendingPathComponent("generation_config.json")))
        guard let temperature = defaults.temperature, let topP = defaults.topP, let topK = defaults.topK,
              temperature.isFinite, temperature >= 0, topP.isFinite, (0...1).contains(topP), topK >= 0,
              defaults.doSample != false else {
            throw FlintError.unsupported("generation_config.json must declare temperature/top_p/top_k with sampling enabled")
        }
        if !temperatureExplicit { self.temperature = temperature; temperatureExplicit = false }
        if !topPExplicit { self.topP = topP; topPExplicit = false }
        if !topKExplicit { self.topK = topK; topKExplicit = false }
    }

    /// Instance-specific disposable KV cache location for the model-aware safe
    /// default. Lives under the OS temp directory: OS-reclaimed, never
    /// pollutes a project directory. A random instance component prevents
    /// two servers with the same served ID sharing mutable cache storage.
    /// Explicit --kv-cache-dir is required for intentional persistence.
    public static func defaultDisposableKVCacheDir(servedModelID: String) -> String {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mei-kv-cache", isDirectory: true)
        return root
            .appendingPathComponent(sanitizedModelKey(servedModelID), isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .path
    }

    /// Model IDs contain '/' (org/repo) and often spaces (quant labels);
    /// collapse everything non-safe into a single path component.
    private static func sanitizedModelKey(_ servedModelID: String) -> String {
        String(servedModelID.map { ch in
            ch.isLetter || ch.isNumber || ch == "-" || ch == "_" || ch == "." ? ch : "-"
        })
    }
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
    /// - Parameter recommendedWorkingSetBytes: the device's Metal recommended
    ///   working set, injected so tests can exercise both sides of the
    ///   affordability check. Defaults to asking the GPU.
    static func parse(
        arguments: [String] = Array(CommandLine.arguments.dropFirst()),
        recommendedWorkingSetBytes: Int? = GPU.maxRecommendedWorkingSetBytes().map { Int($0) },
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ServerConfig {
        var modelDirectory: String?
        var servedModelID: String?
        var requestedOptimizationProfile: ModelOptimizationProfile = .auto
        var prefillStepSizeExplicit = false
        var ssmAnchorBoundariesExplicit = false
        var mlxPressExplicit = false
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
            case "--enable-flint": config.enableFlint = try parseBool(flag, value())
            case "--enable-flint-jangh": config.enableFlintJANGH = try parseBool(flag, value())
            case "--host": config.host = try value()
            case "--port":
                config.port = try parseInt(flag, value())
            case "--context-cap":
                config.contextCap = try parseInt(flag, value())
            case "--max-tokens":
                config.maxTokensDefault = try parseInt(flag, value())
                config.maxTokensExplicit = true
            case "--prefill-step-size":
                config.prefillStepSize = try parseInt(flag, value())
                prefillStepSizeExplicit = true
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
            case "--cache-mode":
                guard let mode = CacheMode(rawValue: try value()) else {
                    throw ConfigError.invalidValue("--cache-mode expects auto, off, memory, or disk")
                }
                config.cacheMode = mode
            case "--mlxpress":
                config.mlxPressPolicy = try WeightAdvicePolicy(value())
                config.mlxPressPolicySource = "cli"
                mlxPressExplicit = true
            case "--log-requests":
                config.logRequests = try parseBool(flag, value())
            case "--request-log":
                config.requestLogPath = try value()
            case "--memory-limit-bytes":
                config.memoryLimitBytes = try parseInt(flag, value())
            case "--cache-limit-bytes":
                config.cacheLimitBytes = try parseInt(flag, value())
            case "--prefix-cache-limit-bytes":
                config.prefixCacheLimitBytes = try parseInt(flag, value())
            case "--kv-cache-dir":
                config.kvCacheDir = try value()
                config.kvCacheDirExplicit = true
            case "--ssm-rederive":
                config.enableSSMReDerive = try parseBool(flag, value())
            case "--compiled-decode":
                config.enableCompiledDecode = try parseBool(flag, value())
            case "--compiled-decode-threshold":
                config.compiledDecodeMaxPromptOffset = try parseInt(flag, value())
            case "--max-kv-window":
                config.maxKVWindowSize = try parseInt(flag, value())
            case "--model-profile":
                let raw = try value()
                guard let tuning = ModelTuningRegistry.named(raw) else {
                    throw ConfigError.invalidValue(
                        "--model-profile '\(raw)' is not a supported model. "
                        + "Known profiles: "
                        + ModelTuningRegistry.names.joined(separator: ", ")
                        + ". Omit the flag to run on architecture defaults.")
                }
                config.modelTuning = tuning
            case "--ssm-anchor-boundaries":
                config.ssmAnchorBoundaryCount = try parseInt(flag, value())
                ssmAnchorBoundariesExplicit = true
            case "--load-mmap":
                config.useMmapSafetensors = try parseBool(flag, value())
            case "--max-queue-depth":
                config.maxQueueDepth = try parseInt(flag, value())
            case "--queue-wait-ms":
                config.queueWaitTimeoutMilliseconds = try parseInt(flag, value())
            case "--max-active-generations":
                // Explicit opt-in bounded overlap. The settled plan caps the
                // experiment at two active generations and keeps one as the
                // default; anything else is a configuration error, not a
                // silently-clamped request.
                let requested = try parseInt(flag, value())
                guard (1...2).contains(requested) else {
                    throw ConfigError.invalidValue(
                        "--max-active-generations must be 1 (strict single-flight, default) "
                        + "or 2 (opt-in bounded overlap); got \(requested). Values above 2 are "
                        + "not supported: Mei has no continuous batching and no preemption.")
                }
                config.maxActiveGenerations = requested
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
        // The served ID is a contract with clients, not a tuning knob: it is
        // whatever they will put in the "model" field. When the operator does
        // not state one they have no expectation to violate, so default to the
        // bundle's directory name and say so at startup.
        //
        // Deliberately NOT inferred from the bundle: config.json carries no
        // name or path field, and a README's front-matter names the base model
        // rather than the artifact — checked across every model in the bench
        // repo, where served IDs are HF-style org/name while directories carry
        // only the name (and one has `-aligned` appended). Guessing the org
        // would produce confidently wrong IDs.
        let resolvedServedModelID = servedModelID
            ?? URL(fileURLWithPath: modelDirectory).lastPathComponent
        config.servedModelIDWasDefaulted = (servedModelID == nil)
        config.modelDirectory = modelDirectory
        config.servedModelID = resolvedServedModelID

        // A named model profile supplies the numbers we MEASURED for that
        // model. An explicit flag still wins — the operator asked for it.
        if let tuning = config.modelTuning {
            // Naming the model selects its architecture handling too, so there
            // is no separate --optimization-profile to keep in sync.
            requestedOptimizationProfile = tuning.optimizationProfile
            if !prefillStepSizeExplicit, let step = tuning.prefillStepSize {
                // A profile states the step it was MEASURED at, so naming a
                // model gives the same answers on every machine that can
                // afford it. Where the device cannot, a thin working-set
                // margin is not worth defending: clamp down and say so, rather
                // than run out of working set mid-prefill. 0.4.1 made this
                // choice silently and invisibly; the operator is told now,
                // because prefill chunking changes what the model writes.
                if step > 512,
                   !ModelOptimizationProfile.canAffordPrefill1024(
                       recommendedWorkingSetBytes: recommendedWorkingSetBytes) {
                    config.prefillStepClampedFrom = step
                    config.prefillStepSize = 512
                } else {
                    config.prefillStepSize = step
                }
                prefillStepSizeExplicit = true   // keep detection from re-deriving
            }
            if !ssmAnchorBoundariesExplicit, let anchors = tuning.ssmAnchorBoundaries {
                config.ssmAnchorBoundaryCount = anchors
            }
            if !config.maxTokensExplicit, let cap = tuning.maxTokens {
                config.maxTokensDefault = cap
                config.maxTokensExplicit = true
            }
        }
        config.requestedOptimizationProfile = requestedOptimizationProfile
        config.optimizationProfile = ModelOptimizationProfile.resolve(
            requested: requestedOptimizationProfile,
            modelDirectory: modelDirectory)
        if !prefillStepSizeExplicit {
            // No named profile: the architecture default, device-aware for
            // the ornith family. Restores behaviour 0.4.1 shipped and the
            // 0.5.0 branch lost (commit 93f9588 is not in this history).
            config.prefillStepSize = ModelOptimizationProfile.prefillStepSize(
                modelDirectory: config.modelDirectory,
                profile: config.optimizationProfile,
                recommendedWorkingSetBytes: recommendedWorkingSetBytes)
        }

        // Architecture-validated generation cap. An explicit --max-tokens
        // always wins; see ModelOptimizationProfile.defaultMaxTokens for the
        // runaway evidence behind the ornith value.
        if !config.maxTokensExplicit,
           let profileMax = config.optimizationProfile.defaultMaxTokens {
            config.maxTokensDefault = profileMax
        }

        if !mlxPressExplicit,
           let key = ["MLXPRESS", "JANGPRESS"].first(where: { environment[$0] != nil }) {
            config.mlxPressPolicy = try WeightAdvicePolicy(environment[key]!)
            config.mlxPressPolicySource = "environment:" + key
        }
        try config.resolveRuntimePolicies()

        guard config.prefillStepSize > 0 else {
            throw ConfigError.invalidValue("--prefill-step-size must be > 0")
        }
        guard config.contextCap > 0 else {
            throw ConfigError.invalidValue("--context-cap must be > 0")
        }
        guard config.maxQueueDepth >= 0 else {
            throw ConfigError.invalidValue("--max-queue-depth must be >= 0 (0 = no queue)")
        }
        guard config.queueWaitTimeoutMilliseconds >= 0 else {
            throw ConfigError.invalidValue("--queue-wait-ms must be >= 0 (0 = wait forever)")
        }
        guard config.prefixCacheLimitBytes > 0 else {
            throw ConfigError.invalidValue("--prefix-cache-limit-bytes must be > 0")
        }
        try config.applyFlintGenerationDefaults()
        return config
    }

    static let usage = """
    mei — native Swift/MLX OpenAI-compatible inference server

    Usage: mei --model-dir DIR --served-model-id ID [options]

    Required:
      --model-dir DIR        Local directory with model config + safetensors
      --served-model-id ID   Exact model ID served by GET /v1/models
    Fetching a model:
      mei pull <profile> [--dest DIR]
                             Download the exact repo and revision this profile
                             was measured on, then verify the weights are
                             naturally aligned. Uses the `hf` CLI.

      --model-profile NAME   Serve a supported model with its MEASURED settings:
                             ornith-1.5-35b-a3b | qwen3.6-35b-a3b-text | qwen3.6-35b-a3b
                             Selects architecture handling, prefill step, anchor
                             boundaries and max-tokens together. Omit it and Mei
                             falls back to architecture defaults detected from
                             config.json, which are safe but not tuned. Explicit
                             flags always win over the profile.
      --version               Print the Mei release version
      --enable-flint-jangh BOOL  EXPERIMENTAL JANGH2 adapter (also enable Flint)
      --enable-flint BOOL     EXPERIMENTAL standard MLX affine Flint support
                             (default false; JANGH2 requires its separate adapter flag)

    Network:
      --host HOST            (default 127.0.0.1)
      --port PORT            (default 8024)

    Context / capacity:
      --context-cap TOKENS   Reject prompts beyond this many tokens (default 65536)
      --max-tokens TOKENS    Server-side per-request generation cap (default 32768)
      --prefill-step-size N  Chunked prefill window (default 64 for generic;
                             auto Ornith profile uses validated 512; gemma4
                             bundles default to measured 256)
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

    Scheduling (strict single-flight execution with bounded admission;
    see docs/INFERENCE-SCHEDULING.md):
      --max-queue-depth N    How many requests may wait for the execution slot
                             (default 32; 0 = no queue, refuse immediately
                             whenever the slot is busy). Over-limit requests
                             get HTTP 429 queue_full.
      --queue-wait-ms N      Queue wait budget in milliseconds before a
                             waiting request gets HTTP 503 queue_timeout
                             (default 600000; 0 = wait forever). Requests
                             cancelled while queued or running are removed
                             without a success finish; a request body may
                             carry the Mei-local \"mei_priority\" field
                             (\"interactive\" default, or \"background\";
                             the standard \"priority\" field is ignored).
      --max-active-generations N
                             How many generations may run at once: 1
                             (default) = strict single-flight; 2 = explicit
                             opt-in bounded overlap. The overlap mode is an
                             experiment with measured trade-offs; the default
                             never changes without this flag.

    Memory:
      --memory-limit-bytes N    Explicit MLX allocator limit (0/omitted = backend
                                default, reported by /v1/mei/status;
                                set this when the default limit is below the
                                model working set, which otherwise hangs)
      --cache-mode MODE         Prefix reuse: auto (default), off, memory, disk.
                                Auto selects disk when required by the model.
                                Explicit memory refuses unsupported topologies.
      --mlxpress POLICY         Experimental weight advice: off (default), auto,
                                or 1..95 percent. CLI overrides MLXPRESS/JANGPRESS.
      --cache-limit-bytes N     Freed allocator buffer cache: explicit 0 disables;
                                omitted uses backend default. Not a RAM ceiling.
      --prefix-cache-limit-bytes N Retained prefix KV/companion budget
                                (default 536870912; separate from buffer pool)
      --kv-cache-dir DIR        Explicit disk-cache directory. Omit for an
                                instance-specific temporary path when disk is
                                selected. Conflicts with memory/off modes.
                                Legacy --cache-reuse remains supported; prefer
                                --cache-mode for new configurations.

    Misc:
      --log-requests BOOL  Log each request's token counts (default false)
      --request-log PATH   Append one JSON line per generation run to PATH:
                              prompt/cached/completion tokens, prefill_ms,
                              generate_ms, wall_ms and tok/s. Covers every
                              response path, including streaming turns that
                              never ask for a usage block.
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