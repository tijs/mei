import Foundation

#if canImport(Darwin)
import Darwin
#endif

/// Selects safe runtime defaults for the active model bundle.
///
/// `auto` is deliberately conservative: only the exact validated
/// qwen3_5_moe family is recognized as Ornith-compatible. Everything else,
/// including missing or malformed metadata, resolves to `generic`.
public enum ModelOptimizationProfile: String, CaseIterable, Sendable, Equatable {
    case auto
    case generic
    case ornith

    public var isOrnith: Bool { self == .ornith }

    public var defaultPrefillStepSize: Int {
        isOrnith ? 512 : 64
    }


    /// Recommended working set below which a 1024-token prefill step is not
    /// affordable, and the conservative 512 is used instead.
    ///
    /// MEASURED on the release pin, 54,016-token prompt, cache reuse off:
    ///   512  -> 311 tok/s prefill, MLX peak 22.72 GB
    ///   1024 -> 337 tok/s prefill (+8.4%), MLX peak 23.77 GB
    /// So 1024 buys ~8% for about 1.05 GB of peak. At the full 65,536-token cap
    /// that lands near 26.2 GB against a 26.8 GB recommended working set — it
    /// fits a 32 GB machine with roughly 0.6 GB to spare, and does not fit
    /// comfortably anywhere smaller. Running out of working set is a far worse
    /// outcome than an 8% slower prefill, so this is a device check rather than
    /// an unconditional default.
    public static let prefill1024MinimumWorkingSetBytes = 26_000_000_000

    /// Whether this device can afford a 1024-token prefill step. A nil working
    /// set means we could not ask, and we do not assume the generous answer.
    public static func canAffordPrefill1024(
        recommendedWorkingSetBytes: Int?
    ) -> Bool {
        guard let ws = recommendedWorkingSetBytes else { return false }
        return ws >= prefill1024MinimumWorkingSetBytes
    }

    /// Operator-facing explanation for a clamped prefill step.
    ///
    /// Lives here, not at the print site, so its content is assertable without
    /// a low-memory device — the numbers in it are easy to get backwards, and
    /// no machine this project owns can reach the branch that emits it.
    public static func prefillClampMessage(from original: Int, to clamped: Int) -> String {
        """
        mei: WARNING prefill step reduced \(original) -> \(clamped). This \
        device reports a recommended working set below \
        \(prefill1024MinimumWorkingSetBytes) bytes, so the \(original)-token \
        step this model profile was measured at does not fit. Expect roughly \
        8% slower prefill. Note also that chunked prefill is not \
        answer-invariant on this architecture, so generated output may differ \
        from the profile's measurements. Pass --prefill-step-size \(original) \
        to override.
        """
    }

    /// Architecture-validated chunked-prefill step, for when no named profile
    /// supplies one. An explicit `--prefill-step-size` always wins.
    public static func prefillStepSize(
        modelDirectory: String, profile: ModelOptimizationProfile,
        recommendedWorkingSetBytes: Int? = nil
    ) -> Int {
        if profile.isOrnith,
           canAffordPrefill1024(recommendedWorkingSetBytes: recommendedWorkingSetBytes) {
            return 1024
        }
        return profile.defaultPrefillStepSize
    }

    public static func resolve(
        requested: ModelOptimizationProfile,
        modelDirectory: String
    ) -> ModelOptimizationProfile {
        switch requested {
        case .auto:
            return detect(modelDirectory: modelDirectory)
        case .generic, .ornith:
            return requested
        }
    }

    /// Detect only from valid model metadata; model names and paths are not
    /// enough to activate the memory-sensitive Ornith profile.
    public static func detect(modelDirectory: String) -> ModelOptimizationProfile {
        let ornithTypes: Set<String> = ["qwen3_5_moe", "qwen3_5_moe_text"]
        return collectedModelTypes(in: modelDirectory).intersection(ornithTypes).isEmpty
            ? .generic
            : .ornith
    }

    /// Model types whose in-process prefix reuse requires the disk-backed KV
    /// tier, so cache-reuse on without an explicit `--kv-cache-dir` defaults
    /// them to a disposable on-disk cache. Two empirically distinct reasons:
    ///
    /// - Dense Qwen3.5/Qwen3.8-lineage checkpoints (`qwen3_5`/`qwen3_5_text`)
    ///   CRASH with the in-memory-only paged KV tier (`Fatal error:
    ///   SmallVector out of range`, vmlx mlx/c/array.cpp:335; trigger isolated
    ///   by the 2026-09-02 bounded 2x2 — prefill step excluded, KV tier
    ///   implicated).
    ///   exact-repeat restore returns cached=0 on the paged in-memory tier;
    ///   the same requests restore 6173/6174 cached on the disk tier (probe
    ///   reuse rides the disk tier for this architecture too.
    ///
    /// The qwen3_5_moe family is here too, as of 0.4.2. It was previously
    /// excluded to keep Ornith runs on an operator-controlled cache, and that
    /// exclusion silently cost every bare invocation its prefix reuse —
    /// including WITHIN a single conversation, which is the ordinary chat case
    /// and has nothing to do with the opt-in cross-conversation feature.
    ///
    /// That lineage's hybrid cache (MambaCache on the GatedDelta layers,
    /// RotatingKVCache on the rest) cannot restore from the paged in-memory
    /// tier at all — the paged store writes zero blocks — so with no disk tier
    /// there is nowhere for a turn boundary to live and every turn re-prefills
    /// the whole transcript.
    ///
    /// MEASURED on Ornith 1.5, growing conversation, no anchors, 0.4.1 build:
    ///   turn   bare                  with --kv-cache-dir
    ///   1      61.55 s  cached 0     61.90 s  cached 0
    ///   2      61.35 s  cached 0      1.68 s  cached 20,385
    ///   3      61.72 s  cached 0      1.60 s  cached 20,407
    ///   4      61.62 s  cached 0      1.69 s  cached 20,441
    ///   total  246 s                  67 s
    /// 3.7x over four turns, widening with conversation length. An explicit
    /// --kv-cache-dir still wins, and --cache-reuse false still disables
    /// caching entirely, so operator control is preserved where it is asked
    /// for rather than assumed by omission.
    public static let diskKVRequiredModelTypes: Set<String> =
        ["qwen3_5", "qwen3_5_text",
         "qwen3_5_moe", "qwen3_5_moe_text"]

    /// True when the bundle's metadata contains any model_type that needs
    /// the disk KV tier for prefix reuse. Missing/malformed metadata returns
    /// false (the safe default must not fire on unreadable state — it only
    /// needs to fire on empirically verified model families).
    public static func needsDiskKVTier(modelDirectory: String) -> Bool {
        !collectedModelTypes(in: modelDirectory)
            .isDisjoint(with: diskKVRequiredModelTypes)
    }

    /// lineage joined the disk-tier-required set; kept as an alias so
    /// external consumers of the 0.1.0 public API keep compiling).
    public static let denseQwen35KVUnsafeModelTypes: Set<String> = diskKVRequiredModelTypes

    /// @deprecated — use `needsDiskKVTier(_:)`; behavior is identical.
    public static func denseQwen35NeedsDiskKVTier(modelDirectory: String) -> Bool {
        needsDiskKVTier(modelDirectory: modelDirectory)
    }

    /// All `model_type` values reachable in config.json (root + nested
    /// text_config etc.), lowercased; empty on unreadable metadata.
    private static func collectedModelTypes(in modelDirectory: String) -> Set<String> {
        let url = URL(fileURLWithPath: modelDirectory, isDirectory: true)
            .appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data),
              let root = object as? [String: Any]
        else {
            return []
        }
        var modelTypes = Set<String>()
        collectModelTypes(in: root, into: &modelTypes)
        return modelTypes
    }

    /// Apply only the validated Ornith memory safeguard. Automatic detection
    /// sets it only when neither supported environment control was supplied;
    /// an explicit `ornith` profile passes `force: true` for reproducibility.
    public func applyRuntimeEnvironment(force: Bool = false) {
        guard isOrnith else { return }
        #if canImport(Darwin)
        let hasExplicitOverride = getenv("VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES") != nil
            || getenv("BENCH_NO_FUSED_GATE_UP") != nil
        if force || !hasExplicitOverride {
            setenv("VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES", "0", 1)
            setenv("BENCH_NO_FUSED_GATE_UP", "1", 1)
        }

        // vmlx's shapeless micro-fusions (compiledSwiGLU/GeGLU/SigmoidGate plus
        // the Qwen35 helpers) are gated off by
        // HardwareInfo.isCompiledDecodeSupported. On this lineage they are a
        // free win and the operator should not have to know the env var exists.
        //
        // MEASURED, same binary both legs, round-robin, warm-up, n=10, quiet
        // machine: Ornith short decode 66.36 -> 72.32 tok/s (+9.0%), 30k
        // 40.36 -> 41.45 (+2.7%); Qwen3.6 text-only 66.65 -> 73.21 (+9.8%),
        // 30k 41.93 -> 43.38 (+3.5%).
        //
        // CORRECTNESS GATE, the one that matters because the documented failure
        // mode is silent numerical corruption rather than a crash: four
        // 700-token greedy temp-0 generations token-for-token identical with
        // the flag on and off, plus an identical native tool call. Re-verified
        // on this exact vmlx pin.
        //
        // Its upstream default-off reason is Osaurus #1173 — decode corruption
        // after switching models inside one process. Mei is
        // one-model-per-server-process, so that is structurally unreachable
        // here. An explicit setting always wins, so anyone who distrusts it can
        // export VMLX_ENABLE_UNSAFE_COMPILE=0.
        if getenv("VMLX_ENABLE_UNSAFE_COMPILE") == nil {
            setenv("VMLX_ENABLE_UNSAFE_COMPILE", "1", 1)
        }
        #endif
    }

    /// Server-side generation cap for this lineage.
    ///
    /// The global default of 32768 lets a degenerate turn run away: one request
    /// in a benchmark run generated 32,768 tokens and hit the cap after 1,156 s,
    /// burning 19.3 of that run's 40.1 minutes of generation on its own. Across
    /// all coding rows that cost 14.5% of total wall against llama.cpp's 0.7%.
    ///
    /// 8192 cannot truncate legitimate work here: per request p99 is 2,933
    /// tokens and the largest non-degenerate completion observed was 4,674. It
    /// only bounds callers that send no max_tokens of their own.
    public var defaultMaxTokens: Int? { isOrnith ? 8192 : nil }

    private static func collectModelTypes(
        in value: Any,
        into result: inout Set<String>
    ) {
        if let dictionary = value as? [String: Any] {
            for (key, child) in dictionary {
                if key == "model_type", let modelType = child as? String {
                    result.insert(modelType.lowercased())
                }
                collectModelTypes(in: child, into: &result)
            }
        } else if let array = value as? [Any] {
            for child in array {
                collectModelTypes(in: child, into: &result)
            }
        }
    }
}
