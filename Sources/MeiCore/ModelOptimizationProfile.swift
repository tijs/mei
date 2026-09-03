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

    /// Model types whose validated chunked-prefill step is 256. Measured
    /// 2026-09-03 on mlx-community/gemma-4-26b-a4b-it-4bit (generic profile,
    /// disk-KV default, 30k fresh fill, port 8024): step 256 fills at
    /// 266.5/265.3/266.2 pps (3 cold repeats) vs 139 (mean of 64-step rows),
    /// +91%; peak 27.23 GB unchanged vs the 64-step row; 30k loaded decode
    /// unchanged (~7.4 t/s, attention-bound); probe_mei acceptance pass-set
    /// identical to the 64-step baseline (only the pre-existing user-gated
    /// Gemma string-args tool-schema mismatch fails). Evidence:
    /// artifacts/probe-longctx-gemma4-pref{64,128,256,512}-*, probe-mei-gemma4-pref256-*,
    /// artifacts/gemma4-prefill-step-sweep-20260903.md.
    public static let prefill256ModelTypes: Set<String> = ["gemma4", "gemma4_text"]

    /// Architecture-validated chunked-prefill step. The Ornith profile keeps
    /// its validated 512; `gemma4`-lineage bundles default to the measured
    /// 256; everything else stays on the conservative 64. An explicit
    /// `--prefill-step-size` always wins.
    public static func prefillStepSize(
        modelDirectory: String, profile: ModelOptimizationProfile
    ) -> Int {
        if profile.isOrnith { return profile.defaultPrefillStepSize }
        if !collectedModelTypes(in: modelDirectory)
            .isDisjoint(with: prefill256ModelTypes) { return 256 }
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
    /// - Gemma 4 bundles (`gemma4`/`gemma4_text`) do NOT crash but their
    ///   exact-repeat restore returns cached=0 on the paged in-memory tier;
    ///   the same requests restore 6173/6174 cached on the disk tier (probe
    ///   evidence 2026-09-03, mlx-community/gemma-4-26b-a4b-it-4bit) —
    ///   reuse rides the disk tier for this architecture too.
    ///
    /// The MoE/hybrid qwen3_5_moe family is intentionally NOT in the set:
    /// Ornith runs keep their operator-controlled cache configuration.
    public static let diskKVRequiredModelTypes: Set<String> =
        ["qwen3_5", "qwen3_5_text", "gemma4", "gemma4_text"]

    /// True when the bundle's metadata contains any model_type that needs
    /// the disk KV tier for prefix reuse. Missing/malformed metadata returns
    /// false (the safe default must not fire on unreadable state — it only
    /// needs to fire on empirically verified model families).
    public static func needsDiskKVTier(modelDirectory: String) -> Bool {
        !collectedModelTypes(in: modelDirectory)
            .isDisjoint(with: diskKVRequiredModelTypes)
    }

    /// @deprecated — use `diskKVRequiredModelTypes` (renamed when the gemma4
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
        guard force || !hasExplicitOverride else { return }
        setenv("VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES", "0", 1)
        setenv("BENCH_NO_FUSED_GATE_UP", "1", 1)
        #endif
    }

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
