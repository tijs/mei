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

    /// Dense Qwen3.5/Qwen3.8-lineage MLX checkpoints (`qwen3_5` /
    /// `qwen3_5_text`) crash with the in-memory-only paged KV cache tier
    /// (`Fatal error: SmallVector out of range`, vmlx mlx/c/array.cpp:335;
    /// trigger isolated by the 2026-09-02 bounded 2x2 — prefill step
    /// excluded, KV tier implicated). Whenever prefix reuse is enabled these
    /// models must land on the disk-backed tier, so cache-reuse on without an
    /// explicit `--kv-cache-dir` defaults them to a disposable on-disk cache.
    /// The MoE/hybrid qwen3_5_moe family is intentionally NOT in the set:
    /// Ornith runs keep their operator-controlled cache configuration.
    public static let denseQwen35KVUnsafeModelTypes: Set<String> = ["qwen3_5", "qwen3_5_text"]

    /// True when the bundle's metadata contains any dense qwen3_5 family
    /// model_type. Missing/malformed metadata returns false (the safe
    /// default must not fire on unreadable state — it only needs to fire on
    /// the empirically verified crashing family).
    public static func denseQwen35NeedsDiskKVTier(modelDirectory: String) -> Bool {
        !collectedModelTypes(in: modelDirectory)
            .isDisjoint(with: denseQwen35KVUnsafeModelTypes)
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
