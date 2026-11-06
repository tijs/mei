import Foundation
import MLXLMCommon

/// One selector for the prefix coordinator, independent of allocator buffers.
public enum CacheMode: String, Sendable, Codable {
    case auto, off, memory, disk
}

/// Legacy source configuration, retained to reject obsolete requests clearly.
/// Only `.off` is supported by the owned-weight runtime.
public enum WeightAdvicePolicy: Sendable, Equatable {
    case off, auto
    case percent(Int)

    public init(_ value: String) throws {
        switch value.lowercased() {
        case "off", "false", "0": self = .off
        case "auto": self = .auto
        default:
            guard let n = Int(value), (1...95).contains(n) else {
                throw ConfigError.invalidValue("--mlxpress expects off, auto, or an integer from 1 to 95")
            }
            self = .percent(n)
        }
    }

    public var description: String {
        switch self {
        case .off: "off"
        case .auto: "auto"
        case .percent(let n): String(n)
        }
    }
}

/// Requested settings accompany live allocator counters; none is a process-RAM
/// ceiling. Residency is requested, subject to runtime/device/system admission.
public struct MeiRuntimePolicies: Sendable, Codable {
    public var cacheModeRequested: String
    public var cacheModeEffective: String
    public var cacheDecision: String
    public var kvCacheDirectory: String?
    public var disposableKVCache: Bool
    public var prefixCacheLimitBytes: Int
    public var allocatorCacheLimitRequestedBytes: Int?
    public var memoryLimitRequestedBytes: Int?
    public var weightStorage: String
    /// A request to the runtime's bounded residency policy, not a grant or RAM ceiling.
    public var keepModelInMemoryRequested: Bool
}

public extension ServerConfig {
    /// Bind explicit allocator requests into loading as well as inference.
    internal var modelLoadConfiguration: LoadConfiguration {
        var configuration = LoadConfiguration(
            maxResidentBytes: cacheLimitExplicit ? .absolute(UInt64(cacheLimitBytes)) : .default,
            memoryLimit: memoryLimitBytes > 0 ? .absolute(UInt64(memoryLimitBytes)) : .default,
            useMmapSafetensors: false)
        configuration.keepModelInMemory = keepModelInMemory
        return configuration
    }

    var effectiveCacheMode: CacheMode {
        !cacheReuse ? .off : (kvCacheDir.isEmpty ? .memory : .disk)
    }

    /// Called both after CLI parsing and before library loading. Resolving twice
    /// preserves the same instance-specific directory and performs no I/O.
    mutating func resolveRuntimePolicies() throws {
        guard memoryLimitBytes >= 0, cacheLimitBytes >= 0 else {
            throw ConfigError.invalidValue("memory and allocator cache limits must be >= 0")
        }
        guard prefixCacheLimitBytes > 0 else {
            throw ConfigError.invalidValue("--prefix-cache-limit-bytes must be > 0")
        }
        guard mlxPressPolicy == .off else {
            throw ConfigError.invalidValue(
                "MLXPress cold-tier advice was removed upstream; remove --mlxpress/MLXPRESS/JANGPRESS. "
                + "Weights use owned storage; --keep-model-in-memory controls a separate residency request.")
        }
        guard !useMmapSafetensors else {
            throw ConfigError.invalidValue(
                "--load-mmap true is no longer supported: weights use owned storage. Remove --load-mmap; "
                + "--keep-model-in-memory controls a separate residency request.")
        }
        let requiresDisk = ModelOptimizationProfile.needsDiskKVTier(modelDirectory: modelDirectory)
        let autoNeedsDisk = requiresDisk || ssmAnchorBoundaryCount > 0
        if disposableKVCache,
           cacheMode == .off || cacheMode == .memory || (cacheMode == .auto && !autoNeedsDisk) {
            kvCacheDir = ""
        }
        if cacheMode != .auto, let requested = cacheReuseRequested, requested != (cacheMode != .off) {
            throw ConfigError.invalidValue("--cache-mode conflicts with --cache-reuse; use one selector")
        }
        switch cacheMode {
        case .off:
            guard kvCacheDir.isEmpty, !kvCacheDirExplicit else {
                throw ConfigError.invalidValue("--cache-mode off conflicts with --kv-cache-dir")
            }
            cacheReuseValue = false
            cacheDecision = "explicit off"
        case .memory:
            guard kvCacheDir.isEmpty, !kvCacheDirExplicit else {
                throw ConfigError.invalidValue("--cache-mode memory conflicts with --kv-cache-dir")
            }
            guard !requiresDisk, ssmAnchorBoundaryCount == 0 else {
                throw ConfigError.invalidValue("this model or SSM anchors require disk-backed reuse; choose --cache-mode disk or off")
            }
            cacheReuseValue = true
            cacheDecision = "explicit memory"
        case .disk:
            guard !kvCacheDirExplicit || !kvCacheDir.isEmpty else {
                throw ConfigError.invalidValue("--cache-mode disk requires a nonempty --kv-cache-dir or an omitted directory")
            }
            cacheReuseValue = true
            cacheDecision = "explicit disk"
            resolveDisposableDirectoryIfNeeded()
        case .auto:
            cacheReuseValue = cacheReuseRequested ?? true
            guard cacheReuseValue else {
                cacheDecision = "disabled by cache-reuse"
                return
            }
            if !kvCacheDir.isEmpty {
                if disposableKVCache {
                    cacheDecision = requiresDisk ? "model requires disk-backed reuse" : "SSM anchors require disk-backed reuse"
                } else {
                    cacheDecision = "explicit directory"
                }
            } else if requiresDisk || ssmAnchorBoundaryCount > 0 {
                guard !kvCacheDirExplicit else {
                    throw ConfigError.invalidValue("empty --kv-cache-dir cannot provide required disk reuse; choose --cache-mode off")
                }
                cacheDecision = requiresDisk ? "model requires disk-backed reuse" : "SSM anchors require disk-backed reuse"
                autoEnabledAnchorKVCacheDir = ssmAnchorBoundaryCount > 0 && !requiresDisk
                resolveDisposableDirectoryIfNeeded()
            } else {
                cacheDecision = "model supports memory reuse"
            }
        }
    }

    private mutating func resolveDisposableDirectoryIfNeeded() {
        if kvCacheDir.isEmpty {
            kvCacheDir = Self.defaultDisposableKVCacheDir(servedModelID: servedModelID)
            disposableKVCache = true
        }
    }

    func runtimePolicies() -> MeiRuntimePolicies {
        MeiRuntimePolicies(
            cacheModeRequested: cacheMode.rawValue,
            cacheModeEffective: effectiveCacheMode.rawValue,
            cacheDecision: cacheDecision,
            kvCacheDirectory: effectiveCacheMode == .disk ? kvCacheDir : nil,
            disposableKVCache: effectiveCacheMode == .disk && disposableKVCache,
            prefixCacheLimitBytes: prefixCacheLimitBytes,
            allocatorCacheLimitRequestedBytes: cacheLimitExplicit ? cacheLimitBytes : nil,
            memoryLimitRequestedBytes: memoryLimitBytes > 0 ? memoryLimitBytes : nil,
            weightStorage: "owned",
            keepModelInMemoryRequested: keepModelInMemory)
    }
}
