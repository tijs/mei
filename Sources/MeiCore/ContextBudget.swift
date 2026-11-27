import Darwin
import Foundation
import MLX
import os

/// The unit is one formatted prompt or generated token. Memory estimates include
/// a fixed prefill allowance and growing attention state, independent of whether
/// the prefix coordinator happens to find a reusable entry.
struct ContextProfile: Sendable {
    var maximum: Int?
    var autoLimit: Int
    var bytesPerToken: Int
    var workspace: Int
    var fallback: Int

    static func resolve(directory: String, fallback: Int) -> Self {
        let url = URL(fileURLWithPath: directory).appendingPathComponent("config.json")
        let object = (try? Data(contentsOf: url)).flatMap {
            try? JSONSerialization.jsonObject(with: $0, options: [.json5Allowed]) as? [String: Any]
        } ?? [:]
        let text = object["text_config"] as? [String: Any] ?? object
        let maximum = (text["max_position_embeddings"] ?? text["model_max_length"]) as? Int
        let family = object["model_type"] as? String
        let limit: Int
        let cost: Int
        // These envelopes are per generation (KV plus sequence-dependent scratch).
        // A long rotating/global window costs more per token than a local layer.
        // Unrecognized exports keep the existing conservative window.
        switch family {
        case "inkling2_loop":
            let profile = InklingSupport.profile(directory: directory)
            limit = profile == .original ? fallback : 32_768
            cost = (profile == .publishedOriginal ? 152 : 192) << 10
        case "sensenova_loop":
            limit = 26_624
            // Q6 and Q8 grew 8.94–8.95 GiB at 26,368 input + 32 output
            // on the 32 GiB M1 Max. Separately charge unfaulted weights.
            cost = 384 << 10
        case "qwen3_5_moe", "qwen3_5_moe_text":
            switch ModelTuningIdentity.detect(modelDirectory: directory)?.name {
            case "groot-19b-a3b-text":
                limit = 65_536
                cost = 96 << 10
            case "qwen3.6-35b-a3b":
                limit = 26_624
                cost = 144 << 10
            case "ornith-1.5-35b-a3b":
                limit = 26_624
                cost = 160 << 10
            case "qwen3.6-35b-a3b-text":
                limit = 32_768
                cost = 128 << 10
            default:
                limit = fallback
                cost = 128 << 10
            }
        default:
            limit = fallback
            cost = 128 << 10
        }
        // For a checkpoint outside the measured geometry, use at least the
        // size of its two FP16 attention tensors. Recurrent layers have
        // fixed state; reserve their workspace independently.
        let layerTypes = text["layer_types"] as? [String]
        let attentionLayers = layerTypes?.filter { $0 != "linear_attention" }.count
            ?? text["num_hidden_layers"] as? Int
        var tokenCost = cost
        if family != "inkling2_loop", let attentionLayers, (1...4096).contains(attentionLayers),
           let heads = text["num_key_value_heads"] as? Int, (1...4096).contains(heads),
           let dimension = text["head_dim"] as? Int, (1...4096).contains(dimension) {
            tokenCost = max(cost, attentionLayers * heads * dimension * 4)
        }
        let ceiling = max(1, min(maximum ?? fallback, limit))
        let unmeasuredLimit = family == "sensenova_loop" ? 4096 : 8192
        return Self(maximum: maximum, autoLimit: ceiling, bytesPerToken: tokenCost,
                    workspace: 768 << 20, fallback: min(unmeasuredLimit, fallback, ceiling))
    }
}

/// Swap counters cover the life of the host. Only recent growth is evidence
/// that a new generation would compete with paging.
struct SwapActivity: Sendable {
    private var last: (pages: UInt64, at: TimeInterval)?
    private var busyUntil: TimeInterval = 0

    mutating func isBusy(pages: UInt64, pageSize: Int, now: TimeInterval) -> Bool {
        if let last, now - last.at <= 30, pages > last.pages,
           pages - last.pages >= UInt64((32 << 20) / max(1, pageSize)) {
            busyUntil = now + 10
        }
        last = (pages, now)
        return now < busyUntil
    }
}

struct ContextHeadroom {
    private static let host = mach_host_self()
    private static let swapActivity = OSAllocatedUnfairLock(initialState: SwapActivity())
    let bytes: Int?
    let pressure: Int32

    static func sample(allocatorLimit: Int) -> Self {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: stats) / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(host, HOST_VM_INFO64, $0, &count)
            }
        }
        var usage = rusage_info_v2()
        let usageResult = withUnsafeMutablePointer(to: &usage) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V2, $0)
            }
        }
        var pressure: Int32 = 0
        var size = MemoryLayout.size(ofValue: pressure)
        let pressureResult = sysctlbyname("kern.memorystatus_vm_pressure_level", &pressure, &size, nil, 0)
        var pageSize: vm_size_t = 0
        guard result == KERN_SUCCESS, usageResult == 0, pressureResult == 0,
              host_page_size(host, &pageSize) == KERN_SUCCESS else {
            return Self(bytes: nil, pressure: pressure)
        }
        // Clean file-backed pages are reclaimable even on the active list.
        // Mach includes speculative pages in both free and external counts.
        let free = max(0, Int(stats.free_count) - Int(stats.speculative_count))
        let available = (free + Int(stats.external_page_count)) * Int(pageSize)
        let swapping = swapActivity.withLock { [pages = stats.swapouts, bytes = Int(pageSize)] in
            $0.isBusy(pages: pages, pageSize: bytes, now: ProcessInfo.processInfo.systemUptime)
        }
        let mlx = Memory.snapshot()
        let bytes = admissibleBytes(physical: physical, reclaimable: available,
                                    footprint: Int(usage.ri_phys_footprint), active: mlx.activeMemory,
                                    deviceLimit: GPU.maxRecommendedWorkingSetBytes() ?? physical,
                                    allocatorLimit: allocatorLimit)
        return Self(bytes: bytes, pressure: swapping ? max(2, pressure) : pressure)
    }

    static func admissibleBytes(physical: Int, reclaimable: Int, footprint: Int, active: Int,
                                deviceLimit: Int, allocatorLimit: Int) -> Int {
        // A mapped checkpoint may have allocated weights that have not yet
        // faulted into the process. Leave room for those pages on first use.
        let pendingResidency = max(0, active - footprint)
        let hostRoom = reclaimable - min(2 << 30, physical / 16) - pendingResidency
        let footprintRoom = physical - min(8 << 30, physical / 4) - max(footprint, active)
        // MLX's default allocator threshold waits for scheduled work and can
        // be crossed briefly. An explicit operator limit remains an upper bound.
        let metalRoom = min(allocatorLimit > 0 ? allocatorLimit : deviceLimit, deviceLimit) - active
        // Each bound measures the same unified memory through a different lens.
        return max(0, min(hostRoom, footprintRoom, metalRoom))
    }
}

struct ContextReservations {
    private var outstanding: [UUID: Int] = [:]
    var isIdle: Bool { outstanding.isEmpty }

    func available(cap: Int, profile: ContextProfile, headroom: ContextHeadroom) -> Int {
        guard headroom.pressure < 2 else { return 0 }
        let reserved = outstanding.values.reduce(0, +)
        let bytes = headroom.bytes ?? (profile.workspace + profile.fallback * profile.bytesPerToken)
        return min(cap, max(0, bytes - reserved - profile.workspace) / profile.bytesPerToken)
    }

    static func replyLimit(prompt: Int, requested: Int, cap: Int) throws -> Int {
        guard prompt < cap else { throw EngineError.overContextCap(promptTokens: prompt, cap: cap) }
        guard requested > 0 else { throw EngineError.invalidOutputLimit }
        return min(requested, cap - prompt)
    }

    mutating func admit(id: UUID, cap: Int, prompt: Int, requested: Int, profile: ContextProfile,
                        headroom: ContextHeadroom) throws -> Int {
        let reply = try Self.replyLimit(prompt: prompt, requested: requested, cap: cap)
        let total = prompt + reply
        let fits = available(cap: cap, profile: profile, headroom: headroom)
        guard total <= fits else { throw EngineError.memoryCapacity(required: total, available: fits) }
        outstanding[id] = profile.workspace + total * profile.bytesPerToken
        return reply
    }

    mutating func release(_ id: UUID) { outstanding.removeValue(forKey: id) }
}

/// Cancel the underlying generation on a critical pressure event. The caller
/// waits for the backend task to drain, just as it does on a client disconnect.
func withContextPressureGuard<T: Sendable>(
    source: any DispatchSourceProtocol = DispatchSource.makeMemoryPressureSource(eventMask: .critical, queue: .global()),
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    let critical = OSAllocatedUnfairLock(initialState: false)
    let work = Task { try await operation() }
    source.setEventHandler {
        critical.withLock { $0 = true }
        work.cancel()
    }
    source.resume()
    defer { source.cancel() }
    return try await withTaskCancellationHandler {
        do {
            let value = try await work.value
            try Task.checkCancellation()
            if critical.withLock({ $0 }) { throw EngineError.criticalMemoryPressure }
            return value
        } catch {
            if !Task.isCancelled && critical.withLock({ $0 }) { throw EngineError.criticalMemoryPressure }
            throw error
        }
    } onCancel: { work.cancel() }
}
