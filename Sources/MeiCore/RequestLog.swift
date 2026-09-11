import Foundation

/// Append-only JSONL log of one line per completed generation run.
///
/// Mei already reported `prefill_ms`, `generate_ms` and `cached_tokens` in
/// every response's `usage` block, but nothing on the benchmark side kept
/// them: an agent harness makes many turns per task and only the task's
/// total wall time survived. That made it impossible to say where an
/// agentic task's time actually goes — the 2026-09-09 measurement showed
/// Mei spending ~66% of coding-suite wall time not generating against
/// llama.cpp's ~25%, with no way to attribute it per turn.
///
/// `--request-log <path>` closes that. One line per run, written after the
/// run is finalized and before it leaves the engine actor, so every response
/// shape is covered — non-streaming chat, streaming chat (which reports
/// usage only when the client asks for it), and `/v1/completions`.
///
/// Deliberately a plain appended file rather than a structured logger: the
/// consumer is a benchmark script, the volume is a few hundred lines per
/// suite, and a partially written run must never lose the lines before it.
public enum RequestLog {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var handle: FileHandle?
    nonisolated(unsafe) private static var startedAt = Date()

    /// Opens (or creates) the log at `path`. Called once at startup. A path
    /// that cannot be opened disables logging rather than failing the
    /// server: instrumentation must never take down a benchmark run.
    public static func configure(path: String) {
        lock.lock()
        defer { lock.unlock() }
        let fm = FileManager.default
        let dir = (path as NSString).deletingLastPathComponent
        if !dir.isEmpty { try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true) }
        if !fm.fileExists(atPath: path) { fm.createFile(atPath: path, contents: nil) }
        guard let h = FileHandle(forWritingAtPath: path) else {
            FileHandle.standardError.write(Data("mei: could not open --request-log at \(path); request logging disabled\n".utf8))
            return
        }
        h.seekToEndOfFile()
        handle = h
        startedAt = Date()
    }

    public static var isEnabled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return handle != nil
    }

    /// Records one finished run. `kind` distinguishes the response path so a
    /// reader can tell a streaming agent turn from a probe's plain
    /// completion without guessing from the shape of the numbers.
    public static func record(_ run: GenerationRun, kind: String) {
        lock.lock()
        defer { lock.unlock() }
        guard let handle else { return }
        let now = Date()
        // A hand-built object rather than JSONEncoder: the field set is
        // fixed, the ordering is stable for eyeballing a tail -f, and every
        // value is already a number or a bare identifier.
        func num(_ v: Double) -> String { String(format: "%.3f", v) }
        let line = """
        {"t":\(num(now.timeIntervalSince1970)),\
        "uptime_s":\(num(now.timeIntervalSince(startedAt))),\
        "kind":"\(kind)",\
        "prompt_tokens":\(run.promptTokenCount),\
        "cached_tokens":\(run.cachedTokenCount),\
        "cache_hit":\(run.cacheHit),\
        "completion_tokens":\(run.completionTokenCount),\
        "prefill_ms":\(num(run.prefillMilliseconds)),\
        "generate_ms":\(num(run.generateMilliseconds)),\
        "wall_ms":\(num(run.wallMilliseconds)),\
        "prompt_tps":\(num(run.promptTokensPerSecond)),\
        "decode_tps":\(num(run.decodeTokensPerSecond)),\
        "tool_calls":\(run.toolCalls.count),\
        "finish":"\(run.finishReason ?? "")",\
        "mem_peak_bytes":\(run.memoryPeakBytes)}

        """
        handle.write(Data(line.utf8))
    }
}
