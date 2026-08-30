import Foundation

/// Deterministic computation of early structural SSM anchor boundaries
/// (patch 0005 plumbing).
///
/// The pinned engine's SSM companion store keeps only the LARGEST prompt
/// boundaries (prompt end, generation-suffix-stripped end, block boundaries
/// nearest the end) — exactly right for the strict-extension agentic
/// pattern, but a mid-transcript DIVERGING edit (identical turns 1..N, a
/// changed tool/thinking block at turn N+1) falls back to a full prefill
/// because no retained boundary near the transcript start has companion
/// state. Storing anchors at early ROLE-TURN boundaries lets the
/// coordinator restore from the nearest retained boundary instead
/// (TTFT lever, not a decode tok/s lever; see
/// artifacts/design-anchor-ssm-0005.md).
///
/// This file is pure logic — the tokenizer renderer is injected as a
/// closure, so the computation is unit-testable without MLX or Metal. It
/// never touches the GPU and never claims performance; it only computes
/// deterministic offset lists, default [] (upstream behavior).
public enum SSMAnchorBoundaries {
    public struct Result: Sendable, Equatable {
        public var offsets: [Int]
        public var warning: String?
    }

    /// Compute at most `k` early role-boundary offsets into the rendered
    /// prompt:
    ///   - candidate boundaries are the token offsets at which each
    ///     template `user` message STARTS (template dictionaries carry
    ///     "role") — the stable system-prefix anchor plus each turn start;
    ///   - offsets come from a prefix-additive renderer: the token count
    ///     of `messages[0..<i]` must equal the request's own full-render
    ///     token count up to that boundary;
    ///   - self-check: rendering the WHOLE message list must reproduce
    ///     `fullTokenCount` exactly (this is the request's own rendered
    ///     token sequence). On any violation the computation returns []
    ///     with a warning — the current (always-correct) behavior is the
    ///     fallback, never a possibly-misplaced offset.
    ///   - returned offsets are strictly increasing, deduped, and within
    ///     the engine's accepted range (0 < offset < prompt length; the
    ///     engine further filters offset <= length and Set-dedupes).
    public static func compute(
        template: [[String: any Sendable]],
        fullTokenCount: Int,
        k: Int,
        renderPrefixCount: (Int) throws -> Int
    ) rethrows -> Result {
        guard k > 0, !template.isEmpty, fullTokenCount > 0 else {
            return Result(offsets: [], warning: nil)
        }
        let userIndices = template.indices.filter {
            template[$0]["role"] as? String == "user"
        }
        guard !userIndices.isEmpty else {
            return Result(offsets: [], warning: nil)
        }
        var offsets: [Int] = []
        for idx in userIndices.prefix(k) {
            let count = try renderPrefixCount(idx)
            if count > 0 && count < fullTokenCount {
                offsets.append(count)
            }
        }
        // Additivity self-check (the request's own rendering path must be
        // reproducible by the prefix renderer).
        do {
            let full = try renderPrefixCount(template.count)
            if full != fullTokenCount {
                return Result(
                    offsets: [],
                    warning: "chat template is not prefix-additive for this "
                        + "transcript (prefix render \(full) tokens != full "
                        + "\(fullTokenCount)); anchors disabled")
            }
        } catch {
            return Result(
                offsets: [],
                warning: "prefix render failed; anchors disabled")
        }
        var seen = Set<Int>()
        let sorted = offsets.filter { seen.insert($0).inserted }.sorted()
        return Result(offsets: sorted, warning: nil)
    }
}