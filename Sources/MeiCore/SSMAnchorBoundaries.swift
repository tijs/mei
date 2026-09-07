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

    /// Divergence-based anchor computation (2026-09-07).
    ///
    /// `compute` assumes a prefix-additive renderer: the render of
    /// `messages[0..<i]` must be a token prefix of the full render. Qwen 3.5/3.6
    /// templates violate this at the system boundary — rendering the system
    /// message ALONE (with tools) yields MORE tokens than the full
    /// system+user render (measured 21,834 vs 20,394 on the Hermes prompt), so
    /// the only anchor that matters, the end of the shared system+tools
    /// prefix, was silently dropped and the disk tier never restored it.
    ///
    /// This variant makes no additivity assumption. For each of the first `k`
    /// user messages it renders the FULL template with that message's content
    /// replaced by a sentinel and takes the longest common token prefix with
    /// the real render. Every offset it returns is, by construction, a
    /// position up to which the real token sequence is shared with a prompt
    /// that differs only from that message on — exactly what a later
    /// conversation with the same system prompt looks like.
    public static func computeByDivergence(
        template: [[String: any Sendable]],
        fullTokens: [Int],
        k: Int,
        renderFull: ([[String: any Sendable]]) throws -> [Int]
    ) rethrows -> Result {
        guard k > 0, !template.isEmpty, !fullTokens.isEmpty else {
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
            var variant = template
            variant[idx]["content"] = "\u{1F}mei-anchor-divergence-probe\u{1F} 0123456789"
            let alt = try renderFull(variant)
            let n = min(alt.count, fullTokens.count)
            var lcp = 0
            while lcp < n && alt[lcp] == fullTokens[lcp] { lcp += 1 }
            if lcp > 0 && lcp < fullTokens.count {
                offsets.append(lcp)
            }
        }
        var seen = Set<Int>()
        let sorted = offsets.filter { seen.insert($0).inserted }.sorted()
        return Result(offsets: sorted, warning: nil)
    }
}