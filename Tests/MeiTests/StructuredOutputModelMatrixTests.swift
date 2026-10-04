import MLX
import MLXHuggingFace
import MLXLMCommon
import VMLXTokenizers
import XCTest

@testable import MeiCore

/// Real-tokenizer regressions for the structured-output model matrix.
///
/// These tests load the staged tokenizer.json of the exact checkpoints whose
/// strict canary failed live on Mei 0.7.0 (`MeiAcceptanceTests` RED on
/// 2026-10-04) and drive the production construction path
/// (`TokenizerFragmentTable` + `StructuredGeneration.plan`) with the real
/// vocabulary — no model weights, no server, no GPU work. When a checkpoint
/// is not staged locally the test skips, so ordinary runs stay model-free.
///
/// The live failure modes pinned here:
/// - Ornith aligned emitted `{"s\u00e` and dead-ended (`noLegalContinuation`);
/// - Qwen3.6 text-only and vision emitted whitespace for the whole 64-token
///   budget and never started the value (`incompleteAtStop`).
final class StructuredOutputModelMatrixTests: XCTestCase {

    // MARK: - Staged checkpoints

    struct StagedModel {
        let label: String
        let directory: String
        let vocabularySize: Int
        let endOfSequenceTokenIds: Set<Int>
    }

    /// The three staged matrix checkpoints, with the vocabulary width and EOS
    /// ids their `config.json` declares (the production Engine reads the same
    /// values). Paths mirror the Kiem model-provenance records; a missing root
    /// skips, never fails.
    static let stagedModels: [StagedModel] = [
        StagedModel(
            label: "qwen36-text-only",
            directory: "/Users/tijs/.cache/mei/models/Qwen3.6-35B-A3B-4bit-textonly",
            vocabularySize: 248_320,
            endOfSequenceTokenIds: [248_046, 248_044]),
        StagedModel(
            label: "qwen36-vision",
            directory: "/Users/tijs/.local/share/local-model-bench/mei-models/Qwen3.6-35B-A3B-4bit",
            vocabularySize: 248_320,
            endOfSequenceTokenIds: [248_046, 248_044]),
        StagedModel(
            label: "ornith-aligned",
            directory: "/Users/tijs/.local/share/local-model-bench/mei-models/Ornith-1.5-35B-A3B-MLX-4bit-aligned",
            vocabularySize: 248_320,
            endOfSequenceTokenIds: [248_046, 248_044]),
    ]

    private static var availableModels: [StagedModel] {
        stagedModels.filter {
            FileManager.default.fileExists(atPath: $0.directory + "/tokenizer.json")
        }
    }

    // MARK: - Helpers

    /// The exact CoCore canary request's `response_format`, decoded through the
    /// same `ChatRequest` DTO the live server uses.
    private func canaryFormat() throws -> ResponseFormat {
        try ChatRequest(json: CoCoreCanary.structuredOutputBodyData(model: "matrix"))
            .responseFormat
    }

    private func loadTable(for model: StagedModel) async throws -> TokenizerFragmentTable {
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: URL(fileURLWithPath: model.directory))
        return try TokenizerFragmentTable(
            tokenizer: tokenizer, vocabularySize: model.vocabularySize,
            additionalEndOfSequenceTokenIds: model.endOfSequenceTokenIds)
    }

    /// First token id whose fragment equals `fragment` (the fragment table is
    /// immutable and id-stable, so this is deterministic).
    private func firstToken(
        in table: TokenizerFragmentTable, fragment: String
    ) -> Int? {
        let bytes = Array(fragment.utf8)
        for id in 0..<table.vocabularySize
        where table.fragment(forTokenId: id) == bytes {
            return id
        }
        return nil
    }

    private func isWhitespaceOnly(_ fragment: [UInt8]?) -> Bool {
        guard let fragment, !fragment.isEmpty else { return false }
        return fragment.allSatisfy { $0 == 0x20 || $0 == 0x09 || $0 == 0x0A || $0 == 0x0D }
    }

    /// Allowed ids for the current processor state, read through the same
    /// mask the vmlx `LogitProcessor` seam consumes.
    private func allowedIds(
        _ processor: JSONGrammarLogitProcessor, table: TokenizerFragmentTable
    ) -> [Int] {
        let masked = processor.process(
            logits: MLXArray(Array(repeating: Float(0), count: table.vocabularySize)))
        let row = masked.asArray(Float.self)
        return row.indices.filter { row[$0] != -Float.infinity }
    }

    // MARK: - Canonical path and live escape path

    func testStagedCheckpointsDriveTheCanaryAndTheLiveEscapePathCannotDeadEnd() async throws {
        let models = Self.availableModels
        guard !models.isEmpty else {
            throw XCTSkip("no staged structured-output matrix model roots are present")
        }
        let format = try canaryFormat()

        for model in models {
            let table = try await loadTable(for: model)

            // 1) The canonical canary document must be spellable, admissible,
            //    and complete — the tokenizer/grammar/runtime contract.
            let canonicalPlan = try XCTUnwrap(
                try StructuredGeneration.plan(for: format, table: table),
                "\(model.label): the canary format must compile")
            var canonical = canonicalPlan.processor
            for fragment in ["{\"", "status", "\":\"", "ok", "\"}"] {
                let id = try XCTUnwrap(
                    firstToken(in: table, fragment: fragment),
                    "\(model.label): no token spells \(fragment)")
                XCTAssertTrue(
                    canonical.isAllowed(tokenId: id),
                    "\(model.label): canonical fragment \(fragment) must be allowed")
                canonical.didSample(token: MLXArray([Int32(id)]))
            }
            XCTAssertTrue(
                canonicalPlan.runRecord.rootValueCompleted,
                "\(model.label): the canonical path must complete the grammar")
            XCTAssertNil(canonicalPlan.runRecord.failure, "\(model.label)")

            // 2) The live Ornith escape prefix `{"s\u00` was admissible; the
            //    next live nibble `e` dead-ended the mask. It must now be
            //    masked, and the mask must stay non-empty at every step.
            let escapePlan = try XCTUnwrap(try StructuredGeneration.plan(for: format, table: table))
            var escape = escapePlan.processor
            for fragment in ["{", "\"", "s", "\\", "u", "0", "0"] {
                let id = try XCTUnwrap(
                    firstToken(in: table, fragment: fragment),
                    "\(model.label): no token spells \(fragment)")
                XCTAssertTrue(
                    escape.isAllowed(tokenId: id),
                    "\(model.label): live prefix fragment \(fragment) must be allowed")
                escape.didSample(token: MLXArray([Int32(id)]))
            }
            XCTAssertFalse(
                allowedIds(escape, table: table).isEmpty,
                "\(model.label): the mask must never be empty on the live escape prefix")
            let eId = try XCTUnwrap(
                firstToken(in: table, fragment: "e"), "\(model.label): no token spells `e`")
            XCTAssertFalse(
                escape.isAllowed(tokenId: eId),
                "\(model.label): the live `\\u00e` nibble must be masked — no completion can continue `status`")
            let sevenId = try XCTUnwrap(
                firstToken(in: table, fragment: "7"), "\(model.label): no token spells `7`")
            XCTAssertTrue(
                escape.isAllowed(tokenId: sevenId),
                "\(model.label): `\\u007…` stays admissible on the way to `t`")

            // 3) A sampler that ignores the mask must fail closed instead of
            //    walking into the dead-end state.
            let poisonPlan = try XCTUnwrap(try StructuredGeneration.plan(for: format, table: table))
            var poisoned = poisonPlan.processor
            for fragment in ["{", "\"", "s", "\\", "u", "0", "0"] {
                let id = try XCTUnwrap(firstToken(in: table, fragment: fragment))
                poisoned.didSample(token: MLXArray([Int32(id)]))
            }
            poisoned.didSample(token: MLXArray([Int32(eId)]))
            XCTAssertEqual(
                poisonPlan.runRecord.failure, .illegalToken(eId),
                "\(model.label): the dead nibble must be rejected as an illegal token")
        }
    }

    // MARK: - Whitespace stall

    func testStagedCheckpointsBoundWhitespaceRunsAndCompleteTheCanary() async throws {
        let models = Self.availableModels
        guard !models.isEmpty else {
            throw XCTSkip("no staged structured-output matrix model roots are present")
        }
        let format = try canaryFormat()

        for model in models {
            let table = try await loadTable(for: model)
            let whitespaceIds = (0..<table.vocabularySize).filter {
                isWhitespaceOnly(table.fragment(forTokenId: $0))
            }
            XCTAssertFalse(whitespaceIds.isEmpty, "\(model.label): whitespace tokens must exist")

            // The live Qwen3.6 root state: whitespace was admissible, the
            // model preferred it, and it padded the whole budget. One
            // whitespace token stays legal; a second consecutive one may not.
            let rootPlan = try XCTUnwrap(try StructuredGeneration.plan(for: format, table: table))
            var root = rootPlan.processor
            let rootAllowed = allowedIds(root, table: table)
            let firstWhitespace = try XCTUnwrap(
                whitespaceIds.first { rootAllowed.contains($0) },
                "\(model.label): one whitespace token must stay legal before the value")
            root.didSample(token: MLXArray([Int32(firstWhitespace)]))
            let afterWhitespace = allowedIds(root, table: table)
            XCTAssertFalse(
                afterWhitespace.contains { whitespaceIds.contains($0) },
                "\(model.label): a second consecutive whitespace-only token must be masked")
            XCTAssertTrue(
                afterWhitespace.contains {
                    table.fragment(forTokenId: $0)?.first == UInt8(ascii: "{")
                },
                "\(model.label): a `{`-leading token must remain available after the whitespace")

            // A whitespace-preferring walk must complete the exact canary
            // within the 64-token budget the CoCore canary requests.
            let walkPlan = try XCTUnwrap(try StructuredGeneration.plan(for: format, table: table))
            var walk = walkPlan.processor
            var bytes: [UInt8] = []
            var steps = 0
            while steps < 64, !walkPlan.runRecord.rootValueCompleted {
                let allowed = allowedIds(walk, table: table)
                guard
                    let pick = whitespaceIds.first(where: { allowed.contains($0) })
                        ?? allowed.first
                else {
                    return XCTFail("\(model.label): the mask must never be empty")
                }
                walk.didSample(token: MLXArray([Int32(pick)]))
                bytes += table.fragment(forTokenId: pick) ?? []
                steps += 1
            }
            XCTAssertTrue(
                walkPlan.runRecord.rootValueCompleted,
                "\(model.label): a whitespace-preferring walk must still complete within 64 tokens")
            XCTAssertTrue(
                CoCoreCanary.structuredOutputPassed(content: String(decoding: bytes, as: UTF8.self)),
                "\(model.label): the walk must emit the canary document, got \(String(decoding: bytes, as: UTF8.self))")
            XCTAssertNil(
                StructuredGeneration.postGenerationError(
                    plan: walkPlan, stopReason: .stop, toolCallCount: 0),
                "\(model.label)")
        }
    }
}
