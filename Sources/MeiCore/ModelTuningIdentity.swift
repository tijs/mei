import CryptoKit
import Foundation

/// Recognizes metadata from the exact curated model revisions. This selects
/// settings; it is not a substitute for verifying downloaded weight payloads.
/// A renamed directory is fine. A re-export with different metadata stays on
/// architecture defaults until its settings are separately qualified.
enum ModelTuningIdentity {
    struct Fingerprint: Sendable {
        let profile: String
        let revision: String
        let hashes: [String: String]
    }

    static let files = ["config.json", "generation_config.json", "tokenizer_config.json"]
    private static let generationHash = "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"

    // Metadata from the artifact revisions in ModelTuningRegistry, independently
    // preserved with the release fixtures. Never match model_type or a path name.
    static let fingerprints: [Fingerprint] = [
        Fingerprint(profile: "ornith-1.5-35b-a3b",
            revision: "ddce5cd6e3d8bc720a5bac5a68c22f406f90403d", hashes: [
                "config.json": "fc27b922c3506e955271c68120be9079ad8bba7266c8086c02dfbb8c98723bd5",
                "generation_config.json": generationHash,
                "tokenizer_config.json": "386ba246ba3dfa4d3a21d8bc8712eba38de558e5ac94c2c90436d8f94d519d5e",
            ]),
        Fingerprint(profile: "qwen3.6-35b-a3b-text",
            revision: "693d7a0f4d0c1feb97d8e885ceb2c67d3eb98a56", hashes: [
                "config.json": "ac8c9c4a36ce2205ada3cf938002166529403475f830e4366c818c18569f78b0",
                "generation_config.json": generationHash,
                "tokenizer_config.json": "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182",
            ]),
        Fingerprint(profile: "qwen3.6-35b-a3b",
            revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46", hashes: [
                "config.json": "a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767",
                "generation_config.json": generationHash,
                "tokenizer_config.json": "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182",
            ]),
        Fingerprint(profile: "groot-19b-a3b-text",
            revision: "53f52e0eb8d67d2941010688b2a6b8807d9a3bea", hashes: [
                "config.json": "b8752d33e18bde028b4af79b29b24242df1bb9b33a8772b357d0262e8097bc58",
                "generation_config.json": generationHash,
                "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
            ]),
    ]

    static func detect(modelDirectory: String,
                       candidates: [Fingerprint] = fingerprints) -> ModelTuning? {
        var hashes = [String: String]()
        for name in files {
            let url = URL(fileURLWithPath: modelDirectory).appendingPathComponent(name)
            // Bound startup reads to small model metadata, never tokenizers or
            // weight shards. Reject malformed JSON even in a test/custom table.
            guard let data = boundedMetadata(at: url),
                  let json = try? JSONSerialization.jsonObject(with: data),
                  json is [String: Any]
            else { return nil }
            hashes[name] = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        let matches = candidates.compactMap { candidate -> ModelTuning? in
            guard candidate.hashes == hashes,
                  let tuning = ModelTuningRegistry.named(candidate.profile),
                  tuning.revision == candidate.revision
            else { return nil }
            return tuning
        }
        // An accidental duplicate/ambiguous registry entry must not silently
        // choose whichever profile was listed first.
        return matches.count == 1 ? matches[0] : nil
    }

    private static func boundedMetadata(at url: URL) -> Data? {
        guard let file = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? file.close() }
        guard let data = try? file.read(upToCount: 1_048_577), data.count <= 1_048_576 else {
            return nil
        }
        return data
    }
}

extension ModelTuning {
    public static let retainedGrootKernelSwitches = [
        "MEI_EXPERIMENTAL_ROUTER_TAIL",
        "MEI_EXPERIMENTAL_GDN",
        "MEI_EXPERIMENTAL_BLOCKED_GDN",
    ]

    /// Stock decode plus eligible blocked prefill, selected with the model.
    /// The precise router/GDN decode combination regressed short requests under
    /// normal compilation defaults. Small prefill chunks use the stock fallback.
    /// Explicit low-level
    /// operator settings stand down the whole automatic group, not just one
    /// member. Keep the existing unsafe/micro-fusion policy unchanged.
    public func retainedKernelEnvironment(prefillStepSize: Int, compiledDecode: Bool,
                                          environment: [String: String]) -> [String: String] {
        guard let groot = ModelTuningRegistry.named("groot-19b-a3b-text"), self == groot,
              [256, 4096].contains(prefillStepSize), !compiledDecode,
              Self.retainedGrootKernelSwitches.allSatisfy({ environment[$0] == nil }),
              ModelOptimizationProfile.compiledRoutedMoEDecodeOptOutSwitches.keys.allSatisfy({
                  environment[$0] == nil || environment[$0] == "0"
              })
        else { return [:] }
        return [
            "MEI_EXPERIMENTAL_ROUTER_TAIL": "0",
            "MEI_EXPERIMENTAL_GDN": "0",
            "MEI_EXPERIMENTAL_BLOCKED_GDN": "1",
        ]
    }
}
