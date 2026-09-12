import Foundation

/// A curated settings bundle for one specific supported model, selected BY NAME.
///
/// The distinction that matters: `ModelOptimizationProfile` detects the
/// ARCHITECTURE from `config.json`'s `model_type` — a declared field that
/// determines topology, and the thing that keeps an unknown model safe. This
/// type is different. It carries the per-MODEL numbers we measured, and it is
/// never inferred, because the models we support are not distinguishable from
/// their metadata: Ornith 1.5 and Qwen3.6 text-only report the identical
/// `model_type`, architecture and layer topology. Guessing between them would
/// mean keying behaviour off incidental fields like `transformers_version` or a
/// config hash, which breaks the moment an upstream re-export changes them.
///
/// So the operator names the model and gets the measured settings. There are
/// only a handful of supported models, which is what makes curation viable.
public struct ModelTuning: Sendable, Equatable {
    /// The name passed to `--model-profile`.
    public let name: String
    /// Architecture family. Carried here rather than exposed as a second flag:
    /// naming the model should select everything about how it is served.
    public let optimizationProfile: ModelOptimizationProfile
    /// Which model this is for, as a human would say it.
    public let model: String
    /// Chunked-prefill step. Different optimum per model; see `provenance`.
    public let prefillStepSize: Int?
    /// Cross-conversation anchor boundaries. 0 disables.
    public let ssmAnchorBoundaries: Int?
    /// Server-side generation cap when a request sends none.
    public let maxTokens: Int?
    /// Why these values. Every number here should be traceable to a measurement.
    public let provenance: String
    /// Settings were measured on a repacked, naturally-aligned copy of the
    /// weights. The published checkpoint is not that copy.
    public let requiresAlignedWeights: Bool

    /// HuggingFace repo these settings were measured on.
    public let repo: String
    /// Pinned revision. Not `main`: the settings are calibrated to a specific
    /// artifact, and an upstream re-export that silently invalidates them is
    /// miserable to debug months later.
    public let revision: String

    public init(name: String, model: String,
                optimizationProfile: ModelOptimizationProfile,
                prefillStepSize: Int?,
                ssmAnchorBoundaries: Int?, maxTokens: Int?, provenance: String,
                repo: String, revision: String,
                requiresAlignedWeights: Bool = false) {
        self.name = name
        self.model = model
        self.optimizationProfile = optimizationProfile
        self.prefillStepSize = prefillStepSize
        self.ssmAnchorBoundaries = ssmAnchorBoundaries
        self.maxTokens = maxTokens
        self.provenance = provenance
        self.repo = repo
        self.revision = revision
        self.requiresAlignedWeights = requiresAlignedWeights
    }
}

public enum ModelTuningRegistry {
    public static let all: [ModelTuning] = [
        ModelTuning(
            name: "ornith-1.5-35b-a3b",
            model: "ornith-ai/Ornith-1.5-35B-A3B (MLX 4-bit)",
            optimizationProfile: .ornith,
            prefillStepSize: nil,   // device-aware; see prefillStepSize(modelDirectory:)
            ssmAnchorBoundaries: 0,
            maxTokens: 8192,
            provenance: """
                Anchors OFF. They cut prefill 4.64 -> 2.82 s/turn, but on this \
                model they also cost hermes_ops-multi-step-chain, reproducibly \
                (0/3 across three run pairs), where the model stops calling \
                search_files/read_file/patch. Prefill step is device-aware \
                (512, or 1024 where the working set allows). max-tokens 8192 \
                bounds a runaway turn that once burned 19.3 of a run's 40.1 \
                generating minutes. The pinned repo is the naturally-aligned
                repack: the published checkpoint is 97% misaligned, which costs
                4.7 GB of memory and 3.2x on an 80k prefill, and these settings
                were measured on the aligned copy.
                """,
            repo: "Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned",
            revision: "ddce5cd6e3d8bc720a5bac5a68c22f406f90403d",
            requiresAlignedWeights: true),
        ModelTuning(
            name: "qwen3.6-35b-a3b-text",
            model: "Tostibrown/Qwen3.6-35B-A3B-4bit-textonly",
            optimizationProfile: .ornith,
            prefillStepSize: 1024,
            ssmAnchorBoundaries: 2,
            maxTokens: 8192,
            provenance: """
                Anchors ON. Prefill 4.60 -> 2.23 s/turn (-52%), and zero task \
                cost across three run pairs on the prompt suite — including \
                hermes_ops-multi-step-chain, which anchors break on Ornith. \
                Same architecture as Ornith, different trajectory.
                """,
            repo: "Tostibrown/Qwen3.6-35B-A3B-4bit-textonly",
            revision: "693d7a0f4d0c1feb97d8e885ceb2c67d3eb98a56"),
        ModelTuning(
            name: "qwen3.6-35b-a3b",
            model: "mlx-community/Qwen3.6-35B-A3B-4bit (with vision tower)",
            optimizationProfile: .ornith,
            prefillStepSize: 1024,
            ssmAnchorBoundaries: 0,
            maxTokens: 8192,
            provenance: """
                Anchors OFF pending measurement: this variant has not been \
                A/B'd. Defaults otherwise follow the text-only sibling.
                """,
            repo: "mlx-community/Qwen3.6-35B-A3B-4bit",
            revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46"),
    ]

    public static func named(_ name: String) -> ModelTuning? {
        let key = name.lowercased()
        return all.first { $0.name == key }
    }

    public static var names: [String] { all.map(\.name) }
}
