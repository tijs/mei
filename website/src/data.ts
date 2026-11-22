const releaseVersion = "0.14.0";
export const release = {
  version: releaseVersion,
  url: `https://github.com/tijs/mei/releases/tag/v${releaseVersion}`,
  archive: `https://github.com/tijs/mei/releases/download/v${releaseVersion}/mei-${releaseVersion}-macos-arm64.tar.gz`,
};

export const inkling = {
  guide: "https://github.com/tijs/mei/blob/v0.14.0/docs/INKLING.md",
  repo: "nativ-community/Inkling-2-Mini-nvfp4",
  revision: "26f71eef259a1dd7fc476778d67020351e0dfb8c",
};

export const inklingCommands = `hf download ${inkling.repo} \\
  --revision ${inkling.revision} \\
  --exclude 'mtp.safetensors' --local-dir ./inkling-2-mini
mei --model-dir ./inkling-2-mini`;

export const models = [
  {
    id: "groot",
    name: "Groot",
    modelCard: "https://huggingface.co/GrootTeam/Groot-19B-A3B-261023",
    fullName: "19B parameters · text-only · 4.5 bits/weight",
    profile: "groot-19b-a3b-text",
    automaticSettings:
      "Mei automatically loads Groot’s measured settings: 256-token prefill, conversation anchors off, stock decode kernels and eligible blocked prefill. No kernel flags are needed.",
    repo: "Tostibrown/Groot-19B-A3B-261023-rco-role-4.5bpw-textonly",
    revision: "53f52e0eb8d67d2941010688b2a6b8807d9a3bea",
    description:
      "Our suggested first model for a 32 GB M1 Max, with an automatic serving profile for local assistants and tool calls.",
    preparation:
      "Uses hermitdave’s MLX version. We removed image-processing weights and unused multi-token prediction components, and corrected the settings describing weight precision. The retained language-model weights are unchanged.",
  },
  {
    id: "qwen",
    name: "Qwen3.6",
    modelCard: "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
    fullName: "35B parameters · text-only · 4-bit weights",
    profile: "qwen3.6-35b-a3b-text",
    automaticSettings:
      "Mei automatically loads Qwen’s measured settings: 1024-token prefill on a fitting Mac and two conversation anchors. Smaller devices use the documented prefill fallback.",
    repo: "Tostibrown/Qwen3.6-35B-A3B-4bit-textonly",
    revision: "693d7a0f4d0c1feb97d8e885ceb2c67d3eb98a56",
    description:
      "A 35B mixture-of-experts text model with 4-bit weights and an automatic serving profile.",
    preparation:
      "Uses mlx-community’s 4-bit MLX version with the image-processing weights removed. The text weights, tokenizer and chat template are unchanged. This version accepts text only.",
  },
  {
    id: "ornith",
    name: "Ornith 1.5",
    modelCard: "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B",
    fullName: "35B parameters · MLX · 4-bit weights",
    profile: "ornith-1.5-35b-a3b",
    automaticSettings:
      "Mei automatically loads Ornith’s measured settings: 1024-token prefill on a fitting Mac and conversation anchors off. Smaller devices use the documented prefill fallback.",
    repo: "Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned",
    revision: "ddce5cd6e3d8bc720a5bac5a68c22f406f90403d",
    description:
      "Ornith’s 35B mixture-of-experts model, with aligned 4-bit MLX weights and an automatic serving profile.",
    preparation:
      "Uses Ornith’s 4-bit MLX version. We changed the file padding to align the stored weights. The weight values are unchanged.",
  },
] as const;

export function modelCommands(model: (typeof models)[number]): string {
  return [
    `mei pull ${model.profile}`,
    "",
    "mei \\",
    `  --model-dir ~/.cache/mei/models/${model.repo.split("/").at(-1)} \\`,
    `  --served-model-id ${model.id}`,
  ].join("\n");
}
