import measured from "../public/data/mei-0.11.2-20261116/measurements.json" with { type: "json" };

const releaseVersion = "0.11.2";
export const release = {
  version: releaseVersion,
  url: `https://github.com/tijs/mei/releases/tag/v${releaseVersion}`,
  archive: `https://github.com/tijs/mei/releases/download/v${releaseVersion}/mei-${releaseVersion}-macos-arm64.tar.gz`,
};

// Frozen, audited observations. Release links can advance independently.
const measurementSource = "https://github.com/tijs/mei/blob/297a4d6c5f75c68a821ec64c5c975e371c493168/website/public/data";
export const evidence = {
  commit: "297a4d6c5f75c68a821ec64c5c975e371c493168",
  date: measured.date,
  dateLabel: new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" }).format(new Date(`${measured.date}T00:00:00Z`)),
  hardware: measured.hardware,
  runtime: `Mei ${measured.runtime.version}`,
  method: measured.method.benchmarkRelease,
  methodCommit: measured.method.benchmarkCommit,
  seeds: measured.method.seeds,
  source: `${measurementSource}/mei-0.11.2-20261116/report.md`,
  contextSource: `${measurementSource}/mei-0.11.2-20261116/report.md#generation-speed`,
  rawSource: `${measurementSource}/mei-0.11.2-20261116/measurements.json`,
  historicalSource: `${measurementSource}/mei-0.11.0-20261107/results.json`,
  evaluationSource:
    "https://github.com/tijs/local-model-bench/blob/abb0cbd7caa23c4ea41817559cc8dde3b0e0fabf/results/QUEUED_CHALLENGERS_FINAL.md",
};

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
      `We suggest starting with Groot on a 32 GB M1 Max. It passed ${measured.models.groot.household} of 24 assistant tasks and all ${measured.models.groot.python} Python tasks. It also had the lowest measured memory increase of these three models.`,
    preparation:
      "Uses hermitdave’s MLX version. We removed image-processing weights and unused multi-token prediction components, and corrected the settings describing weight precision. The retained language-model weights are unchanged.",
    household: measured.models.groot.household,
    python: measured.models.groot.python,
    taskSeconds: measured.models.groot.taskSeconds,
    wiredGiB: measured.models.groot.wiredGiB,
    weightsGiB: measured.models.groot.weightsGiB,
    shortDecode: measured.models.groot.shortDecode,
    longDecode: measured.models.groot.longDecode,
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
      `Qwen finished the test tasks faster on average than Groot and Ornith. It passed ${measured.models.qwen.household} of 24 assistant tasks and all ${measured.models.qwen.python} Python tasks. Its measured memory increase was higher than Groot’s.`,
    preparation:
      "Uses mlx-community’s 4-bit MLX version with the image-processing weights removed. The text weights, tokenizer and chat template are unchanged. This version accepts text only.",
    household: measured.models.qwen.household,
    python: measured.models.qwen.python,
    taskSeconds: measured.models.qwen.taskSeconds,
    wiredGiB: measured.models.qwen.wiredGiB,
    weightsGiB: measured.models.qwen.weightsGiB,
    shortDecode: measured.models.qwen.shortDecode,
    longDecode: measured.models.qwen.longDecode,
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
      `Ornith passed ${measured.models.ornith.household} of 24 assistant tasks and all ${measured.models.ornith.python} Python tasks. It took longer per task on average than Qwen in this cohort. Mei includes separate settings for running this model.`,
    preparation:
      "Uses Ornith’s 4-bit MLX version. We changed the file padding to align the stored weights. The weight values are unchanged.",
    household: measured.models.ornith.household,
    python: measured.models.ornith.python,
    taskSeconds: measured.models.ornith.taskSeconds,
    wiredGiB: measured.models.ornith.wiredGiB,
    weightsGiB: measured.models.ornith.weightsGiB,
    shortDecode: measured.models.ornith.shortDecode,
    longDecode: measured.models.ornith.longDecode,
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

export const views = ["quality", "time", "memory", "decode"] as const;
export type View = (typeof views)[number];
export const labels: Record<View, string> = {
  quality: "Tasks passed",
  time: "Task time",
  memory: "Memory",
  decode: "Generation speed",
};

export const methods = {
  quality:
    "We tested each model twice. Each run used 12 simulated household-assistant tasks and 3 small Python tasks. Failed attempts count too. The results measure each model with its Mei settings; they do not tell us how well it will work in your application.",
  time: "Average time per task, including failed attempts. We averaged the 15 tasks in each run, then averaged the two runs. Loading the model and running startup checks are excluded. Check the pass counts as well as the time.",
  memory:
    "We measured how much the Mac’s wired memory increased from before each run. Wired memory cannot be moved to swap. Each bar averages the highest increase from the two runs. Other applications can affect this measurement. It is not total model memory and does not tell you how much RAM you need. The model-weight sizes are measured separately; do not add them to these values.",
  decode:
    "This measures how fast the model generates text after reading the prompt. Each result averages 10 responses across two separate server starts. Each response generated 128 tokens by choosing the most likely next token. First requests and warmup requests are excluded. These speeds exclude the time spent reading the prompt and do not measure answer quality at 30,000 tokens.",
};
