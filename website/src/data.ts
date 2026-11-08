// Frozen, published evidence. Change this snapshot deliberately, not on each build.
export const evidence = {
  commit: "abb0cbd7caa23c4ea41817559cc8dde3b0e0fabf",
  date: "2026-11-07",
  hardware: "M1 Max · 32 GB unified memory",
  runtime: "Mei 0.11.0",
  method: "benchmark-v9.11.5",
  seeds: [17, 29],
  source:
    "https://github.com/tijs/local-model-bench/blob/abb0cbd7caa23c4ea41817559cc8dde3b0e0fabf/results/FOUR_MODEL_MEI011_FINAL.md",
  contextSource:
    "https://github.com/tijs/local-model-bench/blob/abb0cbd7caa23c4ea41817559cc8dde3b0e0fabf/results/CONTEXT_SPEED_MEI011.md",
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
    repo: "Tostibrown/Groot-19B-A3B-261023-rco-role-4.5bpw-textonly",
    revision: "53f52e0eb8d67d2941010688b2a6b8807d9a3bea",
    description:
      "We suggest starting with Groot on a 32 GB M1 Max. It passed 22 of 24 assistant tasks and all 6 Python tasks. It also had the lowest measured memory increase of these three models.",
    household: 22,
    python: 6,
    taskSeconds: 38.9,
    wiredGiB: 13.72,
    weightsGiB: 11.45,
    shortDecode: 55.6,
    longDecode: 44.5,
  },
  {
    id: "qwen",
    name: "Qwen3.6",
    modelCard: "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
    fullName: "35B parameters · text-only · 4-bit weights",
    profile: "qwen3.6-35b-a3b-text",
    repo: "Tostibrown/Qwen3.6-35B-A3B-4bit-textonly",
    revision: "693d7a0f4d0c1feb97d8e885ceb2c67d3eb98a56",
    description:
      "Qwen finished the test tasks faster on average than Groot and Ornith. It passed 18 of 24 assistant tasks and all 6 Python tasks. Its measured memory increase was higher than Groot’s.",
    household: 18,
    python: 6,
    taskSeconds: 29.44,
    wiredGiB: 20.34,
    weightsGiB: 18.17,
    shortDecode: 67.2,
    longDecode: 50.8,
  },
  {
    id: "ornith",
    name: "Ornith 1.5",
    modelCard: "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B",
    fullName: "35B parameters · MLX · 4-bit weights",
    profile: "ornith-1.5-35b-a3b",
    repo: "Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned",
    revision: "ddce5cd6e3d8bc720a5bac5a68c22f406f90403d",
    description:
      "Ornith passed 19 of 24 assistant tasks and 4 of 6 Python tasks. It took longer per task on average than Qwen and Groot. Mei includes separate settings for running this model.",
    household: 19,
    python: 4,
    taskSeconds: 44.24,
    wiredGiB: 19.86,
    weightsGiB: 18.17,
    shortDecode: 67.2,
    longDecode: 50.7,
  },
] as const;

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
