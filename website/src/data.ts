// Frozen, published evidence. Change this snapshot deliberately, not on each build.
export const evidence = {
  commit: 'abb0cbd7caa23c4ea41817559cc8dde3b0e0fabf',
  date: '2026-11-07',
  hardware: 'M1 Max · 32 GB unified memory',
  runtime: 'Mei 0.11.0',
  method: 'benchmark-v9.11.5',
  seeds: [17, 29],
  source: 'https://github.com/tijs/local-model-bench/blob/abb0cbd7caa23c4ea41817559cc8dde3b0e0fabf/results/FOUR_MODEL_MEI011_FINAL.md',
  contextSource: 'https://github.com/tijs/local-model-bench/blob/abb0cbd7caa23c4ea41817559cc8dde3b0e0fabf/results/CONTEXT_SPEED_MEI011.md',
  evaluationSource: 'https://github.com/tijs/local-model-bench/blob/abb0cbd7caa23c4ea41817559cc8dde3b0e0fabf/results/QUEUED_CHALLENGERS_FINAL.md',
};

export const models = [
  { id: 'groot', name: 'Groot', fullName: '19B · A3B · text-only · 4.5 bpw', profile: 'groot-19b-a3b-text',
    repo: 'Tostibrown/Groot-19B-A3B-261023-rco-role-4.5bpw-textonly', revision: '53f52e0eb8d67d2941010688b2a6b8807d9a3bea',
    label: 'A good place to start', description: 'The strongest assistant-workflow result of our three tuned profiles in this test, with the lowest measured memory increase. A useful starting point for local assistants and light coding.',
    household: 22, python: 6, taskSeconds: 38.90, wiredGiB: 13.72, weightsGiB: 11.45, shortDecode: 55.6, longDecode: 44.5 },
  { id: 'qwen', name: 'Qwen3.6', fullName: '35B · A3B · text-only · 4-bit', profile: 'qwen3.6-35b-a3b-text',
    repo: 'Tostibrown/Qwen3.6-35B-A3B-4bit-textonly', revision: '693d7a0f4d0c1feb97d8e885ceb2c67d3eb98a56',
    label: 'When task time matters', description: 'The shortest average task time in this comparison. It passed all six Python tasks, with fewer assistant-workflow passes than Groot. Budget more memory for this profile.',
    household: 18, python: 6, taskSeconds: 29.44, wiredGiB: 20.34, weightsGiB: 18.17, shortDecode: 67.2, longDecode: 50.8 },
  { id: 'ornith', name: 'Ornith 1.5', fullName: '35B · A3B · MLX · 4-bit', profile: 'ornith-1.5-35b-a3b',
    repo: 'Tostibrown/Ornith-1.5-35B-A3B-MLX-4bit-aligned', revision: 'ddce5cd6e3d8bc720a5bac5a68c22f406f90403d',
    label: 'Another tuned option', description: 'A separately tuned assistant and tool-calling profile. It passed one more assistant workflow than Qwen, but fewer Python tasks and took longer on average in this test.',
    household: 19, python: 4, taskSeconds: 44.24, wiredGiB: 19.86, weightsGiB: 18.17, shortDecode: 67.2, longDecode: 50.7 },
] as const;

export const views = ['quality', 'time', 'memory', 'decode'] as const;
export type View = typeof views[number];
export const labels: Record<View, string> = { quality: 'Task success', time: 'Task time', memory: 'Memory', decode: 'Decode speed' };

export const methods = {
  quality: 'Two runs per model: 12 synthetic household-assistant workflows and 3 isolated Python tasks per run. All attempts count, including failures. These are serving-profile results, not production application acceptance.',
  time: 'Mean wall time across all 15 tasks per run, then equal weight for the two runs. Includes failed attempts; excludes model startup and admission. Read alongside task success: a fast failure is still a failure.',
  memory: 'Average of two per-run peak system-wide wired-memory increases over a pre-run baseline. This is partial accounting, affected by background activity—not total model memory or a RAM requirement. Weight-storage size is a separate measurement and must not be added to these bars.',
  decode: 'Each prompt length averages two fresh processes with five measured greedy 128-token decodes each. Cold requests and warmups are excluded. Decode speed excludes prefill and is not whole-request latency. This does not measure answer quality at 30,000 tokens.',
};
