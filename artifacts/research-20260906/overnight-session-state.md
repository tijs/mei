# Overnight session state 2026-09-06/07: Nemotron solved, two benchmarks chained, traps and queued next steps

# Overnight session state — 2026-09-06/07 (autonomous, other agent paused)

Running record so any agent can pick up mid-flight. GPU work is serialised
through one chain; nothing else should start a Mei server until it finishes.

## In flight right now

1. **Qwen3.6 text-only full benchmark** — `configs/Qwen3.6-35B-A3B-textonly/mei.yaml`,
   log `/tmp/bench_qwen36_textonly.log`. Progress at time of writing:
   **sanity 2/2 PASS, hermes_ops 6/6 PASS**, now in the coding suites.
   Per-task hermes_ops tok/s 1.84–10.09.
2. **Chained Nemotron re-benchmark** — `/tmp/bench_chain.sh` (pid recorded in
   `/tmp/bench_chain.log`) waits for (1) to exit, runs `unload_all.sh`, then runs
   `configs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B/mei.yaml` through the newly
   wired proxy tool-call recovery. Output `/tmp/bench_nemotron_fixed.log`.
   This is the live validation of tonight's main finding.

## Settled tonight

- **Nemotron root cause** — model drops the `<tool_call>` wrapper under
  Hermes-shaped prompts; vmlx's startTag-gated `XMLFunctionParser` never fires;
  call lands in `content` as text; one turn; no KV reuse; cold 22k prefill every
  task; `completion_tokens/wall_seconds` collapses to 1.02. Decode is ~68–70
  tok/s, the lineup's fastest. Ruled out with live evidence: chat template, tool
  count, thinking mode (`--enable-thinking false` gives byte-identical
  failures), raw prompt length, prompt-level instruction.
- **Fix needed no new code** — `bench_local_proxy.py`'s existing `qwen3_coder`
  parser splits on the bare `<function=` and falls back to whole-text with no
  wrapper. Offline-verified against the real captured forms; correctly returns
  zero calls on the corrupted long-prompt form and on plain prose. Cannot
  clobber Mei's native path.
- **C1b's disappointing +3.7%** — explained mechanically:
  `Qwen35.swift:332` disables the GDN input-projection fusion whenever
  `CompiledDecodeTrace.isActive`, so compiled decode *trades* the fusion for the
  graph-rebuild saving instead of stacking. My earlier +12–17% projection costed
  the saving against a baseline that had already banked the fusion.
- **D1/D3 unblock spec** — MLXPress axis-E is fully built in vmlx; Mei just
  never passes `jangPress`. One argument at `Engine.swift:90`
  (`jangPress: .default`) enables it and makes `MLXPRESS=N` live. **I recorded
  an incorrect "env vars need no code change" shortcut first and corrected it**:
  `LoadConfiguration.init` defaults `jangPress: .disabled`, which never consults
  the env.

## Process traps hit tonight, worth not repeating

- `runner/start_mei_server.sh` **does not forward `--enable-thinking`** (the Mei
  binary supports it). Launch the binary directly for that flag.
- `run_bench.py:487` reads `orch["proxy_parser"]` as a **hard key access**. A
  config with `needs_proxy: true` and no `proxy_parser` raises `KeyError`
  mid-run, after the model has loaded. My first patch put the name at the top
  level and would have crashed exactly that way.
- Naming a scratch script `bisect.py` shadowed stdlib `bisect` and broke
  `urllib` with a circular-import error.
- **The first filler I used for the length sweep was one sentence repeated
  hundreds of times.** It produced a *different* answer from realistic varied
  prose (6,259 degenerate tokens still emitted a tool call; 6,003 varied tokens
  did not), which nearly led me to a wrong "hard length threshold" conclusion.
  Degenerate filler is not a valid stand-in for real context.
- `preserve_repository()` in `run_fixture_suite.py` is armed during coding
  suites and reverts `configs/` edits. Check
  `ps ax -o command | grep -c "[r]un_fixture_suite"` before touching configs.
  Pending edits parked in `/tmp/PENDING_AFTER_BENCH.md`.

## Queued next (in order)

1. Read the Qwen3.6 text-only result; append leaderboard; compare against stock
   Qwen3.6 rows (expect the +23.8% short-decode advantage to show up as better
   end-to-end throughput, and confirm no quality regression from the strip).
2. Read the Nemotron re-run; if hermes_ops moves off 1/8, the fix is validated
   and the speed gate should be re-evaluated on the new rows.
3. Apply the parked config fixes in `/tmp/PENDING_AFTER_BENCH.md`.
4. C2 (`compileSeparatedDecode: true` at `MLXLLM/Models/Qwen35.swift:870`,
   Ornith-only) — needs a fork change, rebuild and re-pin, so it is the largest
   remaining unit and should not start until the GPU chain is clear.
5. D3 via the one-argument Mei change, then an `MLXPRESS=0/70/95` sweep judged
   on resident footprint and peak, not tok/s.

#proj/mei
