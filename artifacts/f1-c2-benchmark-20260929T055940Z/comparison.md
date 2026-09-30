# F1/C2 current-pin benchmark

Date: 2026-09-30
Binary: `.build/release/mei`
Binary SHA-256: `aefaf68c7fc4a0836f63b37a9c61529965c49fc35a44a7a704ec2477247f8d6c`
Server settings: context cap 65536, prefill step 1024, temperature 0, top-p 1, top-k 1, cache reuse false, max tokens 32 for timing rows.

Each arm used a fresh Mei process and fresh request log. Each text target has 10 short-context and 10 30,000-token timing requests. The real `probe_mei.py` gate passed for every arm, including model identity, plain completion, stream/non-stream parity, and native tool calls. Every arm stopped its server cleanly. All 100 timing rows are present and parseable.

## Policy engagement

- Ornith candidate: both plan switches were `0 (source mei-profile)`; the opt-out line was present; `Qwen4Exp compiled_routed_switch_glu` did not engage.
- Ornith control: both switches were `1 (source operator)`; the opt-out was skipped; `Qwen4Exp compiled_routed_switch_glu=active` was logged.
- Qwen3.6 text-only candidate: both switches were `0 (source mei-profile)`; the opt-out line was present; the routed switch GLU did not engage.
- Qwen3.6 text-only control: both switches were `1 (source operator)`; the opt-out was skipped; `Qwen4Exp compiled_routed_switch_glu=active` was logged.
- Qwen3.6 vision candidate: both switches remained `unset (source upstream-default)`; no opt-out line was emitted. The bundle's vision metadata was loaded and the real text/tool probe passed.

## Results

The percentages below are candidate (opt-out) versus control (compiled routed-MoE region forced on); positive means the opt-out was faster.

| Model | Context | Candidate decode tok/s | Control decode tok/s | Decode delta | Candidate wall s | Control wall s | Wall delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ornith 1.5 35B | short (13 prompt) | 56.2287 | 55.3267 | +1.630% | 0.701531 | 0.716545 | -2.095% |
| Ornith 1.5 35B | loaded (30,000 prompt) | 46.2794 | 46.1649 | +0.248% | 58.352454 | 58.362613 | -0.017% |
| Qwen3.6 35B text-only | short (13 prompt) | 57.0381 | 57.4347 | -0.691% | 0.698930 | 0.689523 | +1.364% |
| Qwen3.6 35B text-only | loaded (30,000 prompt) | 44.9533 | 44.7964 | +0.350% | 58.734434 | 58.675094 | +0.101% |

All candidate/control timing rows returned 32 completion tokens. The short-context output SHA-256 was `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` and the loaded-context output SHA-256 was `4e0b7037f631632d09cb9fb1809568ad17bf38c859bb80ac58e7e50cf0e0a47f` for every text arm. No paired row differed in output hash or token counts. All native tool-call probes returned `add_numbers(a=15,b=27)` in both streaming and non-streaming forms.

## Verdict

The policy is engaged exactly as intended and is correctness-safe in this campaign, but the prior C2 performance theory is **not reproduced on this current Mei/vmlx pin**. The expected roughly 5–10% opt-out win is absent: Ornith shows only a small short-context improvement and a neutral loaded-context result, while text-only Qwen3.6 is slightly slower short-context and only fractionally faster at 30k. The benchmark supports keeping the change as a fail-closed, reversible policy with operator control, but it does not support claiming a material performance improvement from the opt-out.

The candidate and control arms were run sequentially (candidate before control for each text model), not interleaved round-robin; therefore the tiny sub-percent differences should be treated as neutral/measurement-level, not as a precise estimate of a small effect. The large planned gain is nevertheless not present in these 10-repeat legs.

Raw evidence is in this directory: each arm's `result.json`, `request.jsonl`, `server.log`, `probe.json`, and runner log. The vision run is a no-op-policy control rather than a forced-on comparison, because forcing the two environment switches on a vision bundle would intentionally override the policy under test.
