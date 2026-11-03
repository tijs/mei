# Mei 0.10.0

Mei 0.10 improves status responsiveness, raw completions and bounded cache
behavior. The release notes and attached qualification evidence identify the
exact application, vmlx, MLX core, executable and source-built Metal artifacts.
An unreleased source branch does not establish that those gates have passed.

Status counters no longer wait for the inference actor or an unused disk-quota
calculation. Streaming success headers wait for a visible SSE frame, allowing
an earlier generation error to retain its HTTP error status. Buffered raw
completions use `text_completion` and `choices[].text`, preserve literal
reasoning/tool markers and honor explicit stop strings. Raw `stream: true`
returns an explicit unsupported-request error.

The decode path reads diagnostic flags individually instead of copying the
whole process environment for every decoder layer. Metal buffer-cache admission
keeps returned buffers within the configured allowance and lets recent shapes
displace older entries. These changes preserve model arithmetic and sampling;
CPU lookup savings and allocator regressions do not establish a uniform model
speed improvement.

Optional prefix-cache stores now decline nonempty checkpoints when the kernel
reports warning or critical memory pressure. Normal-pressure byte budgeting and
reclaimed-mmap footprint accounting remain. A log explains the skip; the answer
still completes normally, and a later request may need to prefill again. This
point-in-time signal neither reserves RAM nor guarantees that every later
prefill fits.

Native semantic probes inherit the serving profile's advertised output budget,
including reasoning. Mechanical timing/cache/context probes retain explicit
small limits and establish their declared runtime contracts. Admission,
generated-text parity, task success and actual Peixinho app acceptance are
different claims. Use the [current benchmark summary](https://github.com/tijs/local-model-bench/blob/main/results/AGENT_SUMMARY.md)
for the selected task cohort; historical and partial results keep their original
runtime identities.

Install stable with `brew install tijs/tap/mei`, or upgrade an existing install
with `brew upgrade tijs/tap/mei`, and check `mei --version`. The separately
installed preview formula may trail stable. The bundle contains runtime and
source documentation, not model weights. See [runtime policies](RUNTIME-POLICIES.md)
for explicit memory/cache controls and [model guidance](MODELS.md) for immutable
presets and artifact provenance.
