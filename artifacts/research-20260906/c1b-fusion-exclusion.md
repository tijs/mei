# Why C1b (compiled decode) underperformed: it is mutually exclusive with the GDN fusion, by design (2026-09-06)

Code-level explanation for the C1b result (note "C1b compiled decode — failed
correctness gate"), which measured compiled decode engaging correctly yet
delivering only **+3.7% at 500 tokens / +4.6% at 1000** — far below the +12–17%
I had projected from the graph-rebuild arithmetic. No GPU used; this is source
reading against the pinned fork.

## The mechanism

`Libraries/MLXLLM/Models/Qwen35.swift:332`:

```swift
private func fusedDecodeInputs(_ inputs: MLXArray) -> [MLXArray]? {
    guard !CompiledDecodeTrace.isActive,
        let segments = ensureFusedInputSegments()
    else { return nil }
```

**When the compiled-decode trace is active, the GDN input-projection fusion is
switched off.** The rationale is in the comment at :234 — *"the compiled decode
trace must not capture the lazily built fused arrays."*

So enabling compiled decode does not simply add graph-replay on top of the
existing fast path. It **trades one optimization for another**:

- **gains** ~3.3 ms/token of CPU graph rebuild (`compiled_forward` 1.28–1.46 ms
  vs eager `model_forward` 4.693 ms)
- **loses** the fused GDN input projections — 4 quantized matmuls collapsed into
  1, across all **30** GDN layers (the `groups=[4]` log line)

The C1b note's own observation matches exactly: *"Compiled assembly rejected the
fused GDN tail and did not activate fused GDN input projections or the compiled
MoE router, while eager did."* That was recorded as a hypothesis; this is the
source line that makes it a fact.

**This is why my +12–17% projection was wrong.** I costed the graph-rebuild
saving in isolation, against a baseline that had already banked the GDN fusion,
without checking whether the two could coexist. They cannot.

## The guard looks more conservative than it needs to be

`flushRun()` already calls `MLX.eval(weight, scales, biases)` on the
concatenated arrays, so by the time any second forward pass runs they are fully
materialised, not lazy. The stated hazard — a trace capturing *lazily built*
arrays — applies to the **first** forward pass, not to a process that has
already done one.

A concrete, testable change for anyone revisiting this: **force the fusion to
materialise during warm-up, before `setupCompiledDecode` traces**, then drop the
`!CompiledDecodeTrace.isActive` condition. If that holds, compiled decode would
keep the GDN fusion and the ~3.3 ms graph-rebuild saving would land on top
instead of being cancelled out.

## But performance is not the blocker

C1b failed on **correctness**, not speed: reasoning output diverged on every
long repeat, client re-encoded token IDs differed, and 1000-token visible
content diverged (first divergence around reasoning position 988). Peak memory
also rose 21.17 → 23.45 GB.

So the fusion interaction explains the *disappointing speed*, and would be worth
fixing if compiled decode were otherwise sound — but it does not explain or
excuse the divergence. **Compiled decode stays disabled.** Any future attempt
should treat these as two separate defects: the mutual exclusion above, and an
unexplained numerical divergence that must be root-caused before the path is
trusted at all.

## Wider lesson for this optimization programme

Two of the levers in this plan (`ensureFusedGateUp`, `ensureFusedInputSegments`)
build lazily-materialised fused weight banks, and a third (compiled decode)
records a graph. These interact:

- `VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES=0` disables the MoE gate+up fusion for
  memory reasons — and therefore also removes it as a compile-trace hazard.
- `CompiledDecodeTrace.isActive` disables the GDN fusion.

**No lever in this family should be costed in isolation again.** Before
projecting a stacked gain, check the guards: `grep` for
`CompiledDecodeTrace.isActive` and for the env gates, and confirm the levers can
actually be active at the same time. My +12–17% projection is the counter-example.

#proj/mei
