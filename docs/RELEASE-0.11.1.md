# Mei 0.11.1

Mei 0.11.1 updates the pinned Apple MLX core and fixes a structured-output
stall that could make a complete JSON answer report `length` instead of
`stop`. Existing server commands and model profiles keep working.

- Structured generation now bounds trailing whitespace after a complete JSON
  value. In the Gemma checks, both streaming and non-streaming requests now
  return a normal stop without spending the remaining budget on padding.
- The runtime pins Apple MLX core through `fc344c98`, updates the attention
  bridge, and regenerates the Metal shaders. Release mode disables graph
  debug checks consistently across the two native C++ targets.
- The staging CLI reads the configured host model store, including external
  volumes, without requiring a symlink.

Qwen, Ornith, Groot, and Gemma passed all 13 native probes and all nine live
Swift API tests each. The 16-process comparison against 0.11.0 on a 32 GB
M1 Max verified identical unconstrained outputs, usage counts and finish
reasons. First-token latency, decode throughput, and total time each changed
by less than 0.5% on the matched short and 30,000-token prompts.
[Qualification evidence](https://github.com/tijs/local-model-bench/tree/main/artifacts/mei-0111-schema-qualification-20261108).

The qualified runtime source was `3c3807a836edf54f22857fdc8eb3047588badebc`.
The tagged source adds website and release documentation without changing
the Swift source or package lock. vMLX is pinned at `e280bd2b87dff614991cb6f9213adaf797eb2db4`,
the MLX core at `b06cc4dc5af50e281b1b6309b2f7b79719fa9427`, and the C bridge at
`1169bf6796e05e5857f8d40af5a66b361df3ae5b`.
