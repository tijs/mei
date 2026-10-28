# Qwen startup allocation and memory-reporting fixes

VMLX5b2ffe15 tried the VLM factory first for a text-only Qwen bundle. It built
and read language weights before awaiting mandatory processor metadata; missing
sidecars then triggered LLM fallback and repeated weight loading. Logs showed
two dtype-materialization passes. This is unnecessary startup work; it does not
prove doubled steady-state residency or explain every warning-pressure event.

VMLXf6510fc384bcddb5db9efc96e1dd5ab3afc49346 checks registry eligibility and
text-only Qwen tower declaration first, then reads/validates processor metadata
and registration before any model creation or weight IO. Valid vision metadata
keeps its normal path. Text-only refusal is unsupported-model classification,
so it cannot mask a later real LLM weight/configuration error as a processor-file
error. No weight, sampler, decoder or math kernel is changed.

Mei tests exercise the actual public VLM factory entry point with model/tokenizer
allocation probes. On5b2ffe15:8 tests,5 expected failures reaching model creation
too early. Onf6510fc3:all8 pass, including valid vision and unsupported-model
controls. With existing policy/config tests:52/52. Full offline suite before
peak-counter fix:442 tests,2 opt-in skips,0 failures. No live endpoint implied.

A separate Mei reporting bug reset peakMemory before capturing loadMemory.
Capture now precedes the inference-epoch reset; a synthetic4MiB allocation
regression verifies both retained load peak and reset. Policy/preflight plus
metrics regression:15/15 pass. Final broader offline run is recorded separately.

This unit does not establish lower Qwen steady-state memory or a completed
benchmark. Actual memory audit/rows remain in local-model-bench artifacts/
qwen36-memory-audit-20261028. New supervised native admission and fresh pinned
config are required before retry. Primary Mei checkout and model files untouched.

Final source:443 offline tests,2 opt-in skips,0 failures. No live server was
started for these tests. Both native admission and memory measurements remain
separate from this regression evidence.
