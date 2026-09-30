# Cmlx real-model gate

- Root: `/Users/tijs/projects/mei/artifacts/cmlx-gate-20260930T141121Z`
- Results: 7 (3 behavior, 4 performance)
- Gate verdict: **pass**

## Checks

- `all_server_runs_stopped`: `True`
- `all_runs_passed`: `True`
- `source_behavior_determinism`: `True`
- `source_behavior_count`: `2`
- `stock_behavior_count`: `1`
- `cross_library_plain_output_equal`: `True`
- `cross_library_parity_output_equal`: `True`
- `cross_library_tool_equal`: `True`
- `cross_library_cache_check_equal`: `True`
- `cross_library_context_boundary_equal`: `True`
- `cross_library_tool_valid`: `True`
- `performance_source_short_runs`: `2`
- `performance_source_short_mean_tps`: `46.6472336294498`
- `performance_source_short_mean_decode_tps`: `53.68816666666666`
- `performance_source_short_mean_prefill_ms`: `161.14883333333336`
- `performance_source_loaded_runs`: `2`
- `performance_source_loaded_mean_tps`: `1.078603267393171`
- `performance_source_loaded_mean_decode_tps`: `45.6665`
- `performance_source_loaded_mean_prefill_ms`: `57428.55433333333`
- `performance_stock_short_runs`: `2`
- `performance_stock_short_mean_tps`: `46.70384380694861`
- `performance_stock_short_mean_decode_tps`: `53.75883333333333`
- `performance_stock_short_mean_prefill_ms`: `161.20633333333333`
- `performance_stock_loaded_runs`: `2`
- `performance_stock_loaded_mean_tps`: `1.081097421038145`
- `performance_stock_loaded_mean_decode_tps`: `45.60133333333333`
- `performance_stock_loaded_mean_prefill_ms`: `57326.80266666666`
- `performance_short_source_vs_stock_decode_tps_delta_pct`: `-0.1314512653734501`
- `performance_short_source_vs_stock_prefill_ms_delta_pct`: `-0.0356685738153284`
- `performance_loaded_source_vs_stock_decode_tps_delta_pct`: `0.14290517821116122`
- `performance_loaded_source_vs_stock_prefill_ms_delta_pct`: `0.17749405502049953`

## Behavior legs

### source — `/Users/tijs/projects/mei/artifacts/cmlx-gate-20260930T141121Z/legs/source-behavior-1/result.json`
- status: `passed`; stopped: `True`; probe: `passed` / rc `0`
- tool: `{"arguments": {"a": 15, "b": 27}, "finish_reason": "tool_calls", "name": "add_numbers"}`
- context: `{"context_exact_cap": "passed", "context_over_cap_rejected": "passed"}`
- cache: `{"cache_growing_turn1": {"cached_tokens": 0, "status": "passed"}, "cache_growing_turn2_reuses_slot": {"cached_tokens": 785, "status": "passed"}, "cache_repeat_1": {"cached_tokens": 0, "status": "passed"}, "cache_repeat_2": {"cached_tokens": 6165, "status": "passed"}}`

### source — `/Users/tijs/projects/mei/artifacts/cmlx-gate-20260930T141121Z/legs/source-behavior-2/result.json`
- status: `passed`; stopped: `True`; probe: `passed` / rc `0`
- tool: `{"arguments": {"a": 15, "b": 27}, "finish_reason": "tool_calls", "name": "add_numbers"}`
- context: `{"context_exact_cap": "passed", "context_over_cap_rejected": "passed"}`
- cache: `{"cache_growing_turn1": {"cached_tokens": 0, "status": "passed"}, "cache_growing_turn2_reuses_slot": {"cached_tokens": 785, "status": "passed"}, "cache_repeat_1": {"cached_tokens": 0, "status": "passed"}, "cache_repeat_2": {"cached_tokens": 6165, "status": "passed"}}`

### stock — `/Users/tijs/projects/mei/artifacts/cmlx-gate-20260930T141121Z/legs/stock-behavior-1/result.json`
- status: `passed`; stopped: `True`; probe: `passed` / rc `0`
- tool: `{"arguments": {"a": 15, "b": 27}, "finish_reason": "tool_calls", "name": "add_numbers"}`
- context: `{"context_exact_cap": "passed", "context_over_cap_rejected": "passed"}`
- cache: `{"cache_growing_turn1": {"cached_tokens": 0, "status": "passed"}, "cache_growing_turn2_reuses_slot": {"cached_tokens": 785, "status": "passed"}, "cache_repeat_1": {"cached_tokens": 0, "status": "passed"}, "cache_repeat_2": {"cached_tokens": 6165, "status": "passed"}}`

## Performance legs

| Arm | Status | Rows | Short wall mean (s) | Loaded wall mean (s) | Short end-to-end tok/s | Loaded end-to-end tok/s |
|---|---:|---:|---:|---:|---:|---:|
| source | passed | 6 | 1.4452666253333337 | 59.250872472 | 44.88828246288276 | 1.0801731089361808 |
| source | passed | 6 | 1.3221994719999997 | 59.42270509733333 | 48.406184796016845 | 1.077033425850161 |
| stock | passed | 6 | 1.4474473889999995 | 59.25750440299999 | 44.80500271438285 | 1.080049100161135 |
| stock | passed | 6 | 1.316824902666667 | 59.141910417 | 48.602684899514365 | 1.082145741915155 |

## Interpretation

The source-built and stock libraries are intentionally compared as separate artifacts. Cross-library text identity is reported, not assumed: the plan explicitly allows numerical output differences. Tool-call structure, cache behavior, and exact context-cap behavior are separate acceptance gates. Performance values are descriptive measurements; round-robin order and repeat counts must be reviewed before attributing a difference to the metallib.
