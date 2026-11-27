# Mei 0.19.0

Start Mei with a model name to use automatic context sizing. For chat and raw completions, the formatted prompt and generated reply share one token budget. Cached prompt tokens count once. Mei reserves enough memory for the admitted reply before generation starts.

Use `--context-cap 32768` for a fixed ceiling, or `--context-cap auto` to return to automatic sizing. A numeric ceiling still observes the model's positional limit and machine memory. The status endpoint `/v1/mei/status` exposes the selected mode, the server ceiling and an estimate of the tokens available to a new request.

If the prompt fills the entire window, shorten it to leave room for the reply. If memory admission returns HTTP 503, retry after the active request finishes or reduce the requested output length.
