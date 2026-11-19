# Scheduler lifecycle checks

The repository owns `scheduler.qnt` and its replay fixtures. Studio's broad
`model-serving-runtime` component also includes HTTP/cache code, so this model
intentionally covers only `Sources/MeiCore/Scheduling.swift`: bounded admission,
priority/FIFO grants, cancellation, deadline settlement and cleanup ownership.
It does not model MLX, HTTP delivery, cache correctness or throughput.

## Reproduce

Use the pinned Quint CLI (Node/npm) and the Xcode Swift toolchain:

```sh
npm exec --yes --package=@informalsystems/quint@0.33.1 -- quint typecheck verification/scheduler/scheduler.qnt
npm exec --yes --package=@informalsystems/quint@0.33.1 -- quint run verification/scheduler/scheduler.qnt --backend typescript --invariant safety --seed 20261119 --max-samples 10000 --max-steps 40 --verbosity 1
trace_dir=$(mktemp -d)
npm exec --yes --package=@informalsystems/quint@0.33.1 -- quint run verification/scheduler/scheduler.qnt --backend typescript --invariant safety --seed 20261120 --max-samples 512 --max-steps 40 --n-traces 512 --out-itf "$trace_dir/mei-scheduler-{seq}.itf.json" --verbosity 1
uv run --no-project python tools/convert_scheduler_traces.py "$trace_dir" --output verification/scheduler/traces.json
bash scripts/check_scheduler.sh
```

The script compiles the actual scheduler and the normal scheduling tests in a
disposable Swift package, without MLX or model weights. The generated replay test
also runs in the normal Mei suite. It compares grants (by requiring the modeled
request to enter its operation), active/queued counts, priority mix, admission
counts exact terminal outcomes and failure/cancellation counters, and exactly-once
release after every action. Manual deadlines and explicit
latches control ordering; wall-clock deadlines are only fail-fast watchdogs.
Each trace owns its tasks and settles them before trying a healthy next request.
A failed replay also attempts bounded cleanup; an implementation that cannot
settle fails the watchdog rather than leaving the test waiting indefinitely.
The converter chooses 32 coverage witnesses from a separate 512-trace corpus,
including queue-full refusal, and adds a live two-slot/queued prefix to exercise
teardown at the bound. Generation uses a fresh directory to exclude stale files.

Four request IDs are each used once per trace; active capacity and queue bounds
are independently 1 or 2. The model separates granted, running and cleaning states.
Cancellation after grant prevents operation start; cancellation during cleanup
holds the slot until cleanup completes. Operation errors release the slot too.
Timeout and cancellation remove only a queued request. Priority applies only at
grant time; it cannot preempt a running operation.

The checked safety invariant covers queue membership/uniqueness, queue/active
bounds and grant/release conservation. The actual Swift replay additionally
checks observable ordering and settlement. Random simulation is not an exhaustive
proof. Progress requires eventual operation/cleanup completion; sustained
interactive arrivals may starve background requests. No liveness guarantee is
claimed. Request IDs do not wrap in the bounded model; production UInt64 rollover
and arbitrary queue sizes are outside these checks.

`RequestScheduler` has internal optional hooks for manual deadlines and the
post-grant/pre-start boundary. Its public initializer leaves both unset, retaining
the ordinary production clock and no additional suspension before the cancellation
check. No inference API or admission default changes.
