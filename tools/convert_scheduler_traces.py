"""Convert Quint ITF traces into the committed Swift replay fixture."""

import argparse
import json
from pathlib import Path


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if "#bigint" in value:
            return int(value["#bigint"])
        if "#set" in value:
            return sorted(decode(value["#set"]))
        if "#map" in value:
            return [
                decode(pair[1])
                for pair in sorted(value["#map"], key=lambda p: decode(p[0]))
            ]
        return {key: decode(item) for key, item in value.items() if key != "#meta"}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pattern", default="mei-scheduler-*.itf.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.directory.glob(args.pattern))
    if len(paths) < 32:
        raise ValueError("need at least 32 Quint traces")
    traces = [
        {"states": decode(json.loads(path.read_text())["states"])} for path in paths
    ]

    def features(trace):
        result = set()
        for state in trace["states"]:
            result.add(state["event"])
            result.add(f"bounds:{state['capacity']}:{state['queueLimit']}")
            if state["event"] == "arrive" and state["phase"][state["request"]] == 5:
                result.add("queueFull")
        return result

    required_features = set().union(*(features(trace) for trace in traces))
    if "queueFull" not in required_features:
        raise ValueError("corpus must include queue-full rejection")
    selected = []
    remaining = list(traces)
    uncovered = set(required_features)
    while uncovered:
        best = max(
            range(len(remaining)), key=lambda i: len(features(remaining[i]) & uncovered)
        )
        trace = remaining.pop(best)
        selected.append(trace)
        uncovered -= features(trace)
    selected.extend(remaining[: max(0, 32 - len(selected))])
    # Exercise teardown at a bounded prefix with live work, as well as normal
    # complete traces. Every prefix is still an actual Quint execution.
    for trace in traces:
        prefix = next(
            (
                i
                for i, state in enumerate(trace["states"])
                if state["queue"] and sum(p in (2, 3, 4) for p in state["phase"]) == 2
            ),
            None,
        )
        if prefix is not None:
            selected.append({"states": trace["states"][: prefix + 1]})
            break
    else:
        raise ValueError("corpus must include a two-slot queued teardown prefix")
    traces = selected
    events = {state["event"] for trace in traces for state in trace["states"]}
    required = {
        "arrive",
        "start",
        "cancelQueued",
        "cancelGranted",
        "timeout",
        "cancelActive",
        "cleanup",
        "fail",
        "release",
    }
    if not required <= events:
        raise ValueError(f"missing actions: {required - events}")
    args.output.write_text(json.dumps(traces, separators=(",", ":")) + "\n")
    print(f"Converted {len(traces)} traces; actions: {sorted(events)}")


if __name__ == "__main__":
    main()
