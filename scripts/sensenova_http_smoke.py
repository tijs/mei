"""Disposable localhost acceptance checks against an already guarded Mei.

Does not launch/stop servers, execute tools, or touch model/cache files.
All prompts are synthetic; raw responses and failures are preserved.
"""

import argparse
import hashlib
import http.client
import json
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse


class Check:
    def __init__(self, args):
        url = urlparse(args.base_url)
        assert url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1"}
        self.host, self.port = url.hostname, url.port
        self.out, self.model = args.out, args.model_id
        self.out.mkdir(parents=True, exist_ok=False)
        self.rows = []
        source = Path(__file__).resolve()
        shutil.copyfile(source, self.out / "runner-source.py")
        self.save(
            "runner-provenance",
            {
                "source": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "arguments": sys.argv[1:],
            },
        )

    def save(self, name, data):
        (self.out / (name + ".json")).write_text(json.dumps(data, indent=2))

    def request(self, name, path, body=None, status=200):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=180)
        began = time.monotonic()
        conn.request(
            "POST" if body is not None else "GET",
            path,
            body=json.dumps(body) if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read().decode()
        conn.close()
        row = {
            "name": name,
            "status": response.status,
            "request": body,
            "elapsed_s": time.monotonic() - began,
            "raw": raw,
        }
        self.rows.append(row)
        self.save(name, row)
        assert response.status == status, row
        return json.loads(raw)

    def chat(self, messages, **extra):
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "seed": 17,
            "max_tokens": 16,
            **extra,
        }

    def open_stream(self, body):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=180)
        conn.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(
                {**body, "stream": True, "stream_options": {"include_usage": True}}
            ),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 200, response.read()
        return conn, response

    def stream(self, name, body, cancel=False, on_first=None, expect_error=False):
        began = time.monotonic()
        conn, response = self.open_stream(body)
        frames, text, done, finishes = [], "", False, []
        try:
            while line := response.readline():
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                frame = json.loads(data)
                frames.append(frame)
                if "error" in frame:
                    assert expect_error, frame
                    continue
                for choice in frame.get("choices", []):
                    text += choice.get("delta", {}).get("content", "") or ""
                    if choice.get("finish_reason") is not None:
                        finishes.append(choice["finish_reason"])
                if cancel and text:
                    if on_first:
                        on_first()
                    break
        finally:
            response.close()
            conn.close()
            self.save(
                name,
                {
                    "request": body,
                    "frames": frames,
                    "text": text,
                    "done": done,
                    "finishes": finishes,
                    "elapsed_s": time.monotonic() - began,
                    "cancelled_by_client": cancel,
                },
            )
        if expect_error:
            errors = [frame["error"] for frame in frames if "error" in frame]
            assert len(errors) == 1 and errors[0]["code"] == "stream_error", frames
            assert "structured output constraint failed" in errors[0]["message"], frames
            assert not finishes, finishes
        elif not cancel:
            assert done and len(finishes) == 1, (done, finishes)
            assert any(frame.get("usage") for frame in frames), frames
        return text, finishes

    def basic(self):
        self.request("health", "/healthz")
        models = self.request("models", "/v1/models")
        assert self.model in {row["id"] for row in models["data"]}
        self.request("status-before", "/v1/mei/status")
        messages = [
            {
                "role": "user",
                "content": "Based only on this record, what is Noor's birthday? Record: Noor lives in Delft and enjoys cycling. If absent reply exactly UNKNOWN.",
            }
        ]
        body = self.chat(messages)
        answer = self.request("buffered", "/v1/chat/completions", body)
        assert answer["choices"][0]["message"]["content"] == "UNKNOWN", answer
        assert answer["choices"][0]["finish_reason"] == "stop", answer
        text, finish = self.stream("sse", body)
        assert text == "UNKNOWN" and finish == ["stop"]
        stopped = self.request(
            "stop-substring", "/v1/chat/completions", {**body, "stop": ["KNOWN"]}
        )
        assert stopped["choices"][0]["message"]["content"] == "UN", stopped
        assert stopped["choices"][0]["finish_reason"] == "stop", stopped
        stopped_text, stopped_finish = self.stream(
            "sse-stop-substring", {**body, "stop": ["KNOWN"]}
        )
        assert stopped_text == "UN" and stopped_finish == ["stop"]
        short = self.request(
            "length",
            "/v1/chat/completions",
            self.chat(
                [
                    {
                        "role": "user",
                        "content": "Return exactly one JSON object with keys city and count. Use city Utrecht and count 3. No Markdown.",
                    }
                ],
                max_tokens=1,
            ),
        )
        assert short["choices"][0]["finish_reason"] == "length", short
        assert short["usage"]["completion_tokens"] == 1, short
        for kind in ["image_url", "video", "audio"]:
            invalid = self.chat(
                [{"role": "user", "content": [{"type": kind, kind: "synthetic"}]}],
                stream=True,
            )
            error = self.request(
                "reject-" + kind, "/v1/chat/completions", invalid, status=400
            )
            assert "text content only" in error["error"]["message"], error
        healthy = self.request("healthy-after-errors", "/v1/chat/completions", body)
        assert healthy["choices"][0]["message"]["content"] == "UNKNOWN", healthy
        self.request("status-after", "/v1/mei/status")

    def wait_idle(self, name):
        for attempt in range(60):
            status = self.request(f"{name}-{attempt}", "/v1/mei/status")
            if status["scheduler"]["activeGenerations"] == 0:
                return status
            time.sleep(0.25)
        raise AssertionError("generation slot did not settle after disconnect")

    def lifecycle(self, server_log):
        assert server_log and server_log.is_file()
        before = self.request("lifecycle-before", "/v1/mei/status")
        offset = server_log.stat().st_size
        body = self.chat(
            [
                {
                    "role": "user",
                    "content": "PREFILL-CANCEL synthetic background. " * 150
                    + "List integers from 1 to 1000, one per line.",
                }
            ],
            max_tokens=128,
        )
        # Mei sends SSE headers after the first wire frame, which may be
        # after prefill. Observe the running request without awaiting headers.
        conn = http.client.HTTPConnection(self.host, self.port, timeout=180)
        conn.request(
            "POST",
            "/v1/chat/completions",
            json.dumps({**body, "stream": True}),
            {"Content-Type": "application/json"},
        )
        phase = None
        try:
            for _ in range(300):
                with server_log.open("rb") as log:
                    log.seek(offset)
                    text = log.read().decode(errors="replace")
                progress = list(
                    re.finditer(
                        r"mei: pp stage=(\w+) completed=(\d+) total=(\d+)", text
                    )
                )
                if progress:
                    latest = progress[-1]
                    if latest[1] == "prefill" and 0 < int(latest[2]) < int(latest[3]):
                        phase = latest.group()
                if phase:
                    break
                time.sleep(0.1)
            assert phase, "no observed partial-prefill boundary before disconnect"
        finally:
            conn.close()
            self.save("prefill-disconnect", {"request": body, "observed_phase": phase})
        after_prefill = self.wait_idle("prefill-settle")
        assert (
            after_prefill["scheduler"]["cancelledActive"]
            > before["scheduler"]["cancelledActive"]
        ), after_prefill
        if before.get("cacheReuse"):
            retry = self.request(
                "same-prompt-after-prefill-cancel",
                "/v1/chat/completions",
                {**body, "max_tokens": 1},
            )
            assert retry["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, retry

        unknown = self.chat(
            [
                {
                    "role": "user",
                    "content": "Based only on this record, what is Noor's birthday? Record: Noor lives in Delft and enjoys cycling. If absent reply exactly UNKNOWN.",
                }
            ]
        )
        healthy = self.request(
            "healthy-after-prefill-cancel", "/v1/chat/completions", unknown
        )
        assert healthy["choices"][0]["message"]["content"] == "UNKNOWN", healthy
        pool = ThreadPoolExecutor(max_workers=1)
        queued = []

        def enqueue():
            queued.append(
                pool.submit(
                    self.request, "queued-followup", "/v1/chat/completions", unknown
                )
            )
            for attempt in range(50):
                status = self.request(f"queue-observation-{attempt}", "/v1/mei/status")
                if status["scheduler"]["queueDepth"] == 1:
                    return
                time.sleep(0.1)
            raise AssertionError("second request was not observed queued")

        try:
            text, finishes = self.stream(
                "decode-disconnect",
                self.chat(
                    [
                        {
                            "role": "user",
                            "content": "List integers from 1 to 1000, one per line, continuing until the list is complete.",
                        }
                    ],
                    max_tokens=128,
                ),
                cancel=True,
                on_first=enqueue,
            )
            assert text and not finishes
            answer = queued[0].result(timeout=90)
            assert answer["choices"][0]["message"]["content"] == "UNKNOWN", answer
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        after = self.wait_idle("decode-settle")
        assert (
            after["scheduler"]["cancelledActive"]
            > after_prefill["scheduler"]["cancelledActive"]
        ), after
        assert after["scheduler"]["peakActiveGenerations"] == 1, after

    def native_tools(self):
        def schema_function(name, key):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Look up a synthetic catalog record by {key}.",
                    "parameters": {
                        "type": "object",
                        "properties": {key: {"type": "string"}},
                        "required": [key],
                        "additionalProperties": False,
                    },
                },
            }

        route = "/v1/chat/completions"
        tools = [schema_function("catalog_lookup", "sku")]
        question = {
            "role": "user",
            "content": "Use catalog_lookup to look up SKU K17. Use its returned color to answer with just the color.",
        }

        def get_call(name, schemas):
            payload = self.chat([question], tools=schemas, max_tokens=160)
            result = self.request(name, route, payload)
            choice = result["choices"][0]
            assert choice["finish_reason"] == "tool_calls", result
            calls = choice["message"]["tool_calls"]
            assert len(calls) == 1, result
            call = calls[0]
            assert call["function"]["name"] == "catalog_lookup", call
            assert json.loads(call["function"]["arguments"]) == {"sku": "K17"}, call
            return payload, choice["message"], call["id"]

        get_call("one-tool", tools)
        streaming = self.chat([question], tools=tools, max_tokens=160)
        _, finishes = self.stream("tool-sse", streaming)
        assert finishes == ["tool_calls"], finishes
        streamed = json.loads((self.out / "tool-sse.json").read_text())
        name, arguments, identifiers = "", "", set()
        for frame in streamed["frames"]:
            for choice in frame.get("choices", []):
                for event in choice.get("delta", {}).get("tool_calls", []):
                    assert event["index"] == 0, event
                    if event.get("id"):
                        identifiers.add(event["id"])
                    function = event.get("function", {})
                    name += function.get("name") or ""
                    arguments += function.get("arguments") or ""
        assert len(identifiers) == 1 and name == "catalog_lookup", streamed
        assert json.loads(arguments) == {"sku": "K17"}, streamed
        payload, assistant, call_id = get_call(
            "two-tool-schemas", [schema_function("catalog_search", "query"), *tools]
        )
        payload["messages"].extend(
            [
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": '{"color":"teal"}',
                },
            ]
        )
        result = self.request("tool-result-continuation", route, payload)
        assert result["choices"][0]["finish_reason"] == "stop", result
        assert result["choices"][0]["message"]["content"].strip().lower() == "teal", (
            result
        )

        payload = self.chat(
            [
                {
                    "role": "user",
                    "content": "Reply with exactly OK. Do not call any tools.",
                }
            ],
            tools=tools,
            max_tokens=32,
        )
        result = self.request("no-tool", route, payload)
        assert result["choices"][0]["finish_reason"] == "stop", result
        assert result["choices"][0]["message"]["content"] == "OK", result

    def structured(self):
        route = "/v1/chat/completions"
        payload = self.chat(
            [
                {
                    "role": "user",
                    "content": "Return exactly one JSON object whose city is Utrecht and count is 3.",
                }
            ],
            max_tokens=96,
        )
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["city", "count"],
            "additionalProperties": False,
        }
        grammar = {
            "type": "json_schema",
            "json_schema": {"name": "record", "strict": True, "schema": schema},
        }
        for name, response_format in (
            ("json-object", {"type": "json_object"}),
            ("json-schema", grammar),
        ):
            reply = self.request(
                name, route, {**payload, "response_format": response_format}
            )
            choice = reply["choices"][0]
            assert choice["finish_reason"] == "stop", reply
            assert json.loads(choice["message"]["content"]) == {
                "city": "Utrecht",
                "count": 3,
            }, reply
        text, finishes = self.stream(
            "json-schema-sse", {**payload, "response_format": grammar}
        )
        assert finishes == ["stop"] and json.loads(text) == {
            "city": "Utrecht",
            "count": 3,
        }, text
        self.request(
            "reject-tools-with-schema",
            route,
            {
                **payload,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "test", "parameters": {"type": "object"}},
                    }
                ],
                "response_format": grammar,
            },
            status=400,
        )
        exhausted = {**payload, "response_format": grammar, "max_tokens": 1}
        error = self.request("schema-length-fails-closed", route, exhausted, status=500)
        assert "structured output constraint failed" in error["error"]["message"], error
        self.stream("schema-length-sse-fails-closed", exhausted, expect_error=True)
        reply = self.request("schema-recovers", route, {**payload, "response_format": grammar})
        assert reply["choices"][0]["finish_reason"] == "stop", reply

    def cache(
        self,
        baseline=None,
        warm_required=False,
        expect_no_reuse=False,
        fixture="copy-v1",
        restart_hit_required=False,
    ):
        system = {
            "role": "system",
            "content": "Return only the code from the last user message, with no punctuation or explanation.",
        }
        padding = "Background: Noor lives in Delft and enjoys cycling. " * 10
        user = {"role": "user", "content": padding + "Code: CEDAR"}
        base = [system, user]
        growing = "Code: MAPLE"
        divergent = padding + "Code: LILAC"
        if fixture == "continuation-v2":
            body = (
                padding
                + "Based only on the record, what is Noor's birthday? If absent reply exactly UNKNOWN."
            )
            base = [{"role": "user", "content": body}]
            growing = (
                "Now return one JSON object with city Utrecht and count 3. No Markdown."
            )
            divergent = (
                padding + "What city does Noor live in? Reply with just the city."
            )
        cases = {"base": base}
        summary = {}

        def run(name, messages):
            response = self.request(
                "cache-" + name,
                "/v1/chat/completions",
                self.chat(messages, max_tokens=12),
            )
            choice = response["choices"][0]
            usage = response["usage"]
            summary[name] = {
                "text": choice["message"]["content"],
                "finish": choice["finish_reason"],
                "prompt_tokens": usage["prompt_tokens"],
                "cached_tokens": usage.get("prompt_tokens_details", {}).get(
                    "cached_tokens", 0
                ),
            }
            return summary[name]["text"]

        base_text = run("base", base)
        cases["growing"] = base + [
            {"role": "assistant", "content": base_text},
            {"role": "user", "content": growing},
        ]
        cases["diverging"] = [
            *base[:-1],
            {"role": "user", "content": divergent},
        ]
        run("repeat", base)
        run("growing", cases["growing"])
        run("diverging", cases["diverging"])
        self.save("cache-summary", summary)
        self.save("cache-cases", cases)
        # The original copy task requires semantically complete responses.
        # The continuation fixture checks cache equivalence even at its cap.
        if fixture == "copy-v1":
            assert all(row["finish"] == "stop" for row in summary.values()), summary
        else:
            assert all(
                row["finish"] in {"stop", "length"} and row["text"]
                for row in summary.values()
            ), summary
        assert summary["base"]["text"] == summary["repeat"]["text"], summary
        assert len({summary[name]["text"] for name in cases}) == 3, summary
        if baseline:
            expected = json.loads(baseline.read_text())
            for name in summary:
                for key in ["text", "finish", "prompt_tokens"]:
                    assert summary[name][key] == expected[name][key], (
                        name,
                        key,
                        summary,
                        expected,
                    )
        if warm_required:
            assert summary["repeat"]["cached_tokens"] > 0, summary
            assert summary["growing"]["cached_tokens"] > 0, summary
        if restart_hit_required:
            assert summary["base"]["cached_tokens"] > 0, summary
        status = self.request("status-after", "/v1/mei/status")
        if expect_no_reuse:
            assert all(row["cached_tokens"] == 0 for row in summary.values()), summary
            if status.get("cache"):
                assert status["cache"]["retainedBytes"] == 0, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18024")
    parser.add_argument("--model-id", default="experimental-sensenova-loop-q8-fp32")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=["basic", "cache", "lifecycle", "tools", "structured"],
        default="basic",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--warm-required", action="store_true")
    parser.add_argument("--expect-no-reuse", action="store_true")
    parser.add_argument("--restart-hit-required", action="store_true")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument(
        "--cache-fixture", choices=["copy-v1", "continuation-v2"], default="copy-v1"
    )
    args = parser.parse_args()
    check = Check(args)
    try:
        if args.suite == "basic":
            check.basic()
        elif args.suite == "lifecycle":
            check.lifecycle(args.server_log)
        elif args.suite == "tools":
            check.native_tools()
        elif args.suite == "structured":
            check.structured()
        else:
            check.cache(
                args.baseline,
                args.warm_required,
                args.expect_no_reuse,
                args.cache_fixture,
                args.restart_hit_required,
            )
    except BaseException:
        check.save("outcome", {"pass": False, "error": traceback.format_exc()})
        raise
    check.save(
        "outcome",
        {
            "pass": True,
            "scope": args.suite + " HTTP acceptance only; not benchmark quality",
        },
    )


if __name__ == "__main__":
    main()
