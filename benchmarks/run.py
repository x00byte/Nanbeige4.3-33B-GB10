"""Measure streamed OpenAI requests with fixed prompts and explicit cache isolation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import threading
import time
import urllib.request
import uuid

SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "square": {"type": "integer"},
                    "tag": {"type": "string", "enum": ["alpha", "beta"]},
                },
                "required": ["id", "square", "tag"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)


def engine_counters(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()

    def total(name):
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in lines
            if line.startswith(name + "{")
        )

    return {
        "completed": total("vllm:request_success_total"),
        "running": total("vllm:num_requests_running"),
        "waiting": total("vllm:num_requests_waiting"),
    }


def wait_idle(base, dashboard):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        active = False
        if dashboard:
            for kind in ("decode", "prefill"):
                with urllib.request.urlopen(
                    dashboard + "/api/bench/" + kind, timeout=15
                ) as response:
                    job = json.load(response)
                active |= job.get("status") in ("queued", "running")
        counters = engine_counters(base)
        if not active and not counters["running"] and not counters["waiting"]:
            return counters
        time.sleep(2)
    raise TimeoutError("Other traffic did not become idle for an isolated measurement")


def request(
    base,
    model,
    workload,
    concurrency,
    repeat,
    index,
    tokens,
    barrier,
    namespace,
    temperature=0,
    top_p=1,
):
    prompts = {
        "code": f"Implement a Python LRU cache class named CacheVariant{index}, with locking, detailed comments and example tests. Return the complete Python code.",
        "prose": f"Write a detailed essay about improving public transport in coastal city number {index}. Discuss budgets, buses, trains, accessibility and practical examples. Continue for at least 1000 words.",
        "structured": f"Return a JSON object containing six rows with id 0 through 5, square equal to id times id, and tag alpha. This is dataset variant {index}.",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompts[workload]}],
        "max_tokens": tokens if workload != "structured" else 384,
        "temperature": temperature,
        "top_p": top_p,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_salt": f"nbg-matrix-v1-{namespace}-{workload}-{concurrency}-{repeat}-{index}",
    }
    if workload == "structured":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "rows", "strict": True, "schema": SCHEMA},
        }
    else:
        body["ignore_eos"] = True
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait(timeout=60)
    start = time.monotonic()
    events, content, reasoning, usage, finish, done = [], "", "", {}, None, False
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(
                    "Server rejected streamed generation: " + json.dumps(event["error"])
                )
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if piece or thought:
                    events.append(time.monotonic())
                content += piece
                reasoning += thought
    end = time.monotonic()
    assert done and content and usage.get("completion_tokens"), "Incomplete stream"
    assert not reasoning, "Thinking unexpectedly enabled"
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    assert cached == 0, f"Prompt cache isolation failed: {cached}"
    valid = None
    if workload == "structured":
        import jsonschema

        obj = json.loads(content)
        jsonschema.validate(obj, SCHEMA)
        valid = obj["rows"] == [
            {"id": i, "square": i * i, "tag": "alpha"} for i in range(6)
        ]
        assert valid, obj
        assert finish == "stop", finish
    else:
        assert usage["completion_tokens"] == tokens, usage
    return {
        "index": index,
        "started_monotonic": start,
        "ended_monotonic": end,
        "first_event_monotonic": events[0],
        "last_event_monotonic": events[-1],
        "ttft_s": events[0] - start,
        "elapsed_s": end - start,
        "usage": usage,
        "finish_reason": finish,
        "structured_semantics_passed": valid,
        "content": content,
        "reasoning": reasoning,
        "event_count": len(events),
        "request_body": body,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Nanbeige4.3-33B")
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency", default="1,2,3,4,5,6,7,8,12")
    p.add_argument("--workloads", default="prose,code,structured")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=1)
    p.add_argument(
        "--dashboard",
        help="Optional dashboard URL; wait while its benchmarks are active",
    )
    o = p.parse_args()
    if o.temperature < 0 or not 0 < o.top_p <= 1:
        p.error("Temperature must be nonnegative and top-p must be in (0, 1]")
    o.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "label": o.label,
        "started": utc(),
        "passed": False,
        "method": {
            "thinking": False,
            "temperature": o.temperature,
            "top_p": o.top_p,
            "seed": 42,
            "generation_tokens": o.tokens,
            "structured_max_tokens": 384,
            "structured_schema": SCHEMA,
            "cache": "unique salt per request",
            "wall_tps": "sum(server completion_tokens) / (last request end - first request start)",
            "stream_window_tps": "sum(server completion_tokens) / (last token event - first token event); includes overlapping prefill",
            "note": "Single pass unless repeats > 1; short prompts; not a full-context or model-quality evaluation",
        },
        "batches": [],
    }
    save(o.output, report)
    namespace = uuid.uuid4().hex
    try:
        for repeat in range(o.repeats):
            for workload in o.workloads.split(","):
                for concurrency in map(int, o.concurrency.split(",")):
                    before = wait_idle(o.base, o.dashboard)
                    barrier = threading.Barrier(concurrency)
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        rows = list(
                            pool.map(
                                lambda i: request(
                                    o.base,
                                    o.model,
                                    workload,
                                    concurrency,
                                    repeat,
                                    i,
                                    o.tokens,
                                    barrier,
                                    namespace,
                                    o.temperature,
                                    o.top_p,
                                ),
                                range(concurrency),
                            )
                        )
                    tokens = sum(x["usage"]["completion_tokens"] for x in rows)
                    wall = max(x["ended_monotonic"] for x in rows) - min(
                        x["started_monotonic"] for x in rows
                    )
                    window = max(x["last_event_monotonic"] for x in rows) - min(
                        x["first_event_monotonic"] for x in rows
                    )
                    batch = {
                        "workload": workload,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "completed": utc(),
                        "output_tokens": tokens,
                        "wall_s": wall,
                        "wall_tps": tokens / wall,
                        "stream_window_s": window,
                        "stream_window_tps": tokens / window,
                        "median_ttft_s": statistics.median(x["ttft_s"] for x in rows),
                        "max_ttft_s": max(x["ttft_s"] for x in rows),
                        "requests": rows,
                    }
                    report["batches"].append(batch)
                    deadline = time.monotonic() + 2
                    while True:
                        after = engine_counters(o.base)
                        if (
                            after["completed"] - before["completed"] >= concurrency
                            or time.monotonic() >= deadline
                        ):
                            break
                        time.sleep(0.1)
                    batch["engine_counters_before"] = before
                    batch["engine_counters_after"] = after
                    batch["isolated_counter_check"] = (
                        after["completed"] - before["completed"] == concurrency
                    )
                    save(o.output, report)
                    assert batch["isolated_counter_check"], (
                        "Other traffic overlapped this batch; rerun after it becomes idle"
                    )
                    print(
                        f"{o.label} {workload} C{concurrency}: {tokens / wall:.2f} wall tok/s, {tokens / window:.2f} stream-window tok/s",
                        flush=True,
                    )
        report["passed"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["completed"] = utc()
        save(o.output, report)


if __name__ == "__main__":
    main()
