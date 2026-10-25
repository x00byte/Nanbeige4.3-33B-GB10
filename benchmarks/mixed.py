"""Validate mixed arithmetic, reasoning, tools, and JSON across 12 concurrent streams."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Nanbeige4.3-33B")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--long-prefix",
        action="store_true",
        help="Include a cold 16K-prefix request in the mixed batch",
    )
    o = p.parse_args()
    barrier = threading.Barrier(12)
    namespace = uuid.uuid4().hex

    def request(index):
        kind = ("arithmetic", "reasoning", "tools", "structured")[index % 4]
        a, b = 21 + index, 3
        body = {
            "model": o.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Calculate {a} times {b}. Reply with the integer.",
                }
            ],
            "max_tokens": 192,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache_salt": f"mixed-{namespace}-{index}",
            "chat_template_kwargs": {"enable_thinking": kind == "reasoning"},
        }
        if o.long_prefix and index == 0:
            body["messages"][0]["content"] = (
                "Stable reference text about deployment, caching and model serving.\n"
                * 1400
                + "\n"
                + body["messages"][0]["content"]
            )
        if kind == "tools":
            body["messages"][0]["content"] = "Use the weather tool to check Paris."
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
            body["tool_choice"] = "auto"
        elif kind == "structured":
            body["messages"][0]["content"] += " Return the result in the answer field."
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "integer"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }
        req = urllib.request.Request(
            o.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        barrier.wait(timeout=60)
        start = time.monotonic()
        row = {
            "index": index,
            "kind": kind,
            "content": "",
            "reasoning": "",
            "tools": {},
            "usage": {},
        }
        done = False
        with urllib.request.urlopen(req, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    done = True
                    break
                event = json.loads(line[6:])
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                row["usage"] = event.get("usage") or row["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    row["content"] += delta.get("content") or ""
                    row["reasoning"] += (
                        delta.get("reasoning") or delta.get("reasoning_content") or ""
                    )
                    for tool in delta.get("tool_calls") or []:
                        entry = row["tools"].setdefault(
                            tool.get("index", 0), {"name": "", "arguments": ""}
                        )
                        function = tool.get("function", {})
                        entry["name"] += function.get("name") or ""
                        entry["arguments"] += function.get("arguments") or ""
        assert done, "Incomplete stream"
        if kind == "tools":
            assert len(row["tools"]) == 1
            tool = next(iter(row["tools"].values()))
            assert (
                tool["name"] == "weather"
                and json.loads(tool["arguments"])["city"].lower().strip() == "paris"
            )
        elif kind == "structured":
            assert json.loads(row["content"]) == {"answer": a * b}
        else:
            assert str(a * b) in row["content"], row
            if kind == "reasoning":
                assert row["reasoning"], row
        row.update(elapsed_s=time.monotonic() - start, passed=True)
        return row

    report = {
        "passed": False,
        "concurrency": 12,
        "long_prefix": o.long_prefix,
        "note": __doc__,
    }
    o.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            report["requests"] = list(pool.map(request, range(12)))
        report["passed"] = True
        print("12 mixed streams passed: arithmetic, reasoning, tools, JSON")
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        o.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
