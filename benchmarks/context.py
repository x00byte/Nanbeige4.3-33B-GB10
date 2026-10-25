"""Probe cold context capacity; this does not measure long-context accuracy."""

import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid


def post(base, path, body):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=3600)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Nanbeige4.3-33B")
    p.add_argument("--lengths", default="16384,65536,262128")
    p.add_argument("--output", required=True, type=Path)
    o = p.parse_args()
    report = {"passed": False, "note": __doc__, "cases": []}

    def save():
        o.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = o.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(o.output)

    def tokenize(**kwargs):
        with post(o.base, "/tokenize", {"model": o.model, **kwargs}) as response:
            return json.load(response)["tokens"]

    template = tokenize(
        messages=[{"role": "user", "content": "\nReply with READY only."}],
        chat_template_kwargs={"enable_thinking": False},
        add_generation_prompt=True,
    )
    # The pinned template's user message begins with <|im_start|>user\n.
    starts = [
        i + 3
        for i in range(len(template) - 2)
        if template[i : i + 3] == [166100, 2714, 13]
    ]
    if len(starts) != 1:
        raise RuntimeError(
            "Unexpected chat template; inspect before constructing capacity probe"
        )
    prefix, suffix = template[: starts[0]], template[starts[0] :]
    unit = tokenize(prompt="Stable reference text.\n", add_special_tokens=False)
    save()
    try:
        for length in map(int, o.lengths.split(",")):
            extra = length - len(template)
            if extra < 0 or length + 8 > 262144:
                raise ValueError(
                    "Context length must fit the pinned profile, including output"
                )
            ids = (
                prefix
                + (unit * ((extra + len(unit) - 1) // len(unit)))[:extra]
                + suffix
            )
            assert len(ids) == length
            body = {
                "model": o.model,
                "prompt": ids,
                "max_tokens": 8,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "cache_salt": "context-" + uuid.uuid4().hex,
            }
            start = time.monotonic()
            row = {
                "requested_prompt_tokens": length,
                "content": "",
                "usage": {},
                "ttft_s": None,
            }
            done = False
            with post(o.base, "/v1/completions", body) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    if line.strip() == b"data: [DONE]":
                        done = True
                        break
                    data = json.loads(line[6:])
                    if data.get("error"):
                        raise RuntimeError(str(data["error"]))
                    row["usage"] = data.get("usage") or row["usage"]
                    for choice in data.get("choices", []):
                        text = choice.get("text", "")
                        if text and row["ttft_s"] is None:
                            row["ttft_s"] = time.monotonic() - start
                        row["content"] += text
            row["elapsed_s"] = time.monotonic() - start
            row["passed"] = (
                done
                and "READY" in row["content"]
                and row["usage"].get("prompt_tokens") == length
            )
            report["cases"].append(row)
            save()
            if not row["passed"]:
                raise RuntimeError("Capacity response failed: " + str(row))
            print(
                f"{length:,} cold prompt tokens: TTFT {row['ttft_s']:.2f}s, {row['content']!r}",
                flush=True,
            )
        report["passed"] = True
        save()
    except BaseException as error:
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
