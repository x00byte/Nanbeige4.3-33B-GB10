"""Plot completed cold-prefill cases from the head and record independent capacity evidence."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--chart", type=Path, default=Path("assets/prefill.png"))
    o = p.parse_args()

    def read(name):
        return json.loads((o.results / name).read_text())

    baseline = (
        read("context-baseline-partial.json")["cases"]
        + read("context-native64k-head.json")["cases"]
    )
    optimized = read("context-capacity-head.json")
    worker = read("context-capacity-worker.json")
    assert optimized["passed"] and worker["passed"]
    rows = []
    for length in (16384, 65536):
        before = next(x for x in baseline if x["requested_prompt_tokens"] == length)
        after = next(
            x for x in optimized["cases"] if x["requested_prompt_tokens"] == length
        )
        for case in (before, after):
            assert case["passed"] and case["usage"]["prompt_tokens"] == length
            assert case["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        rows.append((length, before["ttft_s"], after["ttft_s"]))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, (length, before, after) in zip(axes, rows):
        bars = ax.bar(
            ["Native prefill\nfull draft", "Flash prefill\n8K draft"],
            [before, after],
            color=["#556575", "#bd7728"],
            width=0.55,
        )
        ax.bar_label(bars, labels=[f"{before:.1f}s", f"{after:.1f}s"], padding=4)
        ax.set_ylim(0, max(before, after) * 1.18)
        ax.set_title(f"{length:,} cold prompt tokens")
        ax.set_ylabel("Time to first token (seconds)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Nanbeige4.3-33B NVFP4 · one head GB10 · lower is better")
    o.chart.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o.chart, dpi=180)
    lines = [
        "# Cold prefill and context capacity",
        "",
        "Both throughput comparisons below ran on the same head GB10 with its existing 2,000 MHz clock cap. "
        "Thinking was off, each prompt had a fresh cache salt, and the answer was READY. "
        "These are single cold-cache cases, not a model-quality evaluation.",
        "",
        "| Prompt tokens | Native + full draft TTFT | Flash prefill + 8K draft TTFT | Speedup |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for length, before, after in rows:
        lines.append(
            f"| {length:,} | {before:.2f}s | {after:.2f}s | {before / after:.2f}× |"
        )
    full = next(x for x in worker["cases"] if x["requested_prompt_tokens"] == 262128)
    assert full["passed"]
    lines += [
        "",
        "The completed 16K native reference case was retained from a sweep that was then stopped for optimization. "
        "The 64K reference completed as a separate run. The optimized build changes both the prefill kernel and the draft attention window.",
        "",
        f"An independent single-GB10 worker instance processed a cold 262,128-token prompt and returned READY "
        f"in {full['ttft_s']:.2f}s to first token. It used the same target, windowed draft, 262,144 context limit, "
        "12 slots, 74 GiB KV allocation, and patched runtime. This validates capacity, not long-context retrieval accuracy. "
        "The worker driver differs from the head, so its time is not used for the head speedup calculation.",
        "",
        "Raw records: [16K native](context-baseline-partial.json), [64K native](context-native64k-head.json), "
        "[optimized head](context-capacity-head.json), [full worker context](context-capacity-worker.json).",
        "",
        "## Credits",
        "",
        "Models and quantizations: [Nanbeige](https://huggingface.co/Nanbeige), "
        "[primitive-ai](https://huggingface.co/primitive-ai/Nanbeige4.3-33B-NVFP4), "
        "[kaitchup](https://huggingface.co/kaitchup/Nanbeige4.1-3B-NVFP4). "
        "Runtime and attention: [vLLM](https://github.com/vllm-project/vllm), "
        "[FlashAttention](https://github.com/Dao-AILab/flash-attention), "
        "[eugr B12X image](https://hub.docker.com/r/eugr/spark-vllm-b12x).",
        "",
    ]
    (o.results / "PREFILL.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
