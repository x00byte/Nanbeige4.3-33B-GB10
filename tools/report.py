"""Build tables and a chart from completed benchmark records."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--markdown", type=Path, default=Path("results/SUMMARY.md"))
    p.add_argument("--chart", type=Path, default=Path("assets/throughput.png"))
    o = p.parse_args()
    reports = [json.loads(path.read_text()) for path in o.inputs]
    assert all(report["passed"] for report in reports), (
        "Refusing to summarize an incomplete run"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    lines = [
        "# Measured throughput",
        "",
        "Aggregate output tokens per second, including request latency and prefill. "
        "Thinking off, greedy decoding, unique prompt-cache salt per request. "
        "Code and prose use 128 output tokens; structured output runs to a valid JSON document. "
        "Single measurements unless a run explicitly repeats a case.",
        "",
    ]
    for report, source in zip(reports, o.inputs):
        cs = sorted({batch["concurrency"] for batch in report["batches"]})
        lines += [
            "## " + report["label"],
            "",
            "| Workload | " + " | ".join("C" + str(c) for c in cs) + " |",
            "| --- | " + " | ".join("---:" for _ in cs) + " |",
        ]
        for ax, workload in zip(axes, ["prose", "code", "structured"]):
            values = []
            for c in cs:
                rates = [
                    b["wall_tps"]
                    for b in report["batches"]
                    if b["concurrency"] == c and b["workload"] == workload
                ]
                values.append(statistics.median(rates) if rates else None)
            lines.append(
                "| "
                + workload
                + " | "
                + " | ".join(f"{v:.2f}" if v else "—" for v in values)
                + " |"
            )
            points = [(c, v) for c, v in zip(cs, values) if v is not None]
            if points:
                ax.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    marker="o",
                    label=report["label"],
                )
            ax.set_title(workload.capitalize())
            ax.set_xlabel("Concurrent requests")
            ax.set_ylabel("Aggregate output tok/s")
            ax.set_xticks([1, 2, 4, 6, 8, 12])
            ax.grid(alpha=0.2)
        lines += ["", f"Raw data: [{source.name}]({source.name}).", ""]
    axes[0].legend(fontsize=7)
    fig.suptitle("Nanbeige4.3-33B NVFP4 · one NVIDIA GB10 · INT4 KV cache")
    o.chart.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o.chart, dpi=180)
    o.markdown.parent.mkdir(parents=True, exist_ok=True)
    lines += [
        "## Credits",
        "",
        "Model: [Nanbeige](https://huggingface.co/Nanbeige/Nanbeige4.3-33B); "
        "quantization: [primitive-ai](https://huggingface.co/primitive-ai/Nanbeige4.3-33B-NVFP4); "
        "small draft: [kaitchup](https://huggingface.co/kaitchup/Nanbeige4.1-3B-NVFP4); "
        "runtime: [vLLM](https://github.com/vllm-project/vllm) and "
        "[eugr B12X image](https://hub.docker.com/r/eugr/spark-vllm-b12x).",
        "",
    ]
    o.markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
