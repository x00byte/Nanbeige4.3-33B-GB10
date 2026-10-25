# Cold prefill and context capacity

Both throughput comparisons below ran on the same head GB10 with its existing 2,000 MHz clock cap. Thinking was off, each prompt had a fresh cache salt, and the answer was READY. These are single cold-cache cases, not a model-quality evaluation.

| Prompt tokens | Native + full draft TTFT | Flash prefill + 8K draft TTFT | Speedup |
| ---: | ---: | ---: | ---: |
| 16,384 | 38.62s | 18.10s | 2.13× |
| 65,536 | 496.03s | 107.02s | 4.63× |

The completed 16K native reference case was retained from a sweep that was then stopped for optimization. The 64K reference completed as a separate run. The optimized build changes both the prefill kernel and the draft attention window.

An independent single-GB10 worker instance processed a cold 262,128-token prompt and returned READY in 943.45s to first token. It used the same target, windowed draft, 262,144 context limit, 12 slots, 74 GiB KV allocation, and patched runtime. This validates capacity, not long-context retrieval accuracy. The worker driver differs from the head, so its time is not used for the head speedup calculation.

Raw records: [16K native](context-baseline-partial.json), [64K native](context-native64k-head.json), [optimized head](context-capacity-head.json), [full worker context](context-capacity-worker.json).

## Credits

Models and quantizations: [Nanbeige](https://huggingface.co/Nanbeige), [primitive-ai](https://huggingface.co/primitive-ai/Nanbeige4.3-33B-NVFP4), [kaitchup](https://huggingface.co/kaitchup/Nanbeige4.1-3B-NVFP4). Runtime and attention: [vLLM](https://github.com/vllm-project/vllm), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [eugr B12X image](https://hub.docker.com/r/eugr/spark-vllm-b12x).
