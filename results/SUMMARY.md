# Measured throughput

Aggregate output tokens per second, including request latency and prefill. Thinking off, greedy decoding, unique prompt-cache salt per request. Code and prose use 128 output tokens; structured output runs to a valid JSON document. Single measurements unless a run explicitly repeats a case.

## nvfp4-int4kv-baseline

| Workload | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose | 10.72 | 20.93 | 30.80 | 41.33 | 51.37 | 61.39 | 71.18 | 80.84 | 116.18 |
| code | 10.68 | 21.24 | 31.49 | 41.82 | 51.62 | 61.66 | 71.56 | 81.37 | 117.13 |
| structured | 10.88 | 21.26 | 31.54 | 41.89 | 51.65 | 61.62 | 71.31 | 80.20 | 115.64 |

Raw data: [nvfp4-int4kv-baseline.json](nvfp4-int4kv-baseline.json).

## nvfp4-small-draft3

| Workload | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose | 13.44 | 26.36 | 38.40 | 50.82 | 62.54 | 74.41 | 85.11 | 96.13 | 131.93 |
| code | 18.73 | 33.63 | 48.76 | 64.60 | 79.42 | 94.83 | 109.22 | 123.42 | 167.31 |
| structured | 25.67 | 50.08 | 73.97 | 96.80 | 122.45 | 142.47 | 159.86 | 181.12 | 258.65 |

Raw data: [nvfp4-small-draft3.json](nvfp4-small-draft3.json).

## nvfp4-flash-prefill-v3-adaptive4

| Workload | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prose | 13.32 | 25.78 | 38.37 | 50.34 | 62.08 | 73.02 | 84.61 | 94.95 | 133.94 |
| code | 17.21 | 32.20 | 48.02 | 61.92 | 74.58 | 90.31 | 103.14 | 117.94 | 157.11 |
| structured | 26.32 | 51.11 | 75.27 | 99.68 | 121.79 | 144.98 | 166.07 | 188.13 | 264.41 |

Raw data: [nvfp4-flash-prefill-v3-adaptive4.json](nvfp4-flash-prefill-v3-adaptive4.json).

## nvfp4-dflash2-pilot7

| Workload | C1 | C4 | C8 | C12 |
| --- | ---: | ---: | ---: | ---: |
| prose | 9.83 | 37.23 | 69.90 | 90.20 |
| code | 13.70 | 48.06 | 92.41 | 107.28 |
| structured | 15.48 | 58.76 | 111.77 | 160.82 |

Raw data: [nvfp4-dflash2-pilot7.json](nvfp4-dflash2-pilot7.json).

## Credits

Model: [Nanbeige](https://huggingface.co/Nanbeige/Nanbeige4.3-33B); quantization: [primitive-ai](https://huggingface.co/primitive-ai/Nanbeige4.3-33B-NVFP4); small draft: [kaitchup](https://huggingface.co/kaitchup/Nanbeige4.1-3B-NVFP4); runtime: [vLLM](https://github.com/vllm-project/vllm) and [eugr B12X image](https://hub.docker.com/r/eugr/spark-vllm-b12x).
