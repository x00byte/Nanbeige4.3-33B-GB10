# Optimization experiments

Every serving measurement uses one NVIDIA GB10. A second GB10 ran the separate
teacher and training jobs for the DFlash2 pilot; it does not participate in the
published serving recipe. All measured target checkpoints are NVFP4.

Aggregate output tok/s including request overhead; greedy short-prompt tests.

| Candidate | C1 code | C12 prose | C12 code | C12 JSON | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| [nvfp4-int4kv-baseline](../results/nvfp4-int4kv-baseline.json) | 10.68 | 116.18 | 117.13 | 115.64 | passed |
| [nvfp4-w4a4](../results/nvfp4-w4a4.json) | 10.60 | 117.62 | 118.66 | 118.70 | passed |
| [nvfp4-gpu-lookup3](../results/nvfp4-gpu-lookup3.json) | 10.18 | 111.60 | 112.80 | — | rejected: streamed schema error |
| [nvfp4-small-draft2](../results/nvfp4-small-draft2.json) | 17.15 | 138.25 | 167.77 | 222.36 | passed |
| [nvfp4-small-draft3](../results/nvfp4-small-draft3.json) | 18.73 | 131.93 | 167.31 | 258.65 | passed |
| [nvfp4-dflash2-pilot7](../results/nvfp4-dflash2-pilot7.json) | 13.70 | 90.20 | 107.28 | 160.82 | passed |
| [nvfp4-adaptive-draft4](../results/nvfp4-adaptive-draft4.json) | 17.66 | 135.61 | 162.94 | 270.26 | passed |
| [nvfp4-flash-prefill-v3-adaptive4](../results/nvfp4-flash-prefill-v3-adaptive4.json) | 17.21 | 133.94 | 157.11 | 264.41 | selected |

## Quantized pretrained draft

The 33B target has 166,912 vocabulary rows. The pretrained
[Nanbeige4.1-3B NVFP4 draft](https://huggingface.co/kaitchup/Nanbeige4.1-3B-NVFP4)
has 166,144 rows, with exactly the same token-to-ID mapping, including special
tokens. Native draft-model speculation requires matching vocabulary sizes.

The adaptation appends 768 zero rows to the draft's input embedding and output
head. It preserves all existing rows and all quantized decoder tensors. The
portable adaptation script hashes and reads back all 963 original tensor
payloads and checks every appended byte is zero. This adds 7,864,320 bytes; it
does not requantize the pretrained checkpoint. The target checkpoint is used
unchanged. Tokenizer equivalence and tensor preservation do not establish broad
model-quality parity; the target still verifies speculative proposals.

The final draft enables an 8,192-token sliding attention window on all 32
layers through explicit `layer_types`. Llama ignores the window field without
those layer types; the earlier draft-budget reference sweeps used full
attention. The hybrid cache manager is disabled to preserve the tested full
shared allocation while restricting draft attention compute. The
target retains its 262,144-token context. Total KV allocation is 74 GiB across
both models, with 1,073,952 main-model tokens reported by the engine. INT4 KV
uses per-token, per-head scales. These cache tokens are shared by all requests.

Two- and three-token draft budgets were tested separately. Three improves
structured output and single-request code; two is slightly faster on prose.
Synchronous scheduling is explicit because the tested asynchronous GPU-lookup
candidate produced intermittent structured-output errors.

## Trained DFlash2 pilot

This experiment used
[vLLM Speculators](https://github.com/vllm-project/speculators/tree/21192d6ff51bb2aaa2ab0aa1d72e6f0b8f6a7d84),
source commit `21192d6ff51bb2aaa2ab0aa1d72e6f0b8f6a7d84`, with its experimental
[DFlash2 training objective](https://github.com/vllm-project/speculators/blob/21192d6ff51bb2aaa2ab0aa1d72e6f0b8f6a7d84/docs/user_guide/algorithms/dflash2.md).
The target's Llama architecture works with the supported converted draft; this
did not require running the full-precision 33B checkpoint.

| Training setting | Value |
| --- | --- |
| Teacher | The same pinned NVFP4 33B target |
| Draft | Two transformer layers, hidden size 7,168, MLP 8,192 |
| Attention | 16 query heads, 4 KV heads, head dimension 128 |
| Trainable parameters | 718,694,912 |
| Block / proposal budget | 8 / 7 tokens |
| Features | Target hidden-state indices 2, 26, 49 |
| Mask token | 166102, existing `<\|endoftext\|>` embedding |
| Selector | Rank 256, top 16 |
| Dataset | 128 teacher-generated responses: 64 code, 32 prose, 32 JSON |
| Training sequence length | At most 1,024 tokens |
| Optimization | Fused AdamW, LR 0.0003, seed 42 |
| Second-stage step cap | 512; selected state records global step 513 |
| Serving window | 2,048 tokens, explicitly enabled in converted Qwen3 config |
| Export | 1,437,393,752-byte draft weight file; target embeddings/head shared |

The second stage warm-started from the initial pilot. The initial padded mask
row had an unsuitable near-zero embedding; using an existing control-token
embedding improved training. The source's unshuffled 90/10 split leaves only
JSON samples in its small validation set. Those internal acceptance figures
are therefore unsuitable as general code/prose claims. Benchmark prompts were
separate from the teacher-generation prompts.

The actual serving pilot passed 12 concurrent arithmetic checks, automatic
tool calls, schema-constrained JSON, reasoning, and prefix reuse. Its measured
prose, code, and structured throughput was lower than the pretrained small
draft, so it is not the default. Its large hidden dimension and short, narrow
training run remain practical limitations. The experiment does not establish
that a larger, better-trained DFlash2 model could not improve on this result.

Only its architecture and provenance are published here; experimental trained
weights, optimizer state, and the training dataset are not required by the
selected serving recipe and are not included in Git.

## Other candidates

The selected build also accelerates long prefill. A scoped path dequantizes
the current single sequence's INT4 cache into a reusable 1 GiB FP16 workspace,
then calls FlashAttention 2 in the rotated cache basis. Query rotation and
intermediate output rounding retain the original model dtype. The target's
usual decode kernel consumes the persistent INT4 cache. Larger fallback
prefills use M=64/N=64 tiles for the target and M=32/N=32 for the small draft.

The worker prototype tested strided, interleaved K/V storage, shuffled page
tables, odd query counts, and KV lengths up to 262,128. Attention-only speedups
were 3.05× at 16K KV, 4.01× at 64K, and 4.43× near 262K with 4,096 query tokens.
These are kernel measurements, not whole-model speedups. The 128-token case
was slower, so the serving patch keeps the native path below 256 query tokens.

Chunked prefill uses the GPU's actual written prefix length when dequantizing
and constructing FlashAttention sequence lengths. The CPU's planned context
is only a launch upper bound. Cache-page offsets use 64-bit arithmetic because
vLLM interleaves layers and scales in the shared allocation. Regression cases
cover a planned 262K context with only 4K written and physical offsets beyond
2 GB; both details are necessary for the serving cache layout.

Two small sampled cases were also checked against float32 attention over
exactly dequantized cache values, including the native output rounding. The
FlashAttention path's relative L2 error was 0.31–0.34%; the native kernel's was
0.45–4.88% in those cases. This tests local numerical behavior, not model
quality. Greedy output can change with the prefill arithmetic. A direct FP16
dot-product substitution was explored separately and was not selected.

The [Dockerfile](../Dockerfile) pins the base runtime. The patch refuses to
apply if the upstream INT4 implementation's SHA-256 differs. Source hashes,
image identity, and kernel records are retained in
[patch-provenance.json](../results/patch-provenance.json). The reusable workspace
uses the engine's ordered GPU execution stream; asynchronous scheduling is off.

- Forcing calibrated NVFP4 activation quantization gave small throughput gains
  over automatic activation selection at higher concurrency.
- GPU n-gram lookup with three speculative tokens helped predictable JSON but
  intermittently returned streamed HTTP 500 errors when advancing the grammar
  FSM during concurrent schema-constrained generation. Its failed raw report
  is retained for comparison; it is excluded from successful-run charts.
- Native FP8 KV would reduce cache capacity below the requested million-token
  pool within this memory budget. Native NVFP4 KV support in this runtime is
  restricted to SM100; GB10 uses the Triton INT4 backend instead.

## Published alternatives checked

Checked on 2026-10-25 against the model publishers and runtime repositories.
The [official Nanbeige DSpark checkpoint](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-DSpark)
is trained for Nanbeige4.2-3B, whose hidden dimension and layer structure differ
from this 33B target. It cannot be substituted as a trained 33B DSpark head.
The Hugging Face model search did not expose a matching 4.3-33B DSpark or
DFlash checkpoint at the time of this check.

[Tencent DSpine](https://github.com/tencent/dspine) is a newer parallel drafting
approach, but its released checkpoint targets Qwen3-8B; it is not a tested
Nanbeige/GB10 recipe. [DSpark's paper](https://arxiv.org/abs/2607.05147) describes
model-conditioned parallel drafting and confidence scheduling. Applying that
architecture here requires target-specific training and measurement.

[DeepSeek GB10 DSpark recipes](https://github.com/vladimir-voinea/dspark-vllm-gb10)
target DeepSeek's own draft and attention architecture, with two-node examples.
Their speeds do not establish performance for a dense Llama-family 33B on one
GB10. [MLX DSpark](https://github.com/ARahim3/mlx-dspark) targets Apple Silicon.
Neither is a drop-in runtime for this NVFP4 checkpoint.

Newer Nanbeige4.2 small-model NVFP4 conversions also exist, but that base uses a
looped Nanbeige architecture. They were not benchmarked as replacement drafts
in this release. This recipe is the strongest configuration validated in the
recorded experiments, not a claim that every possible drafter or runtime has
been exhausted. A well-trained 33B-specific parallel drafter remains a plausible
route to larger decoding gains; the small DFlash2 pilot is not evidence against
that possibility.

## Measurement limits

These are short-output serving tests, not a broad model-quality assessment.
Structured outputs are parsed and checked against both schema and arithmetic.
Code and prose are checked for successful nonempty streams, not code execution
or semantic correctness. Greedy text can differ between configurations because
batching and kernels change numerical behavior; bitwise output parity is not
claimed. The capacity probe checks allocation and response at long context,
not retrieval or reasoning accuracy across that context.

## Credits

[Nanbeige](https://huggingface.co/Nanbeige) provides both base architectures;
[primitive-ai](https://huggingface.co/primitive-ai/Nanbeige4.3-33B-NVFP4) and
[kaitchup](https://huggingface.co/kaitchup/Nanbeige4.1-3B-NVFP4) provide the target
and draft quantizations. [vLLM](https://github.com/vllm-project/vllm) supplies
verified speculative decoding and cache/scheduling implementations.
[Speculators contributors](https://github.com/vllm-project/speculators) supply
the experimental DFlash2 architecture and trainer. The
[eugr B12X runtime](https://hub.docker.com/r/eugr/spark-vllm-b12x) supplies GB10
kernels, the direct-I/O loader, and the tested runtime integration.
