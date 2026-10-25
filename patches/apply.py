"""Apply the scoped prefill changes only to the exact tested vLLM source."""

import hashlib
from pathlib import Path
import shutil

SOURCE = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/int4_per_token_head.py"
)
EXPECTED_SHA256 = "d91968f280cd8eaed7570ab255f0276b40e833e18a25066d371d9ca8149425f2"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Pinned source no longer matches the expected patch context")
    return source.replace(old, new)


def main():
    original = SOURCE.read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError(
            "INT4 source hash mismatch; refusing to patch a different runtime"
        )
    source = original.decode()
    source = replace_once(
        source,
        "    BLOCK_Q = BLOCK_M // num_queries_per_kv\n",
        "    # Nanbeige prefill: retain the original decode launch parameters.\n"
        "    if max_seqlen_q >= 256 and head_size == 128:\n"
        "        if num_queries_per_kv == 7:\n"
        "            BLOCK_M = 64\n"
        "        elif num_queries_per_kv == 5:\n"
        "            BLOCK_M = 32\n"
        "    BLOCK_Q = BLOCK_M // num_queries_per_kv\n",
    )
    source = replace_once(
        source,
        "    TILE_SIZE_DECODE = _get_tile_size(\n",
        "    if max_seqlen_q >= 256 and head_size == 128 and num_queries_per_kv == 7:\n"
        "        TILE_SIZE_PREFILL = 64\n"
        "    TILE_SIZE_DECODE = _get_tile_size(\n",
    )
    source = replace_once(
        source,
        "    q_orig_dtype = q.dtype\n",
        "    # A transient dequantized view accelerates single-sequence long prefill.\n"
        "    # Persistent cache storage and the decode path retain their INT4 format.\n"
        "    if (max_seqlen_q >= 256 and len(seqused_k) == 1\n"
        "            and 256 <= max_seqlen_k <= 262144\n"
        "            and q.shape[1:] == (56, 128) and k_cache.shape[1:3] == (32, 8)\n"
        "            and q.dtype == torch.bfloat16 and tuple(window_size) == (-1, -1)\n"
        "            and not softcap and sinks is None and alibi_slopes is None\n"
        "            and qq_bias is None and output_scale is None and mm_prefix_range is None):\n"
        "        from vllm.v1.attention.ops.nanbeige_flash_prefill import flash_prefill\n"
        "        flash_prefill(q, k_cache, v_cache, out, cu_seqlens_q, max_seqlen_k,\n"
        "                      block_table, k_scale_cache, v_scale_cache, softmax_scale, seqused_k)\n"
        "        return\n"
        "    q_orig_dtype = q.dtype\n",
    )
    compile(source, str(SOURCE), "exec")
    SOURCE.write_text(source)
    shutil.copy2(
        Path(__file__).with_name("flash_prefill.py"),
        SOURCE.with_name("nanbeige_flash_prefill.py"),
    )
    print("Applied Nanbeige flash-prefill-v3 to the pinned INT4 implementation")


if __name__ == "__main__":
    main()
