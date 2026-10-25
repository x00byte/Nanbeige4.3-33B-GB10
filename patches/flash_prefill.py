# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Packed INT4 layout adapted from vLLM's int4_per_token_head.py.
"""Single-sequence FlashAttention prefill over a transient dequantized INT4 view."""

import torch
from vllm.triton_utils import triton, tl


@triton.jit
def _dequant(
    K,
    V,
    KS,
    VS,
    BT,
    KO,
    VO,
    SEQ,
    CU,
    H: tl.constexpr,
    D: tl.constexpr,
    KB: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    VB: tl.constexpr,
    VT: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    SB: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    TB: tl.constexpr,
    TT: tl.constexpr,
    TH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Chunked prefill exposes a planned CPU maximum but only this GPU prefix is written.
    actual_length = tl.load(SEQ)
    if tl.program_id(0) == 0:
        tl.store(CU, 0)
        tl.store(CU + 1, actual_length)
    valid = x < actual_length * H * D
    d = x % D
    h = (x // D) % H
    t = x // (D * H)
    b = tl.load(BT + t // 32, mask=valid, other=0).to(tl.int64)
    slot = t % 32
    kp = tl.load(K + b * KB + slot * KT + h * KH + (d // 2) * KD, mask=valid, other=0)
    vp = tl.load(V + b * VB + slot * VT + h * VH + (d // 2) * VD, mask=valid, other=0)
    ks = tl.load(KS + b * SB + slot * ST + h * SH, mask=valid, other=0).to(
        tl.int32, bitcast=True
    )
    vs = tl.load(VS + b * TB + slot * TT + h * TH, mask=valid, other=0).to(
        tl.int32, bitcast=True
    )
    kv = (
        ((kp.to(tl.int32) >> (4 * (d % 2))) & 15).to(tl.float32)
        - (ks & 15).to(tl.float32)
    ) * (ks & -16).to(tl.float32, bitcast=True)
    vv = (
        ((vp.to(tl.int32) >> (4 * (d % 2))) & 15).to(tl.float32)
        - (vs & 15).to(tl.float32)
    ) * (vs & -16).to(tl.float32, bitcast=True)
    tl.store(KO + x, kv, mask=valid)
    tl.store(VO + x, vv, mask=valid)


_buffers = {}


def dequantize(k, v, ks, vs, table, length, seq):
    heads = k.shape[2]
    key = (k.device, heads)
    capacity = 262144
    if length > capacity:
        raise ValueError("Prefill workspace limit exceeded")
    if key not in _buffers:
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "Nanbeige FlashAttention prefill: 1 GiB workspace; KV block strides %s/%s bytes",
            k.stride(0),
            v.stride(0),
        )
        _buffers[key] = (
            torch.empty((capacity, heads, 128), dtype=torch.float16, device=k.device),
            torch.empty((capacity, heads, 128), dtype=torch.float16, device=k.device),
            torch.empty(2, dtype=torch.int32, device=k.device),
        )
    ko, vo = _buffers[key][0][:length], _buffers[key][1][:length]
    cu = _buffers[key][2]
    _dequant[(triton.cdiv(length * heads * 128, 512),)](
        k,
        v,
        ks,
        vs,
        table,
        ko,
        vo,
        seq,
        cu,
        heads,
        128,
        *k.stride(),
        *v.stride(),
        *ks.stride(),
        *vs.stride(),
        512,
    )
    return ko, vo, cu


def flash_prefill(q, k, v, out, cu_q, length, table, ks, vs, scale, seq):
    from vllm.v1.attention.ops.int4_per_token_head import single_rht
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    ko, vo, cu_k = dequantize(k, v, ks, vs, table, length, seq)
    # Retain the original BF16 query rotation and the BF16 intermediate output rounding.
    qr = single_rht(q.float()).to(q.dtype).to(torch.float16)
    rotated = flash_attn_varlen_func(
        qr,
        ko,
        vo,
        max_seqlen_q=q.shape[0],
        cu_seqlens_q=cu_q,
        max_seqlen_k=length,
        cu_seqlens_k=cu_k,
        softmax_scale=scale / 128,
        causal=True,
        fa_version=2,
    )
    rotated = rotated.to(q.dtype)
    out.copy_((single_rht(rotated.float(), inverse=True) / 128).to(q.dtype))
