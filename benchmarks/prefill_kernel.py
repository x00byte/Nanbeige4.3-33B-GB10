"""Compare the prefill patch with the pinned native kernel on an otherwise idle GPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", required=True, type=Path)
o = p.parse_args()
if subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
).strip():
    raise RuntimeError("Use an idle GPU; this script never stops another process")
import torch

sys.path.insert(0, str(ROOT / "patches"))
from flash_prefill import flash_prefill, dequantize
from vllm.v1.attention.ops import int4_per_token_head as orig

if (
    hashlib.sha256(Path(orig.__file__).read_bytes()).hexdigest()
    != "d91968f280cd8eaed7570ab255f0276b40e833e18a25066d371d9ca8149425f2"
):
    raise RuntimeError(
        "Run this comparison in the pinned BASE image, not the patched serving image"
    )
report = {
    "passed": False,
    "started": datetime.now(timezone.utc).isoformat(),
    "cases": [],
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "method": {
        "timing": "median of 5 CUDA event measurements, after warmup",
        "reference": "float32 attention over exactly dequantized cache values, sampled query prefix",
        "maximum_reference_relative_l2_error": 0.01,
    },
}


def save():
    o.output.parent.mkdir(parents=True, exist_ok=True)
    o.output.write_text(json.dumps(report, indent=2) + "\n")


torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False


def run_case(qlen, klen, upper, layers):
    heads, kvheads = 56, 8
    nb = (klen + 31) // 32
    # Native backend uses strided views into interleaved K/V storage.
    cache = torch.empty(
        (nb, layers, kvheads, 32, 128), dtype=torch.uint8, device="cuda"
    )
    k, v = cache[:, layers // 2].transpose(1, 2).split(64, dim=-1)
    ks = torch.empty((nb, kvheads, 32), dtype=torch.float32, device="cuda").transpose(
        1, 2
    )
    vs = torch.empty_like(ks)
    keys = torch.randn((nb * 32, kvheads, 128), dtype=torch.bfloat16, device="cuda")
    values = torch.randn_like(keys)
    orig.reshape_and_cache_int4(
        keys,
        values,
        k,
        v,
        torch.arange(nb * 32, device="cuda", dtype=torch.long),
        k_scale_cache=ks,
        v_scale_cache=vs,
    )
    del keys, values
    table = torch.randperm(nb, dtype=torch.int32, device="cuda").unsqueeze(0)
    q = torch.randn((qlen, heads, 128), dtype=torch.bfloat16, device="cuda")
    a = torch.empty_like(q)
    b = torch.empty_like(q)
    cu = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
    seq = torch.tensor([klen], dtype=torch.int32, device="cuda")
    args = dict(
        cu_seqlens_q=cu,
        max_seqlen_q=qlen,
        seqused_k=seq,
        max_seqlen_k=klen,
        softmax_scale=128**-0.5,
        window_size=(-1, -1),
        block_table=table,
        softcap=0,
        sinks=None,
        alibi_slopes=None,
        use_alibi_sqrt=False,
        qq_bias=None,
        output_scale=None,
        mm_prefix_range=None,
        k_scale_cache=ks,
        v_scale_cache=vs,
    )

    def baseline():
        orig.unified_attention_int4(q, k, v, a, **args)

    def candidate():
        flash_prefill(q, k, v, b, cu, upper, table, ks, vs, 128**-0.5, seq)

    baseline()
    candidate()
    torch.cuda.synchronize()
    ko, vo, actual_cu = dequantize(k, v, ks, vs, table, upper, seq)
    assert actual_cu.cpu().tolist() == [0, klen]
    samples = torch.tensor([0, klen // 2, klen - 1], dtype=torch.long, device="cuda")
    pages = table[0, samples // 32].long()
    slots = samples % 32
    for packed, scales, result in [(k, ks, ko), (v, vs, vo)]:
        values = packed[pages, slots]
        bits = scales[pages, slots].contiguous().view(torch.int32)
        scale = (bits & -16).view(torch.float32)
        zp = (bits & 15).float()
        unpacked = (
            torch.stack((values & 15, values >> 4), dim=-1)
            .reshape(3, kvheads, 128)
            .float()
        )
        expected = ((unpacked - zp[..., None]) * scale[..., None]).to(torch.float16)
        torch.testing.assert_close(result[samples], expected, atol=0, rtol=0)
    del (
        packed,
        scales,
        result,
        values,
        bits,
        scale,
        zp,
        unpacked,
        expected,
        samples,
        pages,
        slots,
        ko,
        vo,
        actual_cu,
    )
    row = {
        "dequantized_samples_exact": True,
        "gpu_prefix_length_matches": True,
        "query_tokens": qlen,
        "kv_tokens": klen,
        "planned_context_upper_bound": upper,
        "interleaved_layers": layers,
        "max_k_byte_offset": (nb - 1) * k.stride(0),
        "max_abs_diff_native": (a.float() - b.float()).abs().max().item(),
        "mean_abs_diff_native": (a.float() - b.float()).abs().mean().item(),
    }
    assert torch.isfinite(b).all()
    # High precision reference for a small prefix of the queries, using exact dequantized cache.
    if klen <= 8192:
        ids = table.reshape(-1).long()
        kp = k[ids].reshape(-1, kvheads, 64)[:klen]
        vp = v[ids].reshape(-1, kvheads, 64)[:klen]

        def full(p, s):
            bits = s[ids].reshape(-1, kvheads)[:klen].contiguous().view(torch.int32)
            scale = (bits & -16).view(torch.float32)
            zp = (bits & 15).float()
            vals = (
                torch.stack((p & 15, p >> 4), dim=-1)
                .reshape(klen, kvheads, 128)
                .float()
            )
            return ((vals - zp[..., None]) * scale[..., None]).repeat_interleave(
                7, dim=1
            )

        kr = full(kp, ks)
        vr = full(vp, vs)
        n = min(32, qlen)
        qr = orig.single_rht(q[:n].float()).to(q.dtype).float()
        scores = torch.einsum("qhd,khd->hqk", qr, kr) * (128**-0.5 / 128)
        mask = (
            torch.arange(klen, device="cuda")[None, :]
            <= (klen - qlen + torch.arange(n, device="cuda"))[:, None]
        )
        scores = scores.masked_fill(~mask[None], float("-inf"))
        rotated = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vr).to(q.dtype)
        golden = (orig.single_rht(rotated.float(), inverse=True) / 128).to(q.dtype)
        row["native_mae_to_fp32_reference"] = (
            (a[:n].float() - golden.float()).abs().mean().item()
        )
        row["flash_mae_to_fp32_reference"] = (
            (b[:n].float() - golden.float()).abs().mean().item()
        )
        row["flash_max_abs_error_to_fp32_reference"] = (
            (b[:n].float() - golden.float()).abs().max().item()
        )
        row["native_relative_l2_error"] = (
            a[:n].float() - golden.float()
        ).norm().item() / max(golden.float().norm().item(), 1e-12)
        row["flash_relative_l2_error"] = (
            b[:n].float() - golden.float()
        ).norm().item() / max(golden.float().norm().item(), 1e-12)
        print("REFERENCE", row, flush=True)
        assert row["flash_relative_l2_error"] < 0.01, row
        del kp, vp, kr, vr, qr, scores, mask, rotated, golden
    for label, fn in [("native", baseline), ("flash", candidate)]:
        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        row[label + "_median_ms"] = statistics.median(samples)
    row["speedup"] = row["native_median_ms"] / row["flash_median_ms"]
    row["passed"] = True
    report["cases"].append(row)
    save()
    print(row, flush=True)


for qlen, klen, upper, layers in [
    (128, 128, 128, 1),
    (257, 8192, 8192, 1),
    (4096, 4096, 262128, 1),
    (4096, 16384, 262128, 1),
    (4096, 32768, 262128, 84),
    (4096, 65536, 262128, 1),
    (4096, 262128, 262128, 1),
]:
    run_case(qlen, klen, upper, layers)
report["passed"] = True
report["completed"] = datetime.now(timezone.utc).isoformat()
save()
print("FLASH PREFILL PASSED", flush=True)
