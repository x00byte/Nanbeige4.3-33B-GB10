"""Pad the pinned small draft's unused vocabulary rows without requantization."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while data := stream.read(8 * 1024**2):
            h.update(data)
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def vocabulary(path):
    t = json.loads((path / "tokenizer.json").read_text())
    return t["model"]["vocab"], sorted(
        (x["id"], x["content"]) for x in t["added_tokens"]
    )


def adapt(source, target, dest, full_attention=False):
    manifest = json.loads((ROOT / "profiles/draft-source-manifest.json").read_text())
    if source == dest or dest.exists():
        raise RuntimeError(
            "Output must be a new directory; source weights are kept unchanged"
        )
    if (source / ".gx10-incomplete").exists() or (target / ".gx10-incomplete").exists():
        raise RuntimeError("Finish both downloads first")
    for item in manifest["files"]:
        path = source / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError("Draft source failed verification: " + item["path"])
    if vocabulary(source) != vocabulary(target):
        raise RuntimeError("Target and draft token-to-ID mappings must match exactly")
    c = json.loads((source / "config.json").read_text())
    t = json.loads((target / "config.json").read_text())
    if (c["vocab_size"], t["vocab_size"], c["hidden_size"]) != (166144, 166912, 2560):
        raise RuntimeError("This adaptation is specific to the pinned checkpoints")
    dest.mkdir(parents=True)
    marker = dest / ".gx10-incomplete"
    marker.touch()
    for path in source.iterdir():
        if (
            path.is_file()
            and not path.name.startswith(".gx10")
            and path.suffix != ".safetensors"
        ):
            shutil.copy2(path, dest / path.name)
    c.update(vocab_size=166912, sliding_window=8192)
    if not full_attention:
        c["layer_types"] = ["sliding_attention"] * c["num_hidden_layers"]
    (dest / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    with (source / "model.safetensors").open("rb", buffering=0) as inp:
        old_length = struct.unpack("<Q", inp.read(8))[0]
        old = json.loads(inp.read(old_length))
        entries = sorted(
            ((k, v) for k, v in old.items() if k != "__metadata__"),
            key=lambda pair: pair[1]["data_offsets"][0],
        )
        header = {"__metadata__": old["__metadata__"]} if "__metadata__" in old else {}
        offset, paddings, original_hashes = 0, {}, {}
        for name, entry in entries:
            entry = {**entry, "shape": list(entry["shape"])}
            padding = 0
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                if entry["dtype"] != "BF16" or entry["shape"] != [166144, 2560]:
                    raise RuntimeError("Unexpected vocabulary tensor: " + name)
                entry["shape"][0] = 166912
                padding = 768 * 2560 * 2
                paddings[name] = padding
            size = entry["data_offsets"][1] - entry["data_offsets"][0] + padding
            entry["data_offsets"] = [offset, offset + size]
            header[name] = entry
            offset += size
        if len(paddings) != 2:
            raise RuntimeError("Expected embedding and output-head tensors")
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        output = dest / "model.safetensors"
        with output.open("wb", buffering=0) as out:
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            for name, entry in entries:
                inp.seek(8 + old_length + entry["data_offsets"][0])
                remaining = entry["data_offsets"][1] - entry["data_offsets"][0]
                h = hashlib.sha256()
                while remaining:
                    data = inp.read(min(8 * 1024**2, remaining))
                    if not data:
                        raise RuntimeError("Truncated source tensor: " + name)
                    h.update(data)
                    out.write(data)
                    remaining -= len(data)
                    os.fdatasync(out.fileno())
                    os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    os.posix_fadvise(inp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                original_hashes[name] = h.hexdigest()
                out.write(bytes(paddings.get(name, 0)))
            os.fdatasync(out.fileno())
            os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    # Read the written file back: prove every existing byte survived and every new row is zero.
    with output.open("rb", buffering=0) as stream:
        for name, entry in entries:
            stream.seek(8 + len(raw) + header[name]["data_offsets"][0])
            remaining = entry["data_offsets"][1] - entry["data_offsets"][0]
            h = hashlib.sha256()
            while remaining:
                data = stream.read(min(8 * 1024**2, remaining))
                if not data:
                    raise RuntimeError("Truncated output tensor: " + name)
                h.update(data)
                remaining -= len(data)
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            if h.hexdigest() != original_hashes[name] or any(
                stream.read(paddings.get(name, 0))
            ):
                raise RuntimeError("Output verification failed: " + name)
    report = {
        "source_repo": manifest["repo"],
        "source_revision": manifest["revision"],
        "vocabulary_before": 166144,
        "vocabulary_after": 166912,
        "token_ids_match_target": True,
        "changed_tensors": list(paddings),
        "zero_padding_bytes": sum(paddings.values()),
        "draft_attention": "full" if full_attention else "sliding",
        "draft_sliding_window": None if full_attention else 8192,
        "draft_sliding_window_layers": 0 if full_attention else 32,
        "original_tensor_payload_sha256": original_hashes,
        "verified_original_tensor_payloads": len(original_hashes),
        "padding_zero_verified": True,
        "weight_file_bytes": output.stat().st_size,
        "weight_file_sha256": sha256(output),
    }
    (dest / "adaptation.json").write_text(json.dumps(report, indent=2) + "\n")
    marker.unlink()
    print("Verified padding and all original tensor payloads:", len(original_hashes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", required=True, type=Path)
    p.add_argument("--target-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--full-attention",
        action="store_true",
        help="Reproduce the older full-attention draft reference",
    )
    o = p.parse_args()
    adapt(
        o.source_dir.resolve(),
        o.target_dir.resolve(),
        o.output.resolve(),
        o.full_attention,
    )


if __name__ == "__main__":
    main()
