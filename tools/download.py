"""Download a pinned checkpoint with resumable files and SHA-256 verification."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        while b := f.read(8 * 1024**2):
            h.update(b)
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", choices=["nvfp4", "draft-source"])
    p.add_argument("--output", required=True, type=Path)
    o = p.parse_args()
    source = json.loads(
        (ROOT / "profiles" / (o.profile + "-manifest.json")).read_text()
    )
    dest = o.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / ".gx10-incomplete").touch()
    partial = dest / ".gx10-partial"

    def fetch(item):
        name = item["path"]
        path, tmp = dest / name, partial / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        if (
            path.exists()
            and path.stat().st_size == item["bytes"]
            and digest(path) == item["sha256"]
        ):
            print("Verified " + name, flush=True)
            return
        for attempt in range(6):
            try:
                offset = tmp.stat().st_size if tmp.exists() else 0
                if offset > item["bytes"]:
                    tmp.unlink()
                    offset = 0
                if offset < item["bytes"]:
                    req = urllib.request.Request(
                        f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}/{name}",
                        headers={"Range": f"bytes={offset}-"} if offset else {},
                    )
                    with urllib.request.urlopen(req, timeout=180) as response:
                        if offset and (
                            response.status != 206
                            or not response.headers.get("Content-Range", "").startswith(
                                f"bytes {offset}-"
                            )
                        ):
                            raise RuntimeError("Server did not honor resume offset")
                        with tmp.open("ab" if offset else "wb", buffering=0) as out:
                            while b := response.read(8 * 1024**2):
                                out.write(b)
                                os.fdatasync(out.fileno())
                                os.posix_fadvise(
                                    out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                                )
                if tmp.stat().st_size != item["bytes"] or digest(tmp) != item["sha256"]:
                    tmp.unlink()
                    raise RuntimeError("Size or hash mismatch: " + name)
                tmp.replace(path)
                print("Verified " + name, flush=True)
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(30, 2**attempt))

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(
            pool.map(
                fetch,
                sorted(
                    source["files"], key=lambda x: x["path"].endswith(".safetensors")
                ),
            )
        )
    (dest / ".gx10-download.json").write_text(
        json.dumps({**source, "status": "complete-size-and-hash-verified"}, indent=2)
        + "\n"
    )
    (dest / ".gx10-incomplete").unlink()


if __name__ == "__main__":
    main()
