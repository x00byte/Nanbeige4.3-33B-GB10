"""Start or inspect one scoped serving container without stopping other services."""

import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def command(profile, model, draft, cache, port):
    c = json.loads((ROOT / "profiles" / (profile + ".json")).read_text())
    engine = list(c["engine_args"])
    engine[engine.index("--port") + 1] = str(port)
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        "nanbeige43-" + profile,
        "--gpus",
        "all",
        "--network",
        "host",
        "--shm-size",
        "2g",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--log-opt",
        "max-size=100m",
        "--log-opt",
        "max-file=3",
        "-v",
        str(model) + ":/model:ro",
        "-v",
        str(cache) + ":/cache",
    ]
    args += ["-v", str(draft) + ":/draft:ro"]
    args += [
        "--security-opt",
        "seccomp=unconfined",
        "-v",
        str(cache / "b12x") + ":/root/.cache/b12x",
    ]
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_CACHE_ROOT": "/cache/vllm",
        "TRITON_CACHE_DIR": "/cache/triton",
        "CUDA_CACHE_PATH": "/cache/cuda",
        "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
    }
    env.update(c.get("env", {}))
    for key, value in env.items():
        args += ["-e", key + "=" + value]
    return args + ["--entrypoint", "vllm", c["image"], "serve", *engine]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", choices=["nvfp4", "nvfp4-code"])
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--draft-dir", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--wait-timeout", type=int, default=2400)
    p.add_argument("--print-command", action="store_true")
    o = p.parse_args()
    if not 1 <= o.port <= 65535:
        p.error("Port must be between 1 and 65535")
    if o.wait_timeout < 1:
        p.error("Wait timeout must be positive")
    model, draft, cache = (
        o.model_dir.resolve(),
        o.draft_dir.resolve(),
        (o.cache_dir / o.profile).resolve(),
    )
    args = command(o.profile, model, draft, cache, o.port)
    if o.print_command:
        print(shlex.join(args))
        return
    if (model / ".gx10-incomplete").exists() or (draft / ".gx10-incomplete").exists():
        raise RuntimeError("Checkpoint download is incomplete")
    manifest = json.loads((ROOT / "profiles/nvfp4-manifest.json").read_text())
    for item in manifest["files"]:
        if (
            not (model / item["path"]).is_file()
            or (model / item["path"]).stat().st_size != item["bytes"]
        ):
            raise RuntimeError("Missing or incomplete asset: " + item["path"])
    adaptation = json.loads((draft / "adaptation.json").read_text())
    source = json.loads((ROOT / "profiles/draft-source-manifest.json").read_text())
    config = json.loads((draft / "config.json").read_text())
    if (
        adaptation["source_revision"] != source["revision"]
        or not adaptation["padding_zero_verified"]
        or not adaptation["token_ids_match_target"]
        or adaptation["verified_original_tensor_payloads"] != 963
        or config["vocab_size"] != 166912
        or config["sliding_window"] != 8192
        or (draft / "model.safetensors").stat().st_size
        != adaptation["weight_file_bytes"]
    ):
        raise RuntimeError(
            "Draft adaptation is incomplete or incompatible; rerun tools/adapt_draft.py"
        )
    profile = json.loads((ROOT / "profiles" / (o.profile + ".json")).read_text())
    if (
        profile.get("draft_window_enabled")
        and config.get("layer_types") != ["sliding_attention"] * 32
    ):
        raise RuntimeError(
            "The draft's sliding attention is not enabled; regenerate it with tools/adapt_draft.py"
        )
    if o.profile == "nvfp4-code" and any(
        t != "full_attention" for t in (config.get("layer_types") or [])
    ):
        raise RuntimeError(
            "The reference profile needs a draft generated with --full-attention"
        )
    target_config = json.loads((model / "config.json").read_text())
    token_bytes = sum(
        c["num_hidden_layers"]
        * c["num_key_value_heads"]
        * (c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]) + 8)
        for c in (target_config, config)
    )
    shared_tokens = profile["kv_bytes"] // (32 * token_bytes) * 32
    if shared_tokens < profile["shared_cache_tokens_min"]:
        raise RuntimeError(
            "Combined target/draft KV allocation is below the million-token minimum"
        )
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", o.port)) == 0:
            raise RuntimeError(
                "Port is occupied; choose another port or stop its owner explicitly"
            )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    if gpu.stdout.strip():
        raise RuntimeError(
            "GPU is occupied; stop its owner explicitly before starting this recipe"
        )
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 104 * 2**30:
        raise RuntimeError(
            "Need at least 104 GiB available unified memory before loading"
        )
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", profile["image"]])
    )[0]
    if image["Architecture"] != "arm64":
        raise RuntimeError("This pinned recipe requires the aarch64 image")
    if (
        profile.get("runtime_recipe")
        and image["Config"].get("Labels", {}).get("io.x00byte.nanbeige.recipe")
        != profile["runtime_recipe"]
    ):
        raise RuntimeError(
            "Build the tested patched image first: docker build -t nanbeige43-gb10-nvfp4:20261025 ."
        )
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(args, check=True)
    name = "nanbeige43-" + o.profile
    print("Follow startup: docker logs -f " + name, flush=True)
    deadline = time.monotonic() + o.wait_timeout
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            check=True,
            capture_output=True,
            text=True,
        )
        if state.stdout.strip() != "true":
            raise RuntimeError("Serving container exited; inspect docker logs " + name)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{o.port}/v1/models", timeout=3
            ) as response:
                names = {entry["id"] for entry in json.load(response)["data"]}
            if "Nanbeige4.3-33B" in names:
                print(f"Nanbeige4.3-33B is ready on port {o.port}")
                return
        except (OSError, ValueError):
            pass
        time.sleep(5)
    raise TimeoutError("Model did not become ready; inspect docker logs " + name)


if __name__ == "__main__":
    main()
