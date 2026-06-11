#!/usr/bin/env python3
"""
Prints shell export statements that replicate the notebook's environment setup.
Intended to be eval'd by setup_env.sh:

    eval "$(python3 setup_env.py)"

Mirrors the notebook's §4 (config) + §6 (setup) cells exactly.
"""
from __future__ import annotations

import os
import platform
import socket
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "README.md").exists() and (path / "cookbooks").exists():
            return path
    return start


def free_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def default_framework_repo(root: Path) -> Path:
    for candidate in (root / "packages" / "cosmos-framework", root / "packages" / "cosmos3"):
        if (candidate / "pyproject.toml").exists() and (candidate / "cosmos_framework").exists():
            return candidate
    return root / "packages" / "cosmos3"


def default_uv_group() -> str:
    if platform.machine() == "aarch64":
        return "cu130-train"
    return "cu128-train"


def detect_gpu_count() -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=10, stderr=subprocess.DEVNULL,
        ).decode()
        count = len([l for l in out.strip().splitlines() if l])
        if count > 0:
            return str(count)
    except Exception:
        pass
    return "4"


def default_cache_path(name: str, transfer_root: Path) -> str:
    root = os.environ.get("COSMOS3_CACHE_ROOT")
    if root:
        return str((Path(root).expanduser() / name).resolve())
    return str((transfer_root / ".cache" / name).resolve())


# ── Resolve paths ────────────────────────────────────────────────────────────
COSMOS_ROOT = find_repo_root(Path(__file__).resolve().parent)
COSMOS3_TRANSFER_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "transfer"
COSMOS3_REPO = Path(os.environ.get("COSMOS3_REPO", default_framework_repo(COSMOS_ROOT))).resolve()
COSMOS3_GIT_URL = os.environ.get("COSMOS3_GIT_URL", "https://github.com/NVIDIA/cosmos-framework.git")
COSMOS3_UV_GROUP = os.environ.get("COSMOS3_UV_GROUP", default_uv_group())
COSMOS3_TRANSFER_OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_TRANSFER_OUTPUT_ROOT", COSMOS3_TRANSFER_ROOT / "outputs" / "notebooks")
).resolve()
COSMOS3_NUM_GPUS = os.environ.get("COSMOS3_NUM_GPUS") or detect_gpu_count()
UV_CACHE_DIR = os.environ.get("COSMOS3_UV_CACHE_DIR", default_cache_path("uv", COSMOS3_TRANSFER_ROOT))
HF_HOME = os.environ.get("COSMOS3_HF_HOME", default_cache_path("huggingface", COSMOS3_TRANSFER_ROOT))
ALL_GPUS = ",".join(str(i) for i in range(int(COSMOS3_NUM_GPUS)))
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", ALL_GPUS)
MASTER_ADDR = os.environ.get("COSMOS3_MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("COSMOS3_MASTER_PORT", free_local_port())

# ── Emit shell exports (eval'd by setup_env.sh) ──────────────────────────────
exports = {
    "COSMOS_ROOT": str(COSMOS_ROOT),
    "COSMOS3_TRANSFER_ROOT": str(COSMOS3_TRANSFER_ROOT),
    "COSMOS3_REPO": str(COSMOS3_REPO),
    "COSMOS3_GIT_URL": COSMOS3_GIT_URL,
    "COSMOS3_UV_GROUP": COSMOS3_UV_GROUP,
    "COSMOS3_TRANSFER_OUTPUT_ROOT": str(COSMOS3_TRANSFER_OUTPUT_ROOT),
    "COSMOS3_NUM_GPUS": COSMOS3_NUM_GPUS,
    "UV_CACHE_DIR": UV_CACHE_DIR,
    "HF_HOME": HF_HOME,
    "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
    "COSMOS3_MASTER_ADDR": MASTER_ADDR,
    "COSMOS3_MASTER_PORT": MASTER_PORT,
}

for key, val in exports.items():
    # Shell-safe quoting
    print(f"export {key}={val!r}")

# Mirror the notebook: clear LD_LIBRARY_PATH
print("unset LD_LIBRARY_PATH")
