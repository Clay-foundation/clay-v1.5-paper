"""Provenance block that every results JSON should carry (script, commit, args, versions, UTC time)."""
from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "pandas", "timm", "rasterio",
            "pystac-client", "planetary-computer", "odc-stac", "torchgeo")


def _git(root: Path, *a: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def scrub(v, root: Path):
    """Keep results JSONs free of machine-specific absolute paths."""
    if isinstance(v, str) and v.startswith("/"):
        try:
            return str(Path(v).resolve().relative_to(root.resolve()))
        except ValueError:
            return "<abs>/" + Path(v).name
    return v


def provenance(script: str, args: dict | None = None, root: Path | None = None) -> dict:
    root = Path(root or Path(script).resolve().parent)
    repo = Path(_git(root, "rev-parse", "--show-toplevel") if _git(root, "rev-parse", "--show-toplevel") != "unavailable" else root)
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    try:
        import torch
        cuda = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception:
        cuda = {}
    return {
        "script": Path(script).name,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain", "--", str(Path(script).resolve()))),
        "args": {k: scrub(v, repo) for k, v in (args or {}).items()},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": vers,
        **cuda,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
