"""Download the released Clay v1.5 checkpoint at a pinned Hugging Face revision and verify its SHA-256.

    python fetch_checkpoint.py [--dest checkpoints/v1.5]      # default; ~5.2 GB

The revision and digest are the ones recorded in experiments/results/checkpoint_inspection.json.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clayrt import CKPT_SHA256, sha256  # noqa: E402

REPO, REVISION, FILE = "made-with-clay/Clay", "70200ebcccdf67bf2a0cb9984c77ddee26c10ed2", "v1.5/clay-v1.5.ckpt"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="checkpoints", help="directory to place v1.5/clay-v1.5.ckpt under")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(REPO, FILE, revision=REVISION, local_dir=a.dest))
    got = sha256(p)
    if got != CKPT_SHA256:
        sys.exit(f"SHA-256 mismatch: {got} != {CKPT_SHA256}")
    print(f"{p}  sha256 OK ({CKPT_SHA256})\nexport CLAY_V15_CKPT={p}")
