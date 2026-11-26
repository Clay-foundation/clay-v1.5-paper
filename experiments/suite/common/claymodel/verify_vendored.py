"""Verify the vendored claymodel files against MANIFEST.json (header-stripped body hashes).

    python verify_vendored.py [--upstream PATH_TO_CLAY_MODEL_CHECKOUT]

Exit code 0 only if every file matches. With --upstream, also compares the upstream bytes at
the pinned commit (needs `git`). This is the check to use instead of `git rev-parse` on the
vendored directory (which is not an independent git checkout).
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify(upstream: str | None = None) -> list[str]:
    man = json.loads((HERE / "MANIFEST.json").read_text())
    marker = (man["header_end_marker"] + "\n").encode()
    errs = []
    for name, rec in man["files"].items():
        raw = (HERE / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != rec["vendored_sha256"]:
            errs.append(f"{name}: file hash differs from MANIFEST")
        if marker not in raw:
            errs.append(f"{name}: header marker missing")
            continue
        body = raw.split(marker, 1)[1]
        if hashlib.sha256(body).hexdigest() != rec["body_sha256"]:
            errs.append(f"{name}: body hash differs from MANIFEST")
        if not rec["modified"] and rec["body_sha256"] != rec["upstream_sha256"]:
            errs.append(f"{name}: body is not upstream")
        if upstream:
            up = subprocess.check_output(["git", "-C", upstream, "show", f"{man['upstream_commit']}:claymodel/{name}"])
            if hashlib.sha256(up).hexdigest() != rec["upstream_sha256"]:
                errs.append(f"{name}: upstream checkout differs from MANIFEST upstream hash")
    return errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream")
    a = ap.parse_args()
    e = verify(a.upstream)
    print("vendored claymodel: OK" if not e else "\n".join(e))
    sys.exit(1 if e else 0)
