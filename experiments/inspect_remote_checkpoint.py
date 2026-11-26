"""Inspect public Clay checkpoint files without downloading them or unpickling anything.

Reads only bounded ranges of the public ZIP archives over HTTP (central directory + the tiny `data.pkl`
metadata pickle, interpreted symbolically: no import, no call, no `__setstate__`), plus the advertised
size and LFS SHA-256 from the Hugging Face tree API. It does NOT hash the whole file: the digest printed
is the one the Hub advertises (verify a downloaded copy with `common/fetch_checkpoint.py`).

    python experiments/inspect_remote_checkpoint.py [--out experiments/results] [--only v1.5]

Outputs (default `experiments/results/`, where `run_clay_v15_experiments.py` reads
`clay-v1.5-saved-metadata.json`):
  checkpoint_inspection.json                    all artifacts: revision, URL, size, LFS sha256, params, val_loss, ...
  clay-v1.5-saved-metadata.json                 Lightning hyper-parameters / epoch / global_step of v1.5 (no weights)
  checkpoint_inspection/<tag>-zip-entries.json  archive members
  checkpoint_inspection/<tag>-tensor-metadata.json  tensor shapes (checkpoints only)

The files below live at these exact revisions. The v1.5 checkpoint is on `made-with-clay/Clay` main; the compiled
encoders and v1.0 / v0 checkpoints were moved to `made-with-clay/Clay-legacy` (Sep 2025) and are also present on
`made-with-clay/Clay` at the older revision 5bfb3c77.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import os
import pickletools
import re
import struct
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF = "https://huggingface.co"
CLAY, LEGACY = "made-with-clay/Clay", "made-with-clay/Clay-legacy"
REV_CLAY_MAIN = "70200ebcccdf67bf2a0cb9984c77ddee26c10ed2"
REV_CLAY_COMPILED = "5bfb3c778cf1b51f0ecb6b92395ebdebae893e24"   # "Add compiled clay v1.5 models" (2024-12-04)
REV_LEGACY = "f521faece53b6b15df4deb2fc7215a03e561fb46"
# tag -> (repo, revision, path, kind)
ARTIFACTS = {
    "v1.5": (CLAY, REV_CLAY_MAIN, "v1.5/clay-v1.5.ckpt", "ckpt"),
    "v1.5-encoder-cpu-pt2": (LEGACY, REV_LEGACY, "v1.5/compiled/clay-v1.5-encoder-cpu.pt2", "pt2"),
    "v1.5-encoder-pt2": (LEGACY, REV_LEGACY, "v1.5/compiled/clay-v1.5-encoder.pt2", "pt2"),
    "v1.5-encoder-cpu-pt2-Clay-5bfb3c77": (CLAY, REV_CLAY_COMPILED, "v1.5/compiled/clay-v1.5-encoder-cpu.pt2", "pt2"),
    "v1.0": (LEGACY, REV_LEGACY, "v1/clay-v1-base.ckpt", "ckpt"),          # was Clay-1.0.5.7_epoch-13_val-loss-0.3098.ckpt
    "v0.0": (LEGACY, REV_LEGACY, "v0/Clay-v0.0_epoch-02_val-loss-0.52.ckpt", "ckpt"),
    "v0.1": (LEGACY, REV_LEGACY, "v0/Clay_v0.1_epoch-24_val-loss-0.46.ckpt", "ckpt"),
}


def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "clay-paper-repro"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


class RangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.requests, self.transferred = [], 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError('negative offset')
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b''
        if n > 4_000_000 or self.transferred + n > 32_000_000:
            raise ValueError('range inspection download bound exceeded')
        start = self.pos
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{start+n-1}', 'User-Agent': 'clay-paper-audit'})
        with urllib.request.urlopen(req, timeout=60) as response:
            crange = response.headers.get('Content-Range')
            body = response.read(n+1)
            if response.status != 206 or not crange or not crange.startswith(f'bytes {start}-') or len(body) != n:
                raise ValueError(f'unexpected range response {response.status} {crange} length {len(body)}')
        self.transferred += n
        self.pos += n
        self.requests.append({'start': start, 'bytes': n, 'content_range': crange, 'sha256': hashlib.sha256(body).hexdigest()})
        return body


MARK = object()


def symbolic_pickle(data):
    """Interpret data-building opcodes as plain data. GLOBAL/REDUCE are inert records.

    No imports, function calls, persistent storage loads, object constructors or
    __setstate__ operations are taken from the pickle. Unknown opcodes fail closed.
    """
    stack, memo, counts = [], {}, collections.Counter()
    def marked():
        vals = []
        while stack[-1] is not MARK:
            vals.append(stack.pop())
        stack.pop()
        return vals[::-1]
    for op, arg, pos in pickletools.genops(data):
        name = op.name
        counts[name] += 1
        if name in ('PROTO', 'FRAME'):
            continue
        elif name == 'MARK': stack.append(MARK)
        elif name == 'EMPTY_DICT': stack.append({})
        elif name == 'EMPTY_LIST': stack.append([])
        elif name == 'EMPTY_TUPLE': stack.append(())
        elif name == 'NONE': stack.append(None)
        elif name == 'NEWTRUE': stack.append(True)
        elif name == 'NEWFALSE': stack.append(False)
        elif name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE', 'BININT', 'BININT1', 'BININT2', 'LONG1', 'LONG4', 'INT', 'LONG', 'BINFLOAT', 'FLOAT', 'BINBYTES', 'SHORT_BINBYTES'): stack.append(arg)
        elif name in ('BINPUT', 'LONG_BINPUT', 'PUT'): memo[int(arg)] = stack[-1]
        elif name == 'MEMOIZE': memo[len(memo)] = stack[-1]
        elif name in ('BINGET', 'LONG_BINGET', 'GET'): stack.append(memo[int(arg)])
        elif name == 'GLOBAL': stack.append({'__global__': arg})
        elif name == 'STACK_GLOBAL':
            cls, mod = stack.pop(), stack.pop()
            stack.append({'__global__': mod+' '+cls})
        elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
            n = int(name[-1]); vals = stack[-n:]; del stack[-n:]; stack.append(tuple(vals))
        elif name == 'TUPLE': stack.append(tuple(marked()))
        elif name == 'LIST': stack.append(marked())
        elif name == 'DICT':
            vals = marked(); stack.append(dict(zip(vals[::2], vals[1::2])))
        elif name == 'BINPERSID': stack.append({'__persistent__': stack.pop()})
        elif name == 'REDUCE':
            args, func = stack.pop(), stack.pop()
            if func == {'__global__': 'collections OrderedDict'} and not args:
                stack.append({})
            else:
                stack.append({'__reduce__': func, 'args': args})
        elif name == 'SETITEM':
            value, key = stack.pop(), stack.pop(); stack[-1][key] = value
        elif name == 'SETITEMS':
            vals = marked(); stack[-1].update(zip(vals[::2], vals[1::2]))
        elif name == 'APPEND':
            value = stack.pop(); stack[-1].append(value)
        elif name == 'APPENDS':
            values = marked(); stack[-1].extend(values)
        elif name == 'BUILD':
            state = stack.pop()
            if isinstance(stack[-1], dict): stack[-1]['__build_state__'] = state
            else: raise ValueError(f'unsupported BUILD target at {pos}')
        elif name == 'NEWOBJ':
            args, cls = stack.pop(), stack.pop(); stack.append({'__newobj__': cls, 'args': args})
        elif name == 'POP': stack.pop()
        elif name == 'DUP': stack.append(stack[-1])
        elif name == 'STOP':
            if len(stack) != 1: raise ValueError('unexpected final stack size')
            return stack[0], dict(counts)
        else:
            raise ValueError(f'unsupported inert opcode {name} at {pos}')
    raise ValueError('missing STOP')


def hub_file_info(repo, rev, path):
    """Advertised size and LFS sha256 for `path` at `rev` (Hub tree API; no file download)."""
    d = str(Path(path).parent)
    for e in http_json(f"{HF}/api/models/{repo}/tree/{rev}/{d}" if d != "." else f"{HF}/api/models/{repo}/tree/{rev}"):
        if e["path"] == path:
            return {"size": e["size"], "lfs_sha256": (e.get("lfs") or {}).get("oid")}
    raise FileNotFoundError(f"{repo}@{rev}:{path} not found")


def read_scalar_tensor(archive, prefix, value):
    """Value of a one-element saved tensor (for example a ModelCheckpoint best score) by reading its tiny storage entry."""
    args = value.get("args", ())
    pid = args[0]["__persistent__"]  # ('storage', <FloatStorage>, key, location, numel)
    key, numel = pid[2], pid[4]
    if numel != 1:
        return None
    dtype = (pid[1].get("__global__", "") if isinstance(pid[1], dict) else str(pid[1]))
    raw = archive.read(f"{prefix}/data/{key}")
    return struct.unpack("<d" if "Double" in dtype else "<f", raw[: 8 if "Double" in dtype else 4])[0]


def find_val_loss(decoded, archive, prefix):
    """Return (value, where). Looks for a ModelCheckpoint best_model_score / current_score in the saved callbacks."""
    found = []
    def walk(o, path):
        if isinstance(o, dict):
            if "__reduce__" in o:
                return
            for k, v in o.items():
                walk(v, path + [str(k)])
    cb = decoded.get("callbacks", {})
    for ck, cv in (cb.items() if isinstance(cb, dict) else []):
        if isinstance(cv, dict):
            for fld in ("best_model_score", "current_score"):
                v = cv.get(fld)
                if isinstance(v, dict) and "__reduce__" in v:
                    try:
                        val = read_scalar_tensor(archive, prefix, v)
                    except Exception:
                        val = None
                    if val is not None:
                        found.append((val, f"callbacks[{ck!s:.60}].{fld}", cv.get("monitor")))
    return found


def inspect(tag, repo, rev, path, kind, out_dir):
    info = hub_file_info(repo, rev, path)
    url = f"{HF}/{repo}/resolve/{rev}/{path}"
    remote = RangeReader(url, info["size"])
    res = {"tag": tag, "repo": repo, "revision": rev, "path": path, "public_url": url,
           "advertised_bytes": info["size"], "advertised_lfs_sha256": info["lfs_sha256"], "kind": kind}
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        entries = [{"name": x.filename, "size": x.file_size, "compressed_size": x.compress_size, "crc32": f"{x.CRC:08x}"} for x in infos]
        (out_dir / f"{tag}-zip-entries.json").write_text(json.dumps(entries, indent=1))
        res.update({"zip_entries": len(entries), "zip_uncompressed_bytes": sum(x["size"] for x in entries)})
        pkls = [x for x in infos if x.filename.endswith("/data.pkl")]
        if kind == "ckpt" and pkls:
            pkl = pkls[0]
            prefix = pkl.filename.rsplit("/", 1)[0]
            if pkl.file_size > 4_000_000:
                raise ValueError("pickle metadata too large")
            data = archive.read(pkl)
            res.update({"pickle_bytes": len(data), "pickle_sha256": hashlib.sha256(data).hexdigest()})
            decoded, opcodes = symbolic_pickle(data)
            summary = {k: v for k, v in decoded.items() if k not in ("state_dict", "optimizer_states", "lr_schedulers", "loops", "callbacks")}
            shapes, groups_n, groups_p = {}, collections.Counter(), collections.Counter()
            for key, value in decoded.get("state_dict", {}).items():
                if not isinstance(value, dict) or "__reduce__" not in value:
                    continue
                a = value.get("args", ())
                if len(a) < 4 or not isinstance(a[2], tuple):
                    continue
                shapes[key] = {"shape": list(a[2]), "stride": list(a[3])}
                g = ".".join(key.split(".")[:2])
                groups_n[g] += 1
                groups_p[g] += math.prod(a[2])
            (out_dir / f"{tag}-tensor-metadata.json").write_text(json.dumps(shapes, indent=1))
            vl = find_val_loss(decoded, archive, prefix)
            m = re.search(r"val-loss-([0-9.]+?)(?:\.ckpt|$)", path)
            res.update({"epoch": summary.get("epoch"), "global_step": summary.get("global_step"),
                        "pytorch_lightning_version": summary.get("pytorch-lightning_version"),
                        "state_dict_tensors": len(shapes), "state_dict_elements": sum(groups_p.values()),
                        "tensor_groups": {g: {"tensors": groups_n[g], "elements": groups_p[g]} for g in groups_n},
                        "val_loss_from_checkpoint_callbacks": [{"value": v, "where": w, "monitor": mon} for v, w, mon in vl],
                        "val_loss_from_filename": float(m.group(1)) if m else None,
                        "saved_metadata_keys": list(summary)})
            if tag == "v1.5":
                (out_dir.parent / "clay-v1.5-saved-metadata.json").write_text(json.dumps(summary, indent=2, default=str))
    res["downloaded_bytes"] = remote.transferred
    return res


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.environ.get("CLAY_INSPECT_OUT", str(ROOT / "results")), help="output directory (default experiments/results)")
    ap.add_argument("--only", action="append", choices=list(ARTIFACTS), help="inspect only these tags (repeatable)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "checkpoint_inspection").mkdir(parents=True, exist_ok=True)
    results = []
    for tag, (repo, rev, path, kind) in ARTIFACTS.items():
        if a.only and tag not in a.only:
            continue
        r = inspect(tag, repo, rev, path, kind, out / "checkpoint_inspection")
        results.append(r)
        enc = r.get("tensor_groups", {}).get("model.encoder")
        print(f"{tag:38s} {r['advertised_bytes']:>13,d} B  sha256 {str(r['advertised_lfs_sha256'])[:12]}...  "
              f"encoder={enc}  epoch={r.get('epoch')} step={r.get('global_step')} "
              f"val_loss(callbacks)={[x['value'] for x in r.get('val_loss_from_checkpoint_callbacks', [])]} val_loss(filename)={r.get('val_loss_from_filename')}  "
              f"fetched {r['downloaded_bytes']:,d} B")
    doc = {"provenance": {"script": "inspect_remote_checkpoint.py", "git_commit": git_commit(), "args": {"out": os.path.relpath(a.out, ROOT.parent) if os.path.isabs(a.out) and a.out.startswith(str(ROOT.parent)) else a.out, "only": a.only},
                          "python": sys.version.split()[0], "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
           "artifacts": results}
    (out / "checkpoint_inspection.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
