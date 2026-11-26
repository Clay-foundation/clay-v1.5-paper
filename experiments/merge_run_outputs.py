#!/usr/bin/env python3
"""Merge result JSONs from several runs of the same runner into one committed JSON, without editing any value.

Each input is `LABEL=path.json`. Later inputs override earlier ones per top-level key (and, for the
`cropharvest_togo`-style two-level layout, per cell). Provenance of every input (its `_meta`, producers and
settings) is kept under `_merged_from`, so a reader can trace each cell to the run (script sha256, git commit,
argv, UTC time) that produced it. Values are never recomputed or rounded here.

Usage:
  python experiments/merge_run_outputs.py --out experiments/results/all_doable_table2.json \
      main=runs/table2_x/all_doable_table2.json togo=runs/table2_y/all_doable_table2.json
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


def merge(dst: dict, src: dict, label: str, owner: dict) -> None:
    for k, v in src.items():
        if k in ("_meta", "_merged_from"):
            continue
        if k in ("_producers", "_settings"):
            dst.setdefault(k, {}).update(v)
            continue
        if isinstance(v, dict) and isinstance(dst.get(k), dict) and all(isinstance(x, dict) or not isinstance(x, (int, float)) for x in v.values()):
            for kk, vv in v.items():
                dst[k][kk] = vv
                owner[f"{k}/{kk}"] = label
        else:
            dst[k] = v
            owner[k] = label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+", help="LABEL=path.json")
    a = ap.parse_args()
    out: dict = {}
    owner: dict = {}
    sources = {}
    for item in a.inputs:
        label, path = item.split("=", 1)
        d = json.loads(Path(path).read_text())
        sources[label] = {"file": Path(path).name, "_meta": d.get("_meta") or d.get("meta")}
        merge(out, d, label, owner)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        head = "unknown"
    out["_merged_from"] = {"merge_script": "merge_run_outputs.py", "git_commit": head, "argv": sys.argv[1:],
                           "utc_timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "sources": sources, "cell_owner": owner}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out} from {len(a.inputs)} inputs, {len(owner)} cells")


if __name__ == "__main__":
    main()
