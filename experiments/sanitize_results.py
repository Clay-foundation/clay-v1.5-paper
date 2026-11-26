#!/usr/bin/env python3
"""Make results JSONs free of machine-specific absolute paths and fill the git commit of the code that produced them.

Usage: python experiments/sanitize_results.py FILE.json [FILE.json ...] [--git-commit SHA --git-note TEXT]
Every string that starts with "/" or contains "/home/" / "/Users/" is replaced by "<abs>/<file name>" (the same rule as
suite/common/provenance.scrub). With --git-commit, a provenance block whose git_commit is "unavailable" (runs made on a
host that holds a plain copy of the working tree, not a git checkout) gets that SHA plus a `git_commit_note` saying so.
Prints one line per file with the number of strings changed."""
import argparse, json, re
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--git-commit"); ap.add_argument("--git-note", default="")
a = ap.parse_args()
pat = re.compile(r"(/home/|/Users/)")

def fix(v, n):
    if isinstance(v, str) and (v.startswith("/") or pat.search(v)):
        n[0] += 1
        return "<abs>/" + Path(v.rstrip("/")).name
    if isinstance(v, dict):
        return {k: fix(x, n) for k, x in v.items()}
    if isinstance(v, list):
        return [fix(x, n) for x in v]
    return v

for f in a.files:
    raw = open(f).read(); d = json.loads(raw); n = [0]; compact = raw.count("\n") < 5
    d = fix(d, n)
    pv = d.get("provenance") if isinstance(d, dict) else None
    if a.git_commit and isinstance(pv, dict) and pv.get("git_commit") in ("unavailable", None):
        pv["git_commit"] = a.git_commit; pv["git_commit_note"] = a.git_note
    json.dump(d, open(f, "w"), indent=None if compact else (1 if raw.startswith("{\n ") and not raw.startswith("{\n  ") else 2)); open(f, "a").write("\n")
    print(f, "strings rewritten:", n[0])
