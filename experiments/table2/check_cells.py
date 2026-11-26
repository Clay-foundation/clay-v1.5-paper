#!/usr/bin/env python3
"""Check that every Table 2 cell with a result link reproduces its printed value.

    python3 experiments/table2/check_cells.py [--csv experiments/table2/table2_cells.csv]

For each row of table2_cells.csv with source `rerun_here` or `not_regenerated`, `result_file`/`result_key` must
resolve to a numeric leaf (path segments are joined with `/`; a JSON key may itself contain `/`, so the longest
matching key is tried at each level). A key ending in `*100` multiplies the leaf by 100 (fractions stored as IoU).
The leaf must equal the printed value at the printed precision (2 decimals in Panel D, 3 for Biomass-S3,
1 otherwise). Rows with `third_party:<bibkey>` sources have no link and are only counted.

Leaderboard cells (`third_party:satml_leaderboard`, Panels A and B) are checked against the pinned leaderboard JSON:
`result_key` is `model_id|dataset_id|method_id|band_id`; the value (and the standard error in `notes`) must equal the
cell in `experiments/results/satml_leaderboard_83ca5cb.json`, and the Panel A and B rows must equal what
`from_leaderboard.py` regenerates from that file. Pending cells (`?`, source `pending`) are counted and reported, never
failed. The table body in DRAFT.md and latex/sections/02_model.tex must equal the rendering of the CSV
(`render_table.py`). Exit status 1 on any failure.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_cells_csv as bc
import from_leaderboard as fl
import render_table as rt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
_cache = {}


def load(name):
    if name not in _cache:
        _cache[name] = json.loads((RESULTS / name).read_text())
    return _cache[name]


def resolve(obj, path):
    if path == "":
        return obj
    if not isinstance(obj, dict):
        raise KeyError(path)
    for k in obj:
        if path == k:
            return obj[k]
        if path.startswith(k + "/"):
            try:
                return resolve(obj[k], path[len(k) + 1:])
            except KeyError:
                pass
    raise KeyError(path)


def decimals(panel, task):
    if panel == "D":
        return 2
    if task == "Biomass-S3":
        return 3
    return 1


def check_leaderboard(rows, fails):
    board = fl.Board(fl.load())
    n = 0
    for r in rows:
        if r["source"] != "third_party:satml_leaderboard":
            continue
        where = f"{r['panel']} {r['task']} {r['model']} {r['slot']}"
        if r["result_file"] != fl.PINNED_NAME:
            fails.append(f"{where}: result_file is {r['result_file']!r}, not the pinned leaderboard JSON")
            continue
        try:
            v, se = board.lookup(r["result_key"])
        except (KeyError, StopIteration, ValueError):
            fails.append(f"{where}: {r['result_key']} not found in the pinned leaderboard JSON")
            continue
        if abs(v - float(r["value"])) > 1e-9:
            fails.append(f"{where}: printed {r['value']} vs leaderboard {v} at {r['result_key']}")
            continue
        if "se=" in r["notes"]:
            s_ = float(next(p for p in r["notes"].split(";") if p.startswith("se="))[3:])
            if se is None or abs(se - s_) > 0.005 + 1e-9:
                fails.append(f"{where}: se {s_} vs leaderboard {se}")
                continue
        n += 1
    # the A and B rows must be exactly what the script regenerates from the pinned file
    panel_b = {r["task"] for r in rows if r["panel"] == "B"}
    n_b = len(panel_b) if panel_b else None
    regen = bc.add_own_harness(fl.csv_rows(board, n_b))  # recomputes our own harness cells and ties
    have = [r for r in rows if r["panel"] in "AB"]
    if len(regen) != len(have):
        fails.append(f"Panels A and B: {len(have)} cell rows in the CSV, {len(regen)} regenerated from the leaderboard file")
    for a, b in zip(regen, have):
        for k, v in a.items():
            if b.get(k, "") != v:
                fails.append(f"A/B {b['task']} {b['model']} {b['slot']}: column {k} is {b.get(k)!r}, script gives {v!r}")
                break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "table2_cells.csv"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    fails, n_checked, kinds = [], 0, Counter()
    for r in rows:
        src = r["source"]
        kinds[src.split(":")[0] if src else "n/a"] += 1
        if src not in ("rerun_here", "not_regenerated"):
            continue
        where = f"{r['panel']} {r['task']} {r['model']} {r['slot']}"
        if not r["result_file"] or not r["result_key"]:
            fails.append(f"{where}: no result_file/result_key")
            continue
        key, scale = r["result_key"], 1.0
        if "*" in key:   # a scale factor after the key: `*100`, `*0.01`
            key, _, f_ = key.rpartition("*")
            scale = float(f_)
        try:
            leaf = resolve(load(r["result_file"]), key)
        except (KeyError, FileNotFoundError):
            fails.append(f"{where}: {r['result_file']}:{r['result_key']} does not resolve")
            continue
        if isinstance(leaf, bool) or not isinstance(leaf, (int, float)):
            fails.append(f"{where}: {r['result_key']} is not numeric ({type(leaf).__name__})")
            continue
        d = decimals(r["panel"], r["task"])
        if abs(leaf * scale - float(r["value"])) > 0.5 * 10 ** (-d) + 1e-6:
            fails.append(f"{where}: printed {r['value']} vs {leaf * scale:.6g} at {r['result_key']}")
            continue
        n_checked += 1
    n_lb = check_leaderboard(rows, fails)
    pending = [f"{r['panel']} {r['task']} {r['slot']}" for r in rows if r["source"] == "pending"]
    # table text in both files = rendering of the CSV
    d_text, t_text = rt.DRAFT.read_text(), rt.TEX.read_text()
    _, old_d = rt.splice_md(d_text, [])
    _, old_t = rt.splice_tex(t_text, [])
    if old_d != "\n".join(rt.md_block(rt.render(rt.Md(), a.csv))):
        fails.append("DRAFT.md table body differs from the rendering of the CSV (run render_table.py --write)")
    if old_t != "\n".join(rt.tex_block(rt.render(rt.Tex(), a.csv))):
        fails.append("latex/sections/02_model.tex table body differs from the rendering of the CSV (run render_table.py --write)")
    print(f"rows by source: {dict(kinds)}")
    print(f"checked {n_checked} linked cells in experiments/results/*.json, {n_lb} leaderboard cells against the pinned JSON; "
          f"{len(pending)} pending cells (not failed); {len(fails)} failures")
    print("pending: " + ", ".join(pending))
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
