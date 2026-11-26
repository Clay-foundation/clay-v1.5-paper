#!/usr/bin/env python3
"""Recompute every Table 2 aggregate the paper quotes from table2_cells.csv.

    python3 experiments/table2/summarize.py [--csv experiments/table2/table2_cells.csv] [--json OUT.json]

Rules (stated, applied exactly; change them here, not in the prose):
  * A cell is (panel, task, model, slot); model is `clay`, `other_eo` (best other EO model) or `general` (best general
    model); slot is `single` or `first` / `second` for a printed pair "a / b". Higher is better for every metric.
  * Pending cells (`?`, source `pending`) and `n/a` cells have no value and are excluded from every aggregate.
  * Row "% Best", per slot s that has a Clay value: Clay_s / max(Clay_s, other_eo_s, general_s); the row takes the larger
    slot ratio. A row without any Clay value has no % Best.
  * Per panel and over all panels: number of rows, number of rows with a Clay value, median of the unrounded row
    % Best over those rows. Table 2 prints the overall median; the text quotes the Panel A median.
  * Pending cells are counted and listed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("clay", "other_eo", "general")


def load(path):
    """-> ordered list of (panel, task) keys and {key: {(model, slot): row}}."""
    order, cells = [], defaultdict(dict)
    for r in csv.DictReader(open(path)):
        key = (r["panel"], r["task"])
        if key not in cells:
            order.append(key)
        cells[key][(r["model"], r["slot"])] = r
    return order, cells


def has_value(r):
    return r is not None and r["value"] != ""


def csv_note(r, key):
    return next((p[len(key) + 1:] for p in r["notes"].split(";") if p.startswith(key + "=")), "")


def family(tag):
    """Model family of a peer tag: versions and sizes of one model count once (all OlmoEarth and TerraMind entries;
    DINOv2 and DINOv3 together). The satellite-pretrained DINOv3-SAT is an EO model and stays separate."""
    for prefix, fam in (("Olmo", "OlmoEarth"), ("TM", "TerraMind"), ("DINOv3-SAT", "DINOv3-SAT"), ("DINO", "DINO")):
        if tag.startswith(prefix):
            return fam
    return tag


def row_ratios(cell):
    """{slot: percent} over the slots where Clay has a value."""
    out = {}
    for (m, s), r in cell.items():
        if m != "clay" or not has_value(r):
            continue
        top = max(float(x["value"]) for mm in MODELS if has_value(x := cell.get((mm, s))))
        out[s] = 100.0 * float(r["value"]) / top if top > 0 else float("nan")
    return out


def pct_best(cell):
    r = row_ratios(cell)
    return max(r.values()) if r else None


def summarize(path):
    order, cells = load(path)
    rows = []
    for key in order:
        rows.append({"panel": key[0], "task": key[1], "pct_best": pct_best(cells[key]),
                     "n_pending": sum(1 for r in cells[key].values() if r["source"] == "pending")})
    out = {"rule": "row % Best = max over slots with a Clay value of Clay / max(Clay, other EO, general); "
                   "pending and n/a cells excluded", "per_panel": {}}
    for p in sorted({r["panel"] for r in rows}):
        pr = [r for r in rows if r["panel"] == p]
        vals = [r["pct_best"] for r in pr if r["pct_best"] is not None]
        out["per_panel"][p] = {"rows": len(pr), "rows_with_clay_value": len(vals),
                               "median_pct_best": st.median(vals) if vals else None,
                               "pending_cells": sum(r["n_pending"] for r in pr)}
    allv = [r["pct_best"] for r in rows if r["pct_best"] is not None]
    out["all_panels"] = {"rows": len(rows), "rows_with_clay_value": len(allv),
                         "median_pct_best": st.median(allv) if allv else None}
    # model families behind each comparison column (the holder of each printed value; tie tags are not counted)
    out["families"] = {m: sorted({family(csv_note(r, "tag")) for c in cells.values() for (mm, _), r in c.items()
                                  if mm == m and has_value(r)}) for m in MODELS[1:]}
    out["pending"] = [f"{r['panel']} {r['task']}" + (f" ({r['n_pending']} cells)" if r["n_pending"] > 1 else "")
                      for r in rows if r["n_pending"]]
    out["rows"] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(HERE / "table2_cells.csv"))
    ap.add_argument("--json", help="also write the aggregates to this JSON file")
    a = ap.parse_args()
    out = summarize(a.csv)
    print(f"Table 2 aggregates from {a.csv}\n")
    print("per panel (rows, rows with a Clay value, median % Best, pending cells):")
    for p, d in out["per_panel"].items():
        m = "?" if d["median_pct_best"] is None else f"{d['median_pct_best']:.2f}"
        print(f"  Panel {p}: rows {d['rows']:2d}  with Clay value {d['rows_with_clay_value']:2d}  median {m:>6s}  pending cells {d['pending_cells']}")
    d = out["all_panels"]
    print(f"  all    : rows {d['rows']:2d}  with Clay value {d['rows_with_clay_value']:2d}  median {d['median_pct_best']:.2f}  (printed in Table 2)")
    for m, fams in out["families"].items():
        print(f"  {m}: {len(fams)} model families: {', '.join(fams)}")
    print(f"\npending cells ({sum(v['pending_cells'] for v in out['per_panel'].values())}): " + "; ".join(out["pending"]))
    print("\nper-row % Best:")
    for r in out["rows"]:
        v = "?" if r["pct_best"] is None else f"{r['pct_best']:6.2f}"
        print(f"  {r['panel']} {r['task']:22s} {v}")
    if a.json:
        out_j = {k: v for k, v in out.items() if k != "rows"}
        out_j["rows"] = out["rows"]
        Path(a.json).write_text(json.dumps(out_j, indent=2) + "\n")


if __name__ == "__main__":
    main()
