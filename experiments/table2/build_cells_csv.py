#!/usr/bin/env python3
"""Rebuild table2_cells.csv for the four-panel Table 2.

    python3 experiments/table2/build_cells_csv.py [--out experiments/table2/table2_cells.csv]

Inputs: the pinned SatML leaderboard JSON (Panels A and B, through from_leaderboard.py) and table2_cells_v1.csv, the
cell list of the previous Table 2 (our own runs; Panels C and D here). One output row per printed value, with
columns panel, task, sensor, model, slot, value, metric, source, result_file, result_key, readout, head, seeds, notes.

  model   `clay`, `other_eo` (best other EO model) or `general` (best general model).
  source  `third_party:satml_leaderboard` (result_file/result_key point into the pinned JSON), `rerun_here`,
          `not_regenerated`, `third_party:<bibkey>`, `pending` (printed `?`, being computed) or empty (`n/a`).
  notes   `tag=` short model tag of a peer cell, `model=` leaderboard model name, `marks=` footnote marks that belong to
          the cell, `labelmarks=` marks printed after the row label, `rank=`, `se=`, `pending`, `n/a`, `from=` the old
          column a collapsed peer value came from.

Panels C and D collapse the old DOFA-L, OlmoEarth, Best Peer, TESSERA and AlphaEarth columns into `other_eo` (the
maximum per slot, tag of the model that holds it) and the old DINOv2/v3 column into `general`. The Clay values that
the previous table took from the OlmoEarth v1.3 report (Vessel S1, Vessel L8, Solar Farm, MADOS fine-tune,
m-cashewplant fine-tune) are replaced by pending cells. Nandi is left out of the table: its data is not public, so it
has no Clay value.

Our own runs in the same harness of models that are not on the leaderboard (OWN_HARNESS) compete for the Panel A
peer cell of their column. A run that beats the leaderboard value replaces it (source `own_harness`, mark †, value
read from the harness results CSV); a run that equals it leaves the cell as it is and adds `tie=<tag>`, printed
next to the leaderboard model's tag. A lower run changes nothing.

"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import from_leaderboard as fl

HERE = Path(__file__).resolve().parent
V1 = HERE / "table2_cells_v1.csv"
COLUMNS = ["panel", "task", "sensor", "model", "slot", "value", "metric", "source", "result_file", "result_key",
           "readout", "head", "seeds", "notes"]

# (v1 task name, new task name, sensor and metric text, label marks)
PANEL_C = [
    ("m-pv4ger-seg", "m-pv4ger-seg", "Aerial RGB $0.1\\,\\text{m}$ (mIoU)", ""),
    ("m-chesapeake", "m-chesapeake", "NAIP RGBN $1.0\\,\\text{m}$ (mIoU)", ""),
    ("Biomass-S3", "Biomass-S3", "S3 OLCI $300\\,\\text{m}$, 21b ($R^2$)", "∗"),
    ("LC100Cls-S3", "LC100Cls-S3", "S3 OLCI $300\\,\\text{m}$, 21b (mAP)", "∗‡"),
    ("EuroSAT-S1", "EuroSAT-S1", "S1 land use (OA)", "‡"),
    ("Vessel Detect. (S1)", "Vessel Detect. (S1)", "S1 vessels (FT $F_1$)", ""),
    ("Vessel Detect. (L8)", "Vessel Detect. (L8)", "Landsat-8 vessels (FT $F_1$)", ""),
    ("Solar Farm Detect.", "Solar Farm Detect.", "S2 10b solar (FT mIoU)", ""),
    ("MADOS (debris)", "MADOS", "S2 debris (LP / FT mIoU)", ""),
    ("m-cashewplant", "m-cashewplant", "S2 plantations (LP / FT mIoU)", ""),
]
PANEL_D = [
    ("Africa crop mask", "Cropland mask (B.Acc)"), ("ASTER emissivity", "Thermal emissivity ($R^2$)"),
    ("LCMAP land cover", "CONUS land cover (B.Acc)"), ("LCMAP land use", "CONUS land use (B.Acc)"),
    ("LCMAP cover change", "Cover change (B.Acc)"), ("LCMAP use change", "Use change (B.Acc)"),
    ("Oil palm (Descals)", "Oil palm extent (B.Acc)"), ("GLanCE land cover", "Global land cover (B.Acc)"),
    ("LUCAS land use", "Europe land use (B.Acc)"), ("LUCAS land cover", "Europe land cover (B.Acc)"),
    ("Canada crops (crs.)", "Coarse crop types (B.Acc)"), ("Canada crops (fine)", "Fine crop types (B.Acc)"),
    ("Ethiopia crops", "Smallholder crops (B.Acc)"), ("US tree genera", "Tree genera, 39c (B.Acc)"),
    ("OpenET monthly ET", "S2 flux ($R^2$)"),
]
PENDING_CLAY = {  # (v1 task, v1 slot) printed as `?` now
    ("Vessel Detect. (S1)", "single"), ("Vessel Detect. (L8)", "single"), ("Solar Farm Detect.", "single"),
    ("MADOS (debris)", "second"), ("m-cashewplant", "second"),
}
NO_CLAY = {"Nandi (S2) (crops)"}  # data not public: no Clay cell
PEER_TAG = {"dofa_l": "DOFA", "olmoearth": "Olmo", "tessera": "TES", "alphaearth": "AEF"}
PEER_TAG_FIX = {"Pr-2": "Pr.-2", "Gal": "Gal.", "Pan": "Pan.", "CR": "CR.", "Cop": "Cop."}
EO_PRIORITY = ["olmoearth", "best_peer", "dofa_l", "alphaearth", "tessera"]  # tie-break order
KEEP_MARKS = "♮◊"
RESULTS = HERE.parent / "results"
# (column, tag, model name, harness results CSV under experiments/results): our runs in the same harness of models
# that are not on the leaderboard; linear probe, RGB input
OWN_HARNESS = [("general", "EG2", "EmbeddingGemma 2", "harness_embeddinggemma2.csv"),
               ("other_eo", "FarSLIP", "FarSLIP ViT-B/16", "harness_farslip.csv")]
OWN_MARK = "†"  # printed on a cell that holds one of our own harness runs


def add_own_harness(rows):
    """Let our own harness runs compete for the Panel A peer cells (see the module docstring)."""
    for model, tag, name, fname in OWN_HARNESS:
        own = {r["dataset"]: 100 * float(r["metric_value"])
               for r in csv.DictReader(open(RESULTS / fname)) if r["method"] == "linear"}
        for r in rows:
            if r["panel"] != "A" or r["model"] != model or r["task"] not in own or r["value"] == "":
                continue
            if r["source"] == "own_harness" or f"tie={tag}" in r["notes"]:  # already applied
                continue
            diff = own[r["task"]] - float(r["value"])
            if diff > 0.005:  # best in this column: replaces the leaderboard value and is marked as our run
                r.update(value=f"{round(own[r['task']], 2):g}", source="own_harness", result_file=fname,
                         result_key=f"{r['task']}|linear", readout="default", seeds="1",
                         notes=f"model={name};band=rgb;tag={tag};marks={OWN_MARK}")
            elif abs(diff) <= 0.005:  # equal: the leaderboard cell stays and our tag is printed next to its tag
                r["notes"] += f";tie={tag};tie_file={fname};tie_key={r['task']}|linear"
    return rows


def note_val(notes: str, key: str) -> str:
    for part in notes.split(";"):
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return ""


def load_v1():
    cells = defaultdict(dict)  # v1 task -> {(model, slot): row}
    for r in csv.DictReader(open(V1)):
        cells[r["task"]][(r["model"], r["slot"])] = r
    return cells


def slots_of(c):
    return ["first", "second"] if any(sl != "single" for (_, sl) in c) else ["single"]


def v1_value(c, model, slot):
    """The v1 row holding the value of (model, slot); a model printed once (`single`) covers both slots."""
    r = c.get((model, slot)) or c.get((model, "single"))
    return r if r is not None and r["value"] != "" else None


def peer_tag(r, model):
    if model == "dinov2":
        return "DINOv2" if "dinov2" in r["result_key"] else "DINO"
    if model == "best_peer":
        t = note_val(r["notes"], "peer")
        return PEER_TAG_FIX.get(t, t)
    return PEER_TAG[model]


def own_rows(panel, v1task, task, sensor, labelmarks, v1):
    c = v1[v1task]
    slots = slots_of(c)
    ref0 = next(iter(c.values()))
    out = []

    def empty(model, slot, notes, ref=ref0):
        return {**base(model, slot), "value": "", "metric": ref["metric"], "source": notes.get("source", ""),
                "result_file": "", "result_key": "", "readout": "", "head": ref["head"], "seeds": "",
                "notes": ";".join(n for n in notes["notes"] if n)}

    def base(model, slot):
        return {"panel": panel, "task": task, "sensor": sensor, "model": model, "slot": slot}

    def copy(model, slot, r, notes):
        return {**base(model, slot), "value": r["value"], "metric": r["metric"], "source": r["source"],
                "result_file": r["result_file"], "result_key": r["result_key"], "readout": r["readout"],
                "head": r["head"], "seeds": r["seeds"], "notes": ";".join(n for n in notes if n)}

    for i, slot in enumerate(slots):
        lab = f"labelmarks={labelmarks}" if labelmarks and i == 0 else ""
        r = None if v1task in NO_CLAY else v1_value(c, "clay", slot)
        if (v1task, slot) in PENDING_CLAY:
            out.append(empty("clay", slot, {"source": "pending", "notes": ["pending", lab]}))
        elif r is None:
            out.append(empty("clay", slot, {"notes": ["n/a", "marks=†" if i == 0 else "", lab]}))  # data not public
        else:
            star = "★" if "*" in note_val(r["notes"], "marks") else ""
            out.append(copy("clay", slot, r, [lab, f"marks={star}" if star else ""]))
        cand = []
        for m in EO_PRIORITY:
            rr = v1_value(c, m, slot)
            if rr is not None:
                cand.append((float(rr["value"]), -EO_PRIORITY.index(m), m, rr))
        if cand:
            _, _, m, rr = max(cand, key=lambda t: t[:2])
            marks = "".join(ch for ch in note_val(rr["notes"], "marks") if ch in KEEP_MARKS)
            out.append(copy("other_eo", slot, rr, [f"tag={peer_tag(rr, m)}", f"marks={marks}" if marks else "", f"from={m}"]))
        else:
            out.append(empty("other_eo", slot, {"notes": ["n/a"]}))
        rr = v1_value(c, "dinov2", slot)
        if rr is not None:
            out.append(copy("general", slot, rr, [f"tag={peer_tag(rr, 'dinov2')}"]))
        else:
            out.append(empty("general", slot, {"notes": ["n/a"]}))
    return out


def build(n_b=None):
    board = fl.Board(fl.load())
    out = [{**{k: "" for k in COLUMNS}, **r} for r in add_own_harness(fl.csv_rows(board, n_b))]
    v1 = load_v1()
    for v1task, task, sensor, lm in PANEL_C:
        out += own_rows("C", v1task, task, sensor, lm, v1)
    for v1task, sensor in PANEL_D:
        out += own_rows("D", v1task, v1task, sensor, "", v1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(HERE / "table2_cells.csv"))
    ap.add_argument("--n-b", type=int, default=None, help="keep only the first n Panel B datasets")
    a = ap.parse_args()
    rows = build(a.n_b)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} cell rows to {a.out}")


if __name__ == "__main__":
    main()
