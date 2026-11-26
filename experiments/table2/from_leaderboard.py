#!/usr/bin/env python3
"""Panel A and B rows of Table 2, computed from the pinned SatML leaderboard data.

    python3 experiments/table2/from_leaderboard.py            # print both panels and their rank / leader audit
    python3 experiments/table2/from_leaderboard.py --csv OUT  # write the cell rows (used by build_cells_csv.py)

Input: experiments/results/satml_leaderboard_83ca5cb.json (Taylor Geospatial SatML leaderboard, built from
torchgeo/geo-eval-harness commit 83ca5cb). Rules, applied exactly:

  * A cell is (model, dataset, method, band set). For each model, dataset and method the best band set is used.
  * Clay v1.5 is the leaderboard models `tt_clay_v1_5_large` and `tt_clay_v1_5_large_cls` (two readouts of one model):
    the better of the two is used per row (on an exact tie, the CLS-pooling readout).
  * "EO model" = leaderboard kind GFM, "general model" = kind Generic. Baselines (hand-crafted features, image
    statistics, random features) are not compared.
  * Panel A: the leaderboard's core classification datasets, method `linear` (linear probe).
  * Panel B: segmentation datasets in the priority order below, method `seg-upernet`. Clay has no segmentation cells on the leaderboard yet, so its cells are pending (`?`).
  * Best other EO / best general: the highest value per slot among EO models other than Clay v1.5, and among general
    models; the earlier Clay v1 Base is not offered as a comparator. A cell shows the value and a short tag of the model
    that holds it.
  * % Best = Clay / max(Clay, best other EO, best general); for two-slot cells the larger of the two ratios.
  * Rank = 1 + the number of EO models whose best value is above Clay's, out of the EO models with a value in that
    row. Entries with the same leaderboard name (readout or input variants of one model, e.g. `Scale-MAE Large` with
    and without CLS pooling, or the same checkpoint listed twice) count once, at their best value.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED = HERE.parent / "results" / "satml_leaderboard_83ca5cb.json"
PINNED_NAME = PINNED.name
CLAY_IDS = ("tt_clay_v1_5_large", "tt_clay_v1_5_large_cls")
CLAY_NAME = "Clay v1.5 Large"

# Segmentation datasets in the order the author fixed: if the table has to be cut, the first n are kept.
PANEL_B_ORDER = ["sen1floods11-s2", "sen1floods11-s1", "burn_scars", "pastis", "cloudsen12", "fotw", "flair2", "loveda",
                 "spacenet2", "spacenet7", "hotosm_buildings", "open_cities", "levir-cd-post", "kuro_siwo", "caffe",
                 "openearthmap-sar", "bright-post-sar", "bright-post-sar-rgb", "dynamic_earthnet"]
PANEL_B_SLOTS = [("single", "seg-upernet", "UPerNet")]
# Clay v1.5 segmentation run by the authors through the harness (not on the leaderboard yet).
_CH = Path(__file__).resolve().parents[1] / "results" / "harness_clay_segmentation.json"
CLAY_HARNESS = json.loads(_CH.read_text())["cells"] if _CH.exists() else {}
PANEL_A_SLOTS = [("single", "linear", "linear probe")]
METRIC_LABEL = {"accuracy": "Acc", "micro_mAP": "micro-mAP", "mIoU": "mIoU"}

# Leaderboard model name -> short tag printed in the table. Every tag used must be defined in the table footnote.
TAGS = {
    "DEO Swin-B (RGB)": "DEO-B-RGB",
    "DINOv3-SAT ViT-L/16": "DINOv3-SAT",
    "DOFA Large": "DOFA-L",
    "OlmoEarth v1 Base": "Olmo-v1-B",
    "OlmoEarth v1 Large": "Olmo-v1-L",
    "OlmoEarth v1.1 Base": "Olmo-v1.1-B",
    "OlmoEarth v1.2 Base": "Olmo-v1.2-B",
    "Prithvi-EO v2 600M": "Pr.-2",
    "SatlasNet SwinV2-B (S2 RGB, multi-image)": "Satlas-B-RGB",
    "Scale-MAE Large": "SM-L",
    "TerraMind v1 Base (RGB)": "TM-B-RGB",
    "TerraMind v1 Large (RGB)": "TM-L-RGB",
    "ConvNeXt-L": "ConvNeXt-L",
    "EfficientNetV2-L": "EfficientNetV2-L",
    "DINOv3 ConvNeXt-L": "DINOv3-CNX-L",
    "DINOv3 ViT-L/16": "DINOv3-L",
    "Swin-L": "Swin-L",
    "SwinV2-B": "SwinV2-B",
    "SwinV2-L": "SwinV2-L",
    "ViT-L/16": "ViT-L/16",
}
# Expansion printed in the footnote key, by tag (leaderboard name).
TAG_NAMES = {tag: name for name, tag in TAGS.items()}


def load(path: Path = PINNED) -> dict:
    return json.loads(Path(path).read_text())


class Board:
    def __init__(self, d: dict):
        self.d = d
        self.models = d["models"]
        self.ds = {x["id"]: x for x in d["datasets"]}
        self.methods = [m["id"] for m in d["methods"]]
        self.bands = [b["id"] for b in d["bands"]]
        self.core = list(d["core_datasets"])
        self.ds_index = {x["id"]: i for i, x in enumerate(d["datasets"])}
        self.best: dict[tuple[int, str, str], tuple[float, float | None, str]] = {}
        for mi, di, me, bi, v, se in d["cells"]:
            if v is None:
                continue
            k = (mi, d["datasets"][di]["id"], self.methods[me])
            if k not in self.best or v > self.best[k][0]:
                self.best[k] = (v, se, self.bands[bi])
        self.clay_idx = {i for i, m in enumerate(self.models) if m["id"] in CLAY_IDS}
        assert len(self.clay_idx) == 2, "Clay v1.5 model ids not found in the leaderboard file"

    def ref(self, mi: int, ds: str, method: str, band: str) -> str:
        return f"{self.models[mi]['id']}|{ds}|{method}|{band}"

    def lookup(self, ref: str) -> tuple[float, float | None]:
        """Value and standard error for a `model_id|dataset_id|method_id|band_id` reference (check_cells.py)."""
        mid, ds, method, band = ref.split("|")
        mi = next(i for i, m in enumerate(self.models) if m["id"] == mid)
        di, me, bi = self.ds_index[ds], self.methods.index(method), self.bands.index(band)
        for a, b, c, e, v, se in self.d["cells"]:
            if (a, b, c, e) == (mi, di, me, bi):
                return v, se
        raise KeyError(ref)

    def cell(self, ds: str, method: str, kind: str | None):
        """Best entry in (dataset, method) among models of `kind` (None = Clay v1.5): dict or None."""
        best = None
        for (mi, d_, me), (v, se, band) in self.best.items():
            if d_ != ds or me != method:
                continue
            m = self.models[mi]
            is_clay = mi in self.clay_idx
            if kind is None and not is_clay:
                continue
            if kind is not None and (is_clay or m["kind"] != kind or m["name"].startswith("Clay")):
                continue
            if best is None or v > best["value"] or (is_clay and v == best["value"] and m["id"].endswith("_cls")):
                best = {"value": v, "se": se, "band": band, "model": m["name"], "model_id": m["id"],
                        "ref": self.ref(mi, ds, method, band)}
        if best is not None and kind is not None:
            best["tag"] = TAGS[best["model"]]
        return best

    def rank(self, ds: str, method: str, clay_value: float) -> tuple[int, int]:
        """(rank, n) among EO models, entries with one leaderboard name counted once at their best value."""
        byname: dict[str, float] = {}
        for (mi, d_, me), (v, _, _) in self.best.items():
            if d_ == ds and me == method and self.models[mi]["kind"] == "GFM":
                n_ = self.models[mi]["name"]
                byname[n_] = max(byname.get(n_, float("-inf")), v)
        others = {n_: v for n_, v in byname.items() if n_ != CLAY_NAME}
        return 1 + sum(v > clay_value for v in others.values()), len(others) + 1


def pct_best(clay: dict, eo: dict, gen: dict) -> float | None:
    """Larger over slots of Clay / max(Clay, EO, general); None if Clay has no value."""
    r = []
    for s, c in clay.items():
        if c is None:
            continue
        top = max(x[s]["value"] for x in (clay, eo, gen) if x.get(s))
        r.append(100.0 * c["value"] / top)
    return max(r) if r else None


def rows(board: Board, panel: str, n_b: int | None = None):
    slots = PANEL_A_SLOTS if panel == "A" else PANEL_B_SLOTS
    dsets = board.core if panel == "A" else PANEL_B_ORDER[: n_b or len(PANEL_B_ORDER)]
    out = []
    for ds in dsets:
        info = board.ds[ds]
        r = {"panel": panel, "dataset": ds, "sensor": info["sensor"], "metric": METRIC_LABEL[info["metric"]],
             "clay": {}, "other_eo": {}, "general": {}, "rank": None}
        for slot, method, _ in slots:
            r["clay"][slot] = board.cell(ds, method, None)
            r["other_eo"][slot] = board.cell(ds, method, "GFM")
            r["general"][slot] = board.cell(ds, method, "Generic")
        r["pct_best"] = pct_best(r["clay"], r["other_eo"], r["general"])
        key_slot = slots[-1]
        if r["clay"][key_slot[0]] is not None:
            r["rank"] = board.rank(ds, key_slot[1], r["clay"][key_slot[0]]["value"])
        out.append(r)
    return out


def fmt(v: float) -> str:
    return f"{v:.1f}"


def cell_text(cells: dict, slots, with_tag: bool = True) -> str:
    """Printed form of a cell for the audit listing: `a / b (tagA/tagB)`, n/a for a slot without a value."""
    vals = [cells[s] for s, _, _ in slots]
    txt = " / ".join(fmt(c["value"]) if c else "n/a" for c in vals)
    tags = [c["tag"] for c in vals if c]
    if with_tag and tags:
        txt += f" ({tags[0] if len(set(tags)) == 1 else '/'.join(tags)})"
    return txt


def csv_rows(board: Board, n_b: int | None = None):
    """Cell rows for table2_cells.csv (panel, task, sensor, model, slot, value, metric, source, ..., notes)."""
    out = []
    for panel in "AB":
        slots = PANEL_A_SLOTS if panel == "A" else PANEL_B_SLOTS
        for r in rows(board, panel, n_b):
            sensor = f"{r['sensor']}, {r['metric']}"
            for col in ("clay", "other_eo", "general"):
                for slot, method, head in slots:
                    c = r[col][slot]
                    base = {"panel": panel, "task": r["dataset"], "sensor": sensor, "model": col, "slot": slot,
                            "metric": r["metric"], "head": head, "seeds": "unknown"}
                    if c is None:
                        own = CLAY_HARNESS.get(r["dataset"], {}).get(method) if (col == "clay" and panel == "B") else None
                        if own is not None:
                            out.append({**base, "value": f"{own['value']:.1f}", "source": "rerun_here",
                                        "result_file": "harness_clay_segmentation.json",
                                        "result_key": f"cells/{r['dataset']}/{method}/value", "readout": "",
                                        "seeds": str(own["n_seeds"]), "notes": "harness run, frozen encoder;marks=†"})
                        elif col == "clay":
                            out.append({**base, "value": "", "source": "pending", "result_file": "", "result_key": "",
                                        "readout": "", "notes": "pending"})
                        else:
                            out.append({**base, "value": "", "source": "", "result_file": "", "result_key": "",
                                        "readout": "", "notes": "n/a"})
                        continue
                    notes = [f"model={c['model']}", f"band={c['band']}"]
                    if col == "clay":
                        if c["se"] is not None:
                            notes.append(f"se={c['se']:.2f}")
                        if r["rank"]:
                            notes.append(f"rank={r['rank'][0]} of {r['rank'][1]}")
                        readout = "cls" if c["model_id"].endswith("_cls") else "default"
                    else:
                        notes.append(f"tag={c['tag']}")
                        readout = "default"
                    out.append({**base, "value": f"{c['value']:g}", "source": "third_party:satml_leaderboard",
                                "result_file": PINNED_NAME, "result_key": c["ref"], "readout": readout,
                                "notes": ";".join(notes)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(PINNED))
    ap.add_argument("--n-b", type=int, default=None, help="keep only the first n Panel B datasets")
    ap.add_argument("--csv", help="write the cell rows to this CSV file")
    a = ap.parse_args()
    board = Board(load(Path(a.data)))
    src = board.d["source"]
    print(f"{src['repo']} commit {src['commit'][:7]} ({src['committed_at'][:10]})\n")
    for panel in "AB":
        slots = PANEL_A_SLOTS if panel == "A" else PANEL_B_SLOTS
        print(f"Panel {panel}")
        for r in rows(board, panel, a.n_b):
            c = r["clay"]
            ctxt = " / ".join((f"{x['value']:.1f}" + (f" +- {x['se']:.1f}" if x["se"] is not None else "")) if x else "?"
                              for x in c.values())
            pct = "?" if r["pct_best"] is None else f"{r['pct_best']:.1f}%"
            rk = "-" if r["rank"] is None else f"{r['rank'][0]} of {r['rank'][1]}"
            print(f"  {r['dataset']:20s} Clay {ctxt:18s} {pct:>7s} rank {rk:9s} EO {cell_text(r['other_eo'], slots)} | "
                  f"general {cell_text(r['general'], slots)}")
        print()
    if a.csv:
        out = csv_rows(board, a.n_b)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
