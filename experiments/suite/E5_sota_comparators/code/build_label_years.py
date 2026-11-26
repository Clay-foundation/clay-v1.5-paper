"""Build the (task, row) -> label-year table that joins chip rows to annual AEF/TESSERA point features.

Two modes (both write task,row,year_before,year_after,source; year_after is empty for static tasks):

  exact     --evaluation_dir DIR   DIR = unzipped AEF evaluation.zip (Zenodo 16585402 v1.0.0; MD5 dabf937b8b404ce399afc19fb041dcc4).
            Year = UTC calendar year of the midpoint of the row's valid_time period (per period for change tasks).
            Every row is checked against --rows: released x/y must equal the row's x/y (so a wrong row-index convention
            fails loudly instead of silently joining the wrong year).
  points    --points points.csv [--meta chip_meta.csv]   (fallback when evaluation CSVs are unavailable; this is what the
            committed results use, see E5 RESULTS.md).  points.csv lists (pid, x, y, year) for every point-year that was sampled.  A location with
            exactly one sampled year gives that year (unambiguous: every static task has this property).  A change-task
            location with exactly two consecutive sampled years gives (earlier, later).  Any other location is AMBIGUOUS
            (several rows share it with different year pairs) and gets empty years, so the row is excluded rather than
            guessed unless --meta (E1 chip meta.csv with chip_id, datetime) is given: then an ambiguous change row takes the pair
            (year of chip_before, year of chip_after) if, and only if, that pair is one of the consecutive year pairs sampled at
            the location (source = points_chip_tiebreak).  On the unambiguous change rows the chip-year pair equals the sampled pair on
            99.2-99.5% of rows (the misses are chips acquired on 1 January, the boundary of the label year), and the candidate set
            always contained the chip pair.  There is no nearest-year fallback in either mode.

Usage:
  python build_label_years.py --rows e1_rows.csv --evaluation_dir evaluation/ --out label_years.csv
  python build_label_years.py --rows e1_rows.csv --points points.csv --meta e1/meta.csv --out label_years.csv
"""
import argparse, collections, datetime, json, os
import numpy as np
import pandas as pd

CHG = {"lcmap_lcc", "lcmap_luc"}
REL = {"africa_crop_mask": "africa_crop_mask/africa_crop_mask.csv", "aster_ged": "aster_ged/aster_ged.csv",
       "canada_crops_coarse": "canada_crops/canada_crops_coarse/canada_crops_coarse.csv",
       "canada_crops_fine": "canada_crops/canada_crops_fine/canada_crops_fine.csv", "descals": "descals/descals.csv",
       "ethiopia_crops": "ethiopia_crops/ethiopia_crops.csv", "glance": "glance/glance.csv",
       "lcmap_lc": "lcmap/lcmap_lc/lcmap_lc.csv", "lcmap_lcc": "lcmap/lcmap_lcc/lcmap_lcc.csv",
       "lcmap_lu": "lcmap/lcmap_lu/lcmap_lu.csv", "lcmap_luc": "lcmap/lcmap_luc/lcmap_luc.csv",
       "lucas_lc": "lucas/lucas_lc/lucas_lc.csv", "lucas_lu": "lucas/lucas_lu/lucas_lu.csv",
       "openet_ensemble": "openet/openet_ensemble.csv", "us_trees": "us_trees/us_trees.csv"}


def mid_year(start_ms, end_ms):
    mid = (np.asarray(start_ms, float) + np.asarray(end_ms, float)) / 2 / 1000.0
    return np.array([datetime.datetime.fromtimestamp(m, datetime.timezone.utc).year for m in mid])


def exact(rows, ev_dir):
    out = []
    for t, g in rows.groupby("task"):
        D = pd.read_csv(os.path.join(ev_dir, REL[t])).reset_index(drop=True)
        r = g.row.astype(int).values
        assert r.max() < len(D), (t, "row index out of range")
        d = D.iloc[r]
        assert np.allclose(d.x.values, g.x.values, atol=1e-6) and np.allclose(d.y.values, g.y.values, atol=1e-6), \
            f"{t}: released x/y do not match rows (row-index convention differs?)"
        if t in CHG:
            yb = mid_year(d.valid_time_start_before_ms, d.valid_time_end_before_ms)
            ya = mid_year(d.valid_time_start_after_ms, d.valid_time_end_after_ms)
        else:
            yb = mid_year(d.valid_time_start_ms, d.valid_time_end_ms); ya = np.full(len(d), np.nan)
        out.append(pd.DataFrame({"task": t, "row": r, "year_before": yb, "year_after": ya, "source": "evaluation_csv"}))
    return pd.concat(out)


def from_points(rows, points, meta=""):
    P = pd.read_csv(points)
    c2y = {}
    if meta:
        M = pd.read_csv(meta, dtype={"chip_id": str}); yr = pd.to_datetime(M.datetime.astype(str).str[:10], errors="coerce").dt.year
        c2y = {c: int(y) for c, y in zip(M.chip_id, yr) if pd.notna(y)}
    loc = collections.defaultdict(set)
    for x, y, yr in zip(P.x, P.y, P.year):
        loc[(round(float(x), 7), round(float(y), 7))].add(int(yr))
    out = []
    for t, g in rows.groupby("task"):
        yb, ya, src = [], [], []
        for x, y, cb, ca in zip(g.x, g.y, g.chip_before, g.chip_after):
            ys = sorted(loc.get((round(float(x), 7), round(float(y), 7)), ()))
            if t in CHG:
                pair = (c2y.get(cb), c2y.get(ca))
                if len(ys) == 2 and ys[1] - ys[0] == 1: yb.append(ys[0]); ya.append(ys[1]); src.append("points_unique_pair")
                elif ys and pair[0] is not None and pair[1] is not None and pair[1] == pair[0] + 1 and pair[0] in ys and pair[1] in ys:
                    yb.append(pair[0]); ya.append(pair[1]); src.append("points_chip_tiebreak")
                else: yb.append(np.nan); ya.append(np.nan); src.append("ambiguous" if ys else "no_point")
            else:
                if len(ys) == 1: yb.append(ys[0]); ya.append(np.nan); src.append("points_unique_year")
                else: yb.append(np.nan); ya.append(np.nan); src.append("ambiguous" if ys else "no_point")
        out.append(pd.DataFrame({"task": t, "row": g.row.astype(int).values, "year_before": yb, "year_after": ya, "source": src}))
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="E1 rows.csv (task,row,split,label,partition,x,y,chip,...)")
    ap.add_argument("--evaluation_dir", default="")
    ap.add_argument("--points", default="")
    ap.add_argument("--meta", default="", help="E1 chip meta.csv (chip_id, datetime) for the change-row tie-break in points mode")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = pd.read_csv(a.rows, keep_default_na=False, dtype={"chip": str, "chip_before": str, "chip_after": str})
    R["row"] = R.row.astype(int)
    if a.evaluation_dir: Y = exact(R, a.evaluation_dir)
    elif a.points: Y = from_points(R, a.points, a.meta)
    else: raise SystemExit("give --evaluation_dir (exact) or --points (fallback)")
    Y.to_csv(a.out, index=False)
    print(json.dumps(Y.groupby(["task", "source"]).size().unstack(fill_value=0).to_dict("index"), indent=1))


if __name__ == "__main__":
    main()
