"""Print (and save) every count and Panel C value the manuscript states, from the result JSONs only.

Reads  experiments/results/{e1_e5_complete,e5_sota_comparators,e6_multidate,e6_gap_closed_matched}.json  (and, only for the
old-vs-new column, the Panel C rows of DRAFT.md when present).  Nothing is typed in except the published-paper constants in
PUBLISHED (OpenET), each with its source.

Conventions
  Headline counts use cells that ran on the NOMINAL number of training labels (achieved size == nominal size on every seed); cells
  that ran on fewer labels are shown in brackets as "achieved budget" and are never called max.  `budget_complete` (E1's strict rule:
  every nominally selected row is available) is stored in the JSON and shown in the per-task lines of section 4.
  W / T / L : paired 95% block-bootstrap interval of (Clay readout - comparator) above 0 / including 0 / below 0.
  point+    : point estimate (mean over 5 label seeds) > 0.
  10-shot   : 10 labels per class.  max: NMAX per class, counted over cells whose budget is complete (E1 rule) on the matched
              rows; "all" additionally counts cells that ran on fewer labels than nominal (achieved budget).
  Sets      : every contrast is read from a set whose arms share identical rows and identical training draws
              (see run_e1_e5_complete.py).  Clay vs point arm X: set "clay+X".  Clay vs chip arm: set "chip".

Usage:  python summarize_panelC.py [--results DIR] [--draft DRAFT.md] [--json_out FILE]
"""
import argparse, json, os, re, statistics
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CLAY = ["clay_cls", "clay_center", "clay_mean"]
POINT = ["aef_mean128", "aef_ov32_px", "aef_pixel", "tessera_pixel", "tessera_mean128"]
CHIP = ["olmo_center", "olmo_mean", "prithvi_center", "prithvi_mean", "dofa", "dinov2"]
PUBLISHED = {  # published numbers quoted by the manuscript (inputs, not results of this repository)
    "tessera_v2_openet_annual_window": (0.51, "TESSERA v2 (arXiv 2607.03949v2) Appendix E text and Table 2, TESSERA v2-2B-L annual window; scores are max(0,R2)"),
    "tessera_v2_openet_monthly_window": (0.69, "same, Table 2, monthly inference window"),
    "tessera_v2_openet_alphaearth_row": (0.58, "same, Table 2 AlphaEarth (annual) row"),
    "alphaearth_openet_max_trial_r2": (0.58, "AlphaEarth Foundations (arXiv 2507.22291) Section 'Evaluation' text and Figure 3, max-trial R2 = 0.58 +- 0.01"),
}
TASKS = ["africa_crop_mask", "aster_ged", "lcmap_lc", "lcmap_lu", "lcmap_lcc", "lcmap_luc", "descals", "glance", "lucas_lu", "lucas_lc",
         "canada_crops_coarse", "canada_crops_fine", "ethiopia_crops", "us_trees", "openet_ensemble"]
# Table 2 row label -> task, and column -> arm (columns of Panel C in DRAFT.md)
ROWMAP = {"Africa crop mask": "africa_crop_mask", "ASTER emissivity": "aster_ged", "LCMAP land cover": "lcmap_lc", "LCMAP land use": "lcmap_lu",
          "LCMAP cover change": "lcmap_lcc", "LCMAP use change": "lcmap_luc", "Oil palm (Descals)": "descals", "GLanCE land cover": "glance",
          "LUCAS land use": "lucas_lu", "LUCAS land cover": "lucas_lc", "Canada crops (crs.)": "canada_crops_coarse",
          "Canada crops (fine)": "canada_crops_fine", "Ethiopia crops": "ethiopia_crops", "US tree genera": "us_trees", "OpenET monthly ET": "openet_ensemble"}
COLMAP = [("Clay v1.5", "clay_center"), ("DOFA-L", "dofa"), ("OlmoEarth", "olmo_center"), ("Best Peer (Pr.-2)", "prithvi_center"),
          ("TESSERA", "tessera_pixel"), ("AlphaEarth", "aef_ov32_px"), ("DINOv2/v3", "dinov2")]


MIN_TEST = 30  # a set needs at least this many test rows to be counted


def wtl(cs):
    W = sum(c["lo"] > 0 for c in cs); L = sum(c["hi"] < 0 for c in cs); n = len(cs)
    return {"W": W, "T": n - W - L, "L": L, "point_pos": sum(c["point"] > 0 for c in cs), "n": n}


def fmt(r): return f"{r['W']}/{r['T']}/{r['L']}  point+ {r['point_pos']}/{r['n']}"


def contrast_list(D, setname, name, budget, complete_only):
    out = []
    for t in [t for t in TASKS if t in D["tasks"]]:
        s = D["tasks"][t]["sets"].get(setname)
        if not s or "contrasts" not in s or name not in s["contrasts"]: continue
        if s.get("n_test", 0) < MIN_TEST: continue  # sets with a handful of test rows (e.g. the partial 10 m AEF arm) carry no information
        c = s["contrasts"][name][budget]
        if complete_only:
            bi = D["tasks"][t]["sets"][setname]["budgets"][budget]
            if bi["n_train_achieved"] != bi["n_train_nominal"]: continue  # fewer labels than nominal: not counted as this budget
        out.append(dict(c, task=t))
    return out


def cell(D, t, arm, budget, setname="common"):
    s = D["tasks"][t]["sets"][setname]
    if "arm_results" not in s or arm not in s["arm_results"]: return None
    return s["arm_results"][arm][budget]


def parse_draft(path):
    """Panel C + OpenET rows of Table 2 -> {task: {column: (ten_shot, max)}} (None where n/a)."""
    out = {}
    if not os.path.exists(path): return out
    for line in open(path, encoding="utf-8"):
        if not line.startswith("|"): continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        lab = re.sub(r"[`*]", "", cells[0]).strip()
        if lab not in ROWMAP or len(cells) < 10: continue
        names = ["Clay v1.5", "pct", "DOFA-L", "OlmoEarth", "Best Peer (Pr.-2)", "TESSERA", "AlphaEarth", "DINOv2/v3"]
        vals = {}
        for nm, raw in zip(names, cells[2:10]):
            m = re.findall(r"[-\u2212]?\d+\.\d+", raw.replace("\u2212", "-"))
            vals[nm] = {"values": [float(x) for x in m[:2]], "raw": raw.strip()} if m else None  # raw keeps the star / diamond markers
        out[ROWMAP[lab]] = vals
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(REPO, "experiments", "results"))
    ap.add_argument("--draft", default="", help="optional manuscript file; adds an old-vs-new column for its Panel C rows")
    ap.add_argument("--json_out", default="")
    a = ap.parse_args()
    ld = lambda n: json.load(open(os.path.join(a.results, n))) if os.path.exists(os.path.join(a.results, n)) else None
    D, E5, E6, G = ld("e1_e5_complete.json"), ld("e5_sota_comparators.json"), ld("e6_multidate.json"), ld("e6_gap_closed_matched.json")
    S = {"label_year_sources": D["meta"].get("label_year_sources"), "wtl": {}, "centre_vs_pooled": {}, "cells": {}, "e6": {}, "openet": {}}
    print("== provenance:", D["meta"]["script"], D["meta"]["git_commit"], D["meta"]["utc"], "label years:", D["meta"].get("label_year_sources"))

    bad, nsets = [], 0
    for t, tv in D["tasks"].items():
        for sn, sv in tv["sets"].items():
            if "arm_results" not in sv: continue
            nsets += 1
            for b in ["10", "max"]:
                dig = {(a, r[b]["test_ids_sha256"], tuple(r[b]["train_selected_sha256"]), r[b]["n_test"]) for a, r in sv["arm_results"].items()}
                if len({d[1:] for d in dig}) != 1 or any(r[b]["test_ids_sha256"] != sv["test_ids_sha256"] for r in sv["arm_results"].values()): bad.append((t, sn, b))
    S["integrity"] = {"sets_checked": nsets, "violations": bad}
    print(f"\n== 0. integrity: in every one of {nsets} scored sets all arms carry the same test-id digest, the same per-seed training-draw digests and the same n_test: violations = {bad}")
    print("\n== 0b. per-comparison sample counts: n_test of the scored set (n_train_pool) per task; the same arms-by-rows sets are what every paired contrast below uses")
    S["counts"] = {}
    for t in [t for t in TASKS if t in D["tasks"]]:
        S["counts"][t] = {sn: {"n_test": sv.get("n_test"), "n_train_pool": sv.get("n_train_pool"), "test_ids_sha256": sv.get("test_ids_sha256")} for sn, sv in D["tasks"][t]["sets"].items()}
        print(f"  {t:20s} " + "  ".join(f"{sn.replace('clay+','')}={sv.get('n_test')}({sv.get('n_train_pool')})" for sn, sv in D["tasks"][t]["sets"].items()))
    print("\n== 1. Clay readout vs comparator, paired rows (W/T/L = CI above 0 / crossing 0 / below 0; point+ = point estimate > 0)")
    for R in CLAY:
        for X in POINT + CHIP:
            setname = f"clay+{X}" if X in POINT else "chip"
            line = []
            for b in ["10", "max"]:
                allc = contrast_list(D, setname, f"{R}-{X}", b, False); comp = contrast_list(D, setname, f"{R}-{X}", b, True)
                strict = [c for c in allc if c["budget_complete"]]
                if not allc: continue
                key = f"{R}-{X}|{b}"; S["wtl"][key] = {"nominal_size": wtl(comp) if comp else None, "strict_complete": wtl(strict) if strict else None, "all": wtl(allc), "set": setname}
                line.append(f"{b}: " + (fmt(wtl(comp)) if comp else "n/a") + (f" [strict E1-complete {wtl(strict)['n']}: {fmt(wtl(strict))}]" if strict and len(strict) != len(comp) else "") + (f" [achieved budget, all {wtl(allc)['n']}: {fmt(wtl(allc))}]" if len(allc) != len(comp) else ""))
            if line: print(f"  {R:12s} vs {X:16s} | " + " | ".join(line))

    print("\n== 2. centre vs pooled readout, per backbone (10-shot / max complete), chip set")
    for lhs, rhs in [("clay_center", "clay_mean"), ("clay_center", "clay_cls"), ("clay_cls", "clay_mean"), ("olmo_center", "olmo_mean"), ("prithvi_center", "prithvi_mean")]:
        line = []
        for b in ["10", "max"]:
            comp = contrast_list(D, "chip", f"{lhs}-{rhs}", b, True)  # nominal-size cells
            S["centre_vs_pooled"][f"{lhs}-{rhs}|{b}"] = wtl(comp) if comp else None
            if comp: line.append(f"{b}: {fmt(wtl(comp))}")
        print(f"  {lhs}-{rhs:14s} " + " | ".join(line))

    print("\n== 3. cross-check vs the earlier eval_e5 run (e5_sota_comparators.json, linear head): W/T/L and point+ over the 15 tasks")
    if E5:
        for R in ["clay_center", "clay_cls"]:
            for X in ["aef_mean128", "aef_ov32_px", "tessera_pixel"]:
                for b in ["10", "max"]:
                    cs = [c for c in E5["contrasts"] if c["contrast"] == f"{R}-{X}" and c["budget"] == b and c["head"] == "linear"]
                    if cs: print(f"  eval_e5  {R}-{X} budget {b}: {fmt(wtl(cs))}")

    print("\n== 4. Table 2 Panel C + OpenET cells (balanced accuracy / R2): new value from e1_e5_complete.json tasks.<task>.sets.common.arm_results.<arm>.<budget>.point; old = DRAFT.md")
    old = parse_draft(a.draft); S["cells"] = {}
    for lab, t in [(l, t) for l, t in ROWMAP.items() if t in D["tasks"]]:
        print(f"  {lab} ({t}), common set n_test={D['tasks'][t]['sets']['common'].get('n_test')}, budget_complete 10/max = "
              f"{D['tasks'][t]['sets']['common']['budgets']['10']['budget_complete']}/{D['tasks'][t]['sets']['common']['budgets']['max']['budget_complete']}"
              f"; train rows nominal->achieved (mean over seeds) 10-shot {np.mean(D['tasks'][t]['sets']['common']['budgets']['10']['n_train_nominal']):.0f}->{np.mean(D['tasks'][t]['sets']['common']['budgets']['10']['n_train_achieved']):.0f}, "
              f"max {np.mean(D['tasks'][t]['sets']['common']['budgets']['max']['n_train_nominal']):.0f}->{np.mean(D['tasks'][t]['sets']['common']['budgets']['max']['n_train_achieved']):.0f}")
        for col, arm in COLMAP:
            c10, cmx = cell(D, t, arm, "10"), cell(D, t, arm, "max")
            if c10 is None: continue
            o = old.get(t, {}).get(col)
            S["cells"][f"{t}|{arm}"] = {"new_10": c10["point"], "new_max": cmx["point"], "old": o and o["values"], "old_raw": o and o["raw"],
                                        "key": f"tasks.{t}.sets.common.arm_results.{arm}.{{10,max}}.point"}
            print(f"    {col:18s} {arm:15s} new {c10['point']:.3f} / {cmx['point']:.3f} [{c10['lo']:.2f},{c10['hi']:.2f}]  old {o['raw'] if o else None}")

    print("\n== 5. OpenET: local annual-product cells vs published numbers")
    t = "openet_ensemble"
    for arm in ([] if t not in D["tasks"] else ["clay_cls", "clay_center", "clay_mean", "aef_ov32_px", "aef_mean128", "tessera_pixel", "tessera_mean128"]):
        c10 = cell(D, t, arm, "10", "common" if arm in CLAY + ["aef_ov32_px", "aef_mean128", "tessera_pixel"] else f"clay+{arm}")
        cmx = cell(D, t, arm, "max", "common" if arm in CLAY + ["aef_ov32_px", "aef_mean128", "tessera_pixel"] else f"clay+{arm}")
        if c10: print(f"    {arm:16s} 10-shot {c10['point']:.3f}  max {cmx['point']:.3f}"); S["openet"][arm] = {"10": c10["point"], "max": cmx["point"]}
    for k, (v, src) in PUBLISHED.items(): print(f"    published {k} = {v}  ({src})")
    S["openet"]["published"] = PUBLISHED

    if E6:
        print("\n== 6. E6 original (e6_multidate.json, linear head): four-date minus one-date contrasts, own rows")
        for b in ["10", "max"]:
            for ctr in ["clay4_cat-clay1", "clay4_mean-clay1", "claymean4_mean-claymean1", "dofa4_mean-dofa1", "stats4_cat-stats1", "clay4_cat-stats4_cat"]:
                cs = [c for c in E6["contrasts"] if c["contrast"] == ctr and c["budget"] == b and c["head"] == "linear"]
                if cs: print(f"  {ctr:26s} budget {b}: {fmt(wtl(cs))}"); S["e6"][f"{ctr}|{b}"] = wtl(cs)
        print("  fixed-readout cells for the starred tasks (point [lo,hi]):")
        for t in ["africa_crop_mask", "descals"]:
            for b in ["10", "max"]:
                row = {}
                for s in E6["summaries"]:
                    if s["task"] == t and s["budget"] == b and s["head"] == "linear" and s["arm"] in ("clay1", "clay4_mean", "clay4_cat", "claymean1", "claymean4_mean"):
                        row[s["arm"]] = f"{s['point']:.3f} [{s['lo']:.2f},{s['hi']:.2f}]"
                print(f"    {t} {b}: {row}"); S["e6"][f"star|{t}|{b}"] = row
    if G:
        print(f"\n== 7. E6 gap closed, matched rows, readout CLS, linear head (gap_closed_matched.json); eligibility gap >= {G['meta']['min_gap']}")
        for thr in [0.02, G["meta"]["min_gap"], 0.10]:
            for X in ["aef_mean128", "aef_ov32_px"]:
                for pool in ["mean", "concat"]:
                    for b in ["10", "max"]:
                        vals, ts = [], []
                        for t, tt in G["tasks"].items():
                            c = tt["cells"][X].get("budgets", {}).get(b)
                            if not c or not c.get("run") or c["gap"]["point"] < thr: continue
                            vals.append(c["pooling"][pool]["gap_closed"]); ts.append(t)
                        key = f"thr{thr}|{X}|{pool}|{b}"
                        S["e6"][key] = {"median": statistics.median(vals) if vals else None, "n_eligible": len(vals), "tasks": ts}
                        print(f"  min_gap {thr:.2f} {X:12s} pooling {pool:6s} budget {b:3s}: median closed {S['e6'][key]['median'] if vals else None}  n={len(vals)}  tasks={ts}")
        print("  task-level absolute change (4-date pooled minus 1-date; paired CI) at matched rows:")
        for X in ["aef_mean128", "aef_ov32_px"]:
            for pool in ["mean", "concat", "mean_patchreadout"]:
                for b in ["10", "max"]:
                    cs = []
                    for t, tt in G["tasks"].items():
                        c = tt["cells"][X].get("budgets", {}).get(b)
                        if c and c.get("run"): r = c["pooling"][pool]; cs.append({"lo": r["lo"], "hi": r["hi"], "point": r["abs_change"], "task": t})
                    if cs:
                        S["e6"][f"abs|{X}|{pool}|{b}"] = wtl(cs); print(f"    {X:12s} {pool:18s} {b:3s}: {fmt(wtl(cs))}  tasks run: {len(cs)}")
        print("  starred cells, ONE fixed readout (CLS) and one pooling rule per line: four-date minus one-date, point [paired 95% CI], matched rows (comparator arm only fixes the rows)")
        for t in ["africa_crop_mask", "descals"]:
            for b in ["10", "max"]:
                c = G["tasks"][t]["cells"]["aef_ov32_px"].get("budgets", {}).get(b)
                if not c or not c.get("run"): print(f"    {t} {b}: not run on matched rows"); continue
                r = {p: c["pooling"][p] for p in ["mean", "concat"]}
                one, four = c["arms"]["clay1"]["point"], c["arms"]["clay4_mean"]["point"]
                S["e6"][f"star_fixed|{t}|{b}"] = {"clay1": one, "clay4_mean": four, "pooling": r, "n_test": G["tasks"][t]["cells"]["aef_ov32_px"]["n_test"]}
                print(f"    {t} {b}: clay1 {one:.3f} -> clay4_mean {four:.3f}  mean-pool change {r['mean']['abs_change']:+.3f} [{r['mean']['lo']:+.3f},{r['mean']['hi']:+.3f}]  concat change {r['concat']['abs_change']:+.3f} [{r['concat']['lo']:+.3f},{r['concat']['hi']:+.3f}]")
    if a.json_out:
        json.dump(S, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
