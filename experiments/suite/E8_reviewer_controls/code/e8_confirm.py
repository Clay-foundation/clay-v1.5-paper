"""Extract the manuscript-facing E8 numbers (issues #14 and #29) from results/e8_reviewer_controls.json, each with the JSON key
path it was read from, and write results/e8_confirmed_numbers.json. Nothing is recomputed here: the values are copied from the
committed E8 output so the manuscript can cite one place.
Usage: python e8_confirm.py [--e8 PATH] [--e3 PATH] [--out PATH]   (defaults: experiments/results/*.json)"""
import argparse, json
from pathlib import Path

RES = Path(__file__).resolve().parents[3] / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--e8", default=str(RES / "e8_reviewer_controls.json"))
ap.add_argument("--e3", default=str(RES / "e3_cross_sensor.json"))
ap.add_argument("--out", default=str(RES / "e8_confirmed_numbers.json"))
a = ap.parse_args()
E8 = json.load(open(a.e8)); E3 = json.load(open(a.e3))
src = Path(a.e8).name
out = {"source_file": src, "note": "values copied from the committed E8 output; key paths are relative to the source file"}

# ---- #14: overlap and held-out sensitivity
ov = E8["item2_pretraining_overlap"]
out["overlap"] = {
    "aef_test_rows": {"key": "item2_pretraining_overlap.E1_aef.test_pooled", **ov["E1_aef"]["test_pooled"]},
    "aef_train_rows": {"key": "item2_pretraining_overlap.E1_aef.train_pooled", **ov["E1_aef"]["train_pooled"]},
    "aef_per_task_test": {"key": "item2_pretraining_overlap.E1_aef.per_task_test", **{k: v["frac_inside"] for k, v in ov["E1_aef"]["per_task_test"].items()}},
    "e2_test_chips": {"key": "item2_pretraining_overlap.E2.test_chips", **ov["E2"]["test_chips"]},
    "e3_test_locations": {"key": "item2_pretraining_overlap.E3.test_locations", **ov["E3"]["test_locations"]},
    "e4_partA_chips": {"key": "item2_pretraining_overlap.E4_partA.all_chips", **ov["E4_partA"]["all_chips"]},
    "definition": "centroid inside a sampled Clay pretraining MGRS tile footprint (mgrs_sample_v02, 2,728 tiles); spatial overlap only, no acquisition-date or chip-identity match",
}
rs = E8["item3_e1_sensitivity"]["a_b_resplit_and_distance"]["resplit"]
sens = {}
for task, v in rs.items():
    sens[task] = {"key": f"item3_e1_sensitivity.a_b_resplit_and_distance.resplit.{task}",
                  "aef_split_clay_minus_stats": v["aef_split"]["clay_cls-stats"], "hblock_clay_minus_stats": v["hblock_split"]["clay_cls-stats"],
                  "n_test_aef": v["aef_split"]["n_test"], "n_test_hblock": v["hblock_split"]["n_test"],
                  "n_test_blocks_aef": v["aef_split"]["n_test_blocks"], "n_test_blocks_hblock": v["hblock_split"]["n_test_blocks"]}
out["e1_sensitivity_n10_clay_cls_minus_stats"] = sens
out["e1_sensitivity_summary"] = {
    "n_tasks": len(sens),
    "ci_above_zero_aef_split": sum(v["aef_split_clay_minus_stats"][1] > 0 for v in sens.values()),
    "ci_above_zero_hblock": sum(v["hblock_clay_minus_stats"][1] > 0 for v in sens.values()),
    "point_positive_aef_split": sum(v["aef_split_clay_minus_stats"][0] > 0 for v in sens.values()),
    "point_positive_hblock": sum(v["hblock_clay_minus_stats"][0] > 0 for v in sens.values()),
    "sign_flips": [t for t, v in sens.items() if (v["aef_split_clay_minus_stats"][0] > 0) != (v["hblock_clay_minus_stats"][0] > 0)],
    "bootstrap_unit": "aef split: block-bootstrap on the AEF split; hblock: 100 km blocks (see E8 PROTOCOL)",
}

# ---- #29: chip-size confound (E8 matched context next to E3)
c = E8["item1_context_matched_cross_sensor"]
def row(pair, feat):
    for x in c["classification"]:
        if x["pair"] == pair and x["feature"] == feat:
            return x
    raise KeyError((pair, feat))
def arm(x, k):
    v = x["arms"][k]; return {"macro_f1": v["macro_f1"], "ci": [v["lo"], v["hi"]]}
def con(x, k):
    v = x["contrasts"][k]; return {"diff": v["diff"], "ci": [v["lo"], v["hi"]]}
chip = {"n": c["n"], "protocol": c["protocol"]}
for feat in ("clay_on_cls", "clay_on_center", "clay_off_cls", "dino_cls"):
    pairs = {"e3_320m": "s2_32->naip_320", "e8_2p56km": "s2_256->naip_256"}
    d = {}
    for lab, p in pairs.items():
        try:
            x = row(p, feat)
        except KeyError:
            continue
        d[lab] = {"pair": p, "n_train": x["n_train"], "n_test": x["n_test"], "native": arm(x, "native_full"), "zeroshot": arm(x, "zeroshot"),
                  "zeroshot_restandardised": arm(x, "zeroshot_bstd"), "procrustes_1000": arm(x, "procrustes_n1000"),
                  "delta_zeroshot_minus_native": con(x, "zeroshot-native_full")}
    chip[f"s2_to_naip|{feat}"] = d
chip["same_sensor_chip_size_only|clay_on_cls"] = {
    p: {"native": arm(row(p, "clay_on_cls"), "native_full")["macro_f1"], "zeroshot": arm(row(p, "clay_on_cls"), "zeroshot")["macro_f1"],
        "delta": con(row(p, "clay_on_cls"), "zeroshot-native_full")}
    for p in ("s2_32->s2_256", "s2_256->s2_32", "naip_320->naip_256", "naip_256->naip_320")}
chip["cross_contrasts"] = {k: c["cross_contrasts"][k] for k in (
    "clay_on_cls|delta256-delta32 (zeroshot-native_full)", "clay_on_cls|zeroshot256-zeroshot32", "clay_on_cls|native_naip256-native_naip320",
    "center256 vs cls32|delta(zeroshot-native_full)", "center256 vs cls32|zeroshot")}
chip["retrieval_R@10_s2_to_naip|clay_on_cls"] = [{k: r[k] for k in r if k in ("query", "gallery", "method", "centered", "exclude_same_tile", "r@10", "r@10_lo", "r@10_hi")}
                                                  for r in c["retrieval"] if r["method"] == "clay_on_cls" and r["query"].startswith("s2") and r["gallery"].startswith("naip")]
chip["e3_footprints_m"] = {"s2": 320, "s1": 320, "naip": 320, "landsat": 480,
                           "note": "E3 manifest: S2/S1 32 px at 10 m, NAIP 256 px at 1.25 m, Landsat 16 px at 30 m = 480 m (not 320 m)"}
out["chip_size_confound"] = chip

# ---- E3 per-pair n and ranges, from the committed (documented-order) E3 JSON
rows = {}
for x in E3["classification"]:
    if x["feature"] == "clay_on_cls":
        rows[x["pair"]] = {"n_train": x["n_train"], "n_test": x["n_test"], "native": arm(x, "native_full"), "zeroshot": arm(x, "zeroshot"),
                           "drop": con(x, "zeroshot-native_full"), "procrustes_1000": arm(x, "procrustes_n1000")}
out["e3_committed_documented_order_clay_on_cls"] = {"source_file": Path(a.e3).name, "n_locations": E3["n_locations"], "pairs": rows}
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
