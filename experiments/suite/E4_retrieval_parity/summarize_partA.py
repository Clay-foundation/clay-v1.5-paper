"""Merge the Part A headline numbers (USPVDB PV retrieval) into results/e4_retrieval_parity.json under key "partA".
The full score output (per-seed values, all contrasts) stays in results/e4_partA_results.json.
Usage: python summarize_partA.py --results-a resultsA.json [--e4 ../../results/e4_retrieval_parity.json] [--write]"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--results-a", required=True)
ap.add_argument("--e4", default=str(Path(__file__).resolve().parents[2] / "results" / "e4_retrieval_parity.json"))
ap.add_argument("--write", action="store_true")
a = ap.parse_args()
R = json.load(open(a.results_a))
S = {(s["chip"], s["region"], s["arm"], s["k"], s["mode"], s["metric"]): s for s in R["summary"]}
C = {(c["chip"], c["region"], c["contrast"], c["k"], c["mode"], c["metric"]): c for c in R["contrasts"]}
ARMS = ["clay_cls", "clay_cls_train", "clay_cls_nometa", "clay_mean", "clay_mean_train", "dinov2_rgb", "spectral_stats", "random"]
def pt(s): return {"point": s["point"], "ci95": [s["lo"], s["hi"]]}
head = {"definition": "pooled over the three tiles (CA 11SLU, TX 15STT, NC 18STE); queries are k positives (+20 negatives) from the other two states; "
                      "AP of cosine ranking; 5 query seeds averaged; CI = 1,000 block-bootstrap resamples (10 km blocks)",
        "clay_cls": "documented lat/lon order (the order of the original run); clay_cls_train = training order; clay_cls_nometa = time and lat/lon off",
        "gallery": R["gallery"], "pools": R["pools"], "ap_k5_pos": {}, "ap_k5_pos_by_region": {}, "contrasts_k5_pos_ap": {}, "budget_metrics_k5_pos": {}}
for cs in (64, 256):
    head["ap_k5_pos"][str(cs)] = {arm: pt(S[(cs, "pooled", arm, 5, "pos", "AP")]) for arm in ARMS if (cs, "pooled", arm, 5, "pos", "AP") in S}
    head["ap_k5_pos_by_region"][str(cs)] = {reg: {arm: S[(cs, reg, arm, 5, "pos", "AP")]["point"] for arm in ARMS if (cs, reg, arm, 5, "pos", "AP") in S}
                                            for reg in ("CA", "TX", "NC")}
    head["contrasts_k5_pos_ap"][str(cs)] = {c: {"point": v["point"], "ci95": [v["lo"], v["hi"]]} for (ch, reg, c, k, m, met), v in C.items()
                                            if ch == cs and reg == "pooled" and k == 5 and m == "pos" and met == "AP"}
    head["budget_metrics_k5_pos"][str(cs)] = {met: {arm: pt(S[(cs, "pooled", arm, 5, "pos", met)]) for arm in ("clay_cls", "clay_cls_train", "dinov2_rgb", "spectral_stats", "random")
                                                    if (cs, "pooled", arm, 5, "pos", met) in S} for met in ("P@10", "P@100", "P@1000", "R@100", "R@1000")}
    head.setdefault("ap_k5_pos_pos_minus_neg", {})[str(cs)] = {arm: pt(S[(cs, "pooled", arm, 5, "pos_minus_neg", "AP")]) for arm in ("clay_cls", "clay_mean", "dinov2_rgb")
                                                              if (cs, "pooled", arm, 5, "pos_minus_neg", "AP") in S}
head["n_seeds"] = len({p["seed"] for p in R["per_seed"]})
head["provenance"] = R.get("provenance")
head["scenes"] = R["scenes"]
print(json.dumps({k: head[k] for k in ("ap_k5_pos", "contrasts_k5_pos_ap")}, indent=1)[:3500])
if a.write:
    e4 = json.load(open(a.e4)); e4["partA"] = head
    Path(a.e4).write_text(json.dumps(e4))
    print("wrote partA into", a.e4)
