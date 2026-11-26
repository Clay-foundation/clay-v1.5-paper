"""Compare two Part A score outputs (e.g. the committed one and one regenerated from scratch with partA.py).
Usage: python compare_partA.py A.json B.json --out ../../results/e4_partA_regen_check.json [--label-a committed --label-b regenerated]
Reports, for the headline pooled AP at k = 5 positives (both chip sizes, every arm), both values and the difference, and the
largest absolute difference over all summary entries per arm."""
import argparse, json
from collections import defaultdict

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--out", required=True)
ap.add_argument("--label-a", default="a"); ap.add_argument("--label-b", default="b"); x = ap.parse_args()
A, B = json.load(open(x.a)), json.load(open(x.b))
S = lambda R: {(s["chip"], s["region"], s["arm"], s["k"], s["mode"], s["metric"]): s for s in R["summary"]}  # noqa: E731
sa, sb = S(A), S(B)
head, worst = {}, defaultdict(float)
for k, v in sa.items():
    if k not in sb: continue
    d = abs(v["point"] - sb[k]["point"]); worst[k[2]] = max(worst[k[2]], d)
    if k[1] == "pooled" and k[3] == 5 and k[4] == "pos" and k[5] == "AP":
        head[f"{k[0]}px|{k[2]}"] = {x.label_a: {"point": v["point"], "ci95": [v["lo"], v["hi"]]}, x.label_b: {"point": sb[k]["point"], "ci95": [sb[k]["lo"], sb[k]["hi"]]},
                                      "abs_diff": round(abs(v["point"] - sb[k]["point"]), 5)}
json.dump({"same_keys": set(sa) == set(sb), "n_entries": len(sa), "pooled_ap_k5_pos": head,
           "max_abs_diff_over_all_summary_entries_by_arm": {k: round(v, 5) for k, v in sorted(worst.items())},
           "provenance_a": A.get("provenance"), "provenance_b": B.get("provenance")}, open(x.out, "w"), indent=1)
print("wrote", x.out)
