"""E2c amendment 1: pool contrasts over training seeds. Usage: python pool_seeds.py DIR OUT.json
DIR holds probe_s{k}.json and probe_s{k}.json.boot.npz for k = 0,1,2."""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
D, OUTF = Path(sys.argv[1]), Path(sys.argv[2]); K = [0, 1, 2]
B = {k: dict(np.load(D / f"probe_s{k}.json.boot.npz")) for k in K}
P = {k: json.load(open(D / f"probe_s{k}.json")) for k in K}
MET = {"position_r2_mean": ("pos", 0), "dense_macro_f1": ("f1", 1), "recon_l1": ("rec", 2)}
rng = np.random.default_rng(1729); NB = len(B[0]["s2_A_pos"]); draw = rng.integers(0, len(K), (NB, len(K)))
t = stats.t.ppf(0.975, len(K) - 1); R = {"seeds": K, "n_boot": int(NB), "per_arm": {}, "contrasts": {}}
for s in ["s2", "s1", "ls", "naip"]:
    R["per_arm"][s] = {a: {m: {"per_seed": [float(B[k][f"{s}_{a}_pt"][j]) for k in K],
                              "mean": float(np.mean([B[k][f"{s}_{a}_pt"][j] for k in K]))} for m, (_, j) in MET.items()} for a in "ABC"}
    R["contrasts"][s] = {}
    for b, a in [("B", "A"), ("C", "A"), ("C", "B")]:
        out = {}
        for m, (key, j) in MET.items():
            pts = np.array([B[k][f"{s}_{b}_pt"][j] - B[k][f"{s}_{a}_pt"][j] for k in K])
            bs = np.stack([B[k][f"{s}_{b}_{key}"] - B[k][f"{s}_{a}_{key}"] for k in K], 1)  # [NB, seeds]
            hb = bs[np.arange(NB)[:, None], draw].mean(1)
            mu, sd = pts.mean(), pts.std(ddof=1)
            hier = [float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))]; tint = [float(mu - t * sd / np.sqrt(3)), float(mu + t * sd / np.sqrt(3))]
            robust = bool((hier[0] > 0 or hier[1] < 0) and (tint[0] > 0 or tint[1] < 0) and np.sign(hier[0]) == np.sign(tint[0]))
            out[m] = {"pooled_mean": float(mu), "per_seed": pts.tolist(), "seed_sd": float(sd), "hier_boot_ci95": hier,
                      "seed_t_ci95": tint, "robust": robust}
        R["contrasts"][s][f"{b}_minus_{a}"] = out
json.dump(R, open(OUTF, "w"), indent=1)
for s in R["contrasts"]:
    for m in MET:
        v = R["contrasts"][s]["B_minus_A"][m]
        print(s, m, round(v["pooled_mean"], 4), np.round(v["per_seed"], 4), np.round(v["hier_boot_ci95"], 4), np.round(v["seed_t_ci95"], 4), v["robust"])
