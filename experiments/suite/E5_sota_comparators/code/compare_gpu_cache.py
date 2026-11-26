"""Compare a freshly embedded subset (embed_gpu_arms.py --max_shards N) with the cached full run.

  python compare_gpu_cache.py --new subset.npz --cached e5_gpu_feats.npz [--json_out out.json]
Prints, per arm, the number of chips compared and the min / median cosine similarity and max abs difference between the two runs.
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--cached", required=True); ap.add_argument("--json_out", default="")
    a = ap.parse_args(); n, c = np.load(a.new), np.load(a.cached)
    cidx = {str(i): k for k, i in enumerate(c["ids"].astype(str).tolist())}
    sel = [cidx[str(i)] for i in n["ids"].astype(str).tolist()]
    out = {}
    for k in [k for k in n.files if k != "ids"]:
        x, y = n[k].astype(np.float32), c[k][sel].astype(np.float32)
        cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
        out[k] = {"n": int(len(cos)), "cos_min": float(cos.min()), "cos_median": float(np.median(cos)), "max_abs_diff": float(np.abs(x - y).max())}
        print(k, out[k])
    if a.json_out: json.dump(out, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
