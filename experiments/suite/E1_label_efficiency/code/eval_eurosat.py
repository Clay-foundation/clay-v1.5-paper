"""E1(b): re-score the frozen EuroSAT pilot with the UNCHANGED protocol/evaluate_embeddings.py.

Runs (each a separate invocation of the frozen script):
  clay_full      original manifest + original Clay embeddings (reproduction check)
  dino_full / dofa_full   comparator vector placed in the script's `cls` slot (dino patch-mean in `mean`)
  clay_excl / dino_excl / dofa_excl   test patches whose centroid lies inside a Clay training MGRS tile -> split=excluded
Paired cross-run contrasts (macro-F1, 50/class) use each run's block-confusions.json and the script's own
multiplicity matrix: default_rng(1729).multinomial(n_blocks, 1/n_blocks, size=1000) (same test blocks => paired).
"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np

PILOT = Path(os.environ.get("CLAY_EUROSAT_PILOT_DIR", str(ROOT / "data/eurosat_collected")))
SCRIPT = PILOT / "protocol/evaluate_embeddings.py"
SPECTRAL = PILOT / "results/prepared/spectral_features.npz"
MANIFEST = PILOT / "results/prepared/manifest.csv"
CLAY = PILOT / "results/features/embeddings.npz"


def macro_f1(cm):
    cm = np.asarray(cm, float); d = np.diagonal(cm, axis1=-2, axis2=-1)
    a = cm.sum(-1); p = cm.sum(-2)
    f1 = np.divide(2 * d, a + p, out=np.zeros_like(d), where=a + p != 0)
    return f1.mean(-1)


def run(name, manifest, emb, work):
    out = work / name
    if not (out / "results.json").exists():
        subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest), "--spectral", str(SPECTRAL),
                        "--embeddings", str(emb), "--output", str(out)], check=True, stdout=open(work / f"{name}.log", "w"))
    return json.load(open(out / "results.json")), json.load(open(out / "block-confusions.json"))


def contrast(bcA, bcB, armA="clay_cls", armB="clay_cls", budget=50):
    assert bcA["blocks"] == bcB["blocks"]
    nb = len(bcA["blocks"])
    mult = np.random.default_rng(1729).multinomial(nb, np.full(nb, 1 / nb), size=1000)
    def agg(bc, arm):
        cms = [np.asarray(bc["runs"][f"budget{budget}-seed{s}-{arm}"]) for s in range(5)]
        point = np.mean([macro_f1(c.sum(0)) for c in cms])
        dist = np.mean([macro_f1(np.tensordot(mult, c, axes=(1, 0))) for c in cms], 0)
        return point, dist
    pa, da = agg(bcA, armA); pb, db = agg(bcB, armB)
    d = da - db
    return {"point": float(pa - pb), "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975))}


def main(work="eurosat_eval", overlap="overlap.csv", dino="embeddings_dino.npz", dofa="embeddings_dofa.npz"):
    work = Path(work); work.mkdir(exist_ok=True)
    ov = {r["id"]: r for r in csv.DictReader(open(overlap))}
    rows = list(csv.DictReader(open(MANIFEST, newline="")))
    excl = {r["id"] for r in rows if r["split"] == "test" and ov.get(r["id"], {}).get("in_tile_centroid") == "1"}
    m2 = work / "manifest_excl.csv"
    with open(m2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            r = dict(r)
            if r["id"] in excl:
                r["split"] = "excluded"
            w.writerow(r)
    keep = {r["id"] for r in rows if r["split"] != "excluded" and r["id"] not in excl}
    embs = {"clay": CLAY, "dino": Path(dino), "dofa": Path(dofa)}
    sub = {}
    for k, p in embs.items():
        z = np.load(p); m = np.isin(z["ids"].astype(str), list(keep))
        sp = work / f"emb_{k}_excl.npz"
        if not sp.exists():
            np.savez(sp, ids=z["ids"][m], cls=z["cls"][m], mean=z["mean"][m])
        sub[k] = sp
    R, B = {}, {}
    for k in embs:
        R[f"{k}_full"], B[f"{k}_full"] = run(f"{k}_full", MANIFEST, embs[k], work)
        R[f"{k}_excl"], B[f"{k}_excl"] = run(f"{k}_excl", m2, sub[k], work)
    out = {"n_test_excluded_overlap": len(excl), "runs": {}, "contrasts": {}}
    for k, r in R.items():
        out["runs"][k] = {"primary_contrast_as_scripted": r["primary_contrast"], "n_test": r["n_test"],
                          "n_test_blocks": len(r["test_blocks"]), "summaries": r["summaries"]}
    for s in ["full", "excl"]:
        out["contrasts"][f"clay_cls-dino_cls@50 ({s})"] = contrast(B[f"clay_{s}"], B[f"dino_{s}"])
        out["contrasts"][f"clay_cls-dofa@50 ({s})"] = contrast(B[f"clay_{s}"], B[f"dofa_{s}"])
        out["contrasts"][f"clay_cls-spectral20_plus_xy2@50 ({s})"] = contrast(B[f"clay_{s}"], B[f"clay_{s}"], "clay_cls", "spectral20_plus_xy2")
        for b in (10, 200):
            out["contrasts"][f"clay_cls-dino_cls@{b} ({s})"] = contrast(B[f"clay_{s}"], B[f"dino_{s}"], budget=b)
            out["contrasts"][f"clay_cls-dofa@{b} ({s})"] = contrast(B[f"clay_{s}"], B[f"dofa_{s}"], budget=b)
    json.dump(out, open(work / "eurosat_results.json", "w"), indent=1)
    print(json.dumps(out["contrasts"], indent=1)); print(out["n_test_excluded_overlap"])


if __name__ == "__main__":
    main(*sys.argv[1:])
