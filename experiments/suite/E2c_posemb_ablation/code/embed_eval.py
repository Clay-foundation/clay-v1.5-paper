"""E2c: patch tokens at E2's exact token positions + held-out reconstruction loss, per arm and sensor.
Usage: python embed_eval.py DATADIR E2EMBDIR CKPTDIR OUTDIR [ARMS]"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "code"); sys.path.insert(0, "common")
from common_data import *
DATA, E2E, CK, OUT = map(Path, sys.argv[1:5]); ARMS_RUN = sys.argv[5] if len(sys.argv) > 5 else "ABC"
OUT.mkdir(parents=True, exist_ok=True); dev = "cuda"; t0 = time.time()
def say(*a): print(round(time.time() - t0), *a, flush=True)
models = {}
for a in ARMS_RUN:
    m = SmallClay(a).to(dev).eval(); ck = torch.load(CK / f"model_{a}.pt", map_location=dev, weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=True); models[a] = m
ev = {}
for s in SENSORS:
    pids, lat, lon, blocks, split = chip_index(DATA, s)
    z = np.load(E2E / f"emb_{s}.npz", allow_pickle=True)
    assert list(z["pids"]) == pids and (z["split"] == split).all() and (z["blocks"] == blocks).all(), s
    tpos, tchip = z["tok_pos"].astype(int), z["tok_chip"].astype(int)
    X, spec = load_norm(DATA, s, pids); waves = torch.tensor(spec["waves"], device=dev); gsd = torch.tensor(spec["gsd"], device=dev)
    is_test = split == "test"
    for a, m in models.items():
        cls, mean, P = [], [], []; rec = np.full((len(pids), 2), np.nan, np.float32)
        for i in range(0, len(pids), 64):
            px = torch.from_numpy(X[i:i + 64]).to(dev).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c_, p_ = m.tokens(px, waves, gsd)
                cls.append(c_.cpu().numpy()); mean.append(p_.mean(1).cpu().numpy()); P.append(p_.cpu().numpy().astype(np.float16))
                tb = np.flatnonzero(is_test[i:i + 64])
                if len(tb):
                    for r in range(2):  # two fixed mask draws, identical across arms
                        torch.manual_seed(7_000_000 + 10 * i + r)
                        with torch.no_grad(): _, pc = m(px[tb], waves, gsd)
                        rec[i + tb, r] = pc.float().cpu().numpy()
        P = np.concatenate(P)
        np.savez(OUT / f"emb_{a}_{s}.npz", cls=np.concatenate(cls), mean=np.concatenate(mean),
                 tok_patch=P[tchip, tpos], recon=rec.mean(1))
        ev[f"{a}_{s}"] = {"n": len(pids), "n_test": int(is_test.sum()), "recon_test_mean": float(np.nanmean(rec))}
        say(a, s, ev[f"{a}_{s}"])
    del X
json.dump(ev, open(OUT / "embed_eval_evidence.json", "w"), indent=1)
