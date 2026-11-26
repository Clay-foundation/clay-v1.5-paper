#!/usr/bin/env python3
"""Clay v1.5 on the Table-2 cells that were previously transcribed from the OlmoEarth v1.3 report.

Issue #21: 10 of 38 Clay rows (Vessel S1/L8, Solar Farm, PASTIS, Nandi, MADOS, m-bigearthnet,
m-cashewplant, Sen1Floods11-S1, CropHarvest-Togo) and the Clay fine-tune cells of m-eurosat, m-so2sat,
m-forestnet, m-brick-kiln had no code here.  This script runs every cell that public data allows with the
repo's own protocol and writes one record per cell to experiments/results/clay_missing_cells.json.

Protocol (reused by import, not copied):
  * frozen probes      : experiments/run_standard_benchmarks.py (train-split band z-score, nodata fill, Clay
                         native chip size, val-selected L2 logistic head over {cls, mean}; 1x1 per-token
                         linear head for segmentation).
  * fine-tuning        : optimiser / lr / schedule / augmentation of experiments/suite/E7_finetune/code/
                         train_e7.py (AdamW wd 0.05, head lr, encoder lr = 0.1 x 1e-4, warm-up + cosine,
                         bf16, gradient checkpointing, best-validation epoch, test scored once).  Epoch
                         counts are reduced for the large classification sets (stated in each record).
  * partner / research tasks added later: Vessel S1/L8 and Solar Farm (public OlmoEarth partner_tasks tars, CC BY 4.0;
                         Clay + Faster R-CNN / UPerNet), PASTIS-R (linear probe and fine-tune), MADOS fine-tune.
  * third-party cells  : static record for Nandi only (not public, HTTP 401).

Usage (see the evidence log for the exact commands that produced the committed JSON):
  PYTHONPATH=<dir with h5py> CLAY_V15_CKPT=<ckpt> CLAY_SCRATCH=<scratch> \\
    python experiments/run_clay_missing_cells.py --cells lp_bigearthnet,lp_cashew --out <json>
Cells: static (Nandi only), finalize (mechanical budget check on FT records, run last), lp_mados, lp_bigearthnet, lp_cashew, lp_sen1f_s1,
       lp_cropharvest_togo, lp_pastis, oe_contract, ft_m-eurosat, ft_m-brick-kiln, ft_m-so2sat, ft_m-forestnet, ft_m-bigearthnet,
       ft_m-cashew-plant, ft_sen1f_s1, ft_mados, ft_pastis, prep_vessel-s1, prep_vessel-l8, prep_solar (data preparation, no GPU),
       ft_vessel-s1, ft_vessel-l8, ft_solar.
Flags: --seeds, --epochs (steps for detection cells), --bs, --tag, --lr (single lr for head+encoder; record id gets _lr<value>), --smoke.
Environment: CLAY_V15_CKPT (checkpoint), CLAY_SCRATCH (data/scratch dir), CLAY_H5_PYTHON (python with
h5py, default current), CLAY_E7_DATA (Sen1Floods11 dir, default <scratch>/sen1floods11).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import gc
import io
import json
import math
import os
import sys
import tarfile
import time
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))
import run_standard_benchmarks as rsb  # noqa: E402

SCRIPT = Path(__file__).resolve()
DEFAULT_OUT = Path(os.environ.get("CLAY_MISSING_CELLS_JSON", str(HERE / "experiments/results/clay_missing_cells.json")))
S1F_DIR = Path(os.environ.get("CLAY_E7_DATA", str(rsb.SCRATCH / "sen1floods11")))
S1F_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
CH_BASE = "https://zenodo.org/records/7257688/files"  # CropHarvest labels + features

BANDS_S2_STD = ["02 - Blue", "03 - Green", "04 - Red", "05 - Vegetation Red Edge", "06 - Vegetation Red Edge",
                "07 - Vegetation Red Edge", "08 - NIR", "08A - Vegetation Red Edge", "11 - SWIR", "12 - SWIR"]
GB_EXTRA = {  # tasks that build_tasks_cfg() of run_standard_benchmarks does not define
    "m-bigearthnet": dict(repo_file="classification_v1.0/m-bigearthnet.zip", bands=BANDS_S2_STD, kind="multilabel",
                          waves=rsb.S2_10B_WAVES, gsd=10.0, clay_hw=120),
    "m-cashew-plant": dict(repo_file="segmentation_v1.0/m-cashew-plant.zip", bands=[b + "_*" for b in BANDS_S2_STD],
                           kind="seg", waves=rsb.S2_10B_WAVES, gsd=10.0, clay_hw=256),
}

# Values currently printed in Table 2 (DRAFT.md @ 3deb4bf), all attributed by issue #21 to OlmoEarth v1.3 Table 4 (FT) / 6 (LP, kNN).
OLD = {
    "m-bigearthnet": {"LP": 48.8, "FT": 65.7, "LP_readout_in_source": "kNN"},
    "m-cashew-plant": {"LP": 30.8, "FT": 73.9},
    "sen1floods11-s1": {"LP": 78.9, "FT": 78.5},
    "cropharvest-togo": {"LP": 78.4},
    "m-eurosat": {"FT": 95.1},
    "m-so2sat": {"FT": 61.3},
    "m-forestnet": {"FT": 49.2},
    "m-brick-kiln": {"FT": 98.2},
    "vessel-s1": {"FT": 79.9}, "vessel-l8": {"FT": 70.7}, "solar-farm": {"FT": 82.2},
    "pastis-s2": {"LP": 22.6, "FT": 48.9}, "nandi-s2": {"LP": 57.0, "FT": 30.9}, "mados": {"LP": 47.5, "FT": 68.9},
}
OLD_SOURCE = "OlmoEarth v1.3 report Table 6 (LP/kNN) / Table 4 (FT), Clay ViT Large row; attribution per audit issue 21"


# --------------------------------------------------------------------------------------------- IO
def merge_write(out: Path, cid: str, rec: dict) -> None:
    """Atomically merge one cell record into the results JSON (safe for concurrent runs)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        data = json.loads(out.read_text()) if out.exists() else {"cells": {}}
        data["cells"][cid] = rec
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        os.replace(tmp, out)
    print(f"[written] {cid} -> {out}", flush=True)


DET_BACKBONE_FACTORY = None  # set by run_dofa_missing_cells.py: callable(E7, kind) -> backbone module for the detection cells
DET_MODEL = "clay_v1.5"
TAG = ""  # set by --tag: separate record id for a longer / alternative fine-tune budget
SEED_BUDGET_S = 40 * 60  # protocol: 3 seeds only where one seed trains in under 40 minutes, else the seeds already finished are the result
LR = None  # --lr: one learning rate for head AND encoder (OlmoEarth research-task protocol: a single lr swept over {1e-4,5e-4,1e-3}); default = repo recipe (head 1e-3/1e-4, encoder 1e-5)
SMOKE = False  # --smoke: truncate data and budgets to test a cell end-to-end in about a minute (results go to a smoke file, never committed)


def union_seeds(out: Path, cid: str, runs: list[dict]) -> list[dict]:
    """Per-seed runs may come from separate processes (GPU queue); union them by seed with what is already stored."""
    old = []
    if out.exists():
        old = json.loads(out.read_text())["cells"].get(cid, {}).get("per_seed", [])
    by = {r["seed"]: r for r in old}
    by.update({r["seed"]: r for r in runs})
    return [by[k] for k in sorted(by)]


def prov(extra: dict | None = None) -> dict:
    return rsb.provenance(str(SCRIPT), {"rsb_script_sha256": rsb.sha256_file(rsb.__file__), **(extra or {})})


# --------------------------------------------------------------------------------------------- Clay frozen
class ClayOnly(rsb.ModelSuite):
    """rsb.ModelSuite.extract_clay with only the Clay encoder loaded."""

    def __init__(self):
        from claymodel.model import Encoder
        if rsb.sha256_file(rsb.CLAY_CKPT) != rsb.CLAY_SHA256:
            raise RuntimeError("Clay checkpoint SHA-256 mismatch")
        sd = torch.load(rsb.CLAY_CKPT, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
        state = {k.removeprefix("model.encoder."): v for k, v in sd.items() if k.startswith("model.encoder.")}
        self.clay = Encoder(mask_ratio=0.0, patch_size=8, shuffle=False, dim=1024, depth=24, heads=16,
                            dim_head=64, mlp_ratio=4.0)
        self.clay.load_state_dict(state, strict=True)
        self.clay.eval().requires_grad_(False)


def gb_cfg(task: str) -> dict:
    if task in GB_EXTRA:
        return dict(GB_EXTRA[task], id=task)
    return {c["id"]: c for c in rsb.build_tasks_cfg()}[task]


def load_gb(cfg: dict):
    """GEO-Bench splits through rsb.load_geobench_splits, nodata-filled; so2sat -> 10 S2 bands only."""
    kind = cfg["kind"]
    sp = rsb.load_geobench_splits(cfg["repo_file"], cfg["id"], cfg["bands"], kind=kind)
    meta = dict(getattr(rsb.load_geobench_splits, "last_meta", {}))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = sp["train"], sp["val"], sp["test"]
    if cfg.get("so2sat_s1s2"):
        Xtr, Xva, Xte = Xtr[:, :10], Xva[:, :10], Xte[:, :10]
    means, stds, _, _ = rsb.compute_train_band_stats(Xtr)
    Xtr, Xva, Xte = (rsb.clean_nonfinite(a, means) for a in (Xtr, Xva, Xte))
    waves = rsb.S2_10B_WAVES if cfg.get("so2sat_s1s2") else cfg["waves"]
    return (Xtr, ytr), (Xva, yva), (Xte, yte), means, stds, waves, meta


def clay_feats(suite, X, means, stds, waves, gsd, hw, patches=False):
    """rsb.ModelSuite.extract_clay; the GPU is shared with other jobs, so wait and retry on CUDA OOM."""
    for attempt in range(30):
        try:
            return suite.extract_clay(X, means, stds, waves, gsd, hw, patches)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"[oom] shared GPU full, retry {attempt+1}", flush=True); time.sleep(60)
    raise RuntimeError("GPU stayed full")


# ---- probes not present in rsb ------------------------------------------------------------------
def eval_multilabel_f1(z_tr, y_tr, z_va, y_va, z_te, y_te) -> dict:
    """L2 BCE linear head, C in {0.01,0.1,1,10} chosen on val micro-F1 (threshold 0.5), scored once on test."""
    sc = StandardScaler().fit(z_tr)
    T = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32, device="cuda")
    Xtr, Xva, Xte = T(z_tr), T(z_va), T(z_te)
    Y = torch.tensor(y_tr, dtype=torch.float32, device="cuda")
    best = (-1.0, None, None)
    for c in (0.01, 0.1, 1.0, 10.0):
        W, b = rsb._fit_gpu_linear_lbfgs(Xtr, Y, y_tr.shape[1], 0.5 / (c * len(Xtr)), mode="bce")
        with torch.no_grad():
            pva = (torch.sigmoid(Xva @ W.t() + b) > 0.5).cpu().numpy().astype(int)
            f_va = float(f1_score(y_va.astype(int), pva, average="micro", zero_division=0))
            if f_va > best[0]:
                pte = torch.sigmoid(Xte @ W.t() + b).cpu().numpy()
                best = (f_va, c, pte)
    pte = best[2]
    yt = y_te.astype(int)
    valid = yt.sum(0) > 0
    return {"micro_f1": round(float(f1_score(yt, (pte > 0.5).astype(int), average="micro", zero_division=0)) * 100, 2),
            "macro_map": round(float(average_precision_score(yt[:, valid], pte[:, valid], average="macro")) * 100, 2),
            "val_micro_f1": round(best[0] * 100, 2), "best_C": best[1]}


@torch.no_grad()
def knn_predict(z_tr, y_tr, z_te, k=20, n_classes=None, multilabel=False):
    """Cosine k-NN (k=20 as in the OlmoEarth knn.py default); soft vote, argmax / >=0.5 for multilabel."""
    a = F.normalize(torch.tensor(z_tr, dtype=torch.float32, device="cuda"), dim=1)
    outs = []
    Y = torch.tensor(y_tr, device="cuda")
    for i in range(0, len(z_te), 1024):
        b = F.normalize(torch.tensor(z_te[i:i + 1024], dtype=torch.float32, device="cuda"), dim=1)
        idx = (b @ a.t()).topk(k, dim=1).indices
        if multilabel:
            outs.append(Y[idx].float().mean(1).cpu())
        else:
            outs.append(F.one_hot(Y[idx], n_classes).float().sum(1).cpu())
    return torch.cat(outs).numpy()


def eval_knn(z_tr, y_tr, z_te, y_te, multilabel=False, n_classes=None):
    p = knn_predict(z_tr, y_tr, z_te, 20, n_classes, multilabel)
    if multilabel:
        return {"micro_f1": round(float(f1_score(y_te.astype(int), (p >= 0.5).astype(int), average="micro", zero_division=0)) * 100, 2)}
    return {"oa": round(float(accuracy_score(y_te, p.argmax(1))) * 100, 2)}


def pick(res_by_ro: dict, key: str, val_key: str):
    ro = max(res_by_ro, key=lambda r: res_by_ro[r][val_key])
    return ro, res_by_ro[ro][key]


# --------------------------------------------------------------------------------------------- LP cells
def lp_bigearthnet(suite, out):
    cfg = gb_cfg("m-bigearthnet")
    (Xtr, ytr), (Xva, yva), (Xte, yte), mu, sd, waves, meta = load_gb(cfg)
    hw = Xtr.shape[-1]
    feats = {s: clay_feats(suite, X, mu, sd, waves, cfg["gsd"], hw) for s, X in (("tr", Xtr), ("va", Xva), ("te", Xte))}
    ro_res = {ro: eval_multilabel_f1(feats["tr"][ro], ytr, feats["va"][ro], yva, feats["te"][ro], yte) for ro in ("cls", "mean")}
    ro, v = pick(ro_res, "micro_f1", "val_micro_f1")
    knn = {ro_: eval_knn(feats["tr"][ro_], ytr, feats["te"][ro_], yte, multilabel=True) for ro_ in ("cls", "mean")}
    rec = {
        "table2_row": "m-bigearthnet", "cell": "LP", "status": "rerun_here", "metric": "test micro-F1 (%)",
        "value": v, "readout": ro, "head": "L2 BCE linear head (val-selected C), threshold 0.5",
        "readout_selection": "readout and C chosen on val micro-F1; test scored once", "seeds": "deterministic (no seed)",
        "split_sizes": [len(ytr), len(yva), len(yte)], "chip_hw": hw, "bands": cfg["bands"],
        "details": {"probe_by_readout": ro_res, "knn_k20_cosine_by_readout": knn},
        "old_transcribed": {"value": OLD["m-bigearthnet"]["LP"], "source": OLD_SOURCE,
                            "note": "source value is a native-grid kNN, not a linear probe; compare with details.knn_k20_cosine_by_readout"},
        "geobench_check": meta, "provenance": prov(),
    }
    merge_write(out, "m-bigearthnet/LP", rec)


def lp_cashew(suite, out):
    cfg = gb_cfg("m-cashew-plant")
    (Xtr, ytr), (Xva, yva), (Xte, yte), mu, sd, waves, meta = load_gb(cfg)
    ncls = int(max(ytr.max(), yva.max(), yte.max())) + 1
    f = {s: clay_feats(suite, X, mu, sd, waves, cfg["gsd"], Xtr.shape[-1], True) for s, X in (("tr", Xtr), ("va", Xva), ("te", Xte))}
    r = rsb.eval_dense_segmentation(f["tr"]["patches"], ytr, f["va"]["patches"], yva, f["te"]["patches"], yte, ncls)
    rec = {
        "table2_row": "m-cashewplant", "cell": "LP", "status": "rerun_here", "metric": "test mIoU (%), classes with union>0",
        "value": r["miou"], "readout": "Z_patch (per-token), 1x1 linear head bilinearly upsampled", "head": "multinomial linear, val-selected C",
        "seeds": "deterministic (fixed subsample seed 42)", "split_sizes": [len(ytr), len(yva), len(yte)],
        "chip_hw": int(Xtr.shape[-1]), "n_classes": ncls, "details": r,
        "old_transcribed": {"value": OLD["m-cashew-plant"]["LP"], "source": OLD_SOURCE},
        "geobench_check": meta, "provenance": prov(),
    }
    merge_write(out, "m-cashew-plant/LP", rec)


# ---- Sen1Floods11 (S1) ---------------------------------------------------------------------------
def fetch_s1f_s1():
    """Download S1Hand + LabelHand chips and the official split CSVs (public bucket) into S1F_DIR."""
    import csv
    import rasterio  # noqa: F401
    (S1F_DIR / "S1Hand").mkdir(parents=True, exist_ok=True)
    (S1F_DIR / "LabelHand").mkdir(exist_ok=True)
    def get(u, tries=6):
        for i in range(tries):
            try:
                return urllib.request.urlopen(u, timeout=120).read()
            except Exception as e:  # transient network errors
                if i == tries - 1:
                    raise
                print(f"[retry {i+1}] {u}: {e}", flush=True); time.sleep(3 * (i + 1))
    splits = {}
    for s, f in [("train", "flood_train_data.csv"), ("valid", "flood_valid_data.csv"),
                 ("test", "flood_test_data.csv"), ("bolivia", "flood_bolivia_data.csv")]:
        p = S1F_DIR / f
        if not p.exists():
            p.write_text(get(f"{S1F_BASE}/splits/flood_handlabeled/{f}").decode())
        splits[s] = [r[0].replace("_S1Hand.tif", "") for r in csv.reader(io.StringIO(p.read_text())) if r]

    def dl(c):
        for d in ("S1Hand", "LabelHand"):
            p = S1F_DIR / d / f"{c}_{d}.tif"
            if not p.exists():
                p.write_bytes(get(f"{S1F_BASE}/data/flood_events/HandLabeled/{d}/{c}_{d}.tif"))
    chips = sorted({c for v in splits.values() for c in v})
    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(dl, chips))
    return splits


def load_s1f_s1():
    import rasterio
    splits = fetch_s1f_s1()
    D = {}
    for s, ids in splits.items():
        xs, ys = [], []
        for c in sorted(ids):
            with rasterio.open(S1F_DIR / "S1Hand" / f"{c}_S1Hand.tif") as r:
                xs.append(r.read()[:2].astype(np.float32))
            with rasterio.open(S1F_DIR / "LabelHand" / f"{c}_LabelHand.tif") as r:
                ys.append(r.read(1).astype(np.int16))
        D[s] = (np.stack(xs), np.stack(ys), sorted(ids))
    mu, sd, _, _ = rsb.compute_train_band_stats(D["train"][0])
    for s in D:
        D[s] = (rsb.clean_nonfinite(D[s][0], mu), D[s][1], D[s][2])
    return D, mu, sd


def tile256(X, y):
    N = len(X)
    xt = X.reshape(N, X.shape[1], 2, 256, 2, 256).transpose(0, 2, 4, 1, 3, 5).reshape(N * 4, X.shape[1], 256, 256)
    yt = y.reshape(N, 2, 256, 2, 256).transpose(0, 1, 3, 2, 4).reshape(N * 4, 256, 256)
    return np.ascontiguousarray(xt), np.ascontiguousarray(yt)


class IgnoreCE:
    """Context manager: make rsb's cross-entropy ignore label -1 (Sen1Floods11 no-data) without editing rsb."""

    def __enter__(self):
        import functools
        self.orig = rsb.F
        proxy = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
        proxy.cross_entropy = functools.partial(F.cross_entropy, ignore_index=-1)
        rsb.F = proxy
        return self

    def __exit__(self, *a):
        rsb.F = self.orig


def lp_sen1f_s1(suite, out):
    D, mu, sd = load_s1f_s1()
    waves, gsd = rsb.S1_2B_WAVES, 10.0
    feats, masks = {}, {}
    for s in ("train", "valid", "test", "bolivia"):
        xt, yt = tile256(D[s][0], D[s][1])
        feats[s] = clay_feats(suite, xt, mu, sd, waves, gsd, 256, True)["patches"]
        masks[s] = yt
    res = {}
    with IgnoreCE():
        for te in ("test", "bolivia"):
            res[te] = rsb.eval_dense_segmentation(feats["train"], masks["train"], feats["valid"], masks["valid"],
                                                  feats[te], masks[te], 2)
    rec = {
        "table2_row": "Sen1Floods11 (S1)", "cell": "LP", "status": "rerun_here",
        "metric": "test mIoU (%) over {non-water, water}; fg_iou = water IoU", "value": res["test"]["miou"],
        "readout": "Z_patch (per-token), 1x1 linear head, 512x512 chips tiled to 4 x 256x256",
        "head": "multinomial linear, val-selected C, no-data (-1) ignored in fit and score",
        "seeds": "deterministic (fixed subsample seed 42)",
        "split_sizes": {k: len(D[k][2]) for k in D}, "bands": ["VV_dB", "VH_dB"], "waves_um": waves,
        "details": res, "old_transcribed": {"value": OLD["sen1floods11-s1"]["LP"], "source": OLD_SOURCE},
        "data": f"{S1F_BASE}/data/flood_events/HandLabeled/S1Hand (public GCS bucket), official flood_handlabeled splits",
        "provenance": prov(),
    }
    merge_write(out, "sen1floods11-s1/LP", rec)


# ---- CropHarvest-Togo ----------------------------------------------------------------------------
def load_togo(work: Path):
    import h5py
    import pandas as pd
    work.mkdir(parents=True, exist_ok=True)
    lbl, tar = work / "labels.geojson", work / "features.tar.gz"
    for f, p in (("labels.geojson", lbl), ("features.tar.gz", tar)):
        if not p.exists():
            urllib.request.urlretrieve(f"{CH_BASE}/{f}?download=1", str(p))
    df = pd.json_normalize(json.loads(lbl.read_text())["features"])
    need = {}
    for ds, suffix in (("togo", "togo"), ("togo-eval", "togo-eval")):
        sub = df[df["properties.dataset"] == ds]
        for _, r in sub.iterrows():
            need[f"{int(r['properties.index'])}_{suffix}.h5"] = (ds, int(r["properties.is_crop"]))
    got = {"togo": ([], []), "togo-eval": ([], [])}
    with tarfile.open(tar, "r:gz") as tf:
        for m in tf:
            b = Path(m.name).name
            if b in need:
                with h5py.File(io.BytesIO(tf.extractfile(m).read()), "r") as hf:
                    arr = hf["array"][:].astype(np.float32)
                ds, y = need[b]
                got[ds][0].append(arr); got[ds][1].append(y)
    return (np.stack(got["togo"][0]), np.array(got["togo"][1])), (np.stack(got["togo-eval"][0]), np.array(got["togo-eval"][1]))


def lp_cropharvest_togo(suite, out):
    (Xtr_raw, y_tr), (Xte_raw, y_te) = load_togo(rsb.SCRATCH / "cropharvest_togo")
    s1_tr, s1_te = Xtr_raw[:, :, :2], Xte_raw[:, :, :2]            # (N, 12 months, [VV, VH])
    mu = s1_tr.mean(axis=(0, 1)); sd = s1_tr.std(axis=(0, 1)) + 1e-6
    rng = np.random.RandomState(42)                                 # val carved from TRAIN only (test never used for selection)
    perm = rng.permutation(len(y_tr)); n_va = int(0.15 * len(y_tr))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]

    def embed(S):                                                   # one month = one constant 224x224 image (as the DOFA/Pan arms)
        N, T, C = S.shape
        px = np.broadcast_to(S.reshape(N * T, C, 1, 1), (N * T, C, 16, 16)).astype(np.float32)
        e = suite.extract_clay(np.ascontiguousarray(px), mu, sd, rsb.S1_2B_WAVES, 10.0, 224, False, batch_size=128)
        r = {}
        for ro in ("cls", "mean"):
            E = e[ro].reshape(N, T, -1)
            r[ro] = np.concatenate([E.mean(1), E.std(1), E.max(1)], axis=1)
        return r
    Etr, Ete = embed(s1_tr), embed(s1_te)
    res = {}
    for ro in ("cls", "mean"):
        r = rsb.eval_single_label(Etr[ro][tr_idx], y_tr[tr_idx], Etr[ro][va_idx], y_tr[va_idx], Ete[ro], y_te)
        # refit on 100% of train (C fixed by val); never selects anything on test
        from sklearn.linear_model import LogisticRegression
        sc = StandardScaler().fit(Etr[ro])
        clf = LogisticRegression(C=r["best_C"], max_iter=1000).fit(sc.transform(Etr[ro]), y_tr)
        r["oa_refit_full_train"] = round(float(accuracy_score(y_te, clf.predict(sc.transform(Ete[ro])))) * 100, 2)
        res[ro] = r
    ro = max(res, key=lambda k: res[k]["val_oa"])
    rec = {
        "table2_row": "CropHarvest-Togo", "cell": "LP", "status": "rerun_here",
        "metric": "test accuracy (%) on the official togo-eval set", "value": res[ro]["oa"], "readout": f"{ro} per-month, concat[mean,std,max] over 12 months",
        "head": "L2 logistic, C chosen on a 15% val carved from train; fit on the other 85%; no test-set selection",
        "seeds": "val split seed 42", "split_sizes": [int(len(tr_idx)), int(len(va_idx)), int(len(y_te))],
        "input": "S1 VV/VH channels 0,1 of the 12-month CropHarvest array, one constant image per month",
        "details": res, "old_transcribed": {"value": OLD["cropharvest-togo"]["LP"], "source": OLD_SOURCE + " (attribution of this value to Table 6 is an inference)"},
        "provenance": prov(),
    }
    merge_write(out, "cropharvest-togo/LP", rec)


# ---- MADOS (public: Zenodo 10664073) --------------------------------------------------------------
MADOS_URL = "https://zenodo.org/records/10664073/files/MADOS.zip?download=1"


def load_mados():
    """MADOS tiles, re-implementing the OlmoEarth tile preparation (olmoearth_pretrain evals/datasets/mados_dataset.py:
    S2 L2R bands, 20 m bands bilinearly resized to 240 px, B06 = mean(B05,B07), 240x240 patches cut into 9 tiles of 80 px,
    tiles without any label dropped).  Returns the 10 Clay S2 bands (raw reflectance) and labels (0..14, -1 = unlabelled)
    for the official train/val/test split files, nodata-filled and with train-split band mean/std."""
    import zipfile
    import rasterio
    from rasterio.io import MemoryFile
    zp = Path(os.environ.get("CLAY_MADOS_ZIP", str(rsb.SCRATCH / "mados/MADOS.zip")))
    if not zp.exists():
        zp.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MADOS_URL, str(zp))
    zf = zipfile.ZipFile(zp)
    names = set(zf.namelist())
    root = next(n for n in names if n.endswith("splits/train_X.txt"))[: -len("splits/train_X.txt")]

    def band(a, b, wls, res):
        for wl in wls:
            n = f"{root}Scene_{a}/{res}/Scene_{a}_L2R_rhorc_{wl}_{b}.tif"
            if n in names:
                with MemoryFile(zf.read(n)) as mf, mf.open() as r:
                    return torch.from_numpy(r.read(1).astype(np.float32))[None, None]
        raise FileNotFoundError((a, b, wls, res))

    def rs(t):
        return F.interpolate(t, size=240, mode="bilinear", align_corners=False)
    def split(name):
        X, Y = [], []
        for line in zf.read(f"{root}splits/{name}_X.txt").decode().split():
            a, b = line.split("_")[1], line.split("_")[2]
            B2, B3, B4, B8 = (band(a, b, w, 10) for w in ([492], [559, 560], [665], [833]))
            B5, B7, B8A, B11, B12 = (rs(band(a, b, w, 20)) for w in ([704], [780, 783], [864, 865], [1610, 1614], [2186, 2202]))
            img = torch.cat([B2, B3, B4, B5, (B5 + B7) / 2, B7, B8, B8A, B11, B12], 1)[0]            # 10 Clay bands
            with MemoryFile(zf.read(f"{root}Scene_{a}/10/Scene_{a}_L2R_cl_{b}.tif")) as mf, mf.open() as r:
                m = torch.from_numpy(r.read(1).astype(np.int64))
            for i in range(3):
                for j in range(3):
                    mt = m[i * 80:(i + 1) * 80, j * 80:(j + 1) * 80]
                    if (mt > 0).any():
                        X.append(img[:, i * 80:(i + 1) * 80, j * 80:(j + 1) * 80].numpy()); Y.append((mt - 1).numpy().astype(np.int16))
        return np.stack(X), np.stack(Y)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split("train"), split("val"), split("test")
    mu, sd, _, _ = rsb.compute_train_band_stats(Xtr)
    Xtr, Xva, Xte = (rsb.clean_nonfinite(a, mu) for a in (Xtr, Xva, Xte))
    return (Xtr, ytr), (Xva, yva), (Xte, yte), mu, sd


def lp_mados(suite, out):
    """Frozen LP on MADOS (tiles from load_mados).  Clay input: the 10 Clay S2 bands, z-score with train-split stats."""
    (Xtr, ytr), (Xva, yva), (Xte, yte), mu, sd = load_mados()
    n = 15
    f = {k: clay_feats(suite, X, mu, sd, rsb.S2_10B_WAVES, 10.0, 80, True)["patches"] for k, X in (("tr", Xtr), ("va", Xva), ("te", Xte))}
    with IgnoreCE():
        r = rsb.eval_dense_segmentation(f["tr"], ytr, f["va"], yva, f["te"], yte, n)
    rec = {"table2_row": "MADOS", "cell": "LP", "status": "rerun_here", "metric": "test mIoU (%) over the 15 labelled classes with union>0 (unlabelled=ignored)",
           "value": r["miou"], "readout": "Z_patch (per-token), 1x1 linear head on 80 px tiles (10x10 tokens), bilinear upsampling",
           "head": "multinomial linear, val-selected C", "seeds": "deterministic (fixed subsample seed 42)",
           "split_sizes_tiles": [len(ytr), len(yva), len(yte)], "details": r,
           "caveat": "own re-implementation of the OlmoEarth tile preparation; OlmoEarth's metric/normalisation/probe differ, so the value is Clay under this repo's probe, not a replication of their number",
           "old_transcribed": {"value": OLD["mados"]["LP"], "source": OLD_SOURCE}, "data": MADOS_URL, "provenance": prov()}
    merge_write(out, "mados/LP", rec)


# ---- PASTIS-R S2 (12 monthly composites, 64 px tiles) -----------------------------------------------------
PASTIS_URL = "https://storage.googleapis.com/ai2-olmoearth-projects-public-data/research_benchmarks/pastis_r"
PASTIS_DIR = Path(os.environ.get("CLAY_PASTIS_DIR", str(rsb.SCRATCH / "pastis_r")))
PASTIS_CLAY_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]  # B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 inside the 13-band OlmoEarth layout (the 10 real PASTIS bands)


def load_pastis(split_names=("train", "valid", "test")):
    """OlmoEarth's public processing of PASTIS-R (Zenodo 5735646): 64x64 quadrants of the 128 px patches, one mean composite per
    month (12 timesteps), folds 1-3 train / 4 val / 5 test, 19 classes, void -> -1.  Returns X (N,12,10,64,64) float16 raw
    reflectance DN (the 10 real bands only; the imputed duplicates B01/B09/B10 are dropped), y (N,64,64) int16."""
    out = {}
    for sp in split_names:
        d = PASTIS_DIR / f"pastis_r_{sp}"
        y = torch.load(d / "targets.pt", weights_only=True).numpy().astype(np.int16)
        if SMOKE:
            y = y[:40]
        n = len(y)
        X = np.empty((n, 12, 10, 64, 64), np.float16)
        for i in range(n):
            t = torch.load(d / "s2_images" / f"{i}.pt", weights_only=True)  # (12, 13, 64, 64)
            assert t.shape[0] == 12 and t.shape[1] == 13, t.shape
            X[i] = t[:, PASTIS_CLAY_IDX].numpy().astype(np.float16)
        out[sp] = (X, y)
        print(f"[pastis] {sp}: {X.shape} labels {np.unique(y)[:3]}..{y.max()}", flush=True)
    sub = out["train"][0][:: max(1, len(out["train"][0]) // 800)].astype(np.float32)  # (n,12,10,64,64)
    sub = sub.transpose(0, 1, 3, 4, 2).reshape(-1, 10)
    mu, sd = sub.mean(0).astype(np.float32), np.maximum(sub.std(0), 1e-6).astype(np.float32)
    return out, mu, sd


def lp_pastis(suite, out):
    """Frozen LP on PASTIS-S2.  Clay is single-date, so each of the 12 monthly composites is encoded separately (10 bands, 64 px,
    GSD 10 m, wavelengths in um) and the 8x8 token grids are pooled over time (mean and max, selected on validation mIoU), as the
    OlmoEarth protocol does for single-date baselines (olmoearth_pretrain evals, 'sweep pooling method across time')."""
    D, mu, sd = load_pastis()
    feats = {}
    for sp, (X, y) in D.items():
        fm, fx = [], []
        for i in range(0, len(X), 128):
            xb = X[i:i + 128].astype(np.float32)
            nb = xb.shape[0]
            f = clay_feats(suite, xb.reshape(nb * 12, 10, 64, 64), mu, sd, rsb.S2_10B_WAVES, 10.0, 64, True)["patches"]
            f = torch.from_numpy(f).float().reshape(nb, 12, 8, 8, 1024)
            fm.append(f.mean(1).half().numpy()); fx.append(f.max(1).values.half().numpy())
        feats[sp] = {"mean": np.concatenate(fm), "max": np.concatenate(fx)}
        print(f"[pastis LP] features {sp} done", flush=True)
    res = {}
    with IgnoreCE():
        for pool in ("mean", "max"):
            res[pool] = rsb.eval_dense_segmentation(feats["train"][pool], D["train"][1], feats["valid"][pool], D["valid"][1],
                                                    feats["test"][pool], D["test"][1], 19)
    pool = max(res, key=lambda k: res[k]["val_miou"])
    rec = {"table2_row": "PASTIS-S2", "cell": "LP", "status": "rerun_here", "metric": "test mIoU (%) over the 19 classes (void ignored)",
           "value": res[pool]["miou"], "readout": f"Z_patch per date (8x8 tokens), pooled over the 12 dates by {pool} (selected on val mIoU), 1x1 linear head, bilinear upsampling",
           "head": "multinomial linear, val-selected C", "seeds": "deterministic (fixed subsample seed 42)",
           "split_sizes_tiles": [len(D[k][1]) for k in ("train", "valid", "test")], "details": {"by_pooling": res, "pooling_selected": pool,
           "temporal_protocol": "each monthly composite encoded separately (single-date model), token grids pooled over dates"},
           "bands": BANDS_S2_STD, "waves_um": rsb.S2_10B_WAVES,
           "old_transcribed": {"value": OLD["pastis-s2"]["LP"], "source": OLD_SOURCE},
           "data": PASTIS_URL + " (OlmoEarth public processing of PASTIS-R: 64 px quadrants, monthly means, folds 1-3/4/5)", "provenance": prov()}
    merge_write(out, "pastis-s2/LP", rec)


class PastisFT(nn.Module):
    """Clay encoder applied to each date separately; the four tapped token maps are averaged over the 12 dates, then E7 Neck + UPerHead."""

    def __init__(self, E7, n_out, waves, gsd):
        super().__init__()
        self.bb = E7.ClayBackbone(trainable=True)
        self.bb.waves = torch.tensor(waves, dtype=torch.float32, device="cuda")
        self.bb.gsd = torch.tensor(float(gsd))
        self.neck, self.head = E7.Neck(1024, 8), E7.UPerHead(ncls=n_out)
        self.taps = E7.TAPS

    def forward(self, x):  # x (B,T,10,H,W)
        B, T = x.shape[:2]
        z = torch.zeros(B * T, 4, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = self.bb(x.flatten(0, 1), z, z, taps=self.taps)
            f = [t.reshape(B, T, *t.shape[1:]).mean(1) for t in f]
            o = self.head(self.neck(f))
        return F.interpolate(o.float(), x.shape[-2:], mode="bilinear", align_corners=False)


def ft_pastis_cell(seeds, epochs, out, bs=8):
    E7 = load_e7()
    D, mu_n, sd_n = load_pastis()
    mu = torch.tensor(mu_n, dtype=torch.float32, device="cuda").view(1, 1, -1, 1, 1)
    sd = torch.tensor(sd_n, dtype=torch.float32, device="cuda").view(1, 1, -1, 1, 1)
    n = 19
    Xtr, ytr = torch.from_numpy(D["train"][0]), torch.from_numpy(D["train"][1])
    Xva, yva = torch.from_numpy(D["valid"][0]), torch.from_numpy(D["valid"][1])
    Xte, yte = torch.from_numpy(D["test"][0]), torch.from_numpy(D["test"][1])
    prep = lambda xb: (xb.float().cuda() - mu) / sd

    @torch.no_grad()
    def evaluate(model, X, y):
        model.eval(); conf = np.zeros((n, n), np.int64)
        for i in range(0, len(X), 8):
            p = model(prep(X[i:i + 8])).argmax(1).cpu().numpy()
            conf_update(conf, p.reshape(-1), y[i:i + 8].numpy().reshape(-1), n)
        model.train(); return conf

    runs = []
    prev = union_seeds(out, f"pastis-s2/FT{TAG}", [])
    for seed in seeds:
        if seed in {r["seed"] for r in prev}:
            print(f"[pastis] seed {seed} already stored, skipping", flush=True); continue
        if (runs or prev) and (runs or prev)[-1]["train_s"] > SEED_BUDGET_S:
            print(f"[pastis] seed took {(runs or prev)[-1]['train_s']} s > {SEED_BUDGET_S} s: stopping", flush=True); break
        torch.manual_seed(seed); np.random.seed(seed)
        model = PastisFT(E7, n, rsb.S2_10B_WAVES, 10.0).cuda(); model.train()
        enc = [p for k, p in model.named_parameters() if k.startswith("bb.") and p.requires_grad]
        rest = [p for k, p in model.named_parameters() if not k.startswith("bb.")]
        opt = torch.optim.AdamW([{"params": rest, "lr": 1e-4}, {"params": enc, "lr": 1e-5}], betas=(0.9, 0.999), weight_decay=0.05)
        spe = math.ceil(len(Xtr) / bs); total = spe * epochs; warm = spe * min(2, epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lr(total, warm))
        gen = torch.Generator().manual_seed(seed)
        best = (-1.0, 0, None); curve = []; t0 = time.time()
        for ep in range(epochs):
            perm = torch.randperm(len(Xtr), generator=gen); el = []
            for s in range(spe):
                idx = perm[s * bs:(s + 1) * bs]
                x = prep(Xtr[idx]); y = ytr[idx].long().cuda()
                if torch.rand(1, generator=gen) < 0.5: x = x.flip(-1); y = y.flip(-1)
                if torch.rand(1, generator=gen) < 0.5: x = x.flip(-2); y = y.flip(-2)
                k = int(torch.randint(0, 4, (1,), generator=gen)); x = torch.rot90(x, k, (-2, -1)); y = torch.rot90(y, k, (-2, -1))
                loss = F.cross_entropy(model(x), y, ignore_index=-1)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); el.append(loss.item())
                if s % 100 == 0:
                    print(f"[pastis s{seed}] ep {ep+1} step {s}/{spe} loss {np.mean(el[-100:]):.4f} t {time.time()-t0:.0f}s", flush=True)
            m = float(np.mean(miou_from_conf(evaluate(model, Xva, yva)))) * 100
            curve.append([ep + 1, float(np.mean(el)), m])
            if m > best[0]:
                best = (m, ep + 1, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()})
            print(f"[pastis s{seed}] ep {ep+1}/{epochs} loss {np.mean(el):.4f} val mIoU {m:.2f} best_ep {best[1]} t {time.time()-t0:.0f}s", flush=True)
        model.load_state_dict(best[2])
        ious = miou_from_conf(evaluate(model, Xte, yte))
        runs.append({"seed": seed, "epochs": epochs, "best_val_miou": round(best[0], 2), "best_epoch": best[1], "test_miou": round(float(np.mean(ious)) * 100, 2),
                     "train_s": round(time.time() - t0), "curve": curve})
        del model, opt; gc.collect(); torch.cuda.empty_cache()
        runs_all = union_seeds(out, f"pastis-s2/FT{TAG}", runs); vals = [r["test_miou"] for r in runs_all]
        rec = {"table2_row": "PASTIS-S2", "cell": "FT", "status": "rerun_here", "partial_seeds_done": len(runs_all) < 3,
               "metric": "test mIoU (%) over the 19 classes (void ignored)", "value": round(float(np.mean(vals)), 2),
               "value_std_over_seeds": round(float(np.std(vals)), 2) if len(vals) > 1 else None,
               "head": "E7 UPerNet neck+head on taps (6,12,18,24); each of the 12 monthly composites encoded separately by Clay, tapped token maps averaged over dates before the neck",
               "recipe": {"optimizer": "AdamW wd 0.05", "lr_head": 1e-4, "lr_encoder": 1e-5, "schedule": "2-epoch warm-up + cosine", "epochs": epochs, "batch_size": bs,
                          "input": "64 px tiles, 10 Clay S2 bands, train-split z-score, flips+rot90", "precision": "bf16, gradient checkpointing",
                          "selection": "best-validation mIoU epoch (every epoch); test scored once"},
               "seeds": [r["seed"] for r in runs_all], "per_seed": runs_all,
               "split_sizes_tiles": [len(ytr), len(yva), len(yte)], "bands": BANDS_S2_STD, "waves_um": rsb.S2_10B_WAVES,
               "old_transcribed": {"value": OLD["pastis-s2"]["FT"], "source": OLD_SOURCE},
               "data": PASTIS_URL + " (OlmoEarth public processing of PASTIS-R)", "provenance": prov()}
        merge_write(out, f"pastis-s2/FT{TAG}", rec)


# ---- OlmoEarth partner tasks: Sentinel-1 / Landsat vessel detection (public rslearn datasets, CC BY 4.0) -----------------
PARTNER_URL = "https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks"
PARTNER_DIR = Path(os.environ.get("CLAY_PARTNER_DIR", str(rsb.SCRATCH / "partner_tasks")))
DET_PATCH = 128          # rslearn eval task config: patch_size 128; Clay sees 128 px (16x16 tokens)
DET_BOX = 15             # DetectionTask box_size 15 (box = centre +- 15 px); F1 matches on centre distance <= 15 px
DET_THRS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DET_SPEC = {  # tar name, Clay bands, wavelengths (um), grid GSD (m)
    "vessel-s1": dict(tar="sentinel1_vessels", waves=rsb.S1_2B_WAVES, gsd=10.0, bands=["VV_dB", "VH_dB"], row="Vessel Detect. (S1)"),
    "vessel-l8": dict(tar="landsat_vessels", waves=rsb.L8_6B_WAVES, gsd=15.0, bands=["B4", "B3", "B2", "B5", "B6", "B7"], row="Vessel Detect. (L8)"),
}
L8_30M_SET = "B1_B2_B3_B4_B5_B6_B7_B9_B10_B11"   # band order of the 30 m band set; Clay bands B4,B3,B2,B5,B6,B7 -> positions 3,2,1,4,5,6
L8_CLAY_POS = [3, 2, 1, 4, 5, 6]


def prep_vessels(kind: str):
    """Stream the public tar once and write one compact .npz per window: image (S1: float16 dB at 10 m; L8: uint16 DN of the 6 Clay
    bands at the 30 m band-set grid, upsampled 2x nearest at read time to the 15 m window grid), validity mask, vessel centres
    (window-pixel coordinates), split tag from options.olmoearth_evals_split."""
    import rasterio
    from rasterio.io import MemoryFile
    from PIL import Image
    sp = DET_SPEC[kind]
    tarp = PARTNER_DIR / f"{sp['tar']}.tar"
    outd = PARTNER_DIR / f"{sp['tar']}_cache"
    for k in ("train", "val", "test"):
        (outd / k).mkdir(parents=True, exist_ok=True)
    img_rel = "layers/sentinel1/vv_vh/geotiff.tif" if kind == "vessel-s1" else f"layers/landsat/{L8_30M_SET}/geotiff.tif"
    want = {"metadata.json", "layers/label/data.geojson", "layers/mask/mask/image.png", img_rel}
    counts = {"train": 0, "val": 0, "test": 0}; n_obj = {"train": 0, "val": 0, "test": 0}

    def flush(key, files):
        if key is None or not want.issubset(files):
            return
        m = json.loads(files["metadata.json"]); split = m["options"].get("olmoearth_evals_split")
        if split not in counts:
            return
        with MemoryFile(files[img_rel]) as mf, mf.open() as r:
            arr = r.read().astype(np.float32)
        if kind == "vessel-s1":
            arr = (10.0 * np.log10(np.clip(arr, 1e-6, None))).astype(np.float16); scale = 1
        else:
            arr = arr[L8_CLAY_POS].astype(np.uint16); scale = 2
        mask = (np.asarray(Image.open(io.BytesIO(files["layers/mask/mask/image.png"]))) > 0).astype(np.uint8)
        fc = json.loads(files["layers/label/data.geojson"])
        import shapely.geometry as sg
        pj = fc.get("properties", {})
        assert not fc["features"] or (pj.get("crs") == m["projection"]["crs"] and pj.get("x_resolution") == m["projection"]["x_resolution"]), (pj, m["projection"])
        ctr = [(sg.shape(f["geometry"]).centroid.x - m["bounds"][0], sg.shape(f["geometry"]).centroid.y - m["bounds"][1]) for f in fc["features"]
               if f["properties"].get("category") == "vessel"]
        H, W = m["bounds"][3] - m["bounds"][1], m["bounds"][2] - m["bounds"][0]
        assert mask.shape == (H, W), (mask.shape, H, W)
        np.savez(outd / split / f"{counts[split]:05d}.npz", img=arr, mask=mask, ctr=np.asarray(ctr, np.float32).reshape(-1, 2), scale=scale,
                 res=m["projection"]["x_resolution"], name=m["name"])
        counts[split] += 1; n_obj[split] += len(ctr)

    t0 = t_print = time.time(); cur = None; files = {}
    with tarfile.open(tarp, "r|") as tf:
        for mem in tf:
            if not mem.isfile():
                continue
            parts = mem.name.split("/")
            if parts[0] != "windows" or len(parts) < 4:
                continue
            key = (parts[1], parts[2])
            if key != cur:
                flush(cur, files); cur = key; files = {}
            rel = "/".join(parts[3:])
            if rel in want:
                files[rel] = tf.extractfile(mem).read()
            if time.time() - t_print > 60:
                t_print = time.time(); print(f"[prep {kind}] {counts} {time.time()-t0:.0f}s", flush=True)
        flush(cur, files)
    (outd / "summary.json").write_text(json.dumps({"windows": counts, "vessels": n_obj}))
    print(f"[prep {kind}] done: windows {counts} vessels {n_obj} {time.time()-t0:.0f}s", flush=True)


class DetData:
    """In-memory vessel-detection windows with train-split per-band z-score (valid pixels only) and rslearn-style 128 px patches."""

    def __init__(self, kind: str):
        self.kind = kind; sp = DET_SPEC[kind]
        d = PARTNER_DIR / f"{sp['tar']}_cache"
        self.w = {}
        for k in ("train", "val", "test"):
            L = []
            for f in sorted((d / k).glob("*.npz")):
                z = np.load(f)
                L.append({"img": z["img"], "mask": z["mask"], "ctr": z["ctr"], "scale": int(z["scale"])})
            self.w[k] = L
        C = self.w["train"][0]["img"].shape[0]
        acc = np.zeros((C, 2), np.float64); n = 0
        for wd in self.w["train"][:: max(1, len(self.w["train"]) // 400)]:
            v = self._valid_native(wd)
            if v.sum() == 0:
                continue
            x = wd["img"].astype(np.float64)[:, v]
            acc[:, 0] += x.sum(1); acc[:, 1] += (x ** 2).sum(1); n += x.shape[1]
        self.mu = (acc[:, 0] / n).astype(np.float32)
        self.sd = np.sqrt(np.maximum(acc[:, 1] / n - self.mu.astype(np.float64) ** 2, 1e-12)).astype(np.float32)

    @staticmethod
    def _valid_native(wd):
        m = wd["mask"]
        if wd["scale"] == 2:
            m = m[::2, ::2][: wd["img"].shape[1], : wd["img"].shape[2]]
        return m[: wd["img"].shape[1], : wd["img"].shape[2]] > 0

    def crop(self, wd, r0, c0, P=DET_PATCH):
        """Normalised (C,P,P) float32 crop at window-grid offset (r0,c0), zeros outside the window / on invalid pixels; boxes inside."""
        H, W = wd["mask"].shape
        rr = np.arange(r0, r0 + P); cc = np.arange(c0, c0 + P)
        okr, okc = (rr >= 0) & (rr < H), (cc >= 0) & (cc < W)
        rs, cs = np.clip(rr, 0, H - 1), np.clip(cc, 0, W - 1)
        sc = wd["scale"]
        ir, ic = np.minimum(rs // sc, wd["img"].shape[1] - 1), np.minimum(cs // sc, wd["img"].shape[2] - 1)
        x = wd["img"][:, ir][:, :, ic].astype(np.float32)
        x = (x - self.mu[:, None, None]) / self.sd[:, None, None]
        valid = wd["mask"][rs][:, cs] > 0
        valid &= okr[:, None] & okc[None, :]
        x[:, ~valid] = 0.0
        ctr = wd["ctr"]
        boxes = np.zeros((0, 4), np.float32)
        if len(ctr):
            cx, cy = np.floor(ctr[:, 0]).astype(int) - c0, np.floor(ctr[:, 1]).astype(int) - r0   # int(centroid) as in DetectionTask
            keep = (cx > 0) & (cx < P) & (cy > 0) & (cy < P)                                       # exclude_by_center
            b = np.stack([cx - DET_BOX, cy - DET_BOX, cx + DET_BOX, cy + DET_BOX], 1)[keep].astype(np.float32)
            boxes = np.clip(b, 0, P)
            boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        return x, boxes

    def tiles(self, split):
        """load_all_patches: non-overlapping 128 px grid over each window (edge tiles zero-padded)."""
        out = []
        for i, wd in enumerate(self.w[split]):
            H, W = wd["mask"].shape
            for r0 in range(0, H, DET_PATCH):
                for c0 in range(0, W, DET_PATCH):
                    out.append((i, r0, c0))
        return out[:48] if SMOKE else out


class ClayDetector(nn.Module):
    """E7 Clay encoder -> last-layer 16x16 token map -> torchvision Faster R-CNN (single level, stride 8, anchors 32, as rslearn FasterRCNN)."""

    def __init__(self, E7, waves, gsd, backbone=None):
        super().__init__()
        import collections
        import torchvision.models.detection as det
        from torchvision.models.detection.image_list import ImageList
        from torchvision.ops import MultiScaleRoIAlign
        self.ImageList, self.OD = ImageList, collections.OrderedDict
        if backbone is None:
            self.bb = E7.ClayBackbone(trainable=True)
            self.bb.waves = torch.tensor(waves, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
            self.bb.gsd = torch.tensor(float(gsd))
        else:  # another encoder (run_dofa_missing_cells.py): module with forward(x, z, z, taps) -> [last-layer (B,1024,16,16) token map]
            self.bb = backbone
        Cc = 1024
        ag = det.anchor_utils.AnchorGenerator(((32,),), ((0.5, 1.0, 2.0),))
        self.rpn = det.rpn.RegionProposalNetwork(ag, det.rpn.RPNHead(Cc, 3), 0.7, 0.3, 256, 0.5, dict(training=2000, testing=2000),
                                                 dict(training=2000, testing=2000), 0.7)
        pool = MultiScaleRoIAlign(["feat0"], 7, 2)
        self.roi = det.roi_heads.RoIHeads(pool, det.faster_rcnn.TwoMLPHead(Cc * 49, 1024), det.faster_rcnn.FastRCNNPredictor(1024, 2),
                                          0.5, 0.5, 512, 0.25, None, 0.05, 0.5, 100)

    def forward(self, x, targets=None, enc_grad=True):
        B = x.shape[0]
        z = torch.zeros(B, 4, device=x.device)
        with torch.set_grad_enabled(enc_grad and torch.is_grad_enabled()), torch.autocast("cuda", dtype=torch.bfloat16):
            f = self.bb(x, z, z, taps=(24,))[0]
        feats = self.OD([("feat0", f.float())])
        il = self.ImageList(x, [tuple(x.shape[-2:])] * B)
        props, l1 = self.rpn(il, feats, targets)
        dets, l2 = self.roi(feats, props, il.image_sizes, targets)
        return (dets, {**l1, **l2})


def det_f1(preds, gts):
    """rslearn F1Metric (flatten_classes, centre-distance 15 px, Hungarian matching): best F1 over the score thresholds, plus per-threshold."""
    from scipy.optimize import linear_sum_assignment
    tp = np.zeros(len(DET_THRS), np.int64); fp = tp.copy(); fn = tp.copy()
    for p, g in zip(preds, gts):
        gc = np.stack([(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2], 1) if len(g) else np.zeros((0, 2))
        pb, ps = p["boxes"].cpu().numpy(), p["scores"].cpu().numpy()
        pcs = np.stack([(pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2], 1) if len(pb) else np.zeros((0, 2))
        for ti, th in enumerate(DET_THRS):
            sel = ps >= th; pc = pcs[sel]
            if len(gc) and len(pc):
                cmp = np.sqrt(((gc[:, None] - pc[None]) ** 2).sum(-1)) <= DET_BOX
                r, c = linear_sum_assignment(1 - cmp.astype(np.float32)); t = int(cmp[r, c].sum())
            else:
                t = 0
            tp[ti] += t; fp[ti] += len(pc) - t; fn[ti] += len(gc) - t
    f1 = []
    for ti in range(len(DET_THRS)):
        pr = tp[ti] / (tp[ti] + fp[ti]) if tp[ti] + fp[ti] else 0.0; rc = tp[ti] / (tp[ti] + fn[ti]) if tp[ti] + fn[ti] else 0.0
        f1.append(0.0 if pr + rc < 1e-3 else 2 * pr * rc / (pr + rc))
    return float(max(f1)), f1, dict(tp=tp.tolist(), fp=fp.tolist(), fn=fn.tolist())


@torch.no_grad()
def det_evaluate(model, data, split, bs=8):  # bs 32 needed > 9.8 GiB (RoI heads on 2000 proposals per image) and ran out of memory
    model.eval(); tiles = data.tiles(split); preds, gts = [], []
    for i in range(0, len(tiles), bs):
        xs, bxs = [], []
        for (wi, r0, c0) in tiles[i:i + bs]:
            x, b = data.crop(data.w[split][wi], r0, c0); xs.append(x); bxs.append(b)
        out, _ = model(torch.from_numpy(np.stack(xs)).cuda())
        preds += [{k: v.cpu() for k, v in o.items()} for o in out]; gts += bxs  # keep predictions on the CPU (GPU memory grew with the number of eval tiles)
    model.train()
    return det_f1(preds, gts) + (len(tiles), int(sum(len(g) for g in gts)))


def ft_det_cell(kind, seeds, steps, out, bs=8, eval_every=1000, freeze_frac=0.2):
    if SMOKE:
        steps, eval_every = 30, 15
    """Fine-tune Clay + Faster R-CNN on a partner vessel-detection task (rslearn dataset, OlmoEarth eval split)."""
    E7 = load_e7(); sp = DET_SPEC[kind]
    data = DetData(kind)
    ns = {k: len(v) for k, v in data.w.items()}
    runs = []
    prev = union_seeds(out, f"{kind}/FT{TAG}", [])
    for seed in seeds:
        if seed in {r["seed"] for r in prev}:
            print(f"[{kind}] seed {seed} already stored, skipping", flush=True); continue
        if (runs or prev) and (runs or prev)[-1]["train_s"] > SEED_BUDGET_S:
            print(f"[{kind}] seed took {(runs or prev)[-1]['train_s']} s > {SEED_BUDGET_S} s: stopping", flush=True); break
        torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        model = ClayDetector(E7, sp["waves"], sp["gsd"], backbone=DET_BACKBONE_FACTORY(E7, kind) if DET_BACKBONE_FACTORY else None).cuda(); model.train()
        enc = [p for k, p in model.named_parameters() if k.startswith("bb.") and p.requires_grad]
        rest = [p for k, p in model.named_parameters() if not k.startswith("bb.")]
        opt = torch.optim.AdamW([{"params": rest, "lr": 1e-4}, {"params": enc, "lr": 1e-5}], betas=(0.9, 0.999), weight_decay=0.05)
        warm = min(200, steps // 10); sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lr(steps, warm))
        n_freeze = int(freeze_frac * steps)       # OlmoEarth recipe: encoder frozen for the first 20% of training
        best = (-1.0, 0, None); curve = []; t0 = time.time(); el = []
        for step in range(1, steps + 1):
            xs, bxs = [], []
            for _ in range(bs):
                wd = data.w["train"][int(rng.integers(len(data.w["train"])))]
                H, W = wd["mask"].shape
                r0, c0 = int(rng.integers(0, max(1, H - DET_PATCH + 1))), int(rng.integers(0, max(1, W - DET_PATCH + 1)))
                x, b = data.crop(wd, r0, c0)
                if rng.random() < 0.5: x = x[:, :, ::-1]; b = b.copy(); b[:, [0, 2]] = DET_PATCH - b[:, [2, 0]]
                if rng.random() < 0.5: x = x[:, ::-1]; b = b.copy(); b[:, [1, 3]] = DET_PATCH - b[:, [3, 1]]
                xs.append(np.ascontiguousarray(x)); bxs.append(b)
            x = torch.from_numpy(np.stack(xs)).cuda()
            tg = [{"boxes": torch.from_numpy(b).cuda(), "labels": torch.ones(len(b), dtype=torch.int64, device="cuda")} for b in bxs]
            _, losses = model(x, tg, enc_grad=step > n_freeze)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); el.append(loss.item())
            if step % 100 == 0:
                print(f"[{kind} s{seed}] step {step}/{steps} loss {np.mean(el[-100:]):.4f} t {time.time()-t0:.0f}s", flush=True)
            if step % eval_every == 0 or step == steps:
                f1v, _, _, nt, ng = det_evaluate(model, data, "val")
                curve.append([step, float(np.mean(el[-eval_every:])), round(f1v * 100, 2)])
                if f1v > best[0]:
                    best = (f1v, step, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()})
                print(f"[{kind} s{seed}] step {step} val F1 {f1v*100:.2f} (tiles {nt}, vessels {ng}) best_step {best[1]} t {time.time()-t0:.0f}s", flush=True)
        model.load_state_dict(best[2])
        f1t, f1_by_thr, cnt, nt, ng = det_evaluate(model, data, "test")
        vthr = int(np.argmax(det_evaluate(model, data, "val")[1]))                 # score threshold chosen on val, applied to test
        runs.append({"seed": seed, "steps": steps, "batch_size": bs, "best_val_f1": round(best[0] * 100, 2), "best_step": best[1], "last_step": steps,
                     "test_f1_best_threshold": round(f1t * 100, 2), "test_f1_by_threshold": {str(t): round(v * 100, 2) for t, v in zip(DET_THRS, f1_by_thr)},
                     "test_f1_at_val_selected_threshold": round(f1_by_thr[vthr] * 100, 2), "val_selected_threshold": DET_THRS[vthr],
                     "test_counts_at_best_threshold": {k_: v[int(np.argmax(f1_by_thr))] for k_, v in cnt.items()}, "test_tiles": nt, "test_vessels": ng,
                     "train_s": round(time.time() - t0), "curve": curve})
        del model, opt; gc.collect(); torch.cuda.empty_cache()
        runs_all = union_seeds(out, f"{kind}/FT{TAG}", runs); vals = [r["test_f1_best_threshold"] for r in runs_all]
        rec = {"table2_row": sp["row"], "cell": "FT", "status": "rerun_here", "partial_seeds_done": len(runs_all) < 3,
               "metric": "test F1 (%), centre distance <= 15 px, best over score thresholds {0.05..0.95} (rslearn F1Metric, flatten_classes)",
               "value": round(float(np.mean(vals)), 2), "value_std_over_seeds": round(float(np.std(vals)), 2) if len(vals) > 1 else None,
               "head": "torchvision Faster R-CNN (RPN + 2-MLP box head, one level at stride 8, anchors 32 x {0.5,1,2}) on the Clay last-layer 16x16 token map",
               "recipe": {"optimizer": "AdamW wd 0.05", "lr_head": 1e-4, "lr_encoder": 1e-5, "schedule": "warm-up + cosine", "steps": steps, "batch_size": bs,
                          "encoder_frozen_first_fraction": freeze_frac, "patch": DET_PATCH, "box_half_size": DET_BOX, "precision": "bf16 encoder, fp32 detection head, gradient checkpointing",
                          "selection": f"best-validation F1 checkpoint (every {eval_every} steps); test scored once",
                          "relation_to_olmoearth": "same data, split tag, 128 px patches, 15 px boxes and F1 as data/olmoearth_evals/tasks/*_vessels.yaml; budget is reduced from 100 epochs x 32768 samples"},
               "seeds": [r["seed"] for r in runs_all], "per_seed": runs_all, "n_windows": ns, "bands": sp["bands"], "waves_um": sp["waves"], "gsd_m": sp["gsd"],
               "old_transcribed": {"value": OLD[kind]["FT"], "source": OLD_SOURCE},
               "data": f"{PARTNER_URL}/{sp['tar']}.tar (CC BY 4.0, Allen Institute for AI; olmoearth_evals_split tags train/val/test)", "provenance": prov()}
        rec["model"] = DET_MODEL
        if DET_MODEL != "clay_v1.5":
            rec.pop("old_transcribed", None)
            rec["head"] = rec["head"].replace("Clay last-layer", DET_MODEL + " last-layer")
            rec["recipe"]["relation_to_clay_cell"] = "identical data, head, steps, batch, lr, schedule and metric as the Clay cell; only the backbone (and its own input contract) changes"
        merge_write(out, f"{kind}/FT{TAG}", rec)


# ---- OlmoEarth partner task: Solar Farm segmentation (public rslearn dataset, CC BY 4.0), S2 x 4 dates --------------------------
SOLAR_CACHE = PARTNER_DIR / "solar_farm_cache"
SOLAR_SETS = [("B02_B03_B04_B08", ["B02", "B03", "B04", "B08"]), ("B05_B06_B07_B8A_B11_B12", ["B05", "B06", "B07", "B8A", "B11", "B12"])]
SOLAR_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]   # the 10 Clay S2 bands
SOLAR_LAYERS = ["sentinel2", "sentinel2.1", "sentinel2.2", "sentinel2.3"]


def prep_solar(stream=None):
    """Read the public solar_farm.tar as a stream (stdin: `rclone cat ... | python ... --cells prep_solar`), keep only the 4 Sentinel-2
    dates (10 Clay bands, 20 m bands nearest-upsampled to the 10 m window grid), the validity mask and the label raster; one .npy per window."""
    from rasterio.io import MemoryFile
    from PIL import Image
    for k in ("train", "val", "test"):
        (SOLAR_CACHE / k).mkdir(parents=True, exist_ok=True)
    want = {"metadata.json", "layers/mask/mask/image.png", "layers/label_raster/label/image.png"}
    for ly in SOLAR_LAYERS:
        for sd_, _ in SOLAR_SETS:
            want.add(f"layers/{ly}/{sd_}/geotiff.tif")
    counts = {"train": 0, "val": 0, "test": 0}; skipped = 0; pos = {"train": 0, "val": 0, "test": 0}

    def flush(key, files):
        nonlocal skipped
        if key is None:
            return
        if not want.issubset(files):
            skipped += 1; return
        m = json.loads(files["metadata.json"]); split = m["options"].get("olmoearth_evals_split")
        if split not in counts:
            return
        H, W = m["bounds"][3] - m["bounds"][1], m["bounds"][2] - m["bounds"][0]
        dates = []
        for ly in SOLAR_LAYERS:
            bands = {}
            for sd_, names in SOLAR_SETS:
                with MemoryFile(files[f"layers/{ly}/{sd_}/geotiff.tif"]) as mf, mf.open() as r:
                    a = r.read()
                if a.shape[1:] != (H, W):   # 20 m band set -> 10 m grid: 2x nearest, edge-pad / crop to the window size
                    a = np.repeat(np.repeat(a, 2, 1), 2, 2)
                    a = np.pad(a, ((0, 0), (0, max(0, H - a.shape[1])), (0, max(0, W - a.shape[2]))), mode="edge")[:, :H, :W]
                for nm, ch in zip(names, a):
                    bands[nm] = ch
            dates.append(np.stack([bands[b] for b in SOLAR_ORDER]).astype(np.uint16))
        mask = (np.asarray(Image.open(io.BytesIO(files["layers/mask/mask/image.png"]))) > 0).astype(np.uint8)
        lab = (np.asarray(Image.open(io.BytesIO(files["layers/label_raster/label/image.png"]))) > 0).astype(np.uint8)
        assert mask.shape == (H, W) and lab.shape == (H, W), (mask.shape, lab.shape, H, W)
        i = counts[split]
        np.save(SOLAR_CACHE / split / f"{i:05d}_img.npy", np.stack(dates))
        np.save(SOLAR_CACHE / split / f"{i:05d}_aux.npy", np.stack([mask, lab]))
        counts[split] += 1; pos[split] += int(lab.sum())

    t0 = t_print = time.time(); cur = None; files = {}
    src = stream if stream is not None else sys.stdin.buffer
    with tarfile.open(fileobj=src, mode="r|") as tf:
        for mem in tf:
            if not mem.isfile():
                continue
            parts = mem.name.split("/")
            if parts[0] != "windows" or len(parts) < 4:
                continue
            key = (parts[1], parts[2])
            if key != cur:
                flush(cur, files); cur = key; files = {}
            rel = "/".join(parts[3:])
            if rel in want:
                files[rel] = tf.extractfile(mem).read()
            if time.time() - t_print > 120:
                t_print = time.time(); print(f"[prep solar] {counts} skipped {skipped} {time.time()-t0:.0f}s", flush=True)
        flush(cur, files)
    (SOLAR_CACHE / "summary.json").write_text(json.dumps({"windows": counts, "skipped_incomplete": skipped, "positive_pixels": pos}))
    print(f"[prep solar] done {counts} skipped {skipped} {time.time()-t0:.0f}s", flush=True)


class SolarData:
    """Memory-mapped solar-farm windows; per-band z-score from the train split (valid pixels, 4 dates); 128 px patches."""

    def __init__(self):
        self.w = {}
        for k in ("train", "val", "test"):
            fs = sorted((SOLAR_CACHE / k).glob("*_img.npy"))
            self.w[k] = [(np.load(f, mmap_mode="r"), np.load(str(f).replace("_img", "_aux"), mmap_mode="r")) for f in fs]
        acc = np.zeros((10, 2), np.float64); n = 0
        for img, aux in self.w["train"][:: max(1, len(self.w["train"]) // 300)]:
            v = aux[0] > 0
            x = np.asarray(img[0]).astype(np.float64)[:, v]
            acc[:, 0] += x.sum(1); acc[:, 1] += (x ** 2).sum(1); n += x.shape[1]
        self.mu = (acc[:, 0] / n).astype(np.float32)
        self.sd = np.sqrt(np.maximum(acc[:, 1] / n - self.mu.astype(np.float64) ** 2, 1e-12)).astype(np.float32)

    def crop(self, wd, r0, c0, P=DET_PATCH):
        img, aux = wd
        T, C, H, W = img.shape
        x = np.zeros((T, C, P, P), np.float32); y = np.full((P, P), -1, np.int16)
        r1, c1 = min(r0 + P, H), min(c0 + P, W)
        h, w = r1 - r0, c1 - c0
        blk = np.asarray(img[:, :, r0:r1, c0:c1]).astype(np.float32)
        blk = (blk - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        blk[:, :, aux[0, r0:r1, c0:c1] == 0] = 0.0
        x[:, :, :h, :w] = blk
        y[:h, :w] = aux[1, r0:r1, c0:c1]
        return x, y

    def tiles(self, split):
        out = []
        for i, (img, aux) in enumerate(self.w[split]):
            H, W = img.shape[2:]
            for r0 in range(0, H, DET_PATCH):
                for c0 in range(0, W, DET_PATCH):
                    out.append((i, r0, c0))
        return out[:48] if SMOKE else out


@torch.no_grad()
def solar_evaluate(model, data, split, bs=16):
    model.eval(); conf = np.zeros((2, 2), np.int64); tiles = data.tiles(split)
    for i in range(0, len(tiles), bs):
        xs, ys = zip(*[data.crop(data.w[split][wi], r0, c0) for (wi, r0, c0) in tiles[i:i + bs]])
        p = model(torch.from_numpy(np.stack(xs)).cuda()).argmax(1).cpu().numpy()
        conf_update(conf, p.reshape(-1), np.stack(ys).reshape(-1), 2)
    model.train()
    return conf


def ft_solar_cell(seeds, steps, out, bs=8, eval_every=1000):
    """Fine-tune Clay (each of the 4 S2 dates encoded separately, tapped token maps averaged over dates) + E7 UPerNet on the Solar Farm task."""
    if SMOKE:
        steps, eval_every = 20, 10
    E7 = load_e7(); data = SolarData(); ns = {k: len(v) for k, v in data.w.items()}
    runs = []
    prev = union_seeds(out, f"solar-farm/FT{TAG}", [])
    for seed in seeds:
        if seed in {r["seed"] for r in prev}:
            print(f"[solar] seed {seed} already stored, skipping", flush=True); continue
        if (runs or prev) and (runs or prev)[-1]["train_s"] > SEED_BUDGET_S:
            print(f"[solar] seed took {(runs or prev)[-1]['train_s']} s > {SEED_BUDGET_S} s: stopping", flush=True); break
        torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        model = PastisFT(E7, 2, rsb.S2_10B_WAVES, 10.0).cuda(); model.train()
        enc = [p for k, p in model.named_parameters() if k.startswith("bb.") and p.requires_grad]
        rest = [p for k, p in model.named_parameters() if not k.startswith("bb.")]
        opt = torch.optim.AdamW([{"params": rest, "lr": 1e-4}, {"params": enc, "lr": 1e-5}], betas=(0.9, 0.999), weight_decay=0.05)
        warm = min(200, steps // 10); sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lr(steps, warm))
        best = (-1.0, 0, None); curve = []; t0 = time.time(); el = []
        for step in range(1, steps + 1):
            xs, ys = [], []
            for _ in range(bs):
                img, aux = data.w["train"][int(rng.integers(len(data.w["train"])))]
                H, W = img.shape[2:]
                x, y = data.crop((img, aux), int(rng.integers(0, max(1, H - DET_PATCH + 1))), int(rng.integers(0, max(1, W - DET_PATCH + 1))))
                if rng.random() < 0.5: x, y = x[..., ::-1], y[:, ::-1]
                if rng.random() < 0.5: x, y = x[..., ::-1, :], y[::-1]
                k = int(rng.integers(0, 4)); x, y = np.rot90(x, k, (-2, -1)), np.rot90(y, k, (0, 1))
                xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
            loss = F.cross_entropy(model(torch.from_numpy(np.stack(xs)).cuda()), torch.from_numpy(np.stack(ys)).long().cuda(), ignore_index=-1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); el.append(loss.item())
            if step % 100 == 0:
                print(f"[solar s{seed}] step {step}/{steps} loss {np.mean(el[-100:]):.4f} t {time.time()-t0:.0f}s", flush=True)
            if step % eval_every == 0 or step == steps:
                m = float(np.mean(miou_from_conf(solar_evaluate(model, data, "val")))) * 100
                curve.append([step, float(np.mean(el[-eval_every:])), round(m, 2)])
                if m > best[0]:
                    best = (m, step, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()})
                print(f"[solar s{seed}] step {step} val mIoU {m:.2f} best_step {best[1]} t {time.time()-t0:.0f}s", flush=True)
        model.load_state_dict(best[2])
        conf = solar_evaluate(model, data, "test"); ious = miou_from_conf(conf)
        runs.append({"seed": seed, "steps": steps, "batch_size": bs, "best_val_miou": round(best[0], 2), "best_step": best[1], "last_step": steps,
                     "test_miou": round(float(np.mean(ious)) * 100, 2), "test_iou_bg_solar": [round(i * 100, 2) for i in ious],
                     "test_pixel_accuracy": round(float(np.diag(conf).sum() / conf.sum()) * 100, 2), "train_s": round(time.time() - t0), "curve": curve})
        del model, opt; gc.collect(); torch.cuda.empty_cache()
        runs_all = union_seeds(out, f"solar-farm/FT{TAG}", runs); vals = [r["test_miou"] for r in runs_all]
        rec = {"table2_row": "Solar Farm Detect.", "cell": "FT", "status": "rerun_here", "partial_seeds_done": len(runs_all) < 3,
               "metric": "test mIoU (%) over {background, solar farm}, 128 px non-overlapping patches", "value": round(float(np.mean(vals)), 2),
               "value_std_over_seeds": round(float(np.std(vals)), 2) if len(vals) > 1 else None,
               "head": "E7 UPerNet neck+head on taps (6,12,18,24); each of the 4 S2 dates encoded separately by Clay, tapped token maps averaged over dates",
               "recipe": {"optimizer": "AdamW wd 0.05", "lr_head": 1e-4, "lr_encoder": 1e-5, "schedule": "warm-up + cosine", "steps": steps, "batch_size": bs, "patch": DET_PATCH,
                          "precision": "bf16, gradient checkpointing", "selection": f"best-validation mIoU checkpoint (every {eval_every} steps); test scored once",
                          "relation_to_olmoearth": "same data, split tag, 4 S2 timesteps, 128 px patches as data/olmoearth_evals/tasks/solar_farm_{base,ts}.yaml; budget reduced from 100 epochs x 16384 samples"},
               "seeds": [r["seed"] for r in runs_all], "per_seed": runs_all, "n_windows": ns, "bands": SOLAR_ORDER, "waves_um": rsb.S2_10B_WAVES,
               "old_transcribed": {"value": OLD["solar-farm"]["FT"], "source": OLD_SOURCE},
               "data": f"{PARTNER_URL}/solar_farm.tar (CC BY 4.0, Allen Institute for AI)", "provenance": prov()}
        merge_write(out, f"solar-farm/FT{TAG}", rec)


# ---- OlmoEarth adapter contract diagnostic ---------------------------------------------------------
OE_TASKS = ["m-eurosat", "m-brick-kiln", "m-so2sat", "m-forestnet"]


def oe_contract(suite, out):
    """Measure what the OlmoEarth Clay adapter contract (olmoearth_pretrain evals/models/clay/clay.py: wavelengths x1000,
    chip resized to 128 px, metadata-yaml normalisation, GSD fixed to 10 m) does to Clay on tasks we can run both ways."""
    res = {}
    cid = "diagnostic/olmoearth_adapter_contract"
    if out.exists():  # resume: keep tasks finished by an earlier (interrupted) run
        res = json.loads(out.read_text())["cells"].get(cid, {}).get("details", {})
    for task in OE_TASKS:
        if task in res:
            continue
        cfg = gb_cfg(task)
        (Xtr, ytr), (Xva, yva), (Xte, yte), mu, sd, waves, meta = load_gb(cfg)
        H, hw = Xtr.shape[-1], cfg["clay_hw"]
        arms = {"repo_protocol": (mu, sd, waves, cfg["gsd"], hw),
                "waves_x1000_only": (mu, sd, [w * 1000 for w in waves], cfg["gsd"], hw)}
        if cfg.get("clay_meta"):
            plat, mult = cfg["clay_meta"]
            mi = rsb.CLAY_METADATA[plat]
            order = cfg.get("clay_meta_bands") or list(mi["band_order"])
            m_ = np.array([float(mi["bands"]["mean"][b]) / mult for b in order], np.float32)
            s_ = np.array([float(mi["bands"]["std"][b]) / mult for b in order], np.float32)
            w_ = [float(mi["bands"]["wavelength"][b]) for b in order]
            arms["metadata_norm_um_native"] = (m_, s_, w_, cfg["gsd"], hw)
            arms["olmoearth_contract"] = (m_, s_, [w * 1000 for w in w_], cfg["gsd"] * 128 / H, 128)  # gsd_eff fixed = md gsd
        res[task] = {"splits": [len(ytr), len(yva), len(yte)], "arms": {}}
        for name, (m_, s_, w_, g_, h_) in arms.items():
            f = {k: clay_feats(suite, X, m_, s_, w_, g_, h_) for k, X in (("tr", Xtr), ("va", Xva), ("te", Xte))}
            nc = int(max(ytr.max(), yva.max(), yte.max())) + 1
            lp = {ro: rsb.eval_single_label(f["tr"][ro], ytr, f["va"][ro], yva, f["te"][ro], yte) for ro in ("cls", "mean")}
            kn = {ro: eval_knn(f["tr"][ro], ytr, f["te"][ro], yte, n_classes=nc) for ro in ("cls", "mean")}
            res[task]["arms"][name] = {"lp_oa_by_readout": {k: v["oa"] for k, v in lp.items()},
                                       "lp_oa_val_selected": lp[max(lp, key=lambda r: lp[r]["val_oa"])]["oa"],
                                       "knn20_oa_by_readout": {k: v["oa"] for k, v in kn.items()}}
            print(task, name, res[task]["arms"][name], flush=True)
        _oe_write(out, res)
    _oe_write(out, res)


def _oe_write(out, res):
    rec = {"table2_row": "(diagnostic) OlmoEarth Clay adapter contract", "cell": "diagnostic", "status": "rerun_here",
           "metric": "test OA (%)", "details": res,
           "adapter_source": {"repo": "allenai/olmoearth_pretrain", "file": "olmoearth_pretrain/evals/models/clay/clay.py",
                              "revision_pinned_by_issue_17": "13ca06ffaf0247bd49b8a4fccf705a3a85fe9d6d (2026-10-26)",
                              "wavelengths_x1000": "L220-L224 (metadata wavelength [um] * 1000, comment 'Convert to nm')",
                              "resize_128": "L53 image_resolution=128 and L158-L167 (F.interpolate to 128x128)",
                              "metadata_normaliser": "L134-L156", "fixed_gsd": "L232",
                              "wavelength_units_expected_by_clay": "um: claymodel/model.py L463 feeds metadata wavelengths (um) unchanged",
                              "band_subsets": "L20-L46, L146-L152: S2 -> 10 Clay bands, Landsat -> 6 Clay bands, others dropped"},
           "tasks_done": sorted(res), "provenance": prov()}
    merge_write(out, "diagnostic/olmoearth_adapter_contract", rec)


# --------------------------------------------------------------------------------------------- fine-tuning
def load_e7():
    """Import the E7 fine-tune module (Neck / UPerHead / ClayBackbone) without running its main()."""
    p = HERE / "experiments/suite/E7_finetune/code/train_e7.py"
    src = p.read_text().rsplit("\nmain()", 1)[0]
    m = types.ModuleType("train_e7")
    m.__file__ = str(p)
    old = sys.argv
    sys.argv = [str(p), "--arm", "clay_finetune_upernet"]
    try:
        exec(compile(src, str(p), "exec"), m.__dict__)
    finally:
        sys.argv = old
    return m


def cosine_lr(opt_steps, warm):
    return lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(opt_steps - warm, 1)))


class FTModel(nn.Module):
    """Clay encoder (trainable, E7 ClayBackbone) + task head. cls: mean-pooled last-layer tokens -> Linear; seg: E7 UPerNet."""

    def __init__(self, E7, kind, n_out, waves, gsd):
        super().__init__()
        self.kind = kind
        self.bb = E7.ClayBackbone(trainable=True)
        self.bb.waves = torch.tensor(waves, dtype=torch.float32, device="cuda")
        self.bb.gsd = torch.tensor(float(gsd))
        if kind == "seg":
            self.neck, self.head = E7.Neck(1024, 8), E7.UPerHead(ncls=n_out)
        else:
            self.head = nn.Linear(1024, n_out)
        self.taps = E7.TAPS

    def forward(self, x):
        B = x.shape[0]
        z = torch.zeros(B, 4, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "seg":
                f = self.bb(x, z, z, taps=self.taps)
                o = self.head(self.neck(f))
                return F.interpolate(o.float(), x.shape[-2:], mode="bilinear", align_corners=False)
            f = self.bb(x, z, z, taps=(24,))[0].float().mean((2, 3))
        return self.head(f)


def aug_batch(x, y, gen, seg):
    if torch.rand(1, generator=gen) < 0.5:
        x = x.flip(-1); y = y.flip(-1) if seg else y
    if torch.rand(1, generator=gen) < 0.5:
        x = x.flip(-2); y = y.flip(-2) if seg else y
    k = int(torch.randint(0, 4, (1,), generator=gen))
    x = torch.rot90(x, k, (-2, -1)); y = torch.rot90(y, k, (-2, -1)) if seg else y
    return x, y


def prep_x(xb, mu, sd, hw):
    """normalise (train-split z-score) and resize to the Clay chip size; gsd compensation is done by the caller."""
    x = (xb.float().cuda() - mu) / sd
    if x.shape[-1] != hw:
        x = F.interpolate(x, size=(hw, hw), mode="bilinear", align_corners=False)
    return x


def conf_update(conf, pred, y, n):
    v = (y >= 0) & (y < n)
    conf += np.bincount((y[v].astype(np.int64) * n + pred[v].astype(np.int64)), minlength=n * n).reshape(n, n)


def miou_from_conf(conf):
    inter = np.diag(conf).astype(float); union = conf.sum(0) + conf.sum(1) - inter
    return [float(inter[k] / union[k]) for k in range(len(conf)) if union[k] > 0]


@torch.no_grad()
def predict_cls(model, X, mu, sd, hw, bs):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(model(prep_x(torch.from_numpy(X[i:i + bs]), mu, sd, hw)).float().cpu())
    model.train()
    return torch.cat(out)


def ft_classification(E7, task, seed, epochs, bs, out_extra):
    cfg = gb_cfg(task)
    (Xtr, ytr), (Xva, yva), (Xte, yte), mu_n, sd_n, waves, meta = load_gb(cfg)
    multilabel = cfg["kind"] == "multilabel"
    H = Xtr.shape[-1]; hw = H if (H % 8 == 0 and 32 <= H <= 256) else 256
    gsd_eff = cfg["gsd"] * H / hw
    n_out = ytr.shape[1] if multilabel else int(max(ytr.max(), yva.max(), yte.max())) + 1
    mu = torch.tensor(mu_n, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    sd = torch.tensor(sd_n, dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
    torch.manual_seed(seed); np.random.seed(seed)
    model = FTModel(E7, "cls", n_out, waves, gsd_eff).cuda(); model.train()
    enc = [p for n, p in model.named_parameters() if n.startswith("bb.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("bb.")]
    lr_head, lr_enc = (LR, LR) if LR else (1e-3, 1e-5)
    opt = torch.optim.AdamW([{"params": head, "lr": lr_head}, {"params": enc, "lr": lr_enc}], betas=(0.9, 0.999), weight_decay=0.05)
    spe = math.ceil(len(ytr) / bs); total = spe * epochs; warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lr(total, warm))
    gen = torch.Generator().manual_seed(seed)
    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
    crit = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()

    def score(logits, y):
        if multilabel:
            p = (torch.sigmoid(logits) > 0.5).numpy().astype(int); yi = y.astype(int)
            return float(f1_score(yi, p, average="micro", zero_division=0)) * 100
        return float(accuracy_score(y, logits.argmax(1).numpy())) * 100

    best = (-1.0, 0, None); curve = []; t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(ytr), generator=gen); el = []
        for s in range(spe):
            idx = perm[s * bs:(s + 1) * bs]
            x = prep_x(Xtr_t[idx], mu, sd, hw)
            y = ytr_t[idx].cuda()
            x, _ = aug_batch(x, y, gen, False)
            logits = model(x)
            loss = crit(logits, y.float()) if multilabel else crit(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); el.append(loss.item())
        va = score(predict_cls(model, Xva, mu, sd, hw, 64), yva)
        curve.append([ep + 1, float(np.mean(el)), va])
        if va > best[0]:
            best = (va, ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"[{task} s{seed}] ep {ep+1}/{epochs} loss {np.mean(el):.4f} val {va:.2f} best_ep {best[1]} t {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best[2])
    lt = predict_cls(model, Xte, mu, sd, hw, 64)
    r = {"seed": seed, "lr": LR, "epochs": epochs, "batch_size": bs, "test": round(score(lt, yte), 2), "best_val": round(best[0], 2), "best_epoch": best[1],
         "train_s": round(time.time() - t0), "curve": curve}
    if not multilabel:
        r["test_macro_f1"] = round(float(f1_score(yte, lt.argmax(1).numpy(), average="macro")) * 100, 2)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return r, dict(split_sizes=[len(ytr), len(yva), len(yte)], chip_hw=int(H), clay_hw=hw, gsd_eff=gsd_eff, geobench_check=meta,
                   bands=cfg.get("band_names", cfg["bands"]), n_out=n_out)


def ft_seg(E7, name, seed, epochs, bs, data):
    """data: dict(X_tr,y_tr,X_va,y_va,test sets...) already normalised float16 CPU tensors, waves, gsd, n_cls, crop."""
    torch.manual_seed(seed); np.random.seed(seed)
    n = data["n_cls"]
    model = FTModel(E7, "seg", n, data["waves"], data["gsd"]).cuda(); model.train()
    enc = [p for k, p in model.named_parameters() if k.startswith("bb.") and p.requires_grad]
    rest = [p for k, p in model.named_parameters() if not k.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": rest, "lr": LR or 1e-4}, {"params": enc, "lr": LR or 1e-5}], betas=(0.9, 0.999), weight_decay=0.05)
    spe = math.ceil(len(data["X_tr"]) / bs)
    if len(data["X_tr"]) % bs == 1:
        spe -= 1  # drop a 1-sample last batch (BatchNorm in the UPerNet head needs >= 2 samples per batch; MADOS: 2657 tiles, batch 8)
    total = spe * epochs; warm = spe * min(5, epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lr(total, warm))
    gen = torch.Generator().manual_seed(seed)
    Xtr, ytr, crop = data["X_tr"], data["y_tr"], data["crop"]

    @torch.no_grad()
    def evaluate(X, y):
        model.eval(); conf = np.zeros((n, n), np.int64)
        for i in range(0, len(X), 16):
            for a in range(0, X.shape[-2], crop):
                for b in range(0, X.shape[-1], crop):
                    xb = X[i:i + 16, :, a:a + crop, b:b + crop].float().cuda()
                    p = model(xb).argmax(1).cpu().numpy()
                    conf_update(conf, p.reshape(-1), y[i:i + 16, a:a + crop, b:b + crop].numpy().reshape(-1), n)
        model.train(); return conf

    best = (-1.0, 0, None); curve = []; t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), generator=gen).tolist(); el = []
        for s in range(spe):
            xs, ys = [], []
            if len(perm[s * bs:(s + 1) * bs]) < 2:   # BatchNorm in the UPerNet head needs >1 sample (last batch of an odd-sized set)
                continue
            for c in perm[s * bs:(s + 1) * bs]:
                H = Xtr.shape[-1]
                i, j = (torch.randint(0, H - crop + 1, (2,), generator=gen).tolist() if H > crop else (0, 0))
                x, y = aug_batch(Xtr[c, :, i:i + crop, j:j + crop].float(), ytr[c, i:i + crop, j:j + crop].long(), gen, True)
                xs.append(x); ys.append(y)
            logits = model(torch.stack(xs).cuda())
            loss = F.cross_entropy(logits, torch.stack(ys).cuda(), ignore_index=-1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); el.append(loss.item())
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            m = float(np.mean(miou_from_conf(evaluate(data["X_va"], data["y_va"])))) * 100
            curve.append([ep + 1, float(np.mean(el)), m])
            if m > best[0]:
                best = (m, ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            print(f"[{name} s{seed}] ep {ep+1}/{epochs} loss {np.mean(el):.4f} val mIoU {m:.2f} best_ep {best[1]} t {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best[2])
    r = {"seed": seed, "lr": LR, "epochs": epochs, "best_val_miou": round(best[0], 2), "best_epoch": best[1], "train_s": round(time.time() - t0), "curve": curve}
    for tname in data["tests"]:
        conf = evaluate(*data["tests"][tname]); ious = miou_from_conf(conf)
        r[f"{tname}_miou"] = round(float(np.mean(ious)) * 100, 2)
        if n == 2:
            r[f"{tname}_water_iou"] = round(ious[1] * 100, 2)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return r


def ft_cell(task, seeds, epochs, bs, out):
    E7 = load_e7()
    runs = []; extra = {}
    for sd_ in seeds:
        r, extra = ft_classification(E7, task, sd_, epochs, bs, {})
        runs.append(r)
        _write_ft(out, task, runs, extra, epochs, bs, partial=len(runs) < len(seeds))


def _write_ft(out, task, runs, extra, epochs, bs, partial):
    runs = union_seeds(out, f"{task}/FT{TAG}", runs)
    vals = [r["test"] for r in runs]
    multilabel = task == "m-bigearthnet"
    row = {"m-bigearthnet": "m-bigearthnet", "m-cashew-plant": "m-cashewplant"}.get(task, task)
    rec = {
        "table2_row": row, "cell": "FT", "status": "rerun_here", "partial_seeds_done": len(runs) < 3,
        "metric": "test micro-F1 (%)" if multilabel else "test overall accuracy (%)",
        "value": round(float(np.mean(vals)), 2), "value_std_over_seeds": round(float(np.std(vals)), 2) if len(vals) > 1 else None,
        "readout": "mean of last-layer patch tokens (after final norm), full encoder trainable",
        "head": "Linear on pooled tokens; BCE" if multilabel else "Linear on pooled tokens; CE",
        "recipe": {"optimizer": "AdamW betas (0.9,0.999) wd 0.05", "lr_head": LR or 1e-3, "lr_encoder": LR or 1e-5, "schedule": "10% warm-up + cosine",
                   "epochs": sorted({r["epochs"] for r in runs if "epochs" in r}) or [epochs], "batch_size": bs, "augmentation": "per-batch random flips and rot90",
                   "precision": "bf16 autocast, gradient checkpointing", "selection": "best-validation epoch; test scored once",
                   "relation_to_E7": "E7 optimizer/lr/schedule (E7 head lr 1e-4 for UPerNet; 1e-3 here for the linear head, as E7 clay_frozen_linear); epochs reduced"},
        "seeds": [r["seed"] for r in runs], "per_seed": runs, **extra,
        "old_transcribed": {"value": OLD[task]["FT"], "source": OLD_SOURCE}, "provenance": prov(),
    }
    merge_write(out, f"{task}/FT{TAG}", rec)


def ft_seg_cell(which, seeds, epochs, out):
    E7 = load_e7()
    if which == "m-cashew-plant":
        cfg = gb_cfg(which)
        (Xtr, ytr), (Xva, yva), (Xte, yte), mu, sd, waves, meta = load_gb(cfg)
        n = int(max(ytr.max(), yva.max(), yte.max())) + 1
        nz = lambda X: torch.from_numpy(((X - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float16))
        T = lambda y: torch.from_numpy(y.astype(np.int16))
        data = dict(X_tr=nz(Xtr), y_tr=T(ytr), X_va=nz(Xva), y_va=T(yva), waves=waves, gsd=cfg["gsd"], n_cls=n, crop=256,
                    tests={"test": (nz(Xte), T(yte))})
        extra = dict(split_sizes=[len(ytr), len(yva), len(yte)], n_classes=n, geobench_check=meta)
        key, row, old, metric = "m-cashew-plant/FT", "m-cashewplant", OLD["m-cashew-plant"]["FT"], "test_miou"
    elif which == "mados":
        (Xtr, ytr), (Xva, yva), (Xte, yte), mu, sd = load_mados()
        nz = lambda X: torch.from_numpy(((X - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float16))
        T = lambda y: torch.from_numpy(y.astype(np.int16))
        if SMOKE:
            Xtr, ytr, Xva, yva, Xte, yte = Xtr[:32], ytr[:32], Xva[:16], yva[:16], Xte[:16], yte[:16]
        data = dict(X_tr=nz(Xtr), y_tr=T(ytr), X_va=nz(Xva), y_va=T(yva), waves=rsb.S2_10B_WAVES, gsd=10.0, n_cls=15, crop=80,
                    tests={"test": (nz(Xte), T(yte))})
        extra = dict(split_sizes_tiles=[len(ytr), len(yva), len(yte)], n_classes=15, bands=BANDS_S2_STD, data=MADOS_URL)
        key, row, old, metric = "mados/FT", "MADOS", OLD["mados"]["FT"], "test_miou"
    else:
        D, mu, sd = load_s1f_s1()
        nz = lambda X: torch.from_numpy(((X - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float16))
        T = lambda y: torch.from_numpy(y)
        data = dict(X_tr=nz(D["train"][0]), y_tr=T(D["train"][1]), X_va=nz(D["valid"][0]), y_va=T(D["valid"][1]),
                    waves=rsb.S1_2B_WAVES, gsd=10.0, n_cls=2, crop=256,
                    tests={"test": (nz(D["test"][0]), T(D["test"][1])), "bolivia": (nz(D["bolivia"][0]), T(D["bolivia"][1]))})
        extra = dict(split_sizes={k: len(D[k][2]) for k in D}, bands=["VV_dB", "VH_dB"])
        key, row, old, metric = "sen1floods11-s1/FT", "Sen1Floods11 (S1)", OLD["sen1floods11-s1"]["FT"], "test_miou"
    key += TAG  # tagged runs (--tag/--lr) get their own record, as in ft_cell/ft_det_cell
    runs = []
    prev = union_seeds(out, key, [])
    for s in seeds:
        if s in {r["seed"] for r in prev}:
            print(f"[{which}] seed {s} already in {out.name}, skipping", flush=True); continue
        if (runs or prev) and (runs or prev)[-1]["train_s"] > SEED_BUDGET_S:
            print(f"[{which}] seed took {(runs or prev)[-1]['train_s']} s > {SEED_BUDGET_S} s: stopping", flush=True); break
        runs.append(ft_seg(E7, which, s, epochs, 8, data))
        runs_all = union_seeds(out, key, runs)
        vals = [r[metric] for r in runs_all]
        rec = {"table2_row": row, "cell": "FT", "status": "rerun_here", "partial_seeds_done": len(runs_all) < 3,
               "metric": "test mIoU (%) over classes", "value": round(float(np.mean(vals)), 2),
               "value_std_over_seeds": round(float(np.std(vals)), 2) if len(vals) > 1 else None,
               "head": "E7 UPerNet neck+head on taps (6,12,18,24), full encoder trainable",
               "recipe": {"optimizer": "AdamW wd 0.05", "lr_head": LR or 1e-4, "lr_encoder": LR or 1e-5, "schedule": "5-epoch warm-up + cosine",
                          "epochs": epochs, "batch_size": 8, "crop": data["crop"], "precision": "bf16, gradient checkpointing",
                          "selection": "best-validation mIoU epoch (every 5 epochs); test scored once", "source": "experiments/suite/E7_finetune/code/train_e7.py classes"},
               "seeds": [r["seed"] for r in runs_all], "per_seed": runs_all, **extra,
               "old_transcribed": {"value": old, "source": OLD_SOURCE}, "provenance": prov()}
        merge_write(out, key, rec)


# --------------------------------------------------------------------------------------------- finalize
def finalize(out):
    """Add a mechanical budget check to every fine-tune record (no new numbers): is the best-validation epoch the last
    epoch (still improving when the budget ended) and is the fine-tune below the frozen probe of the same task."""
    data = json.loads(out.read_text())["cells"]
    lp_of = {"m-bigearthnet/FT": "m-bigearthnet/LP", "m-cashew-plant/FT": "m-cashew-plant/LP", "sen1floods11-s1/FT": "sen1floods11-s1/LP"}
    for cid, rec in data.items():
        if "/FT" not in cid or "per_seed" not in rec:
            continue
        runs = rec["per_seed"]
        still_rising = [r["seed"] for r in runs if r.get("epochs") and r.get("best_epoch") == r["epochs"]]
        chk = {"seeds_best_epoch_is_last_epoch": still_rising,
               "reading": "fine-tune budget-limited (validation still rising at the last epoch) for the seeds listed" if still_rising else
                          "best validation epoch before the last epoch for every seed"}
        lp = data.get(lp_of.get(cid.split("_long")[0], ""), {}).get("value")
        if lp is not None:
            chk["frozen_probe_value_same_task"] = lp
            chk["fine_tune_below_frozen_probe"] = bool(rec["value"] < lp)
        rec["budget_check"] = chk
        merge_write(out, cid, rec)


# --------------------------------------------------------------------------------------------- static records
def static_records(out):
    """Only Nandi remains a third-party cell.  Vessel S1/L8, Solar Farm, PASTIS and MADOS fine-tune are run by their own
    cells (prep_*, ft_*, lp_pastis); this function never writes those keys, so it cannot overwrite a real run."""
    why = ("OlmoEarth eval dataset nandi_crop_map is not public: the Hugging Face dataset allenai/olmoearth_projects_nandi returns HTTP 401 "
           "and no copy exists in the public OlmoEarth buckets (checked 2026-11). The value in Table 2 stays third-party (OlmoEarth v1.3).")
    for cid in ("nandi-s2/LP", "nandi-s2/FT"):
        k, c = cid.split("/")
        merge_write(out, cid, {"table2_row": "Nandi (S2)", "cell": c, "status": "third_party_olmoearth", "value": None, "reason": why,
                               "old_transcribed": {"value": OLD[k][c], "source": OLD_SOURCE}, "provenance": prov()})


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True, help="comma-separated cell ids (see module docstring)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=None, help="override fine-tune epochs")
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the FT record id, e.g. _long")
    ap.add_argument("--lr", type=float, default=None, help="single lr for head and encoder (ft_m-*, ft_mados, ft_sen1f_s1); record id gets suffix _lr<value>")
    ap.add_argument("--eval-every", type=int, default=None, help="vessel cells: validation interval in steps (default 1000); one L8 validation pass takes about 4.5 minutes, so longer budgets need a longer interval")
    ap.add_argument("--smoke", action="store_true", help="truncated data/budget end-to-end test; use with a throw-away --out")
    a = ap.parse_args()
    global TAG, SMOKE, LR
    LR = a.lr
    TAG = a.tag + (f"_lr{a.lr:g}" if a.lr else "")
    SMOKE = a.smoke
    out = Path(a.out)
    seeds = [int(s) for s in a.seeds.split(",")]
    rsb.SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    suite = None
    ft_defaults = {"m-eurosat": (15, 32), "m-brick-kiln": (3, 32), "m-so2sat": (3, 32), "m-forestnet": (3, 16), "m-bigearthnet": (2, 32)}  # (epochs, batch); ~1-2k optimiser steps each, budget-limited
    for cell in a.cells.split(","):
        t0 = time.time(); print(f"\n===== {cell} =====", flush=True)
        if cell == "static":
            static_records(out)
        elif cell == "finalize":
            finalize(out)
        elif cell.startswith("prep_vessel"):
            prep_vessels(cell[5:])
        elif cell in ("ft_vessel-s1", "ft_vessel-l8"):
            ft_det_cell(cell[3:], seeds, a.epochs or 6000, out, eval_every=a.eval_every or 1000)
        elif cell == "prep_solar":
            prep_solar()
        elif cell == "ft_solar":
            ft_solar_cell(seeds, a.epochs or 4000, out)
        elif cell == "ft_pastis":
            ft_pastis_cell(seeds, a.epochs or 10, out)
        elif cell.startswith("ft_"):
            task = cell[3:]
            if task in ("sen1f_s1", "m-cashew-plant", "mados"):
                ft_seg_cell(task, seeds, a.epochs or {"sen1f_s1": 100, "mados": 40}.get(task, 20), out)
            else:
                ep, bs = ft_defaults[task]
                ft_cell(task, seeds, a.epochs or ep, a.bs or bs, out)
        else:
            suite = suite or ClayOnly()
            {"lp_bigearthnet": lp_bigearthnet, "lp_cashew": lp_cashew, "lp_sen1f_s1": lp_sen1f_s1,
             "lp_cropharvest_togo": lp_cropharvest_togo, "lp_mados": lp_mados, "lp_pastis": lp_pastis, "oe_contract": oe_contract}[cell](suite, out)
        print(f"===== {cell} done in {time.time()-t0:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
