#!/usr/bin/env python3
"""EmbeddingGemma 2 as a frozen RGB image embedder on the standard benchmark tasks (comparator, not part of E9's committed table).

Same data loaders, splits, RGB / pseudo-RGB construction and probes as run_standard_benchmarks.py (`rsb`, imported; the probe
functions are not re-implemented). Per image one frozen 768-D embedding (sentence-transformers pipeline of the model card:
vision encoder -> mean pooling -> L2 normalisation), image only, no text prefix (model card: prefixes apply to text only).

Input to the model: the RGB chip DINOv2-L receives in rsb (3 bands on tasks with more bands; train-split 1-99 % stretch for non-uint8
data; Sentinel-1 pseudo-RGB [VV, VH, (VV+VH)/2] stretched the same way), bilinearly resized to 224x224 (`--input-size 224`, default;
identical to DINOv2's input) or passed at the stored chip size (`--input-size native`). The model's own processor then resizes
to its fixed vision-token budget (default 280 soft tokens) and rescales to [0, 1] (no mean/std normalisation). Input is
quantised to uint8 for the processor (the committed DINOv2 path keeps float32).

`--model dinov2` runs the DINOv2-L baseline through this script's data path with rsb's own `ModelSuite.extract_dinov2` and the same
probes (reproduction check against standard_benchmarks.json).

Usage:
  python experiments/run_embeddinggemma2.py --output-dir DIR [--tasks m-eurosat,ucmerced] [--limit 200]
         [--model embeddinggemma2|dinov2] [--input-size 224|native] [--resume]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))
import run_standard_benchmarks as rsb  # noqa: E402

MODEL_REPO = "google/embeddinggemma-2"
# Hub commit of the model card read for this evaluation (2026-10-28 snapshot); override with --revision.
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
PROMPT_USED = "none (image-only input; model card: task prefixes apply to text only)"

# m-bigearthnet is not in rsb's task table (it lives in run_all_doable_table2.py); same 10 S2 bands and archive.
BEN_BANDS = ["02 - Blue", "03 - Green", "04 - Red", "05 - Vegetation Red Edge", "06 - Vegetation Red Edge",
             "07 - Vegetation Red Edge", "08 - NIR", "08A - Vegetation Red Edge", "11 - SWIR", "12 - SWIR"]
EXTRA_TASKS = [{
    "id": "m-bigearthnet", "source": "geobench", "repo_file": "classification_v1.0/m-bigearthnet.zip", "bands": BEN_BANDS,
    "waves": rsb.S2_10B_WAVES, "rgb_idx": [2, 1, 0], "gsd": 10.0, "kind": "multilabel_f1", "is_uint8": False,
    "is_unit_float": False, "sensor_label": "Sentinel-2 (10 m, 10b)",
}]
# Image-level readout tasks only (dense tasks m-pv4ger-seg, m-chesapeake are skipped: see report).
TASK_IDS = ["m-eurosat", "m-brick-kiln", "m-so2sat", "m-forestnet", "m-bigearthnet", "m-pv4ger", "ucmerced", "resisc45",
            "eurosat-s1", "lc100cls-s3", "biomass-s3"]


def eval_multilabel_micro_f1(z_tr, y_tr, z_va, y_va, z_te, y_te) -> dict:
    """Micro-F1 head of run_all_doable_table2.score_task_dict (kind multilabel_f1), transcribed with rsb's L-BFGS solver
    because that module cannot be imported without olmoearth_pretrain. C chosen on val micro-F1 at threshold 0.5."""
    scaler = StandardScaler().fit(z_tr)
    Xtr, Xva, Xte = (torch.tensor(scaler.transform(z), dtype=torch.float32, device="cuda") for z in (z_tr, z_va, z_te))
    Ytr = torch.tensor(y_tr, dtype=torch.float32, device="cuda")
    best_f1, best_te, best_c = -1.0, -1.0, None
    for c in (0.01, 0.1, 1.0, 10.0):
        W, b = rsb._fit_gpu_linear_lbfgs(Xtr, Ytr, y_tr.shape[1], 0.5 / (c * len(Xtr)), mode="bce")
        with torch.no_grad():
            p_va = (torch.sigmoid(Xva @ W.t() + b) > 0.5).cpu().numpy().astype(int)
            va_f1 = float(f1_score(y_va.astype(int), p_va, average="micro"))
            if va_f1 > best_f1:
                best_f1, best_c = va_f1, c
                p_te = (torch.sigmoid(Xte @ W.t() + b) > 0.5).cpu().numpy().astype(int)
                best_te = float(f1_score(y_te.astype(int), p_te, average="micro"))
    return {"micro_f1": round(best_te * 100.0, 2), "val_micro_f1": round(best_f1 * 100.0, 2), "best_C": best_c}


def score(kind: str, tr, y_tr, va, y_va, te, y_te) -> dict:
    if kind == "cls":
        return rsb.eval_single_label(tr, y_tr, va, y_va, te, y_te)
    if kind == "multilabel":
        return rsb.eval_multilabel(tr, y_tr, va, y_va, te, y_te)
    if kind == "multilabel_f1":
        return eval_multilabel_micro_f1(tr, y_tr, va, y_va, te, y_te)
    if kind == "biomass":
        m = lambda y: y.reshape(len(y), -1).mean(1)
        return rsb.eval_chip_regression(tr, m(y_tr), va, m(y_va), te, m(y_te))
    raise ValueError(kind)


class GemmaEmbedder:
    def __init__(self, revision: str, input_size: str):
        from sentence_transformers import SentenceTransformer
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32  # model card: never float16
        self.st = SentenceTransformer(MODEL_REPO, revision=revision, device="cuda",
                                      model_kwargs={"torch_dtype": dtype}, config_kwargs={"audio_config": None})
        self.st.eval()
        self.input_size = input_size
        self.dtype = str(dtype).replace("torch.", "")
        self.dim = int(self.st.get_sentence_embedding_dimension())

    @torch.inference_mode()
    def __call__(self, rgb01: np.ndarray, chunk: int = 256, batch_size: int = 32) -> np.ndarray:
        from PIL import Image
        out = []
        for i in range(0, len(rgb01), chunk):
            xb = torch.as_tensor(rgb01[i:i + chunk], dtype=torch.float32)
            if self.input_size == "224" and xb.shape[-2:] != (224, 224):
                xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
            u8 = (xb.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
            imgs = [Image.fromarray(a, mode="RGB") for a in u8]
            e = self.st.encode(imgs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
            out.append(np.asarray(e, dtype=np.float32))
        z = np.concatenate(out, 0)
        assert np.isfinite(z).all(), "non-finite embedding (check dtype: float16 is not supported by the model)"
        return z


class DinoEmbedder:
    """rsb's own DINOv2-L extraction (224x224, ImageNet normalisation, fp16 autocast) on a stub ModelSuite."""
    def __init__(self):
        import timm
        self.suite = rsb.ModelSuite.__new__(rsb.ModelSuite)
        self.suite.dv2 = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
        self.suite.dv2.eval().requires_grad_(False)
        cfg = self.suite.dv2.pretrained_cfg
        self.suite.imnet_mean = torch.tensor(cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        self.suite.imnet_std = torch.tensor(cfg["std"], dtype=torch.float32).view(1, 3, 1, 1)

    def __call__(self, rgb01: np.ndarray) -> dict:
        return self.suite.extract_dinov2(rgb01, False)


def load_splits(cfg: dict):
    if cfg["source"] == "geobench":
        splits = rsb.load_geobench_splits(cfg["repo_file"], cfg["id"], cfg["bands"], kind=cfg["kind_loader"])
        meta = dict(rsb.load_geobench_splits.last_meta)
    elif cfg["source"] == "torchgeo_cls":
        splits, meta = rsb.load_torchgeo_cls(cfg["cls_type"], resize_hw=cfg["resize_hw"]), {}
    else:
        splits, meta = rsb.load_copernicus_s3(is_biomass=cfg["is_biomass"]), {}
    return splits, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=None, help="default: fresh experiments/runs/embeddinggemma2_<UTC>/")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: all image-level tasks)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test only: N seeded-random samples per split")
    ap.add_argument("--model", choices=["embeddinggemma2", "dinov2"], default="embeddinggemma2")
    ap.add_argument("--input-size", choices=["224", "native"], default="224")
    ap.add_argument("--revision", default=MODEL_REVISION)
    a = ap.parse_args()
    if a.limit is not None:
        rsb.SMOKE_LIMIT = a.limit
    torch.cuda.set_per_process_memory_fraction(11.0 / 23.0)
    rsb.SCRATCH.mkdir(parents=True, exist_ok=True)

    script = str(Path(__file__).resolve())
    fname = "embeddinggemma2.json" if a.model == "embeddinggemma2" else "dinov2_check.json"
    out_json = rsb.resolve_output(a.output_dir, a.resume, "embeddinggemma2", fname)
    print(f"[output] {out_json}", flush=True)
    results = {"tasks": {}, "meta": {}}
    if a.resume and out_json.exists():
        results = json.loads(out_json.read_text())

    base = {c["id"]: c for c in rsb.build_tasks_cfg()}
    cfgs = {**base, **{c["id"]: c for c in EXTRA_TASKS}}
    want = a.tasks.split(",") if a.tasks else TASK_IDS
    bad = [t for t in want if t not in cfgs]
    if bad:
        raise SystemExit(f"unknown task ids: {bad}")

    if a.model == "embeddinggemma2":
        emb = GemmaEmbedder(a.revision, a.input_size)
        model_info = {"model": MODEL_REPO, "model_revision": a.revision, "embedding_dim": emb.dim, "dtype": emb.dtype,
                      "prompt": PROMPT_USED, "input_size_arg": a.input_size,
                      "vision_token_budget": "default (processor max_soft_tokens=280)"}
    else:
        emb = DinoEmbedder()
        model_info = {"model": "vit_large_patch14_reg4_dinov2.lvd142m (timm)", "input_size_arg": "224 (rsb extract_dinov2)"}

    import importlib.metadata as md
    libs = {}
    for n in ("torch", "transformers", "sentence-transformers", "timm", "huggingface-hub", "numpy", "scikit-learn", "pillow", "torchgeo"):
        try:
            libs[n] = md.version(n)
        except Exception:
            libs[n] = "n/a"

    for tid in want:
        cfg = dict(cfgs[tid])
        kind = cfg["kind"]
        cfg["kind_loader"] = "multilabel" if kind in ("multilabel", "multilabel_f1") else ("cls" if kind == "cls" else kind)
        fp = rsb.cfg_fingerprint({k: v for k, v in cfg.items() if k != "cls_type"} | {"cls_type": str(cfg.get("cls_type")), "model": model_info})
        if a.resume and rsb.can_skip(results["tasks"].get(tid), script, fp):
            print(f"[resume-skip] {tid}", flush=True)
            continue
        t0 = time.time()
        print(f"\n==================== {tid} ({cfg['sensor_label']}) ====================", flush=True)
        splits, gb_meta = load_splits(cfg)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits["train"], splits["val"], splits["test"]
        del splits
        means, stds, p01, p99 = rsb.compute_train_band_stats(X_tr)
        X_tr, X_va, X_te = (rsb.clean_nonfinite(x, means) for x in (X_tr, X_va, X_te))

        if tid == "eurosat-s1":  # same pseudo-RGB as run_all_doable_table2 (DINOv2-L on SAR); here stretched to [0,1] like the other non-uint8 tasks
            three = lambda X: np.concatenate([X, 0.5 * (X[:, 0:1] + X[:, 1:2])], axis=1)
            X_tr, X_va, X_te = three(X_tr), three(X_va), three(X_te)
            _, _, p01, p99 = rsb.compute_train_band_stats(X_tr)
            rgb_idx = [0, 1, 2]
        elif tid == "m-so2sat":
            X_tr, X_va, X_te = X_tr[:, :10], X_va[:, :10], X_te[:, :10]   # S2 arm (RGB = bands 2,1,0)
            rgb_idx = cfg["rgb_idx"]
        else:
            rgb_idx = cfg["rgb_idx"]
        h_orig = int(X_tr.shape[-2])
        rgb = [rsb.to_rgb01(x, rgb_idx, cfg["is_uint8"], cfg["is_unit_float"], p01, p99) for x in (X_tr, X_va, X_te)]
        del X_tr, X_va, X_te
        gc.collect()

        task_res = {"sensor": cfg["sensor_label"], "kind": kind, "orig_hw": h_orig,
                    "splits": [int(len(y_tr)), int(len(y_va)), int(len(y_te))], "n_train": int(len(y_tr)),
                    "n_val": int(len(y_va)), "n_test": int(len(y_te)), "rgb_source_channels": rgb_idx,
                    "input_size": 224 if a.input_size == "224" or a.model == "dinov2" else h_orig,
                    "models": {}}
        if gb_meta:
            task_res["archive"] = {k: gb_meta[k] for k in ("archive", "archive_sha256", "hf_revision") if k in gb_meta}
        feats = [emb(r) for r in rgb]
        if a.model == "embeddinggemma2":
            task_res["embedding_dim"] = int(feats[0].shape[1])
            task_res["models"]["embeddinggemma2"] = score(kind, feats[0], y_tr, feats[1], y_va, feats[2], y_te)
        else:
            task_res["embedding_dim"] = int(feats[0]["cls"].shape[1])
            for ro in ("cls", "mean"):
                task_res["models"][f"dinov2_l_{ro}"] = score(kind, feats[0][ro], y_tr, feats[1][ro], y_va, feats[2][ro], y_te)
        task_res["elapsed_s"] = round(time.time() - t0, 1)
        task_res["producer"] = {"script_sha256": rsb.sha256_file(script), "cfg": fp, "utc_timestamp": rsb.utc_now()}
        results["tasks"][tid] = task_res
        results["meta"] = rsb.provenance(script, {
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3), "library_versions": libs, **model_info,
            "rsb_script_sha256": rsb.sha256_file(rsb.__file__),
            "geobench_hf_repo": rsb.GEOBENCH_HF_REPO, "geobench_hf_revision": rsb.GEOBENCH_HF_REVISION,
            **({"smoke_limit": rsb.SMOKE_LIMIT} if rsb.SMOKE_LIMIT else {}),
        })
        out_json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"[done {tid} in {task_res['elapsed_s']}s] " + json.dumps(task_res["models"]), flush=True)
        del rgb, feats, y_tr, y_va, y_te
        gc.collect()
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
