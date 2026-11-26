"""E2_tokens embed: Clay (meta ON/OFF, S2 GSD lies) + DINOv2-L patch tokens at fixed token subsets.
Usage: python embed.py DATADIR OUTDIR [sensors...]"""
import hashlib, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "common")
import clayrt
from clayrt import ClayRuntime, to_display_rgb, platform_spec
assert clayrt.LATLON_ORDER == "training", clayrt.LATLON_ORDER
from evalkit import hblock, block_split

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SENSORS = sys.argv[3:] or ["s2", "s1", "ls", "naip"]
PLAT = {"s2": "sentinel-2-l2a", "s1": "sentinel-1-rtc", "ls": "landsat-c2l2-sr", "naip": "naip"}
RGB = {"s2", "ls", "naip"}
LIES = [1.0, 5.0, 30.0, 100.0]
TEST_POS = np.array([r * 32 + c for r in range(0, 32, 2) for c in range(0, 32, 2)])

def train_pos(pid):
    k = [hashlib.sha256(f"{pid}|tok|{t}".encode()).hexdigest() for t in range(1024)]
    return np.sort(np.argsort(k)[:64])

class Dino:
    def __init__(self, name="vit_large_patch14_reg4_dinov2.lvd142m", size=448):
        import timm
        self.m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size).eval().cuda()
        cfg = self.m.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        self.size, self.npre = size, self.m.num_prefix_tokens
    @torch.inference_mode()
    def patches(self, rgb01):
        x = torch.as_tensor(rgb01, dtype=torch.float32)
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        f = self.m.forward_features(((x - self.mean) / self.std).cuda())
        return f[:, self.npre:].float().cpu().numpy()  # B,1024,1024

def band_stats(xn):  # xn B,C,256,256 normalised -> B,1024,2C
    B, C = xn.shape[:2]
    r = xn.reshape(B, C, 32, 8, 32, 8)
    m = r.mean((3, 5)); s = r.std((3, 5))
    return np.concatenate([m, s], 1).reshape(B, 2 * C, 1024).transpose(0, 2, 1)

def unit(a): return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-12)

rt = ClayRuntime(device="cuda"); dino = None
ev = {"latlon_order": clayrt.LATLON_ORDER, "clay": rt.evidence, "gpu": torch.cuda.get_device_name(0)}
for s in SENSORS:
    t0 = time.time()
    if not (DATA / f"log_{s}.jsonl").exists(): print(s, "no log; skipped"); continue
    logs = [json.loads(l) for l in open(DATA / f"log_{s}.jsonl")]
    meta = {r["pid"]: r for r in logs if r.get("ok")}
    pids = sorted(p for p in meta if (DATA / "chips" / s / f"{p}.npz").exists())
    if not pids: print(s, "no chips; skipped", flush=True); continue
    lat = np.array([meta[p]["lat"] for p in pids]); lon = np.array([meta[p]["lon"] for p in pids])
    blocks = hblock(lat, lon); split = block_split(blocks)
    is_test = split == "test"
    spec = platform_spec(PLAT[s]); mu = np.array(spec["means"])[None, :, None, None]; sd = np.array(spec["stds"])[None, :, None, None]
    keys = ["cls", "mean", "cls_off", "mean_off", "cls_doc", "mean_doc", "chipstats"]
    acc = {k: [] for k in keys}; tok = {"patch": [], "bstat": [], "label": [], "purity": [], "pos": [], "chip": []}
    if s in RGB: tok["dino"] = []
    if s == "s2":
        for g in LIES:
            for k in ["cls", "mean"]: acc[f"{k}_g{g:g}"] = []
            tok[f"patch_g{g:g}"] = []; acc[f"cos_cls_g{g:g}"] = []; acc[f"cos_mean_g{g:g}"] = []; acc[f"cos_patch_g{g:g}"] = []
    if s in RGB and dino is None: dino = Dino()
    BS = 16
    for i in range(0, len(pids), BS):
        bp = pids[i:i + BS]
        d = [np.load(DATA / "chips" / s / f"{p}.npz") for p in bp]
        px = np.stack([x["pixels"] for x in d]).astype(np.float32)
        lab = np.stack([x["labels"].reshape(-1) for x in d]); pur = np.stack([x["purity"].reshape(-1) for x in d])
        ll = [(meta[p]["lat"], meta[p]["lon"]) for p in bp]
        dts = [datetime.fromisoformat(meta[p]["datetime"].replace("Z", "+00:00")) for p in bp]
        on = rt.embed(px, platform=PLAT[s], latlon=ll, date=dts, batch_size=BS)
        off = rt.embed(px, platform=PLAT[s], latlon=None, date=None, batch_size=BS)
        doc = rt.embed(px, platform=PLAT[s], latlon=ll, date=dts, batch_size=BS, latlon_order="documented")
        acc["cls_doc"].append(doc["cls"]); acc["mean_doc"].append(doc["mean"])
        acc["cls"].append(on["cls"]); acc["mean"].append(on["mean"]); acc["cls_off"].append(off["cls"]); acc["mean_off"].append(off["mean"])
        xn = (px - mu) / sd
        acc["chipstats"].append(np.concatenate([xn.mean((2, 3)), xn.std((2, 3))], 1))
        bs = band_stats(xn); P = on["patches"].reshape(len(bp), 1024, -1)
        dp = dino.patches(to_display_rgb(px, PLAT[s])) if s in RGB else None
        lies = {}
        if s == "s2":
            for g in LIES:
                o = rt.embed(px, platform=PLAT[s], latlon=ll, date=dts, gsd=g, batch_size=BS); lies[g] = o
                Pg = o["patches"].reshape(len(bp), 1024, -1)
                acc[f"cls_g{g:g}"].append(o["cls"]); acc[f"mean_g{g:g}"].append(o["mean"])
                acc[f"cos_cls_g{g:g}"].append((unit(o["cls"]) * unit(on["cls"])).sum(-1))
                acc[f"cos_mean_g{g:g}"].append((unit(o["mean"]) * unit(on["mean"])).sum(-1))
                acc[f"cos_patch_g{g:g}"].append((unit(Pg) * unit(P)).sum(-1).mean(-1))
        for j, p in enumerate(bp):
            pos = TEST_POS if is_test[i + j] else train_pos(p)
            tok["patch"].append(P[j, pos].astype(np.float16)); tok["bstat"].append(bs[j, pos].astype(np.float32))
            tok["label"].append(lab[j, pos]); tok["purity"].append(pur[j, pos]); tok["pos"].append(pos.astype(np.int16))
            tok["chip"].append(np.full(len(pos), i + j, np.int32))
            if dp is not None: tok["dino"].append(dp[j, pos].astype(np.float16))
            for g, o in lies.items():
                tok[f"patch_g{g:g}"].append(o["patches"].reshape(len(bp), 1024, -1)[j, pos].astype(np.float16))
        if i % (BS * 20) == 0: print(s, i, len(pids), round(time.time() - t0), "s", flush=True)
    outd = {k: np.concatenate(v) for k, v in acc.items()}
    outd.update({f"tok_{k}": np.concatenate(v) for k, v in tok.items()})
    outd.update(pids=np.array(pids), lat=lat, lon=lon, blocks=blocks, split=split)
    np.savez(OUT / f"emb_{s}.npz", **outd)
    ev[s] = {"n_chips": len(pids), "n_test": int(is_test.sum()), "s": round(time.time() - t0),
             "gpu_max_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print(s, ev[s], flush=True)
json.dump(ev, open(OUT / "embed_evidence.json", "w"), indent=1)
