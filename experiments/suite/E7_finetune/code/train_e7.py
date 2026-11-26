"""E7: Sen1Floods11 hand-labelled water segmentation — frozen vs fine-tuned Clay v1.5, U-Net scratch, DOFA-L FT.
Usage: python train_e7.py --arm {clay_frozen_linear,clay_frozen_upernet,clay_finetune_upernet,unet_scratch,dofa_finetune_upernet}
       --seed S [--epochs 100] [--smoke]   (protocol: ../PROTOCOL.md)"""
import argparse, json, math, os, sys, time
from datetime import datetime
import numpy as np, rasterio, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.checkpoint import checkpoint
HERE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(HERE / "experiments/suite/common"))
from clayrt import ClayRuntime, platform_spec, encode_latlon, encode_time

DATA = os.environ.get("CLAY_E7_DATA", str(HERE / "data/E7_finetune"))
BIDX = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]  # S2 L1C 13-band -> Clay blue..swir22
TAPS = (6, 12, 18, 24)
SPEC = platform_spec("sentinel-2-l2a")
P = argparse.ArgumentParser()
P.add_argument("--arm", required=True); P.add_argument("--seed", type=int, default=0)
P.add_argument("--epochs", type=int, default=100); P.add_argument("--smoke", action="store_true")
P.add_argument("--out", default="results")
P.add_argument("--max-gpu-gb", type=float, default=10.0, help="hard cap on this process's CUDA allocations (original runs: 10)")
A = P.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed)
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(A.max_gpu_gb / 24.0)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


def load():
    rows = json.load(open(f"{DATA}/chips.json"))
    m = np.array(SPEC["means"], np.float32)[:, None, None]; s = np.array(SPEC["stds"], np.float32)[:, None, None]
    D = {}
    for r in rows:
        c = r["chip"]
        x = rasterio.open(f"{DATA}/S2Hand/{c}_S2Hand.tif").read()[BIDX].astype(np.float32)
        y = rasterio.open(f"{DATA}/LabelHand/{c}_LabelHand.tif").read(1).astype(np.int64)
        nod = (x == 0).any(0)
        x = (x - m) / s; x[:, nod] = 0.0
        assert np.isfinite(x).all() and x.shape == (10, 512, 512) and y.shape == (512, 512)
        dt = datetime.fromisoformat(r["stac"]["datetime"].replace("Z", "+00:00"))
        D[c] = dict(x=torch.from_numpy(x).half(), y=torch.from_numpy(y), split=r["split"][0],
                    ll=encode_latlon(r["lat"], r["lon"]), tt=encode_time(dt))
    return D


# ---------------- models ----------------
def cbr(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Neck(nn.Module):
    def __init__(self, din, stride, c=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(din, c, 1) for _ in TAPS])
        up = lambda: nn.Sequential(nn.ConvTranspose2d(c, c, 2, 2), nn.BatchNorm2d(c), nn.GELU())
        if stride == 8:
            self.rs = nn.ModuleList([up(), nn.Identity(), nn.MaxPool2d(2), nn.MaxPool2d(4)])
        else:
            self.rs = nn.ModuleList([nn.Sequential(up(), nn.ConvTranspose2d(c, c, 2, 2)), up(), nn.Identity(), nn.MaxPool2d(2)])

    def forward(self, feats):
        return [r(p(f)) for f, p, r in zip(feats, self.proj, self.rs)]


class UPerHead(nn.Module):
    def __init__(self, c=256, ncls=2, scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), cbr(c, c, 1)) for s in scales])
        self.bott = cbr(c * (1 + len(scales)), c)
        self.lat = nn.ModuleList([cbr(c, c, 1) for _ in range(3)])
        self.fpn = nn.ModuleList([cbr(c, c) for _ in range(3)])
        self.fuse = cbr(4 * c, c); self.cls = nn.Conv2d(c, ncls, 1)

    def forward(self, f):
        top = f[3]; hw = top.shape[-2:]
        top = self.bott(torch.cat([top] + [F.interpolate(p(top), hw, mode="bilinear", align_corners=False) for p in self.ppm], 1))
        lats = [l(x) for l, x in zip(self.lat, f[:3])] + [top]
        for i in range(2, -1, -1):
            lats[i] = lats[i] + F.interpolate(lats[i + 1], lats[i].shape[-2:], mode="bilinear", align_corners=False)
        outs = [self.fpn[i](lats[i]) for i in range(3)] + [lats[3]]
        hw0 = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, hw0, mode="bilinear", align_corners=False) for o in outs[1:]]
        return self.cls(self.fuse(torch.cat(outs, 1)))


class ClayBackbone(nn.Module):
    def __init__(self, trainable):
        super().__init__()
        rt = ClayRuntime(device=dev); self.evidence = rt.evidence
        self.enc = rt.enc.float(); self.enc.requires_grad_(trainable); self.trainable = trainable
        self.register_buffer("waves", torch.tensor(SPEC["waves"]), persistent=False)
        self.gsd = torch.tensor(SPEC["gsd"])

    def forward(self, x, tt, ll, taps=TAPS):
        e = self.enc; B = x.shape[0]
        p, _ = e.to_patch_embed(x, self.waves)
        p = e.add_encodings(p, tt, ll, self.gsd.to(x.device))
        h = torch.cat([e.cls_token.expand(B, -1, -1), p], 1)
        out = []; n = int(math.isqrt(p.shape[1]))
        for i, (attn, ff) in enumerate(e.transformer.layers):
            blk = lambda h, attn=attn, ff=ff: (lambda z: ff(z) + z)(attn(h) + h)
            h = checkpoint(blk, h, use_reentrant=False) if (self.trainable and self.training) else blk(h)
            if i + 1 in taps:
                t = e.transformer.norm(h)[:, 1:]
                out.append(t.transpose(1, 2).reshape(B, -1, n, n))
        return out


class DofaBackbone(nn.Module):
    def __init__(self, size=256):
        super().__init__()
        from torchgeo.models import dofa_large_patch16_224, DOFALarge16_Weights
        m = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE)
        pe = m.pos_embed.data; g0 = int(math.isqrt(pe.shape[1] - 1)); g = size // 16
        grid = pe[:, 1:].reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, (g, g), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(1, g * g, -1)
        m.pos_embed = nn.Parameter(torch.cat([pe[:, :1], grid], 1), requires_grad=m.pos_embed.requires_grad)
        self.m = m; self.norm = getattr(m, "norm", None) or m.fc_norm
        self.waves = list(SPEC["waves"])
        self.evidence = {"model": "torchgeo dofa_large_patch16_224 DOFA_MAE", "pos_embed_trainable": bool(m.pos_embed.requires_grad),
                         "params": sum(p.numel() for p in m.parameters()), "norm": "norm" if getattr(m, "norm", None) is not None else "fc_norm"}

    def forward(self, x, tt=None, ll=None):
        m = self.m; B = x.shape[0]
        h, _ = m.patch_embed(x, torch.tensor(self.waves, device=x.device).float())
        h = h + m.pos_embed[:, 1:]
        h = torch.cat([(m.cls_token + m.pos_embed[:, :1]).expand(B, -1, -1), h], 1)
        out = []; n = x.shape[-1] // 16
        for i, blk in enumerate(m.blocks):
            h = checkpoint(blk, h, use_reentrant=False) if self.training else blk(h)
            if i + 1 in TAPS:
                out.append(self.norm(h)[:, 1:].transpose(1, 2).reshape(B, -1, n, n))
        return out


class UNet(nn.Module):
    def __init__(self, cin=10, ncls=2, b=64):
        super().__init__()
        ch = [b, 2 * b, 4 * b, 8 * b, 16 * b]
        blk = lambda i, o: nn.Sequential(cbr(i, o), cbr(o, o))
        self.down = nn.ModuleList([blk(cin, ch[0])] + [blk(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in range(4)])
        self.dec = nn.ModuleList([blk(2 * ch[i], ch[i]) for i in range(4)])
        self.cls = nn.Conv2d(ch[0], ncls, 1)

    def forward(self, x, tt=None, ll=None):
        sk = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2)); sk.append(x)
        x = sk.pop()
        for i in range(3, -1, -1):
            x = self.dec[i](torch.cat([self.up[i](x), sk[i]], 1))
        return self.cls(x)


class Seg(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        if arm == "unet_scratch":
            self.net = UNet(); self.bb = None
        elif arm.startswith("clay"):
            self.bb = ClayBackbone(trainable=(arm == "clay_finetune_upernet"))
        else:
            self.bb = DofaBackbone()
        if arm == "clay_frozen_linear":
            self.head = nn.Conv2d(1024, 2, 1)
        elif arm != "unet_scratch":
            self.neck = Neck(1024, 8 if arm.startswith("clay") else 16); self.head = UPerHead()

    def forward(self, x, tt, ll):
        H = x.shape[-1]
        if self.bb is None:
            return self.net(x)
        frozen = self.arm.startswith("clay_frozen")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if frozen:
                with torch.no_grad():
                    f = self.bb(x, tt, ll, taps=(24,) if self.arm == "clay_frozen_linear" else TAPS)
                f = [t.float() for t in f]
            else:
                f = self.bb(x, tt, ll)
            o = self.head(f[-1]) if self.arm == "clay_frozen_linear" else self.head(self.neck(f))
        return F.interpolate(o.float(), (H, H), mode="bilinear", align_corners=False)


# ---------------- train / eval ----------------
def batch(D, ids, gen, train):
    xs, ys, tts, lls = [], [], [], []
    for c in ids:
        d = D[c]
        i, j = torch.randint(0, 257, (2,), generator=gen).tolist()
        x = d["x"][:, i:i + 256, j:j + 256].float(); y = d["y"][i:i + 256, j:j + 256]
        if torch.rand(1, generator=gen) < 0.5: x, y = x.flip(-1), y.flip(-1)
        if torch.rand(1, generator=gen) < 0.5: x, y = x.flip(-2), y.flip(-2)
        k = int(torch.randint(0, 4, (1,), generator=gen)); x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
        xs.append(x); ys.append(y); tts.append(d["tt"]); lls.append(d["ll"])
    return (torch.stack(xs).to(dev), torch.stack(ys).to(dev), torch.tensor(tts).to(dev), torch.tensor(lls).to(dev))


@torch.no_grad()
def evaluate(model, D, ids, bs=16):
    model.eval(); conf = np.zeros((len(ids), 4), np.int64)  # tp, fp, fn, tn
    tiles = [(ci, a, b) for ci in range(len(ids)) for a in (0, 256) for b in (0, 256)]
    for s in range(0, len(tiles), bs):
        T = tiles[s:s + bs]
        x = torch.stack([D[ids[ci]]["x"][:, a:a + 256, b:b + 256].float() for ci, a, b in T]).to(dev)
        tt = torch.tensor([D[ids[ci]]["tt"] for ci, _, _ in T]).to(dev); ll = torch.tensor([D[ids[ci]]["ll"] for ci, _, _ in T]).to(dev)
        p = model(x, tt, ll).argmax(1).cpu()
        for k, (ci, a, b) in enumerate(T):
            y = D[ids[ci]]["y"][a:a + 256, b:b + 256]; v = y >= 0; pk, yk = p[k][v], y[v]
            conf[ci] += [int(((pk == 1) & (yk == 1)).sum()), int(((pk == 1) & (yk == 0)).sum()),
                         int(((pk == 0) & (yk == 1)).sum()), int(((pk == 0) & (yk == 0)).sum())]
    model.train(); return conf


def iou(conf):
    tp, fp, fn, tn = conf.sum(0)
    w = tp / max(tp + fp + fn, 1); l = tn / max(tn + fn + fp, 1)
    return float(w), float((w + l) / 2)


def main():
    t0 = time.time(); D = load()
    ids = {s: sorted(c for c in D if D[c]["split"] == s) for s in ("train", "valid", "test", "bolivia")}
    print({k: len(v) for k, v in ids.items()}, f"load {time.time()-t0:.0f}s", flush=True)
    model = Seg(A.arm).to(dev); model.train()
    if A.arm.startswith("clay_frozen"): model.bb.eval()
    lr_head = {"clay_frozen_linear": 1e-3, "unet_scratch": 1e-3}.get(A.arm, 1e-4)
    enc = [p for n, p in model.named_parameters() if n.startswith("bb.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("bb.") and p.requires_grad]
    groups = [{"params": rest, "lr": lr_head}] + ([{"params": enc, "lr": 0.1 * lr_head}] if enc else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.05)
    bs = 8; spe = math.ceil(len(ids["train"]) / bs); total = spe * A.epochs; warm = spe * min(5, A.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    gen = torch.Generator().manual_seed(A.seed)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"arm {A.arm} seed {A.seed} trainable {n_train/1e6:.1f}M enc_trainable {sum(p.numel() for p in enc)/1e6:.1f}M", flush=True)
    log = {"arm": A.arm, "seed": A.seed, "epochs": A.epochs, "lr_head": lr_head, "lr_enc": 0.1 * lr_head if enc else None,
           "trainable_params": n_train, "backbone_evidence": getattr(model.bb, "evidence", None), "valid_curve": [], "loss": []}
    best = (-1, None, None); torch.cuda.reset_peak_memory_stats()
    for ep in range(A.epochs):
        perm = torch.randperm(len(ids["train"]), generator=gen).tolist(); el = []
        for s in range(spe):
            x, y, tt, ll = batch(D, [ids["train"][i] for i in perm[s * bs:(s + 1) * bs]], gen, True)
            loss = F.cross_entropy(model(x, tt, ll), y, ignore_index=-1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); el.append(loss.item())
            if A.smoke and s == 3: break
        log["loss"].append(float(np.mean(el)))
        if (ep + 1) % 5 == 0 or ep + 1 == A.epochs:
            cv = evaluate(model, D, ids["valid"]); w, mi = iou(cv)
            log["valid_curve"].append([ep + 1, w, mi])
            if w > best[0]:
                best = (w, ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("bb.") or not A.arm.startswith("clay_frozen")})
            print(f"ep {ep+1} loss {np.mean(el):.4f} valid wIoU {w:.4f} mIoU {mi:.4f} best_ep {best[1]} t {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.2f}G", flush=True)
    model.load_state_dict(best[2], strict=False)
    log.update(best_epoch=best[1], best_valid_wiou=best[0], train_s=round(time.time() - t0), peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    out = {"conf_valid": evaluate(model, D, ids["valid"]), "ids_valid": np.array(ids["valid"])}
    if not A.smoke:
        for s in ("test", "bolivia"):
            out[f"conf_{s}"] = evaluate(model, D, ids[s]); out[f"ids_{s}"] = np.array(ids[s])
            log[f"{s}_wiou"], log[f"{s}_miou"] = iou(out[f"conf_{s}"])
    os.makedirs(A.out, exist_ok=True); tag = f"{A.arm}_s{A.seed}" + ("_smoke" if A.smoke else "")
    np.savez(f"{A.out}/{tag}.npz", **out); json.dump(log, open(f"{A.out}/{tag}.json", "w"), indent=1, default=str)
    print("done", tag, {k: log.get(k) for k in ("best_epoch", "best_valid_wiou", "test_wiou", "bolivia_wiou", "train_s", "peak_mem_gb")}, flush=True)


main()
