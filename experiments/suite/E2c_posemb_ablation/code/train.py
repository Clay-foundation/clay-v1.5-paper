"""E2c: train arms A/B/C of a small Clay MAE in lock-step on identical batches and masks.
Usage: python train.py DATADIR OUTDIR STEPS [ARMS=ABC] [BS=64]"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "code"); sys.path.insert(0, "common")
from common_data import *

DATA, OUT, STEPS = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
ARMS_RUN = sys.argv[4] if len(sys.argv) > 4 else "ABC"; BS = int(sys.argv[5]) if len(sys.argv) > 5 else 64
OUT.mkdir(parents=True, exist_ok=True)
SEED = int(os.environ.get("E2C_SEED", "0"))  # amendment 1: 0 = original run; seeds change init, data order/augmentation and masks only
LR, WD, WARM = 2e-4, 0.05, max(1, int(0.05 * STEPS))
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True
t0 = time.time()
def say(*a): print(round(time.time() - t0), *a, flush=True)

# ---- training data: E2 chips not in E2's test split (train ∪ val); test chips never loaded here ----
D = {}
for s in SENSORS:
    pids, lat, lon, blocks, split = chip_index(DATA, s)
    tr = np.flatnonzero(split != "test")
    X, spec = load_norm(DATA, s, [pids[i] for i in tr])
    D[s] = dict(X=torch.from_numpy(X), waves=torch.tensor(spec["waves"], device=dev), gsd=torch.tensor(spec["gsd"], device=dev),
                pids=[pids[i] for i in tr])
    say("loaded", s, X.shape)
json.dump({s: d["pids"] for s, d in D.items()}, open(OUT / "train_pids.json", "w"))
n = np.array([len(D[s]["pids"]) for s in SENSORS]); prob = n / n.sum()

models, opts = {}, {}
for a in ARMS_RUN:
    torch.manual_seed(SEED)  # identical initial weights for every arm
    m = SmallClay(a).to(dev); models[a] = m
    dec, nodec = [], []
    for nme, p in m.named_parameters(): (nodec if p.ndim < 2 else dec).append(p)
    opts[a] = torch.optim.AdamW([{"params": dec, "weight_decay": WD}, {"params": nodec, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.95))
say("params/arm", sum(p.numel() for p in models[ARMS_RUN[0]].parameters()))
def lr_at(t): return LR * (t + 1) / WARM if t < WARM else LR * 0.5 * (1 + math.cos(math.pi * (t - WARM) / max(1, STEPS - WARM)))

rng = np.random.default_rng(20261105 + SEED); log = []
run = {a: [] for a in ARMS_RUN}
for step in range(STEPS):
    s = SENSORS[rng.choice(4, p=prob)]; d = D[s]
    idx = rng.choice(len(d["pids"]), BS, replace=False); k = int(rng.integers(4)); flip = bool(rng.integers(2))
    px = d["X"][idx].to(dev, non_blocking=True).float()
    px = torch.rot90(px, k, (2, 3)); px = torch.flip(px, (3,)) if flip else px
    for a, m in models.items():
        for g in opts[a].param_groups: g["lr"] = lr_at(step)
        torch.manual_seed(1_000_000 + step + 10_000_000 * SEED)  # identical mask across arms at each step
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = m(px, d["waves"], d["gsd"])
        opts[a].zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opts[a].step()
        run[a].append(float(loss))
    if (step + 1) % 50 == 0:
        rec = {"step": step + 1, "lr": lr_at(step), "sensor_last": s, **{f"loss_{a}": float(np.mean(run[a])) for a in run}}
        run = {a: [] for a in ARMS_RUN}; log.append(rec)
        with open(OUT / "loss_log.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
    if step + 1 == STEPS // 2:
        (OUT / "mid").mkdir(exist_ok=True)
        for a, m in models.items(): torch.save({"arm": a, "state_dict": m.state_dict(), "steps": step + 1}, OUT / "mid" / f"model_{a}.pt")
    if (step + 1) % 1000 == 0 or step == 0:
        say("step", step + 1, {k: round(v, 4) for k, v in log[-1].items() if k.startswith("loss")} if log else None,
            "gpu_max_gb", round(torch.cuda.max_memory_allocated() / 1e9, 2))
for a, m in models.items():
    torch.save({"arm": a, "state_dict": m.state_dict(), "steps": STEPS, "bs": BS, "lr": LR}, OUT / f"model_{a}.pt")
json.dump({"seed": SEED, "steps": STEPS, "bs": BS, "lr": LR, "wd": WD, "warmup": WARM, "arms": ARMS_RUN, "n_train_chips": dict(zip(SENSORS, n.tolist())),
           "sensor_prob": dict(zip(SENSORS, prob.round(4).tolist())), "elapsed_s": round(time.time() - t0),
           "gpu_max_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__}, open(OUT / "train_meta.json", "w"), indent=1)
say("done")
