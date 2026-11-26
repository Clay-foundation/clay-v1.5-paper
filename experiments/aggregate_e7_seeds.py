#!/usr/bin/env python3
"""Combine per-seed Sen1Floods11 (S2) peer-arm results into the committed `remaining_all.json` layout.

`run_geobench_and_e7_complete.py --e7-seeds S` writes one JSON per seed (same arms, same protocol). This script reads
those JSONs (`--seed-json PATH`, one per seed), and for every `sen1floods11_s2_<arm>` cell records the mean test
water-IoU over the seeds plus the per-seed values, min/max and the protocol settings taken from the run itself.
No value is typed by hand; the mean is rounded to 0.1 like the single-run cells. Cells present in fewer seed files than
requested are flagged with `n_seeds` and `matched_to_clay_e7: false`.

Usage: python experiments/aggregate_e7_seeds.py --out OUT.json --seed-json run_s0.json --seed-json run_s1.json ...
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-json", action="append", required=True)
    a = ap.parse_args()
    runs = [json.loads(Path(p).read_text()) for p in a.seed_json]
    out: dict = {"_aggregated_from": {"script": "aggregate_e7_seeds.py", "inputs": [Path(p).name for p in a.seed_json],
                                      "utc_timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                      "run_meta": [r.get("_meta") for r in runs]}}
    arms = sorted({k for r in runs for k in r if k.startswith("sen1floods11_s2_")})
    for k in arms:
        per_seed, settings = {}, None
        for r in runs:
            if k in r:
                st = r["_settings"][k]
                per_seed.update(st["per_seed_test_water_iou"])
                settings = settings or st
        vals = [per_seed[s] for s in sorted(per_seed)]
        out[k] = round(float(np.mean(vals)), 1)
        out.setdefault("_settings", {})[k] = {
            **{x: settings[x] for x in settings if x not in ("seeds", "per_seed_test_water_iou", "seed_min", "seed_max", "matched_to_clay_e7")},
            "seeds": sorted(int(s) for s in per_seed), "n_seeds": len(vals), "per_seed_test_water_iou": per_seed,
            "seed_min": min(vals), "seed_max": max(vals),
            "matched_to_clay_e7": bool(settings["epochs"] == 100 and sorted(int(s) for s in per_seed) == [0, 1, 2]),
        }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}: {len(arms)} arms")


if __name__ == "__main__":
    main()
