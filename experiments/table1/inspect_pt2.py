#!/usr/bin/env python3
"""Download the compiled Clay v1.5 encoders from the Clay-legacy HF repo, hash them, and read the
dynamic-shape constraints and input shapes straight from the .pt2 zip (no torch needed; torch>=2.7 cannot
load these schema-5.1 archives, so we parse serialized_exported_program.json).

Usage: python experiments/table1/inspect_pt2.py [--out experiments/table1/pt2_inspection.json] [--workdir DIR]
Writes sha256, byte size, range constraints (symbols s0=batch, s1=channels) and user-input shapes.
"""
import argparse, hashlib, json, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "suite" / "common"))
from provenance import provenance  # noqa: E402

REPO, REV = "made-with-clay/Clay-legacy", "f521faece53b6b15df4deb2fc7215a03e561fb46"
FILES = ["v1.5/compiled/clay-v1.5-encoder-cpu.pt2", "v1.5/compiled/clay-v1.5-encoder.pt2"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "experiments/table1/pt2_inspection.json"))
ap.add_argument("--workdir", default="pt2_download")
a = ap.parse_args()
wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
res = {}
for f in FILES:
    p = wd / Path(f).name
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REV}/{f}") as r, open(p, "wb") as o:
        while chunk := r.read(1 << 24):
            o.write(chunk); h.update(chunk); n += len(chunk)
    d = json.loads(zipfile.ZipFile(p).read("serialized_exported_program.json"))
    g = d["graph_module"]["graph"]; tv = g["tensor_values"]
    shapes = {}
    for i in g["inputs"]:
        t = i.get("as_tensor")
        if t and t["name"] in ("arg265_1", "arg266_1", "arg267_1", "arg268_1", "arg269_1"):
            shapes[t["name"]] = [x.get("as_int", x.get("as_expr", {}).get("expr_str")) for x in tv[t["name"]]["sizes"]]
    res[f] = {"repo": REPO, "revision": REV, "bytes": n, "sha256": h.hexdigest(),
              "range_constraints": d["range_constraints"], "schema_version": d["schema_version"],
              "last_five_graph_inputs_shapes(pixels,time,latlon,waves,gsd)": shapes}
    print(f, n, h.hexdigest(), d["range_constraints"], shapes, flush=True)
    p.unlink()
res["provenance"] = provenance(__file__, {"workdir": a.workdir}, ROOT)
Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
