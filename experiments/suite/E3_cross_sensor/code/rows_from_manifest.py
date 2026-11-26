"""Build rows.json (loc_id, lat, lon, label) for evaluate.py from the shipped ../manifest.csv.
Usage: python rows_from_manifest.py ../manifest.csv rows.json"""
import csv, json, sys

src, dst = sys.argv[1:3]
rows = [{"loc_id": r["loc_id"], "lat": float(r["lat"]), "lon": float(r["lon"]), "label": int(r["label"])}
        for r in csv.DictReader(open(src, newline=""))]
json.dump(rows, open(dst, "w"))
print(len(rows), "rows ->", dst)
