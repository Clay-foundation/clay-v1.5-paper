"""Pick one S2 MGRS tile per state (CA, TX, NC) by USPVDB PV area (p_year<=2022); pick least-cloudy JJA-2023 scene.
Uses only labels + scene metadata (no model scores)."""
import json, os, glob, pyproj
os.environ.setdefault("PROJ_DATA", pyproj.datadir.get_data_dir())
import geopandas as gpd, numpy as np, pystac_client, planetary_computer
from shapely.geometry import shape
g = gpd.read_file(glob.glob("data/*.geojson")[0])
g = g[g.p_year <= 2022].copy()
g["area_m2"] = g.to_crs(5070).area
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
out = {}
for st in ["CA", "TX", "NC"]:
    s = g[g.p_state == st]
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=list(s.total_bounds), datetime="2023-06-01/2023-08-31",
                            query={"eo:cloud_cover": {"lt": 10}}).items())
    tiles = {}
    for it in items:
        t = it.properties["s2:mgrs_tile"]; tiles.setdefault(t, []).append(it)
    rows = []
    for t, its in tiles.items():
        full = [i for i in its if i.properties.get("s2:nodata_pixel_percentage", 100) < 1]
        if not full: continue
        geom = shape(full[0].geometry)
        inside = s[s.geometry.centroid.within(geom)]
        best = min(full, key=lambda i: i.properties["eo:cloud_cover"])
        rows.append(dict(tile=t, n_sites=int(len(inside)), pv_km2=float(inside.area_m2.sum() / 1e6), item=best.id,
                         cloud=best.properties["eo:cloud_cover"], nodata=best.properties["s2:nodata_pixel_percentage"],
                         baseline=best.properties.get("s2:processing_baseline"), datetime=best.properties["datetime"],
                         epsg=best.properties.get("proj:epsg")))
    rows.sort(key=lambda r: -r["pv_km2"])
    out[st] = rows[:5]
    print(st, len(items), json.dumps(rows[:5], indent=0))
json.dump(out, open("region_candidates.json", "w"), indent=1)
