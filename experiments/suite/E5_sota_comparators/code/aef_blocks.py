import geopandas as gpd, pandas as pd, numpy as np, time
from pyproj import Transformer
t=time.time()
idx=gpd.read_parquet("aef_index.parquet",columns=["path","year","crs","utm_west","utm_north","utm_south","utm_east","geom"])
P=pd.read_csv("points.csv"); g=gpd.GeoDataFrame(P,geometry=gpd.points_from_xy(P.x,P.y),crs=4326)
out=[]
for yr,gp in g.groupby("year"):
    j=gpd.sjoin(gp,idx[idx.year==yr].drop(columns="year").to_crs(4326),predicate="within",how="left")
    j=j[~j.index.duplicated()]
    out.append(j)
J=pd.concat(out); print("matched",J.path.notna().sum(),"of",len(J),round(time.time()-t,1),flush=True)
rows=[]
for crs,gc in J[J.path.notna()].groupby("crs"):
    X,Y=Transformer.from_crs(4326,crs,always_xy=True).transform(gc.x.values,gc.y.values)
    col=((X-gc.utm_west.values)//10).astype(int); row=((Y-gc.utm_south.values)//10).astype(int)  # south-up
    d=gc[["pid","path","year"]].copy(); d["col"]=col; d["row"]=row; rows.append(d)
D=pd.concat(rows); D["blk"]=D.path+"|"+(D.row//1024).astype(str)+"|"+(D.col//1024).astype(str)
print("unique files",D.path.nunique(),"unique blocks",D.blk.nunique(), "rows range", D.row.min(), D.row.max(), D.col.min(), D.col.max())
D.to_csv("aef_point_blocks.csv",index=False)
