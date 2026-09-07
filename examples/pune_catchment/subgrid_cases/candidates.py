import os, json
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import rasterize
from scipy import ndimage
import common as C

shape, prof, recv, level, order, accum, basin = C.topology()
rows, cols = shape
n = rows * cols
A = accum * C.CELL_KM2

# ---- ancillary masks on the shared grid ----
imp, _ = C.read("imp")
core = (imp == 100)
# 300 m buffer == 10 cells; euclidean distance dilation
dist = ndimage.distance_transform_edt(~core, sampling=C.CELL)
urban = core | (dist <= 300.0)
fp, _ = C.read("fp"); fp = fp.astype(bool)
rmask, _ = C.read("rmask"); rmask = rmask.astype(bool)
wb = gpd.read_file(C.P["wb"], layer="waterbody_refinement")
wbr = rasterize(((g, i + 1) for i, g in enumerate(wb.geometry)), out_shape=shape,
                transform=prof["transform"], fill=0, dtype="int32", all_touched=False)
dom = gpd.read_file(C.P["domain"])
domr = rasterize([(g, 1) for g in dom.geometry], out_shape=shape,
                 transform=prof["transform"], fill=0, dtype="uint8").astype(bool)
print("urban km2", urban.sum()*C.CELL_KM2, "core km2", core.sum()*C.CELL_KM2,
      "fp km2", fp.sum()*C.CELL_KM2, "wb km2", (wbr>0).sum()*C.CELL_KM2,
      "domain km2", domr.sum()*C.CELL_KM2)
np.savez_compressed(os.path.join(C.CACHE, "masks.npz"),
                    urban=urban, fp=fp, rmask=rmask, wbr=wbr, dom=domr)

# ---- maximal sub-basin outlets in the 150-300 km2 band ----
Ad = np.where(recv >= 0, A[np.maximum(recv, 0)], np.inf)
cand = np.flatnonzero((A >= 150.0) & (A <= 300.0) & (Ad > 300.0))
print("candidate outlets:", cand.size)

# one downstream-first sweep labels every candidate basin at once
lab = np.full(n, -1, dtype=np.int32)
lab[cand] = np.arange(cand.size, dtype=np.int32)
bnd = np.searchsorted(level[order], np.arange(level.max() + 2))
for L in range(level.max(), -1, -1):
    sel = order[bnd[L]:bnd[L + 1]]
    d = recv[sel]
    ok = (d >= 0) & (lab[sel] < 0)
    s = sel[ok]
    lab[s] = lab[d[ok]]

urf, fpf, rmf = urban.ravel(), fp.ravel(), rmask.ravel()
wbf = wbr.ravel()
rowsi, colsi = np.divmod(np.arange(n), cols)
recs = []
for k, out in enumerate(cand):
    m = lab == k
    cnt = int(m.sum())
    a_km2 = cnt * C.CELL_KM2
    if abs(a_km2 - A[out]) > 1e-6:
        raise RuntimeError("basin area mismatch")
    wbids = np.unique(wbf[m]); wbids = wbids[wbids > 0]
    recs.append(dict(
        k=k, outlet=int(out), row=int(rowsi[out]), col=int(colsi[out]),
        area_km2=round(a_km2, 3),
        urban_pct=round(100 * urf[m].sum() / cnt, 2),
        fp_pct=round(100 * fpf[m].sum() / cnt, 2),
        river_km=round(rmf[m].sum() * C.CELL / 1000.0, 1),
        wb_pct=round(100 * (wbf[m] > 0).sum() / cnt, 2),
        wb_ids=[int(wb.waterbody_id.iloc[i - 1]) for i in wbids],
        wb_types=[str(wb.waterbody_type.iloc[i - 1]) for i in wbids],
    ))
recs.sort(key=lambda r: -r["area_km2"])
json.dump(recs, open(os.path.join(C.CACHE, "candidates.json"), "w"), indent=1)
hdr = f"{'k':>4} {'area':>8} {'urban%':>7} {'fp%':>6} {'riv_km':>7} {'wb%':>6} {'row':>5} {'col':>5}  wb"
print(hdr)
for r in recs:
    print(f"{r['k']:>4} {r['area_km2']:>8.1f} {r['urban_pct']:>7.1f} {r['fp_pct']:>6.2f} "
          f"{r['river_km']:>7.1f} {r['wb_pct']:>6.2f} {r['row']:>5} {r['col']:>5}  {r['wb_ids']} {r['wb_types']}")
np.save(os.path.join(C.CACHE, "cand_labels.npy"), lab)
np.save(os.path.join(C.CACHE, "cand_outlets.npy"), cand)
