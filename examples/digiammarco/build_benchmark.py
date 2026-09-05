"""The actual Di Giammarco et al. (1996) V-catchment, at its published resolution.

Di Giammarco, Todini & Consuegra (1996), "A combined finite element - finite
volume formulation for solving the shallow water equations", and used since as a
standard 2-D overland flow test (Neal et al. 2012, and the local-inertial
literature generally).

Published configuration -- NOT a lookalike:
  * two planes, each 800 m wide x 1000 m long, cross slope 0.05, along slope 0.02
  * a channel 1000 m long and 20 m wide, slope 0.02, in the middle
  * total width 2 x 800 + 20 = 1620 m; total area 1.620 km2
  * rainfall 10.8 mm/h (3e-6 m/s) for 90 min, then nothing
  * Manning n = 0.015 on the planes, 0.15 in the channel
  * computational grid 20 m

Two hard checks come free with this geometry:
  * equilibrium outflow = 1.620e6 m2 * 3e-6 m/s = 4.860 m3/s, and peak outflow can
    never exceed it, because outflow cannot exceed rainfall while storage is still
    filling and only decays afterwards;
  * the published peak is ~4.5 m3/s, reached near the end of the 90 min of rain.
An arm that peaks above 4.86, or long before 5400 s, is wrong regardless of how
good its mass balance looks.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, rasterio
from rasterio.transform import from_origin

OUT = Path("examples/digiammarco"); OUT.mkdir(parents=True, exist_ok=True)
CRS = 'LOCAL_CS["DiGiammarco",UNIT["metre",1]]'
DX = 20.0                       # the published grid
LX, HALF_W, CHAN_HALF = 1000.0, 800.0, 10.0
LY = 2 * (HALF_W + CHAN_HALF)   # 1620 m
CHAN_SLOPE, PLANE_SLOPE = 0.02, 0.05
N_PLANE, N_CHAN = 0.015, 0.15
RAIN_MM_H, RAIN_S = 10.8, 5400.0

def bed(x, y):
    return CHAN_SLOPE * (LX - x) + PLANE_SLOPE * np.maximum(np.abs(y) - CHAN_HALF, 0.0)

ncol, nrow = int(round(LX/DX)), int(round(LY/DX))
xs = (np.arange(ncol) + 0.5) * DX
ys = LY/2 - (np.arange(nrow) + 0.5) * DX
XX, YY = np.meshgrid(xs, ys)
dem = bed(XX, YY)
man = np.where(np.abs(YY) <= CHAN_HALF, N_CHAN, N_PLANE)
tr = from_origin(0.0, LY/2, DX, DX)

def write(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float64", crs=CRS, transform=tr, nodata=-9999.0) as d:
        d.write(np.asarray(arr, dtype="float64"), 1)

write(OUT/"dem_20m.tif", dem)
write(OUT/"manning_20m.tif", man)
write(OUT/"lulc_20m.tif", np.ones_like(dem))
write(OUT/"soil_20m.tif", np.ones_like(dem))

area = LX * LY
equil = area * RAIN_MM_H/1000/3600
chan_rows = np.flatnonzero(np.abs(ys) <= CHAN_HALF)
print(f"grid {nrow} x {ncol} at {DX:.0f} m   area {area/1e6:.3f} km2")
print(f"elevation {dem.min():.2f}-{dem.max():.2f} m")
print(f"channel occupies rows {chan_rows.tolist()} ({len(chan_rows)*DX:.0f} m wide)")
print(f"manning {man.min():.3f} / {man.max():.3f}")
print(f"EQUILIBRIUM OUTFLOW = {equil:.4f} m3/s  (hard upper bound on the peak)")
print(f"published peak ~4.5 m3/s near t = {RAIN_S:.0f} s")
json.dump({"dx":DX,"nrow":nrow,"ncol":ncol,"area_m2":area,"equilibrium_m3_s":equil,
           "chan_rows":chan_rows.tolist(),"rain_mm_h":RAIN_MM_H,"rain_s":RAIN_S,
           "n_plane":N_PLANE,"n_chan":N_CHAN},
          open(OUT/"benchmark.json","w"), indent=2)
print("wrote", OUT)
