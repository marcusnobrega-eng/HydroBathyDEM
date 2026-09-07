"""Fine terrain for the Pune sub-catchment, with the mapped channel burned in.

Pune's DEM is 30 m and its channels are 3-52 m wide (median 8.7 m), so the DEM
does not resolve them at all. Sub-grid detail for river cells therefore cannot
come from the DEM -- it comes from the mapped width / depth / bed-elevation
rasters, which is what HydroBathyDEM produces.

Output is a 5 m grid: the 30 m conditioned DEM resampled bilinearly, with the
channel incised to its mapped bed elevation over its mapped width. At 90 m mesh
cells that gives 18 sub-cells across each face and a channel about 2 pixels wide
-- enough for the face tables to see a channel that is 10% of a cell.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

D = Path("examples/pune_catchment/data/pune_caseA")
O = Path("examples/pune_subgrid")
FINE = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
PAD = 0                                            # 30 m cells of margin

bas = np.load(O/"_basin.npz")
mask30, rows, cols = bas["mask"], bas["rows"], bas["cols"]
r0, r1 = int(rows.min())-PAD, int(rows.max())+PAD+1
c0, c1 = int(cols.min())-PAD, int(cols.max())+PAD+1

def read(name):
    with rasterio.open(D/name) as s:
        return s.read(1, masked=True), s.transform, s.crs
dem30, tr30, crs = read("caseA_dem_hydraulic_conditioned_m.tif")
riv30, _, _ = read("caseA_d8_river_mask_2km2.tif")
wid30, _, _ = read("caseA_d8_river_width_m.tif")
dep30, _, _ = read("caseA_d8_river_depth_m.tif")
bed30, _, _ = read("caseA_d8_river_bed_elevation_m.tif")
urb30, _, _ = read("caseA_urban_mask.tif")

sub = (slice(r0, r1), slice(c0, c1))
dem = dem30[sub].filled(np.nan); m30 = mask30[sub]
riv = riv30[sub].filled(0).astype(bool) & m30
urb = urb30[sub].filled(0).astype(bool)
wid = wid30[sub].filled(np.nan); dep = dep30[sub].filled(np.nan); bedz = bed30[sub].filled(np.nan)
h30, w30 = dem.shape
x0 = tr30.c + c0*tr30.a
y0 = tr30.f + r0*tr30.e                     # tr30.e is negative
print(f"window {h30}x{w30} at 30 m   origin ({x0:.0f}, {y0:.0f})")

f = int(round(30.0/FINE))
hf, wf = h30*f, w30*f
trf = from_origin(x0, y0, FINE, FINE)
# bilinear resample of the terrain, nearest for the categorical layers
with rasterio.open(D/"caseA_dem_hydraulic_conditioned_m.tif") as s:
    fine = s.read(1, masked=True, window=((r0, r1), (c0, c1)),
                  out_shape=(hf, wf), resampling=Resampling.bilinear).filled(np.nan)
def up(a, order=0):
    return np.repeat(np.repeat(a, f, axis=0), f, axis=1)
maskf = up(m30); urbf = up(urb)
print(f"fine grid {hf}x{wf} at {FINE:.0f} m   valid {int(maskf.sum()):,} px "
      f"= {maskf.sum()*FINE*FINE/1e6:.2f} km2")

# ---- burn the mapped channel -------------------------------------------------
rr, cc = np.where(riv)
cx = x0 + (cc + 0.5)*30.0
cy = y0 - (rr + 0.5)*30.0
half0 = 0.5*wid[rr, cc]
bz0 = bedz[rr, cc]
# Densify along the CENTRELINE, not just at cell centres. The D8 river cells are
# 30 m apart, so testing distance to the centres alone burns a string of discs of
# radius ~4 m with 30 m gaps: it captured 0.107 km2 of a 0.378 km2 channel, 28%.
# Each cell is linked to its D8 receiver and the segment is sampled at FINE/2.
from dem_processing.hybrid_mesh import receiver_from_d8_direction
recv_full = receiver_from_d8_direction(
    np.asarray(rasterio.open(D/"caseA_d8_flow_direction.tif").read(1), dtype=np.int32))
H0, W0 = mask30.shape
flat_src = (rr + r0)*W0 + (cc + c0)
flat_dst = recv_full.reshape(-1)[flat_src]
px_l, py_l, half_l, bz_l = [], [], [], []
for k in range(len(rr)):
    xa, ya = cx[k], cy[k]
    d = flat_dst[k]
    if d >= 0:
        dr, dc = divmod(int(d), W0)
        xb = tr30.c + (dc + 0.5)*30.0
        yb = tr30.f + (dr + 0.5)*tr30.e - 0.5*tr30.e + 0.5*tr30.e
        yb = tr30.f + (dr + 0.5)*tr30.e
    else:
        xb, yb = xa, ya
    n = max(2, int(np.hypot(xb-xa, yb-ya)/(0.5*FINE)) + 1)
    px_l.append(np.linspace(xa, xb, n)); py_l.append(np.linspace(ya, yb, n))
    half_l.append(np.full(n, half0[k])); bz_l.append(np.full(n, bz0[k]))
lx = np.concatenate(px_l); ly = np.concatenate(py_l)
half = np.concatenate(half_l); bz = np.concatenate(bz_l)
tree = cKDTree(np.column_stack([lx, ly]))
print(f"centreline densified to {len(lx):,} points from {len(rr)} river cells")
gy, gx = np.mgrid[0:hf, 0:wf]
px = x0 + (gx.ravel() + 0.5)*FINE
py = y0 - (gy.ravel() + 0.5)*FINE
# a fine pixel belongs to the channel if it lies within half-width of the nearest
# centreline point; search radius is the widest channel plus one 30 m cell
dist, idx = tree.query(np.column_stack([px, py]), distance_upper_bound=float(half.max())+30.0)
inb = np.isfinite(dist) & (idx < len(half))
chan = np.zeros(px.size, dtype=bool)
chan[inb] = dist[inb] <= half[idx[inb]]
burn = np.full(px.size, np.nan)
burn[chan] = bz[idx[chan]]
chan = chan.reshape(hf, wf); burn = burn.reshape(hf, wf)
before = np.nanmean(fine[chan])
fine = np.where(chan & np.isfinite(burn), np.minimum(fine, burn), fine)
print(f"channel burned: {int(chan.sum()):,} fine px = "
      f"{chan.sum()*FINE*FINE/1e6:.3f} km2; mean bed drop "
      f"{before-np.nanmean(fine[chan]):.2f} m")

fine = np.where(maskf, fine, np.nan)
man = np.where(chan, 0.035, np.where(urbf, 0.015, 0.060))
lulc = np.where(chan, 3.0, np.where(urbf, 2.0, 1.0))
def w(name, a, dtype="float64", nodata=-9999.0):
    aa = np.where(np.isfinite(a), a, nodata).astype(dtype)
    with rasterio.open(O/name, "w", driver="GTiff", height=hf, width=wf, count=1,
                       dtype=dtype, crs=crs, transform=trf, nodata=nodata,
                       compress="deflate") as d:
        d.write(aa, 1)
w(f"dem_{int(FINE)}m.tif", fine)
w(f"manning_{int(FINE)}m.tif", np.where(maskf, man, np.nan))
w(f"lulc_{int(FINE)}m.tif", np.where(maskf, lulc, np.nan))
w(f"soil_{int(FINE)}m.tif", np.where(maskf, 1.0, np.nan))
w(f"channel_{int(FINE)}m.tif", np.where(maskf, chan.astype(float), np.nan))
print(f"elevation {np.nanmin(fine):.2f}-{np.nanmax(fine):.2f} m")
print(f"manning: channel 0.035, urban 0.015, rural 0.060")
np.savez(O/f"_fine_{int(FINE)}m.npz", origin=np.array([x0, y0]), shape=np.array([hf, wf]),
         res=FINE, crs_wkt=crs.to_wkt())
print("wrote", O)
