"""Sub-grid tables for the Pune sub-catchment mesh."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, rasterio, netCDF4
from dem_processing.subgrid_tables import (
    build_cell_volume_table, build_face_conveyance_table, subgrid_cell_policy)

O = Path("examples/pune_subgrid")
FINE = 5
CELL = float(next((a for a in sys.argv[1:] if not a.startswith("--")), 90.0))
POLICY = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--policy=")), "river")
TAG = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--tag=")),
           f"{int(CELL)}m")
M = np.load(O/f"_mesh_{TAG}.npz", allow_pickle=True)
fclass = M["fclass"]

with rasterio.open(O/f"dem_{FINE}m.tif") as s:
    dem = s.read(1, masked=True).filled(np.nan); tr = s.transform
with rasterio.open(O/f"manning_{FINE}m.tif") as s:
    man = s.read(1, masked=True).filled(np.nan)
h, w = dem.shape
with netCDF4.Dataset(O/f"mesh_{TAG}.nc") as ds:
    e_own = np.asarray(ds["edge_owner"][:]).reshape(-1)
    e_nb = np.asarray(ds["edge_neighbor"][:]).reshape(-1)
    e_len = np.asarray(ds["edge_length_m"][:]).reshape(-1)
    mx = np.asarray(ds["edge_midpoint_x"][:]).reshape(-1)
    my = np.asarray(ds["edge_midpoint_y"][:]).reshape(-1)
    nx_ = np.asarray(ds["edge_normal_x"][:]).reshape(-1)
    ny_ = np.asarray(ds["edge_normal_y"][:]).reshape(-1)
    ca = np.asarray(ds["cell_area_m2"][:]).reshape(-1)
with netCDF4.Dataset(O/f"overlap_{TAG}.nc") as ds:
    oc = np.asarray(ds["overlap_mesh_index"][:]).reshape(-1)
    oi = np.asarray(ds["overlap_raster_index"][:]).reshape(-1)
    oa = np.asarray(ds["overlap_area_m2"][:]).reshape(-1)
N = len(ca)
mask = subgrid_cell_policy(fclass, POLICY)
print(f"policy '{POLICY}': level-pool closure on {int(mask.sum()):,} of {N:,} cells")

dsu = np.flipud(dem).reshape(-1); msu = np.flipud(man).reshape(-1)
zp = dsu[oi]
good = np.isfinite(zp) & (oa > 0)
t0 = time.perf_counter()
tabc = build_cell_volume_table(oc[good], oa[good], zp[good], N, ca,
                               level_pool_mask=mask, cell_bed_mean_m=M["bed"])
print(f"cell tables: {tabc.zeta_m.shape[1]} points max, {time.perf_counter()-t0:.1f}s")

# A face's tables are only ever read when BOTH its cells are sub-grid, so only
# those get a full profile.  The rest get a degenerate 2-point table.  On this mesh
# that is a few hundred faces instead of 25,879.
need = np.zeros(len(e_own), dtype=bool)
internal = e_nb >= 0
need[internal] = mask[e_own[internal]] & mask[e_nb[internal]]
print(f"faces needing a full profile: {int(need.sum()):,} of {len(e_own):,}")
inv = ~tr
def samp(a, X, Y):
    c_, r_ = inv * (X, Y)
    return a[np.clip(np.floor(r_).astype(int), 0, h-1), np.clip(np.floor(c_).astype(int), 0, w-1)]
OFF = 0.5*FINE
floor = np.where(internal, np.minimum(tabc.datum_m[e_own], tabc.datum_m[np.maximum(e_nb, 0)]),
                 tabc.datum_m[e_own])
t0 = time.perf_counter()
profiles = []
for i in range(len(e_own)):
    if not need[i]:
        profiles.append((np.array([0.0, e_len[i]]), np.full(2, floor[i]), np.full(2, 0.05)))
        continue
    L = float(e_len[i]); npts = max(3, int(np.ceil(L/FINE)) + 1)
    s_ = np.linspace(-0.5*L, 0.5*L, npts); tx, ty = -ny_[i], nx_[i]
    Xa, Ya = mx[i]+s_*tx-OFF*nx_[i], my[i]+s_*ty-OFF*ny_[i]
    Xb, Yb = mx[i]+s_*tx+OFF*nx_[i], my[i]+s_*ty+OFF*ny_[i]
    za, zb = samp(dem, Xa, Ya), samp(dem, Xb, Yb)
    za = np.where(np.isfinite(za), za, zb); zb = np.where(np.isfinite(zb), zb, za)
    z = np.maximum(za, zb)
    z = np.where(np.isfinite(z), z, floor[i])
    nn = 0.5*(samp(man, Xa, Ya) + samp(man, Xb, Yb))
    nn = np.where(np.isfinite(nn) & (nn > 0), nn, 0.05)
    profiles.append((s_-s_[0], np.maximum(z, floor[i]), nn))
tabf = build_face_conveyance_table(profiles, e_len)
print(f"face tables: {tabf.zeta_m.shape[1]} points max, {time.perf_counter()-t0:.1f}s")
low = np.minimum(tabc.datum_m[e_own], tabc.datum_m[np.maximum(e_nb, 0)])
bad = int(((low - tabf.datum_m) > 1e-6)[internal].sum())
print(f"cell/face datum invariant violations: {bad}")

out = O/f"subgrid_{TAG}_{POLICY}.nc"
with netCDF4.Dataset(out, "w") as o:
    o.createDimension("cell", N); o.createDimension("cell_pt", tabc.zeta_m.shape[1])
    o.createDimension("face", len(e_own)); o.createDimension("face_pt", tabf.zeta_m.shape[1])
    def v(nm, d, a):
        o.createVariable(nm, "f8" if a.dtype.kind == "f" else "i8", d, zlib=True)[:] = a
    v("cell_datum_m",("cell",),tabc.datum_m); v("cell_zeta_m",("cell","cell_pt"),tabc.zeta_m)
    v("cell_volume_m3",("cell","cell_pt"),tabc.volume_m3)
    v("cell_wet_area_m2",("cell","cell_pt"),tabc.area_m2)
    v("cell_point_count",("cell",),tabc.count); v("cell_plan_area_m2",("cell",),tabc.plan_area_m2)
    v("cell_is_subgrid",("cell",),mask.astype(np.int64))
    v("face_datum_m",("face",),tabf.datum_m); v("face_zeta_m",("face","face_pt"),tabf.zeta_m)
    v("face_flow_area_m2",("face","face_pt"),tabf.area_m2)
    v("face_perimeter_m",("face","face_pt"),tabf.perimeter_m)
    v("face_conveyance",("face","face_pt"),tabf.conveyance)
    v("face_point_count",("face",),tabf.count); v("face_length_m",("face",),tabf.length_m)
    o.setncattr("policy", POLICY)
print("wrote", out)
