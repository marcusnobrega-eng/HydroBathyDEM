"""A 60 m Voronoi mesh over the real Di Giammarco domain, plus sub-grid tables.

The 20 m benchmark grid is the reference: it is the published configuration, and
it has been validated against the exact equilibrium bound (4.860 m3/s).  Nothing
here invents a finer grid -- the sub-grid tables are built from the SAME 20 m DEM
the benchmark is defined on, which is the honest fine-scale information available.

At 60 m the 20 m channel is 1/3 of a cell: the situation sub-grid exists for.
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, geopandas as gpd, rasterio, netCDF4
from shapely import MultiPoint, STRtree, intersection, voronoi_polygons
from shapely.geometry import Polygon, box
from dem_processing.hybrid_mesh import _topology, _write_ugrid
from dem_processing.mesh_product import write_conservative_overlap
from dem_processing.subgrid_tables import build_cell_volume_table, build_face_conveyance_table

OUT = Path("examples/digiammarco")
B = json.load(open(OUT/"benchmark.json"))
CELL = float([a for a in sys.argv[1:] if not a.startswith("--")][0]) \
    if [a for a in sys.argv[1:] if not a.startswith("--")] else 60.0
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
DEMRES = int(POS[1]) if len(POS) > 1 else 20
TAG = f"{int(CELL)}m_d{DEMRES}"
LX, LY = 1000.0, B["nrow"]*B["dx"]
with rasterio.open(OUT/f"dem_{DEMRES}m.tif") as s: dem, tr, crs = s.read(1), s.transform, s.crs
with rasterio.open(OUT/f"manning_{DEMRES}m.tif") as s: man = s.read(1)

domain = box(0.0, -LY/2, LX, LY/2)
gx = np.arange(CELL/2, LX, CELL)
gy = np.arange(-LY/2 + CELL/2, LY/2, CELL)
seeds = np.column_stack([a.ravel() for a in np.meshgrid(gx, gy, indexing="ij")])
cells = [intersection(c, domain)
         for c in voronoi_polygons(MultiPoint(seeds), extend_to=domain).geoms]
cells = [c for c in cells if c.geom_type == "Polygon" and c.area > 1e-9]
faces, edges = _topology(cells)
cells = [Polygon(f) for f in faces]
n = len(cells)

# area-averaged bed and roughness from the 20 m raster
tree = STRtree(cells)
h, w = dem.shape
rows, cols = np.mgrid[0:h, 0:w]
xs = tr.c + (cols.ravel()+0.5)*tr.a
ys = tr.f + (rows.ravel()+0.5)*tr.e
own = tree.query(gpd.points_from_xy(xs, ys), predicate="within")
bed = np.zeros(n); rough = np.zeros(n); cnt = np.zeros(n)
np.add.at(bed, own[1], dem.ravel()[own[0]])
np.add.at(rough, own[1], man.ravel()[own[0]])
np.add.at(cnt, own[1], 1.0)
ok = cnt > 0
bed[ok] /= cnt[ok]; rough[ok] /= cnt[ok]
areas = np.array([c.area for c in cells]); per = np.array([c.length for c in cells])
target = np.full(n, CELL)
gpkg = OUT/f"coarse_{TAG}.gpkg"
for s_ in (gpkg, Path(f"{gpkg}-wal"), Path(f"{gpkg}-shm")): s_.unlink(missing_ok=True)
gpd.GeoDataFrame({"face_id":range(n),"area_m2":areas,"feature_class":["rural"]*n,
                  "bed_m":bed,"manning":rough}, geometry=cells, crs=crs
                 ).to_file(gpkg, layer="mesh", driver="GPKG")
nc = OUT/f"coarse_{TAG}.nc"
rep = _write_ugrid(nc, faces, edges, cells, crs, target, tr, dem.astype(float),
                   ["rural"]*n, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
                   cell_target_width=target)
with netCDF4.Dataset(nc, "a") as ds:
    ds["cell_bed_elevation_m"][:] = bed
    ds["cell_hydraulic_roughness"][:] = rough
ov = write_conservative_overlap(nc, gpkg, OUT/f"dem_{DEMRES}m.tif", OUT/f"coarse_overlap_{TAG}.nc")
print(f"mesh {n} cells at {CELL:.0f} m   A/P min {np.min(areas/per):.2f} m   faces {len(edges)}")
print(f"bed {bed.min():.2f}-{bed.max():.2f} m   manning {rough.min():.4f}-{rough.max():.4f}")
print(f"overlap {ov['overlap_count']:,} pairs   area err {ov['maximum_mesh_area_relative_error']:.2e}")

# ---- sub-grid tables from the SAME 20 m data -----------------------------
with netCDF4.Dataset(nc) as ds:
    e_own = np.asarray(ds["edge_owner"][:]).reshape(-1)
    e_nb  = np.asarray(ds["edge_neighbor"][:]).reshape(-1)
    e_len = np.asarray(ds["edge_length_m"][:]).reshape(-1)
    mx = np.asarray(ds["edge_midpoint_x"][:]).reshape(-1)
    my = np.asarray(ds["edge_midpoint_y"][:]).reshape(-1)
    nx_ = np.asarray(ds["edge_normal_x"][:]).reshape(-1)
    ny_ = np.asarray(ds["edge_normal_y"][:]).reshape(-1)
    cell_area = np.asarray(ds["cell_area_m2"][:]).reshape(-1)
with netCDF4.Dataset(OUT/f"coarse_overlap_{TAG}.nc") as ds:
    oc = np.asarray(ds["overlap_mesh_index"][:]).reshape(-1)
    oi = np.asarray(ds["overlap_raster_index"][:]).reshape(-1)
    oa = np.asarray(ds["overlap_area_m2"][:]).reshape(-1)
dem_su, man_su = np.flipud(dem).reshape(-1), np.flipud(man).reshape(-1)
# Gate the level-pool closure. A cell only gets an elevation-volume curve if its
# terrain is genuinely NOT planar -- if a fitted plane already explains its whole
# relief, the water inside is a sloping sheet, not a pool, and a level-pool table
# over-deepens it (72x on this geometry).  The split is bimodal with an empty band
# between 0.05 and 0.20, so the threshold is not a tuned knob.
from dem_processing.subgrid_tables import _fit_bed_planes
POOL_RATIO_MIN = 0.20
nr_, nc_ = dem.shape
r_, c_ = np.divmod(oi, nc_)
ymin_ = tr.f + tr.e * nr_
px_ = tr.c + (c_ + 0.5) * tr.a
py_ = ymin_ + (r_ + 0.5) * abs(tr.e)
zpatch = dem_su[oi]
plane_ = _fit_bed_planes(oc, oa, zpatch, px_, py_, n)
resid_ = zpatch - plane_
ordr = np.argsort(oc, kind="stable"); cnt_ = np.bincount(oc, minlength=n)
st_ = np.r_[0, np.cumsum(cnt_)[:-1]]
def _pc(v, fn):
    w = v[ordr]
    return np.array([fn(w[st_[i]:st_[i]+cnt_[i]]) if cnt_[i] else 0.0 for i in range(n)])
rng_tot = _pc(zpatch, lambda a_: a_.max()-a_.min())
rng_res = _pc(resid_, lambda a_: a_.max()-a_.min())
pool_ratio = np.divide(rng_res, rng_tot, out=np.zeros(n), where=rng_tot > 1e-9)
# HEC-RAS puts an elevation-volume curve on EVERY cell.  No gating: that was an
# invention of mine and it is datum-inconsistent with the face tables.
print(f"level-pool closure on all {n} cells (HEC-RAS behaviour, no gating)")
tabc = build_cell_volume_table(oc, oa, dem_su[oi], n, cell_area)
inv = ~tr
def sample(a, X, Y):
    c_, r_ = inv * (X, Y)
    r_ = np.clip(np.floor(r_).astype(int), 0, dem.shape[0]-1)
    c_ = np.clip(np.floor(c_).astype(int), 0, dem.shape[1]-1)
    return a[r_, c_]
# A face lies ON the cell boundary, so it belongs to BOTH cells' footprints and its
# low point can never be below either cell's own low point.  Sampling the DEM
# directly on the boundary line breaks that: nearest-neighbour picks up whichever
# pixel is closer, including the downslope neighbour's.  Measured on these meshes,
# 208-3920 faces per resolution had a datum BELOW both adjacent cells, by up to
# 1.0 m, so a barely-wet cell handed a metre of phantom head to half its faces --
# and the sub-grid arm then failed to converge to the flat-prism answer even at
# 20 m, where one DEM pixel per cell makes the two identical by construction.
#
# Fix: sample HALF A PIXEL into each cell and take the per-station MAX.  Water
# crossing the face at that station must clear the higher of the two sides, so the
# max is the controlling sill there.  This keeps the along-face variation (unlike
# raising the whole profile to one constant, which turns the face into a weir and
# inflated conveyance 57x) while guaranteeing
#     face_datum >= min(cell_datum_owner, cell_datum_neighbour).
OFFSET = 0.5 * abs(tr.a)
profiles = []
for i in range(len(e_own)):
    L = float(e_len[i]); npts = max(3, int(np.ceil(L/abs(tr.a)))+1)
    s_ = np.linspace(-0.5*L, 0.5*L, npts)
    tx, ty = -ny_[i], nx_[i]
    Xa, Ya = mx[i] + s_*tx - OFFSET*nx_[i], my[i] + s_*ty - OFFSET*ny_[i]
    Xb, Yb = mx[i] + s_*tx + OFFSET*nx_[i], my[i] + s_*ty + OFFSET*ny_[i]
    za, zb = sample(dem, Xa, Ya), sample(dem, Xb, Yb)
    na_, nb_ = sample(man, Xa, Ya), sample(man, Xb, Yb)
    za = np.where(np.isfinite(za), za, zb); zb = np.where(np.isfinite(zb), zb, za)
    z = np.maximum(za, zb)
    nn = 0.5*(np.where(np.isfinite(na_)&(na_>0), na_, 0.015)
              + np.where(np.isfinite(nb_)&(nb_>0), nb_, 0.015))
    profiles.append((s_-s_[0], np.where(np.isfinite(z), z, 0.0), nn))
# Residual guard for the invariant. Offsetting half a pixel into each cell can
# still land outside its own cell near a corner, so ~25% of faces still dipped
# below both neighbours. Clamp each face profile from below at the MINIMUM of the
# two cell datums. This is not the earlier weir bug: that clamped at the MAXIMUM,
# flattening the whole face into a sill (57x conveyance). Clamping at the minimum
# only lifts terrain that lies below BOTH cells -- exactly the unphysical part --
# and leaves every real along-face variation intact.
floor = np.minimum(tabc.datum_m[e_own], tabc.datum_m[np.maximum(e_nb, 0)])
floor = np.where(e_nb >= 0, floor, tabc.datum_m[e_own])
profiles = [(s_, np.maximum(z_, floor[i]), n_)
            for i, (s_, z_, n_) in enumerate(profiles)]
tabf = build_face_conveyance_table(profiles, e_len)
with netCDF4.Dataset(OUT/f"coarse_subgrid_{TAG}.nc","w") as o:
    o.createDimension("cell", n); o.createDimension("cell_pt", tabc.zeta_m.shape[1])
    o.createDimension("face", len(e_own)); o.createDimension("face_pt", tabf.zeta_m.shape[1])
    def v(nm, d, a):
        o.createVariable(nm, "f8" if a.dtype.kind=="f" else "i8", d)[:] = a
    v("cell_datum_m",("cell",),tabc.datum_m); v("cell_zeta_m",("cell","cell_pt"),tabc.zeta_m)
    v("cell_volume_m3",("cell","cell_pt"),tabc.volume_m3)
    v("cell_wet_area_m2",("cell","cell_pt"),tabc.area_m2)
    v("cell_point_count",("cell",),tabc.count); v("cell_plan_area_m2",("cell",),tabc.plan_area_m2)
    v("face_datum_m",("face",),tabf.datum_m); v("face_zeta_m",("face","face_pt"),tabf.zeta_m)
    v("face_flow_area_m2",("face","face_pt"),tabf.area_m2)
    v("face_perimeter_m",("face","face_pt"),tabf.perimeter_m)
    v("face_conveyance",("face","face_pt"),tabf.conveyance)
    v("face_point_count",("face",),tabf.count); v("face_length_m",("face",),tabf.length_m)
print(f"sub-grid tables: {n} cells x{tabc.zeta_m.shape[1]}, {len(e_own)} faces x{tabf.zeta_m.shape[1]}")
pickle.dump(cells, open(OUT/f"_cells_{TAG}.pkl","wb"))
print("wrote", OUT)
