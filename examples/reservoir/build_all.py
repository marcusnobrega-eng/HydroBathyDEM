"""Reservoir case, rebuilt from scratch with the corrected sub-grid pipeline.

Terrain at 5 m, mesh at 60 m: 144 sub-cells per cell, 12 across each face. Lower
than HEC-RAS practice (25-50x) but the basin is a smooth bowl, so the shoreline --
the feature sub-grid exists to capture here -- is well resolved.

Sub-grid tables are applied ONLY to cells that genuinely pond. Selection is
measured, not assumed: the fraction of a cell's terrain relief left over after
removing an area-weighted least-squares plane. A cell sitting on the plain is
planar, its residual is ~0, and level pool is the wrong closure there. A cell the
shoreline cuts, or one inside the bowl, has a large residual.

`cell_is_subgrid` is written to the table file so the solver can switch the FACE
closure too. Switching only the cell closure is what broke the earlier attempt.
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, geopandas as gpd, rasterio, netCDF4
from rasterio.crs import CRS as RCRS
from rasterio.transform import from_origin
from shapely import MultiPoint, STRtree, intersection, voronoi_polygons
from shapely.geometry import Polygon, box
from dem_processing.hybrid_mesh import _topology, _write_ugrid
from dem_processing.mesh_product import write_conservative_overlap
from dem_processing.subgrid_tables import (
    build_cell_volume_table, build_face_conveyance_table, _fit_bed_planes)

OUT = Path("examples/reservoir"); OUT.mkdir(parents=True, exist_ok=True)
CRS = 'LOCAL_CS["Reservoir",UNIT["metre",1]]'
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
DX, CELL = 5.0, (float(POS[0]) if POS else 60.0)
LX = LY = 600.0
BX, BY, BR, BD, SLOPE = 300.0, 300.0, 75.0, 3.0, 0.003
POOL_RATIO_MIN = 0.20
ALL = "--all" in sys.argv   # HEC-RAS behaviour: tables on every cell

def bed(x, y):
    r = np.hypot(x - BX, y - BY)
    return 10.0 + SLOPE * (LX - x) - BD * np.maximum(1.0 - (r / BR) ** 2, 0.0)

n = int(LX / DX)
xs = (np.arange(n) + 0.5) * DX; ys = LY - (np.arange(n) + 0.5) * DX
XX, YY = np.meshgrid(xs, ys)
dem = bed(XX, YY)
man = np.where(np.hypot(XX - BX, YY - BY) <= BR, 0.030, 0.050)
tr = from_origin(0.0, LY, DX, DX)
def w(p, a):
    with rasterio.open(p, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
                       count=1, dtype="float64", crs=CRS, transform=tr, nodata=-9999.0) as d:
        d.write(a.astype("float64"), 1)
w(OUT/"dem.tif", dem); w(OUT/"manning.tif", man)
w(OUT/"lulc.tif", np.where(man < 0.04, 2.0, 1.0)); w(OUT/"soil.tif", np.ones_like(dem))
print(f"terrain {dem.shape[0]}x{dem.shape[1]} at {DX:.0f} m   elev {dem.min():.2f}-{dem.max():.2f} m")

domain = box(0.0, 0.0, LX, LY)
g = np.arange(CELL/2, LX, CELL)
seeds = np.column_stack([a.ravel() for a in np.meshgrid(g, g, indexing="ij")])
cells = [intersection(c, domain)
         for c in voronoi_polygons(MultiPoint(seeds), extend_to=domain).geoms]
cells = [c for c in cells if c.geom_type == "Polygon" and c.area > 1e-9]
faces, edges = _topology(cells); cells = [Polygon(f) for f in faces]
N = len(cells)
tree = STRtree(cells)
rows, cols = np.mgrid[0:dem.shape[0], 0:dem.shape[1]]
px = tr.c + (cols.ravel()+0.5)*tr.a; py = tr.f + (rows.ravel()+0.5)*tr.e
own = tree.query(gpd.points_from_xy(px, py), predicate="within")
bedc = np.zeros(N); roughc = np.zeros(N); cnt = np.zeros(N)
np.add.at(bedc, own[1], dem.ravel()[own[0]])
np.add.at(roughc, own[1], man.ravel()[own[0]])
np.add.at(cnt, own[1], 1.0)
ok = cnt > 0; bedc[ok] /= cnt[ok]; roughc[ok] /= cnt[ok]
areas = np.array([c.area for c in cells]); per = np.array([c.length for c in cells])
gpkg = OUT/f"mesh_{int(CELL)}m.gpkg"
for s_ in (gpkg, Path(f"{gpkg}-wal"), Path(f"{gpkg}-shm")): s_.unlink(missing_ok=True)
gpd.GeoDataFrame({"face_id": range(N), "area_m2": areas, "feature_class": ["rural"]*N,
                  "bed_m": bedc, "manning": roughc}, geometry=cells, crs=CRS
                 ).to_file(gpkg, layer="mesh", driver="GPKG")
nc = OUT/f"mesh_{int(CELL)}m.nc"
_write_ugrid(nc, faces, edges, cells, RCRS.from_wkt(CRS), np.full(N, CELL), tr, dem.astype(float),
             ["rural"]*N, np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N),
             cell_target_width=np.full(N, CELL))
with netCDF4.Dataset(nc, "a") as ds:
    ds["cell_bed_elevation_m"][:] = bedc
    ds["cell_hydraulic_roughness"][:] = roughc
ov = write_conservative_overlap(nc, gpkg, OUT/"dem.tif", OUT/f"overlap_{int(CELL)}m.nc")
print(f"mesh {N} cells at {CELL:.0f} m   A/P min {np.min(areas/per):.2f} m   faces {len(edges)}")
print(f"overlap {ov['overlap_count']:,} pairs   area err {ov['maximum_mesh_area_relative_error']:.2e}")

with netCDF4.Dataset(nc) as ds:
    e_own = np.asarray(ds["edge_owner"][:]).reshape(-1)
    e_nb = np.asarray(ds["edge_neighbor"][:]).reshape(-1)
    e_len = np.asarray(ds["edge_length_m"][:]).reshape(-1)
    mx = np.asarray(ds["edge_midpoint_x"][:]).reshape(-1)
    my = np.asarray(ds["edge_midpoint_y"][:]).reshape(-1)
    nx_ = np.asarray(ds["edge_normal_x"][:]).reshape(-1)
    ny_ = np.asarray(ds["edge_normal_y"][:]).reshape(-1)
    cell_area = np.asarray(ds["cell_area_m2"][:]).reshape(-1)
with netCDF4.Dataset(OUT/f"overlap_{int(CELL)}m.nc") as ds:
    oc = np.asarray(ds["overlap_mesh_index"][:]).reshape(-1)
    oi = np.asarray(ds["overlap_raster_index"][:]).reshape(-1)
    oa = np.asarray(ds["overlap_area_m2"][:]).reshape(-1)
dem_su, man_su = np.flipud(dem).reshape(-1), np.flipud(man).reshape(-1)
zp = dem_su[oi]
nr_, nc_ = dem.shape
r_, c_ = np.divmod(oi, nc_)
ymin = tr.f + tr.e * nr_
qx = tr.c + (c_ + 0.5)*tr.a; qy = ymin + (r_ + 0.5)*abs(tr.e)

plane = _fit_bed_planes(oc, oa, zp, qx, qy, N)
resid = zp - plane
ordr = np.argsort(oc, kind="stable"); k = np.bincount(oc, minlength=N)
st = np.r_[0, np.cumsum(k)[:-1]]
def pc(v, fn):
    ww = v[ordr]
    return np.array([fn(ww[st[i]:st[i]+k[i]]) if k[i] else 0.0 for i in range(N)])
ratio = np.divide(pc(resid, lambda a: a.max()-a.min()),
                  np.maximum(pc(zp, lambda a: a.max()-a.min()), 1e-9))
mask = np.ones(N, dtype=bool) if ALL else (ratio > POOL_RATIO_MIN)
print(f"sub-grid closure on {int(mask.sum())} of {N} cells (pool ratio > {POOL_RATIO_MIN})")

tabc = build_cell_volume_table(oc, oa, zp, N, cell_area,
                               level_pool_mask=mask, cell_bed_mean_m=bedc)
inv = ~tr
def samp(a, X, Y):
    cc, rr = inv * (X, Y)
    rr = np.clip(np.floor(rr).astype(int), 0, dem.shape[0]-1)
    cc = np.clip(np.floor(cc).astype(int), 0, dem.shape[1]-1)
    return a[rr, cc]
OFF = 0.5*abs(tr.a)
profiles = []
for i in range(len(e_own)):
    L = float(e_len[i]); npts = max(3, int(np.ceil(L/abs(tr.a)))+1)
    s_ = np.linspace(-0.5*L, 0.5*L, npts); tx, ty = -ny_[i], nx_[i]
    Xa, Ya = mx[i]+s_*tx-OFF*nx_[i], my[i]+s_*ty-OFF*ny_[i]
    Xb, Yb = mx[i]+s_*tx+OFF*nx_[i], my[i]+s_*ty+OFF*ny_[i]
    za, zb = samp(dem, Xa, Ya), samp(dem, Xb, Yb)
    nn = 0.5*(samp(man, Xa, Ya) + samp(man, Xb, Yb))
    profiles.append((s_-s_[0], np.maximum(za, zb), np.where(nn > 0, nn, 0.03)))
floor = np.where(e_nb >= 0,
                 np.minimum(tabc.datum_m[e_own], tabc.datum_m[np.maximum(e_nb, 0)]),
                 tabc.datum_m[e_own])
profiles = [(s_, np.maximum(z_, floor[i]), n_) for i, (s_, z_, n_) in enumerate(profiles)]
tabf = build_face_conveyance_table(profiles, e_len)
lower = np.minimum(tabc.datum_m[e_own], tabc.datum_m[np.maximum(e_nb, 0)])
bad = int(((lower - tabf.datum_m) > 1e-6)[e_nb >= 0].sum())
print(f"cell/face datum invariant violations: {bad}")

with netCDF4.Dataset(OUT/(f"subgrid_{int(CELL)}m_all.nc" if ALL else f"subgrid_{int(CELL)}m.nc"), "w") as o:
    o.createDimension("cell", N); o.createDimension("cell_pt", tabc.zeta_m.shape[1])
    o.createDimension("face", len(e_own)); o.createDimension("face_pt", tabf.zeta_m.shape[1])
    def v(nm, d, a):
        o.createVariable(nm, "f8" if a.dtype.kind == "f" else "i8", d)[:] = a
    v("cell_datum_m", ("cell",), tabc.datum_m); v("cell_zeta_m", ("cell","cell_pt"), tabc.zeta_m)
    v("cell_volume_m3", ("cell","cell_pt"), tabc.volume_m3)
    v("cell_wet_area_m2", ("cell","cell_pt"), tabc.area_m2)
    v("cell_point_count", ("cell",), tabc.count); v("cell_plan_area_m2", ("cell",), tabc.plan_area_m2)
    v("cell_is_subgrid", ("cell",), mask.astype(np.int64))
    v("face_datum_m", ("face",), tabf.datum_m); v("face_zeta_m", ("face","face_pt"), tabf.zeta_m)
    v("face_flow_area_m2", ("face","face_pt"), tabf.area_m2)
    v("face_perimeter_m", ("face","face_pt"), tabf.perimeter_m)
    v("face_conveyance", ("face","face_pt"), tabf.conveyance)
    v("face_point_count", ("face",), tabf.count); v("face_length_m", ("face",), tabf.length_m)
print(f"tables: {N} cells x{tabc.zeta_m.shape[1]}, {len(e_own)} faces x{tabf.zeta_m.shape[1]}")
json.dump({"cells":N,"cell_m":CELL,"dem_m":DX,"LX":LX,"LY":LY,"basin_r":BR,"basin_d":BD,
           "slope":SLOPE,"subgrid_cells":int(mask.sum()),"nrow":dem.shape[0],"ncol":dem.shape[1]},
          open(OUT/f"geometry_{int(CELL)}m.json","w"), indent=2)
print("wrote", OUT)
