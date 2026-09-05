"""90 m Voronoi mesh over the Pune sub-catchment, with sub-grid tables.

A uniform 90 m lattice is used on purpose. The question here is what the sub-grid
CLOSURE does, so the mesh must be identical in both arms; a feature-refined mesh
would change cell sizes as well and confound the comparison.

A cell is classed "river" if any of its own fine terrain is burned channel.
`--policy` selects which cells get a level-pool elevation-volume curve, and the
mask is written as `cell_is_subgrid` so the solver switches the FACE closure with
it (a face uses the tables only if both its cells do).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, geopandas as gpd, rasterio, netCDF4
from shapely import MultiPoint, STRtree, intersection, voronoi_polygons
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from dem_processing.hybrid_mesh import _topology, _write_ugrid
from dem_processing.mesh_product import write_conservative_overlap
from dem_processing.subgrid_tables import (
    build_cell_volume_table, build_face_conveyance_table, subgrid_cell_policy)

O = Path("examples/pune_subgrid")
FINE = 5
CELL = float(next((a for a in sys.argv[1:] if not a.startswith("--")), 90.0))
POLICY = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--policy=")), "river")
TAG = f"{int(CELL)}m"

with rasterio.open(O/f"dem_{FINE}m.tif") as s:
    dem = s.read(1, masked=True).filled(np.nan); tr = s.transform; crs = s.crs
with rasterio.open(O/f"manning_{FINE}m.tif") as s:
    man = s.read(1, masked=True).filled(np.nan)
with rasterio.open(O/f"channel_{FINE}m.tif") as s:
    chan = s.read(1, masked=True).filled(0) > 0.5
with rasterio.open(O/f"lulc_{FINE}m.tif") as s:
    lulc = s.read(1, masked=True).filled(np.nan)
valid = np.isfinite(dem)
h, w = dem.shape
print(f"fine terrain {h}x{w} at {FINE} m, valid {int(valid.sum()):,} px")

# domain polygon from the valid fine pixels, dissolved at 90 m to keep it cheap
ys, xs = np.where(valid)
gx = tr.c + (xs + 0.5)*tr.a; gy = tr.f + (ys + 0.5)*tr.e
bx0, bx1 = gx.min()-CELL, gx.max()+CELL
by0, by1 = gy.min()-CELL, gy.max()+CELL
seeds_x = np.arange(bx0 + CELL/2, bx1, CELL)
seeds_y = np.arange(by0 + CELL/2, by1, CELL)
SX, SY = np.meshgrid(seeds_x, seeds_y, indexing="ij")
seeds = np.column_stack([SX.ravel(), SY.ravel()])
# keep only seeds whose 90 m footprint holds valid terrain
inv = ~tr
cc, rr = inv * (seeds[:,0], seeds[:,1])
rr = np.clip(np.floor(rr).astype(int), 0, h-1); cc = np.clip(np.floor(cc).astype(int), 0, w-1)
seeds = seeds[valid[rr, cc]]
print(f"seeds on valid terrain: {len(seeds):,}")
hull = box(bx0, by0, bx1, by1)
raw = [c for c in voronoi_polygons(MultiPoint(seeds), extend_to=hull).geoms
       if c.geom_type == "Polygon" and c.area > 1e-9]
# Clip to the real domain. The Voronoi diagram is extended to a bounding box, so
# edge cells otherwise reach into nodata and the conservative overlap refuses the
# mesh (it requires every cell to be fully covered by raster cells).
from rasterio import features
dom = unary_union([Polygon(g["coordinates"][0], g["coordinates"][1:])
                   for g, v in features.shapes(valid.astype(np.uint8), mask=valid,
                                               transform=tr) if v == 1])
print(f"domain polygon area {dom.area/1e6:.2f} km2")
clipped = []
for c in raw:
    g = intersection(c, dom)
    if g.is_empty:
        continue
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda q: q.area)
    if g.geom_type == "Polygon" and g.area > 0.05*CELL*CELL:
        clipped.append(g)
print(f"cells after clipping to the domain: {len(clipped):,} of {len(raw):,}")
faces, edges = _topology(clipped); cells = [Polygon(f) for f in faces]
N = len(cells)
print(f"mesh {N:,} cells at {CELL:.0f} m, {len(edges):,} faces")

# per-cell aggregates from the fine grid
tree = STRtree(cells)
own = tree.query(gpd.points_from_xy(gx, gy), predicate="within")
cell_idx = own[1]; px_idx = own[0]
bedc = np.zeros(N); rgh = np.zeros(N); cnt = np.zeros(N); nch = np.zeros(N)
dflat = dem[ys, xs]; mflat = man[ys, xs]; chflat = chan[ys, xs]
np.add.at(bedc, cell_idx, dflat[px_idx]); np.add.at(rgh, cell_idx, mflat[px_idx])
np.add.at(cnt, cell_idx, 1.0); np.add.at(nch, cell_idx, chflat[px_idx].astype(float))
keep = cnt >= 0.25*(CELL/FINE)**2          # drop slivers with little real terrain
print(f"cells with >=25% valid fine coverage: {int(keep.sum()):,} of {N:,}")
ok = cnt > 0
bedc[ok] /= cnt[ok]; rgh[ok] /= cnt[ok]
is_river = nch > 0
fclass = np.where(is_river, "river", "rural")
print(f"river cells: {int(is_river.sum()):,} ({is_river.sum()/N*100:.1f}%)")

areas = np.array([c.area for c in cells]); per = np.array([c.length for c in cells])
gpkg = O/f"mesh_{TAG}.gpkg"
for s_ in (gpkg, Path(f"{gpkg}-wal"), Path(f"{gpkg}-shm")): s_.unlink(missing_ok=True)
gpd.GeoDataFrame({"face_id": range(N), "area_m2": areas, "feature_class": fclass,
                  "bed_m": bedc, "manning": rgh}, geometry=cells, crs=crs
                 ).to_file(gpkg, layer="mesh", driver="GPKG")
nc = O/f"mesh_{TAG}.nc"
_write_ugrid(nc, faces, edges, cells, crs, np.full(N, CELL), tr, dem.astype(float),
             list(fclass), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N),
             cell_target_width=np.full(N, CELL))
with netCDF4.Dataset(nc, "a") as ds:
    ds["cell_bed_elevation_m"][:] = bedc
    ds["cell_hydraulic_roughness"][:] = rgh
ov = write_conservative_overlap(nc, gpkg, O/f"dem_{FINE}m.tif", O/f"overlap_{TAG}.nc")
print(f"overlap {ov['overlap_count']:,} pairs, area err {ov['maximum_mesh_area_relative_error']:.1e}")
np.savez(O/f"_mesh_{TAG}.npz", bed=bedc, rough=rgh, is_river=is_river, area=areas,
         fclass=fclass, ap_min=float(np.min(areas/per)))
print(f"A/P min {np.min(areas/per):.2f} m")
json.dump({"cells":N,"cell_m":CELL,"fine_m":FINE,"river_cells":int(is_river.sum()),
           "nrow":h,"ncol":w,"ap_min":float(np.min(areas/per))},
          open(O/f"case_{TAG}.json","w"), indent=2)
print("wrote", O)
