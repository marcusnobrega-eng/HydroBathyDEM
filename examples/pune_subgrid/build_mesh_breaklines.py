"""Pune sub-catchment mesh with the river as a HEC-RAS style breakline corridor.

The previous uniform-lattice mesh routed water fine but broke the RIVER: on a
square 90 m lattice a sinuous channel steps diagonally, a diagonal step lands in a
corner-neighbour that shares no face, and 494 of the channel's 1,166 cell-to-cell
adjacencies carried no face at all. The river fell into 40 disconnected pieces,
largest holding 27% of river cells. Since a face uses the sub-grid tables only
when BOTH its cells are river cells, two thirds of the channel silently reverted
to the flat-prism closure.

`ChannelCorridor` fixes that by construction: it resamples the centreline ONCE and
derives every row from those same stations and normals, so corridor cells follow
the channel and consecutive ones share a face.

Corridor width is the CELL size, not the channel width. Pune's channels are
3-52 m (median 8.7 m); a corridor that narrow would give cells with A/P near 4 m
and destroy the timestep. One 90 m cell wide, following the channel, is the
decision already taken for this catchment -- the 8.7 m channel then lives inside
that cell, carried by the sub-grid tables, which is the whole point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import geopandas as gpd
import netCDF4
import numpy as np
import rasterio
from rasterio import features
from rasterio.crs import CRS as RCRS
from shapely import STRtree, intersection
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from dem_processing.hecras_mesh import ChannelCorridor, computation_points
from dem_processing.hybrid_mesh import _river_reach_paths, _topology, _write_ugrid
from dem_processing.mesh_product import write_conservative_overlap

O = Path("examples/pune_subgrid")
CA = Path("examples/pune_catchment/data/pune_caseA")
FINE = 5
CELL = float(next((a for a in sys.argv[1:] if not a.startswith("--")), 90.0))
NEAR_REPEATS = int(next((a.split("=")[1] for a in sys.argv[1:]
                         if a.startswith("--repeats=")), "1"))
# near_repeats=0 leaves the corridor claiming only its half-width, so background
# points can land right against the outer corridor row and squeeze the Voronoi
# cell between them: that gave 18 interior cells with A/P down to 4.39 m against a
# nominal 22.5 m, which alone would cut the timestep 2.6x.
TAG = f"bl{int(CELL)}m_r{NEAR_REPEATS}"

with rasterio.open(O / f"dem_{FINE}m.tif") as s:
    dem, tr, crs = s.read(1, masked=True).filled(np.nan), s.transform, s.crs
with rasterio.open(O / f"manning_{FINE}m.tif") as s:
    man = s.read(1, masked=True).filled(np.nan)
with rasterio.open(O / f"channel_{FINE}m.tif") as s:
    chan_fine = s.read(1, masked=True).filled(0) > 0.5
valid = np.isfinite(dem)
h, w = dem.shape

# ---- the river network, as junction-to-junction centrelines -----------------
bas = np.load(O / "_basin.npz")
mask30 = bas["mask"]
with rasterio.open(CA / "caseA_d8_flow_direction.tif") as s:
    d8 = s.read(1, masked=True).filled(0).astype(np.int32)
    t30 = s.transform
with rasterio.open(CA / "caseA_d8_river_mask_2km2.tif") as s:
    riv30 = s.read(1, masked=True).filled(0).astype(bool) & mask30
from dem_processing.hybrid_mesh import receiver_from_d8_direction

recv = receiver_from_d8_direction(d8)
paths, orphan = _river_reach_paths(riv30, recv)
H0, W0 = d8.shape


def to_line(path: list[int]) -> LineString | None:
    r, c = np.divmod(np.asarray(path, dtype=np.int64), W0)
    x = t30.c + (c + 0.5) * t30.a
    y = t30.f + (r + 0.5) * t30.e
    pts = list(zip(x.tolist(), y.tolist(), strict=True))
    return LineString(pts) if len(pts) >= 2 else None


lines = [ln for ln in (to_line(p) for p in paths) if ln is not None and ln.length > CELL]
total_km = sum(ln.length for ln in lines) / 1000.0
print(f"river reaches: {len(paths)} raw, {len(lines)} longer than one cell "
      f"({total_km:.1f} km); {len(orphan)} orphan links")

# ---- domain and computation points -----------------------------------------
domain = unary_union([
    Polygon(g["coordinates"][0], g["coordinates"][1:])
    for g, v in features.shapes(valid.astype(np.uint8), mask=valid, transform=tr) if v == 1
])
if domain.geom_type == "MultiPolygon":
    domain = max(domain.geoms, key=lambda q: q.area)
print(f"domain {domain.area / 1e6:.2f} km2")

corridors = [
    ChannelCorridor(centerline=ln, width_m=CELL, along_spacing_m=CELL,
                    cross_spacing_m=CELL, far_spacing_m=CELL,
                    near_repeats=NEAR_REPEATS,
                    name=f"reach{i}")
    for i, ln in enumerate(lines)
]
pts, budget = computation_points(domain, CELL, CELL, corridors=corridors)
print(f"computation points: {len(pts):,}   provenance {budget.counts}")

# ---- minimum point separation ----------------------------------------------
# A Voronoi cell can be no bigger than the gap to its nearest seed, so two points
# closer than a fraction of the nominal spacing force a squeezed cell. Measured
# without this: 18 interior cells at A/P down to 4.39 m against a nominal 22.5 m,
# all sitting 51-87 m from a centreline -- i.e. in the gap between the corridor's
# outer row at +/-90 m and the background lattice. Corridor grading did not close
# it (near_repeats=2 made it worse, 74 cells) because the corridor's claimed band
# and the lattice phase are independent.
#
# Corridor points are kept first so the channel is never thinned; only background
# points are dropped.
from scipy.spatial import cKDTree

MIN_SEP = float(next((a.split("=")[1] for a in sys.argv[1:]
                      if a.startswith("--minsep=")), "0.55")) * CELL
n_corr = sum(v for k, v in budget.counts.items() if k != "background")
corr_pts, back_pts = pts[:n_corr], pts[n_corr:]
if len(corr_pts) and len(back_pts):
    keep = cKDTree(corr_pts).query(back_pts)[0] >= MIN_SEP
    dropped = int((~keep).sum())
    back_pts = back_pts[keep]
else:
    dropped = 0
pts = np.vstack([corr_pts, back_pts]) if len(corr_pts) else back_pts
print(f"  minimum separation {MIN_SEP:.1f} m: dropped {dropped:,} background points "
      f"-> {len(pts):,}")

from shapely import MultiPoint, voronoi_polygons

AP_FLOOR = float(next((a.split("=")[1] for a in sys.argv[1:]
                       if a.startswith("--apfloor=")), "12.0"))


def tessellate(seeds: np.ndarray) -> list[Polygon]:
    raw = [c for c in voronoi_polygons(MultiPoint(seeds),
                                       extend_to=domain.buffer(CELL)).geoms
           if c.geom_type == "Polygon" and c.area > 1e-9]
    out = []
    for c in raw:
        g = intersection(c, domain)
        if g.is_empty:
            continue
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda q: q.area)
        if g.geom_type == "Polygon" and g.area > 0.05 * CELL * CELL:
            out.append(g)
    return out


# Prune seeds whose cell falls below the A/P floor and rebuild.
#
# The timestep scales with min(area/perimeter), so a handful of squeezed cells set
# the cost of the whole run. Chasing their cause was unproductive: they are not
# domain slivers (1-3 km from the boundary), not confluences (only one reach
# within 72 m), and not corridor/background crowding (enforcing a 63 m minimum
# point separation dropped a single point and changed nothing). Removing the seed
# is the direct remedy -- neighbours absorb the area and the tessellation stays
# conforming. River seeds are never pruned, so the channel chain is preserved.
# Protect only the channel LANES, not the corridor's graded outer rows. Protecting
# every corridor point left 7 cells at A/P 4.89 m that the pruner refused to touch
# and which were outer-row cells carrying no channel at all.
from shapely import MultiLineString
from scipy.spatial import cKDTree as _KD

_dense = []
for ln in lines:
    n = max(2, int(ln.length / (0.25 * CELL)) + 1)
    _dense.extend(ln.interpolate(d, normalized=True).coords[0]
                  for d in np.linspace(0.0, 1.0, n))
_ctree = _KD(np.asarray(_dense))
_is_lane = _ctree.query(pts)[0] <= 0.5 * CELL
corr_set = {(round(float(x), 3), round(float(y), 3))
            for x, y in pts[_is_lane]}
print(f"  protected channel-lane seeds: {int(_is_lane.sum()):,} "
      f"(all corridor points: {len(corr_pts):,})")
seeds = pts.copy()
for iteration in range(6):
    cand = tessellate(seeds)
    ar = np.array([c.area for c in cand])
    pe = np.array([c.length for c in cand])
    ap = ar / pe
    small = np.flatnonzero(ap < AP_FLOOR)
    if small.size == 0:
        print(f"  seed pruning: converged after {iteration} pass(es), "
              f"A/P min {ap.min():.2f} m")
        break
    # map each small cell back to the seed inside it
    tre = STRtree([Polygon(c.exterior) for c in cand])
    drop = []
    for i in small:
        hit = np.flatnonzero(
            [cand[i].contains(gpd.points_from_xy([sx], [sy])[0]) for sx, sy in seeds]
        )
        for j in hit:
            key = (round(float(seeds[j, 0]), 3), round(float(seeds[j, 1]), 3))
            if key not in corr_set:
                drop.append(j)
    if not drop:
        print(f"  seed pruning: {small.size} cell(s) below {AP_FLOOR} m are all "
              f"corridor seeds; keeping them (A/P min {ap.min():.2f} m)")
        break
    seeds = np.delete(seeds, np.unique(drop), axis=0)
    print(f"  seed pruning pass {iteration + 1}: dropped {len(np.unique(drop))} seed(s), "
          f"{len(seeds):,} remain (A/P min was {ap.min():.2f} m)")
clipped = tessellate(seeds)
pts = seeds
faces, edges = _topology(clipped)
cells = [Polygon(f) for f in faces]
N = len(cells)
areas = np.array([c.area for c in cells])
per = np.array([c.length for c in cells])
print(f"mesh {N:,} cells, {len(edges):,} faces, A/P min {np.min(areas / per):.2f} m")

# ---- per-cell aggregates and the river class --------------------------------
tree = STRtree(cells)
ys, xs = np.where(valid)
gx = tr.c + (xs + 0.5) * tr.a
gy = tr.f + (ys + 0.5) * tr.e
own = tree.query(gpd.points_from_xy(gx, gy), predicate="within")
bedc = np.zeros(N); rgh = np.zeros(N); cnt = np.zeros(N); nch = np.zeros(N)
np.add.at(bedc, own[1], dem[ys, xs][own[0]])
np.add.at(rgh, own[1], man[ys, xs][own[0]])
np.add.at(cnt, own[1], 1.0)
np.add.at(nch, own[1], chan_fine[ys, xs][own[0]].astype(float))
ok = cnt > 0
bedc[ok] /= cnt[ok]; rgh[ok] /= cnt[ok]
is_river = nch > 0
fclass = np.where(is_river, "river", "rural")
print(f"river cells: {int(is_river.sum()):,} of {N:,} ({is_river.sum() / N * 100:.1f}%)")

gpkg = O / f"mesh_{TAG}.gpkg"
for s_ in (gpkg, Path(f"{gpkg}-wal"), Path(f"{gpkg}-shm")):
    s_.unlink(missing_ok=True)
gpd.GeoDataFrame({"face_id": range(N), "area_m2": areas, "feature_class": fclass,
                  "bed_m": bedc, "manning": rgh}, geometry=cells, crs=crs
                 ).to_file(gpkg, layer="mesh", driver="GPKG")
nc = O / f"mesh_{TAG}.nc"
_write_ugrid(nc, faces, edges, cells, RCRS.from_wkt(crs.to_wkt()), np.full(N, CELL), tr,
             dem.astype(float), list(fclass), np.zeros(N), np.zeros(N), np.zeros(N),
             np.zeros(N), cell_target_width=np.full(N, CELL))
with netCDF4.Dataset(nc, "a") as ds:
    ds["cell_bed_elevation_m"][:] = bedc
    ds["cell_hydraulic_roughness"][:] = rgh

# ---- CONNECTIVITY GATE: check BEFORE spending compute on a run --------------
# `_topology` does not return (owner, neighbour) pairs, so read the written mesh.
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

with netCDF4.Dataset(nc) as ds:
    e_own = np.asarray(ds["edge_owner"][:]).reshape(-1)
    e_nb = np.asarray(ds["edge_neighbor"][:]).reshape(-1)
internal = e_nb >= 0
o, n = e_own[internal], e_nb[internal]
whole = coo_matrix((np.ones(o.size), (o, n)), shape=(N, N))
kw, _ = connected_components(whole + whole.T, directed=False)
rr = is_river[o] & is_river[n]
ar = coo_matrix((np.ones(int(rr.sum())), (o[rr], n[rr])), shape=(N, N))
kr, labr = connected_components(ar + ar.T, directed=False)
sizes = np.array(sorted([c for c in np.bincount(labr[is_river],
                        minlength=labr.max() + 1) if c > 0], reverse=True))
frac = sizes[0] / is_river.sum() * 100
ap = areas / per
print(f"\nCONNECTIVITY  whole mesh {kw} component(s)")
print(f"              river: {len(sizes)} piece(s); largest {sizes[0]} "
      f"({frac:.1f}% of river cells); top {sizes[:8].tolist()}")
print(f"              river-river faces: {int(rr.sum()):,}")
print(f"              uniform lattice was: 40 pieces, largest 27.0%, 672 faces")
print(f"CELL QUALITY  A/P min {ap.min():.2f} m  p1 {np.percentile(ap,1):.2f}  "
      f"p50 {np.percentile(ap,50):.2f}   cells below 10 m: {int((ap<10).sum()):,}")
print(f"              river cells A/P min {ap[is_river].min():.2f} m  "
      f"median {np.median(ap[is_river]):.2f} m")

ov = write_conservative_overlap(nc, gpkg, O / f"dem_{FINE}m.tif", O / f"overlap_{TAG}.nc")
print(f"\noverlap {ov['overlap_count']:,} pairs, area err "
      f"{ov['maximum_mesh_area_relative_error']:.1e}")
np.savez(O / f"_mesh_{TAG}.npz", bed=bedc, rough=rgh, is_river=is_river, area=areas,
         fclass=fclass, ap_min=float(np.min(areas / per)))
json.dump({"cells": N, "cell_m": CELL, "fine_m": FINE, "river_cells": int(is_river.sum()),
           "nrow": h, "ncol": w, "ap_min": float(np.min(areas / per)),
           "river_pieces": int(len(sizes)), "river_largest_pct": float(frac),
           "river_km": total_km},
          open(O / f"case_{TAG}.json", "w"), indent=2)
print("wrote", O)
