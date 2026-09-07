"""Shared loaders / D8 topology for Pune subcatchment delineation (vectorised)."""
import sys, os, json
import numpy as np
import rasterio

REPO = "/Users/mngomes/Documents/GitHub/HydroBathyDEM"
sys.path.insert(0, os.path.join(REPO, "src"))
from dem_processing.hybrid_mesh import receiver_from_d8_direction  # noqa: E402

EX = os.path.join(REPO, "examples/pune_catchment")
P = dict(
    dem=f"{EX}/outputs/hydrobathy_20km2_corrected_dem/dem/DEM_hydraulic_conditioned.tif",
    fdir=f"{EX}/outputs/pune_rivers_2km2_d4_d8/rasters/D8_flow_direction.tif",
    rmask=f"{EX}/outputs/pune_rivers_2km2_d4_d8/rasters/D8_river_mask_2km2.tif",
    rw=f"{EX}/outputs/pune_rivers_2km2_d4_d8/rasters/D8_river_width_m.tif",
    rd=f"{EX}/outputs/pune_rivers_2km2_d4_d8/rasters/D8_river_depth_m.tif",
    rbed=f"{EX}/outputs/pune_rivers_2km2_d4_d8/rasters/D8_river_bed_elevation_m.tif",
    uaa=f"{EX}/outputs/pune_rivers_2km2_d4_d8/rasters/D8_upstream_area_km2.tif",
    imp=f"{EX}/data/pune_corrected_30m/pune_impervious_from_esa_lulc_30m.tif",
    lulc=f"{EX}/data/pune_corrected_30m/pune_lulc_esa_30m.tif",
    soil=f"{EX}/data/pune_corrected_30m/pune_soil_30m.tif",
    fp=f"{EX}/outputs/pune_mesh_feature_candidates_d8_2km2/floodplain_refinement_candidate_mask.tif",
    wb=f"{EX}/outputs/pune_mesh_feature_candidates_d8_2km2/mesh_feature_candidates.gpkg",
    domain=f"{EX}/data/pune_full_valid_domain.geojson",
)
CELL = 30.0
CELL_KM2 = CELL * CELL / 1e6
CACHE = os.path.dirname(os.path.abspath(__file__))


def read(key):
    with rasterio.open(P[key]) as s:
        return s.read(1), s.profile.copy()


def _levels(recv, n):
    """Kahn levels: level 0 = no donors. Returns array level[i]."""
    indeg = np.zeros(n, dtype=np.int32)
    has = recv >= 0
    np.add.at(indeg, recv[has], 1)
    level = np.full(n, -1, dtype=np.int32)
    frontier = np.flonzero if False else np.flatnonzero(indeg == 0)
    deg = indeg.copy()
    L = 0
    seen = 0
    while frontier.size:
        level[frontier] = L
        seen += frontier.size
        d = recv[frontier]
        d = d[d >= 0]
        if d.size:
            np.add.at(deg, d, -1)
            cand = np.unique(d)
            frontier = cand[deg[cand] == 0]
        else:
            frontier = np.empty(0, dtype=np.int64)
        L += 1
    if seen != n:
        raise RuntimeError(f"cycles: {n - seen} cells unresolved")
    return level


def topology(cache=True):
    """(shape, profile, recv, level, order_by_level, accum_cells, basin_label)."""
    npz = os.path.join(CACHE, "d8_topology.npz")
    fdir, prof = read("fdir")
    shape = fdir.shape
    if cache and os.path.exists(npz):
        z = np.load(npz)
        return shape, prof, z["recv"], z["level"], z["order"], z["accum"], z["basin"]
    n = shape[0] * shape[1]
    recv = receiver_from_d8_direction(fdir).ravel().astype(np.int64)
    idx = np.arange(n, dtype=np.int64)
    recv = np.where(fdir.ravel() != 0, recv, -1)
    recv = np.where(recv == idx, -1, recv)
    level = _levels(recv, n)
    order = np.argsort(level, kind="stable")          # upstream-first
    bnd = np.searchsorted(level[order], np.arange(level.max() + 2))
    # accumulation (cells), upstream-first, level by level
    accum = np.ones(n, dtype=np.float64)
    for L in range(level.max() + 1):
        sel = order[bnd[L]:bnd[L + 1]]
        d = recv[sel]
        ok = d >= 0
        np.add.at(accum, d[ok], accum[sel][ok])
    # basin label = terminal cell id, downstream-first (decreasing level)
    basin = idx.copy()
    for L in range(level.max(), -1, -1):
        sel = order[bnd[L]:bnd[L + 1]]
        d = recv[sel]
        ok = d >= 0
        basin[sel[ok]] = basin[d[ok]]
    if cache:
        np.savez_compressed(npz, recv=recv, level=level, order=order,
                            accum=accum, basin=basin)
    return shape, prof, recv, level, order, accum, basin
