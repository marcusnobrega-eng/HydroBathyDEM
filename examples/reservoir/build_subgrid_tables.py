"""Precompute HEC-RAS style sub-grid tables for the V-catchment mesh.

Cell elevation-volume curves come from the conservative overlap (exact areas).
Face conveyance curves come from sampling the fine DEM and Manning raster along
each face line. Both are written to a sidecar NetCDF the solver reads directly,
so the hydraulic model never opens a raster -- the same split HEC-RAS uses,
where tables are built once at mesh time and stored with the geometry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dem_processing.subgrid_tables import (  # noqa: E402
    build_cell_volume_table,
    build_face_conveyance_table,
)

HERE = Path(__file__).resolve().parent
MESH_NC = HERE / "reservoir_mesh.nc"
OVERLAP_NC = HERE / "reservoir_mesh_overlap.nc"
DATUM_MODE = sys.argv[1] if len(sys.argv) > 1 else "horizontal"
GATED = "--gated" in sys.argv
OUT_NC = HERE / (f"reservoir_subgrid_{DATUM_MODE}" + ("_gated" if GATED else "") + ".nc")
FACE_SAMPLE_M = 2.5          # half the 5 m DEM pixel: two samples per pixel crossed


class _Grid:
    """Nearest-cell raster sampling, vectorised.

    ``rasterio.DatasetReader.index`` is scalar-only, and reopening the file per
    face would mean 507 opens for this mesh alone, so index arithmetic is done
    here against a band held in memory.
    """

    def __init__(self, path: Path) -> None:
        with rasterio.open(path) as src:
            self.band = src.read(1, masked=True).filled(np.nan)
            self.transform = src.transform
        self.inv = ~self.transform

    def at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols, rows = self.inv * (np.asarray(xs), np.asarray(ys))
        rows = np.clip(np.floor(rows).astype(np.int64), 0, self.band.shape[0] - 1)
        cols = np.clip(np.floor(cols).astype(np.int64), 0, self.band.shape[1] - 1)
        return self.band[rows, cols]


def main() -> None:
    with Dataset(MESH_NC) as ds:
        cell_area = np.asarray(ds["cell_area_m2"][:], dtype=np.float64).reshape(-1)
        edge_owner = np.asarray(ds["edge_owner"][:], dtype=np.int64).reshape(-1)
        edge_neighbor = np.asarray(ds["edge_neighbor"][:], dtype=np.int64).reshape(-1)
        edge_length = np.asarray(ds["edge_length_m"][:], dtype=np.float64).reshape(-1)
        mx = np.asarray(ds["edge_midpoint_x"][:], dtype=np.float64).reshape(-1)
        my = np.asarray(ds["edge_midpoint_y"][:], dtype=np.float64).reshape(-1)
        nx = np.asarray(ds["edge_normal_x"][:], dtype=np.float64).reshape(-1)
        ny = np.asarray(ds["edge_normal_y"][:], dtype=np.float64).reshape(-1)
        dem_path = Path(ds.getncattr("subgrid_dem_path"))
        man_path = Path(ds.getncattr("subgrid_manning_path"))

    with Dataset(OVERLAP_NC) as ds:
        pair_cell = np.asarray(ds["overlap_mesh_index"][:], dtype=np.int64).reshape(-1)
        pair_index = np.asarray(ds["overlap_raster_index"][:], dtype=np.int64).reshape(-1)
        pair_area = np.asarray(ds["overlap_area_m2"][:], dtype=np.float64).reshape(-1)

    dem_grid, man_grid = _Grid(dem_path), _Grid(man_path)
    dem, man = dem_grid.band, man_grid.band
    # overlap_raster_index is SOUTH-UP: flip before taking a flat index
    dem_su = np.flipud(dem).reshape(-1)
    man_su = np.flipud(man).reshape(-1)
    patch_z = dem_su[pair_index]
    patch_n = man_su[pair_index]

    # patch centroids, needed for the bed-plane fit.  overlap_raster_index is
    # SOUTH-UP, so row 0 is the BOTTOM of the north-up raster.
    n_rows, n_cols = dem.shape
    tr = dem_grid.transform
    row_su, col = np.divmod(pair_index, n_cols)
    y_min = tr.f + tr.e * n_rows            # tr.e is negative for a north-up raster
    patch_x = tr.c + (col + 0.5) * tr.a
    patch_y = y_min + (row_su + 0.5) * abs(tr.e)
    # gate the level-pool closure to cells that genuinely pond (see subgrid_tables)
    from dem_processing.subgrid_tables import _fit_bed_planes
    plane_ = _fit_bed_planes(pair_cell, pair_area, patch_z, patch_x, patch_y, len(cell_area))
    resid_ = patch_z - plane_
    ordr = np.argsort(pair_cell, kind="stable")
    cnt_ = np.bincount(pair_cell, minlength=len(cell_area))
    st_ = np.r_[0, np.cumsum(cnt_)[:-1]]
    def _pc(v, fn):
        w = v[ordr]
        return np.array([fn(w[st_[i]:st_[i]+cnt_[i]]) if cnt_[i] else 0.0
                         for i in range(len(cell_area))])
    rt = _pc(patch_z, lambda a_: a_.max()-a_.min())
    rr = _pc(resid_, lambda a_: a_.max()-a_.min())
    ratio = np.divide(rr, rt, out=np.zeros(len(cell_area)), where=rt > 1e-9)
    with Dataset(MESH_NC) as ds:
        bed_mean = np.asarray(ds["cell_bed_elevation_m"][:], dtype=np.float64).reshape(-1)
    mask = ratio > 0.20 if GATED else np.ones(len(cell_area), dtype=bool)
    print(f"level-pool closure on {int(mask.sum())} of {len(cell_area)} cells")
    cells = build_cell_volume_table(
        pair_cell, pair_area, patch_z, len(cell_area), cell_area,
        datum_mode=DATUM_MODE,
        **({"patch_x": patch_x, "patch_y": patch_y} if DATUM_MODE == "bed_plane" else {}),
        level_pool_mask=mask, cell_bed_mean_m=bed_mean,
    )
    print(f"datum mode: {DATUM_MODE}")
    print(f"cell tables: {len(cell_area)} cells, up to {cells.zeta_m.shape[1]} points")

    # face profiles: walk the face line, which is perpendicular to its normal
    profiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(len(edge_owner)):
        L = float(edge_length[i])
        npts = max(3, int(np.ceil(L / FACE_SAMPLE_M)) + 1)
        s = np.linspace(-0.5 * L, 0.5 * L, npts)
        tx, ty = -ny[i], nx[i]                       # tangent = normal rotated 90 deg
        xs, ys = mx[i] + s * tx, my[i] + s * ty
        z = dem_grid.at(xs, ys)
        n = man_grid.at(xs, ys)
        z = np.where(np.isfinite(z), z, np.nanmax(z) if np.isfinite(z).any() else 0.0)
        n = np.where(np.isfinite(n) & (n > 0), n, 0.03)
        profiles.append((s - s[0], z, n))
    faces = build_face_conveyance_table(profiles, edge_length)
    print(f"face tables: {len(edge_owner)} faces, up to {faces.zeta_m.shape[1]} points")

    with Dataset(OUT_NC, "w") as out:
        out.createDimension("cell", len(cell_area))
        out.createDimension("cell_pt", cells.zeta_m.shape[1])
        out.createDimension("face", len(edge_owner))
        out.createDimension("face_pt", faces.zeta_m.shape[1])
        def var(name, dims, data):
            v = out.createVariable(name, "f8" if data.dtype.kind == "f" else "i8", dims)
            v[:] = data
        var("cell_datum_m", ("cell",), cells.datum_m)
        var("cell_zeta_m", ("cell", "cell_pt"), cells.zeta_m)
        var("cell_volume_m3", ("cell", "cell_pt"), cells.volume_m3)
        var("cell_wet_area_m2", ("cell", "cell_pt"), cells.area_m2)
        var("cell_point_count", ("cell",), cells.count)
        var("cell_plan_area_m2", ("cell",), cells.plan_area_m2)
        var("face_datum_m", ("face",), faces.datum_m)
        var("face_zeta_m", ("face", "face_pt"), faces.zeta_m)
        var("face_flow_area_m2", ("face", "face_pt"), faces.area_m2)
        var("face_perimeter_m", ("face", "face_pt"), faces.perimeter_m)
        var("face_conveyance", ("face", "face_pt"), faces.conveyance)
        var("face_point_count", ("face",), faces.count)
        var("face_length_m", ("face",), faces.length_m)
        out.setncattr("source_dem", str(dem_path))
        out.setncattr("source_manning", str(man_path))
        out.setncattr("face_sample_spacing_m", FACE_SAMPLE_M)
        out.setncattr("cell_datum_mode", DATUM_MODE)
    print(f"wrote {OUT_NC}")

    # sanity: total volume at the highest datum must match the DEM-integrated volume
    top = cells.datum_m + cells.zeta_m[np.arange(len(cell_area)), cells.count - 1]
    v_tab = cells.volume(top)
    print(f"volume at each cell's top break: {v_tab.sum():.3f} m3")
    print(f"cell wet area at top vs plan area: max rel diff "
          f"{np.max(np.abs(cells.area_m2[np.arange(len(cell_area)), cells.count-1] - cells.plan_area_m2) / cells.plan_area_m2):.2e}")
    print(f"datum vs cell terrain: datum span {cells.datum_m.min():.2f}-{cells.datum_m.max():.2f} m, "
          f"residual range median {np.median(cells.zeta_m[np.arange(len(cell_area)), cells.count-1] - cells.zeta_m[:, 0]):.3f} m")


if __name__ == "__main__":
    main()
