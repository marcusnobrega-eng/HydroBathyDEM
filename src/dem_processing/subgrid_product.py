"""Build versioned HydroPol2D subgrid tables from a published mesh package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import rasterio

from .mesh_contract import (
    MESH_CONTRACT_VERSION,
    MESH_PRODUCT_SCHEMA,
    OVERLAP_RASTER_ORDER,
    SUBGRID_CONVEYANCE_CONVENTION,
    apply_contract_attributes,
)
from .subgrid_tables import (
    build_cell_volume_table,
    build_face_conveyance_table,
    subgrid_cell_policy,
)


@dataclass(frozen=True)
class SubgridProductConfig:
    fine_dem: Path
    fine_manning: Path | None = None
    policy: str = "river"
    default_manning_n: float = 0.05
    face_sample_spacing_m: float | None = None
    output_name: str = "hydropol_subgrid_tables.nc"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "SubgridProductConfig":
        aliases = {
            "dem": "fine_dem",
            "dem_path": "fine_dem",
            "manning": "fine_manning",
            "manning_path": "fine_manning",
            "sample_spacing_m": "face_sample_spacing_m",
        }
        normalized = {aliases.get(key, key): value for key, value in values.items()}
        normalized.pop("enabled", None)
        for key in ("fine_dem", "fine_manning"):
            if normalized.get(key) is not None:
                normalized[key] = Path(normalized[key])
        config = cls(**normalized)
        if config.default_manning_n <= 0:
            raise ValueError("subgrid.default_manning_n must be positive.")
        if config.face_sample_spacing_m is not None and config.face_sample_spacing_m <= 0:
            raise ValueError("subgrid.face_sample_spacing_m must be positive.")
        return config


def _read_raster(path: Path) -> tuple[np.ndarray, Any, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Subgrid raster not found: {path}")
    with rasterio.open(path) as source:
        if source.crs is None or not source.crs.is_projected:
            raise ValueError(f"Subgrid raster requires a projected CRS: {path}")
        data = source.read(1, masked=True).filled(np.nan).astype(np.float64)
        return data, source.transform, source.crs


def _sample_nearest(array: np.ndarray, inverse_transform: Any, x, y) -> np.ndarray:
    columns, rows = inverse_transform * (x, y)
    row = np.clip(np.floor(rows).astype(np.int64), 0, array.shape[0] - 1)
    column = np.clip(np.floor(columns).astype(np.int64), 0, array.shape[1] - 1)
    return array[row, column]


def _write_tables(path: Path, cell_table, face_table, mask: np.ndarray, *, config, crs_wkt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as target:
        target.Conventions = "CF-1.13"
        target.title = "HydroBathyDEM cell and face subgrid hydraulic tables"
        target.schema_version = MESH_PRODUCT_SCHEMA
        target.crs_wkt = crs_wkt
        target.policy = config.policy
        target.conveyance_convention = SUBGRID_CONVEYANCE_CONVENTION
        target.default_manning_n = config.default_manning_n
        apply_contract_attributes(target, product="hydraulic_subgrid_tables")
        target.createDimension("cell", len(cell_table.datum_m))
        target.createDimension("cell_point", cell_table.zeta_m.shape[1])
        target.createDimension("face", len(face_table.datum_m))
        target.createDimension("face_point", face_table.zeta_m.shape[1])

        def write(name: str, dimensions: tuple[str, ...], values: np.ndarray, units: str | None = None) -> None:
            dtype = "f8" if np.asarray(values).dtype.kind == "f" else "i8"
            variable = target.createVariable(name, dtype, dimensions, zlib=True)
            if units:
                variable.units = units
            variable[:] = values

        write("cell_datum_m", ("cell",), cell_table.datum_m, "m")
        write("cell_zeta_m", ("cell", "cell_point"), cell_table.zeta_m, "m")
        write("cell_volume_m3", ("cell", "cell_point"), cell_table.volume_m3, "m3")
        write("cell_wet_area_m2", ("cell", "cell_point"), cell_table.area_m2, "m2")
        write("cell_point_count", ("cell",), cell_table.count)
        write("cell_plan_area_m2", ("cell",), cell_table.plan_area_m2, "m2")
        write("cell_is_subgrid", ("cell",), mask.astype(np.int64))
        write("face_datum_m", ("face",), face_table.datum_m, "m")
        write("face_zeta_m", ("face", "face_point"), face_table.zeta_m, "m")
        write("face_flow_area_m2", ("face", "face_point"), face_table.area_m2, "m2")
        write("face_perimeter_m", ("face", "face_point"), face_table.perimeter_m, "m")
        write("face_conveyance", ("face", "face_point"), face_table.conveyance, "m3 s-1")
        write("face_point_count", ("face",), face_table.count)
        write("face_length_m", ("face",), face_table.length_m, "m")


def build_subgrid_product(
    mesh_file: str | Path,
    overlap_file: str | Path,
    config: SubgridProductConfig,
    output_file: str | Path,
) -> dict[str, Any]:
    """Build the cell/face tables consumed unchanged by MATLAB and Python."""
    mesh_file, overlap_file, output_file = map(Path, (mesh_file, overlap_file, output_file))
    dem, transform, crs = _read_raster(config.fine_dem)
    if config.fine_manning is None:
        manning = np.full(dem.shape, config.default_manning_n, dtype=np.float64)
    else:
        manning, manning_transform, manning_crs = _read_raster(config.fine_manning)
        if manning.shape != dem.shape or manning_transform != transform or manning_crs != crs:
            raise ValueError("Fine DEM and Manning rasters must have identical grids and CRS.")

    with netCDF4.Dataset(mesh_file) as mesh:
        mesh_crs = str(getattr(mesh, "crs_wkt", ""))
        if not mesh_crs:
            raise ValueError("Mesh package has no CRS.")
        if crs.to_wkt() != mesh_crs and crs != rasterio.crs.CRS.from_wkt(mesh_crs):
            raise ValueError("Fine DEM and mesh package use different CRS definitions.")
        owner = np.asarray(mesh["edge_owner"][:], dtype=np.int64).reshape(-1)
        neighbour = np.asarray(mesh["edge_neighbor"][:], dtype=np.int64).reshape(-1)
        length = np.asarray(mesh["edge_length_m"][:], dtype=np.float64).reshape(-1)
        midpoint_x = np.asarray(mesh["edge_midpoint_x"][:], dtype=np.float64).reshape(-1)
        midpoint_y = np.asarray(mesh["edge_midpoint_y"][:], dtype=np.float64).reshape(-1)
        normal_x = np.asarray(mesh["edge_normal_x"][:], dtype=np.float64).reshape(-1)
        normal_y = np.asarray(mesh["edge_normal_y"][:], dtype=np.float64).reshape(-1)
        cell_area = np.asarray(mesh["cell_area_m2"][:], dtype=np.float64).reshape(-1)
        cell_bed = np.asarray(mesh["cell_bed_elevation_m"][:], dtype=np.float64).reshape(-1)
        feature_class = np.asarray(mesh["cell_feature_class"][:]).astype(str).reshape(-1)

    with netCDF4.Dataset(overlap_file) as overlap:
        contract_version = str(getattr(overlap, "mesh_contract_version", ""))
        if contract_version != MESH_CONTRACT_VERSION:
            raise ValueError(
                "Overlap file uses mesh contract "
                f"{contract_version or 'unspecified'!r}; expected {MESH_CONTRACT_VERSION!r}."
            )
        overlap_crs = str(getattr(overlap, "crs_wkt", ""))
        if not overlap_crs or rasterio.crs.CRS.from_wkt(overlap_crs) != crs:
            raise ValueError("Fine DEM and overlap table use different CRS definitions.")
        rows = int(getattr(overlap, "raster_rows"))
        columns = int(getattr(overlap, "raster_cols"))
        order = str(getattr(overlap, "raster_index_order", ""))
        if order != OVERLAP_RASTER_ORDER:
            raise ValueError(f"Unsupported overlap raster order: {order!r}.")
        cell_index = np.asarray(overlap["overlap_mesh_index"][:], dtype=np.int64).reshape(-1)
        raster_index = np.asarray(overlap["overlap_raster_index"][:], dtype=np.int64).reshape(-1)
        overlap_area = np.asarray(overlap["overlap_area_m2"][:], dtype=np.float64).reshape(-1)
    if dem.shape != (rows, columns):
        raise ValueError(
            f"Fine DEM shape {dem.shape} does not match overlap raster grid {(rows, columns)}."
        )

    mask = subgrid_cell_policy(feature_class, config.policy)
    terrain_south_up = np.flipud(dem).reshape(-1)
    patch_elevation = terrain_south_up[raster_index]
    good = np.isfinite(patch_elevation) & (overlap_area > 0)
    cell_table = build_cell_volume_table(
        cell_index[good], overlap_area[good], patch_elevation[good], len(cell_area), cell_area,
        level_pool_mask=mask, cell_bed_mean_m=cell_bed,
    )

    internal = neighbour >= 0
    full_profile = np.zeros(len(owner), dtype=bool)
    full_profile[internal] = mask[owner[internal]] & mask[neighbour[internal]]
    floor = np.where(
        internal,
        np.minimum(cell_table.datum_m[owner], cell_table.datum_m[np.maximum(neighbour, 0)]),
        cell_table.datum_m[owner],
    )
    spacing = config.face_sample_spacing_m or min(abs(transform.a), abs(transform.e))
    offset = 0.5 * spacing
    inverse = ~transform
    profiles = []
    for index in range(len(owner)):
        if not full_profile[index]:
            profiles.append((
                np.asarray([0.0, length[index]]),
                np.full(2, floor[index]),
                np.full(2, config.default_manning_n),
            ))
            continue
        point_count = max(3, int(np.ceil(length[index] / spacing)) + 1)
        station = np.linspace(-0.5 * length[index], 0.5 * length[index], point_count)
        tangent_x, tangent_y = -normal_y[index], normal_x[index]
        x_a = midpoint_x[index] + station * tangent_x - offset * normal_x[index]
        y_a = midpoint_y[index] + station * tangent_y - offset * normal_y[index]
        x_b = midpoint_x[index] + station * tangent_x + offset * normal_x[index]
        y_b = midpoint_y[index] + station * tangent_y + offset * normal_y[index]
        z_a = _sample_nearest(dem, inverse, x_a, y_a)
        z_b = _sample_nearest(dem, inverse, x_b, y_b)
        z_a = np.where(np.isfinite(z_a), z_a, z_b)
        z_b = np.where(np.isfinite(z_b), z_b, z_a)
        bed = np.where(np.isfinite(np.maximum(z_a, z_b)), np.maximum(z_a, z_b), floor[index])
        n_a = _sample_nearest(manning, inverse, x_a, y_a)
        n_b = _sample_nearest(manning, inverse, x_b, y_b)
        roughness = 0.5 * (n_a + n_b)
        roughness = np.where(
            np.isfinite(roughness) & (roughness > 0), roughness, config.default_manning_n
        )
        profiles.append((station - station[0], np.maximum(bed, floor[index]), roughness))
    face_table = build_face_conveyance_table(profiles, length)

    low_cell_datum = np.minimum(cell_table.datum_m[owner], cell_table.datum_m[np.maximum(neighbour, 0)])
    datum_violations = int(np.count_nonzero(internal & ((low_cell_datum - face_table.datum_m) > 1e-6)))
    if datum_violations:
        raise ValueError(f"Subgrid cell/face datum invariant failed on {datum_violations} faces.")
    _write_tables(output_file, cell_table, face_table, mask, config=config, crs_wkt=mesh_crs)
    return {
        "subgrid": str(output_file),
        "subgrid_policy": config.policy,
        "subgrid_cell_count": int(mask.sum()),
        "subgrid_profile_face_count": int(full_profile.sum()),
        "subgrid_cell_points": int(cell_table.zeta_m.shape[1]),
        "subgrid_face_points": int(face_table.zeta_m.shape[1]),
        "subgrid_conveyance_convention": SUBGRID_CONVEYANCE_CONVENTION,
    }
