"""User-facing regular/Voronoi mesh product dispatcher."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import netCDF4
import numpy as np
import rasterio
from shapely import STRtree, area, box, intersection

from .config import load_config_file
from .hybrid_mesh import HybridMeshConfig, build_hybrid_mesh


MESH_PRODUCT_SCHEMA = "hydrobathydem-mesh-1.0"


@dataclass(frozen=True)
class RegularMeshConfig:
    dem: Path
    out_dir: Path
    cell_size_m: float | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "RegularMeshConfig":
        aliases = {"dem_path": "dem", "output_dir": "out_dir", "resolution_m": "cell_size_m"}
        normalized = {aliases.get(key, key): value for key, value in values.items()}
        for key in ("dem", "out_dir"):
            if normalized.get(key) is not None:
                normalized[key] = Path(normalized[key])
        config = cls(**normalized)
        if config.cell_size_m is not None and config.cell_size_m <= 0:
            raise ValueError("mesh.regular.cell_size_m must be positive.")
        return config


def _selected_mesh(values: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Normalize the documented nested schema and the legacy flat Voronoi schema."""
    values = dict(values)
    if "mesh" not in values:
        mode = str(values.get("mesh_mode", "voronoi_fv"))
        return mode, values
    schema_version = values.get("schema_version", MESH_PRODUCT_SCHEMA)
    if schema_version != MESH_PRODUCT_SCHEMA:
        raise ValueError(
            f"Unsupported mesh schema_version {schema_version!r}; expected {MESH_PRODUCT_SCHEMA!r}."
        )
    unknown_root = set(values) - {"schema_version", "mesh"}
    if unknown_root:
        raise ValueError(f"Unknown top-level mesh configuration keys: {sorted(unknown_root)}")
    mesh = values["mesh"]
    if not isinstance(mesh, dict):
        raise ValueError("mesh must be a configuration object.")
    mesh = dict(mesh)
    mode = str(mesh.pop("mode", ""))
    if mode not in {"regular", "voronoi_fv"}:
        raise ValueError("mesh.mode must be 'regular' or 'voronoi_fv'.")
    other_mode = "voronoi" if mode == "regular" else "regular"
    if other_mode in mesh:
        raise ValueError(f"mesh.{other_mode} is incompatible with mesh.mode={mode!r}.")
    inputs = mesh.pop("inputs", {})
    selected = mesh.pop("regular" if mode == "regular" else "voronoi", {})
    if not isinstance(inputs, dict) or not isinstance(selected, dict):
        raise ValueError("mesh.inputs and the selected mode block must be configuration objects.")
    return mode, {**inputs, **selected, **mesh}


def build_regular_mesh(config: RegularMeshConfig) -> dict[str, Any]:
    """Validate and package the existing conditioned raster as a regular mesh."""
    if not config.dem.is_file():
        raise FileNotFoundError(f"Conditioned DEM not found: {config.dem}")
    with rasterio.open(config.dem) as source:
        if source.crs is None or not source.crs.is_projected:
            raise ValueError("Regular meshes require a projected DEM CRS.")
        transform = source.transform
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
            raise ValueError("Regular meshes require an unrotated raster grid.")
        if transform.a <= 0 or transform.e >= 0:
            raise ValueError("Regular meshes require a north-up raster grid.")
        dx, dy = abs(transform.a), abs(transform.e)
        if not np.isclose(dx, dy, rtol=0, atol=1e-9 * max(dx, dy, 1)):
            raise ValueError("Regular meshes require square raster cells.")
        if config.cell_size_m is not None and not np.isclose(
            dx, config.cell_size_m, rtol=0, atol=1e-9 * max(dx, 1)
        ):
            raise ValueError(
                f"Configured regular cell size {config.cell_size_m:g} m does not match DEM {dx:g} m. "
                "Resample during DEM conditioning, not during mesh packaging."
            )
        valid = source.read_masks(1) > 0
        product = {
            "schema_version": MESH_PRODUCT_SCHEMA,
            "mesh_mode": "regular",
            "dem": str(config.dem.resolve()),
            "crs_wkt": source.crs.to_wkt(),
            "rows": source.height,
            "columns": source.width,
            "cell_size_m": dx,
            "cell_area_m2": dx * dy,
            "active_cell_count": int(valid.sum()),
            "bounds": list(source.bounds),
            "transform_gdal": list(transform.to_gdal()),
        }
    mesh_directory = config.out_dir / "mesh"
    reports_directory = config.out_dir / "reports"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    reports_directory.mkdir(parents=True, exist_ok=True)
    grid_path = mesh_directory / "hydropol_regular_grid.json"
    qa_path = reports_directory / "regular_mesh_qa.json"
    resolved_path = reports_directory / "mesh_config_resolved.json"
    grid_path.write_text(json.dumps(product, indent=2), encoding="utf-8")
    qa_path.write_text(json.dumps({**product, "passed": True}, indent=2), encoding="utf-8")
    resolved_path.write_text(
        json.dumps({"mesh": {"mode": "regular", "inputs": {"dem": str(config.dem), "out_dir": str(config.out_dir)}, "regular": {"cell_size_m": dx}}}, indent=2),
        encoding="utf-8",
    )
    return {**product, "grid": str(grid_path), "qa": str(qa_path), "resolved_config": str(resolved_path)}


def write_conservative_overlap(
    mesh_file: str | Path,
    mesh_geopackage: str | Path,
    raster_file: str | Path,
    output_file: str | Path,
    *,
    row_chunk_size: int = 32,
) -> dict[str, Any]:
    """Write exact polygon/raster overlap weights used by MATLAB and Python."""
    mesh_file, mesh_geopackage = Path(mesh_file), Path(mesh_geopackage)
    raster_file, output_file = Path(raster_file), Path(output_file)
    polygons = gpd.read_file(mesh_geopackage, layer="mesh", columns=["face_id", "geometry"])
    polygons = polygons.sort_values("face_id").reset_index(drop=True)
    if not np.array_equal(polygons.face_id.to_numpy(), np.arange(len(polygons))):
        raise ValueError("Mesh GeoPackage face_id must be contiguous and zero based.")
    geometry = polygons.geometry.to_numpy()
    tree = STRtree(geometry)
    with netCDF4.Dataset(mesh_file) as mesh_dataset:
        mesh_area = np.asarray(mesh_dataset["cell_area_m2"][:], dtype=np.float64).reshape(-1)
    if mesh_area.size != len(polygons):
        raise ValueError("UGRID and GeoPackage mesh cell counts differ.")

    mesh_indices: list[np.ndarray] = []
    raster_indices: list[np.ndarray] = []
    overlap_areas: list[np.ndarray] = []
    with rasterio.open(raster_file) as raster:
        if polygons.crs is None or raster.crs is None or polygons.crs != raster.crs:
            raise ValueError("Mesh and overlap raster must have the same projected CRS.")
        transform = raster.transform
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
            raise ValueError("Conservative overlap requires an unrotated raster grid.")
        if transform.a <= 0 or transform.e >= 0:
            raise ValueError("Conservative overlap requires a north-up raster grid.")
        rows, columns = raster.height, raster.width
        x_edges = transform.c + np.arange(columns + 1, dtype=np.float64) * transform.a
        north_edges = transform.f + np.arange(rows + 1, dtype=np.float64) * transform.e
        y_edges = north_edges[::-1]
        pixel_area = abs(transform.a * transform.e)
        for south_start in range(0, rows, row_chunk_size):
            south_stop = min(rows, south_start + row_chunk_size)
            south_rows = np.arange(south_start, south_stop, dtype=np.int64)
            columns_index = np.arange(columns, dtype=np.int64)
            rr, cc = np.meshgrid(south_rows, columns_index, indexing="ij")
            cells = box(x_edges[cc.ravel()], y_edges[rr.ravel()], x_edges[cc.ravel() + 1], y_edges[rr.ravel() + 1])
            pairs = tree.query(cells, predicate="intersects")
            if not pairs.size:
                continue
            areas = area(intersection(cells[pairs[0]], geometry[pairs[1]])).astype(np.float64)
            keep = areas > max(pixel_area, 1.0) * 1e-12
            mesh_indices.append(pairs[1, keep].astype(np.int32))
            raster_indices.append((rr.ravel()[pairs[0, keep]] * columns + cc.ravel()[pairs[0, keep]]).astype(np.int64))
            overlap_areas.append(areas[keep])
        raster_area = np.full(rows * columns, pixel_area, dtype=np.float64)

    mesh_index = np.concatenate(mesh_indices) if mesh_indices else np.empty(0, dtype=np.int32)
    raster_index = np.concatenate(raster_indices) if raster_indices else np.empty(0, dtype=np.int64)
    overlap_area = np.concatenate(overlap_areas) if overlap_areas else np.empty(0, dtype=np.float64)
    covered = np.bincount(mesh_index, weights=overlap_area, minlength=mesh_area.size)
    relative_error = np.abs(covered - mesh_area) / np.maximum(mesh_area, 1e-12)
    maximum_error = float(relative_error.max(initial=0.0))
    if maximum_error > 1e-8:
        raise ValueError(f"Conservative overlap does not cover every mesh cell; maximum relative error={maximum_error:.3g}.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(output_file, "w") as dataset:
        dataset.Conventions = "CF-1.13"
        dataset.title = "HydroBathyDEM conservative raster-polygon overlap"
        dataset.schema_version = MESH_PRODUCT_SCHEMA
        dataset.raster_rows = rows
        dataset.raster_cols = columns
        dataset.createDimension("overlap", overlap_area.size)
        dataset.createDimension("mesh_cell", mesh_area.size)
        dataset.createDimension("raster_cell", raster_area.size)
        dataset.createDimension("x_edge", x_edges.size)
        dataset.createDimension("y_edge", y_edges.size)
        dataset.createVariable("overlap_mesh_index", "i4", ("overlap",), zlib=True)[:] = mesh_index
        dataset.createVariable("overlap_raster_index", "i8", ("overlap",), zlib=True)[:] = raster_index
        dataset.createVariable("overlap_area_m2", "f8", ("overlap",), zlib=True)[:] = overlap_area
        dataset.createVariable("mesh_area_m2", "f8", ("mesh_cell",), zlib=True)[:] = mesh_area
        dataset.createVariable("raster_area_m2", "f8", ("raster_cell",), zlib=True)[:] = raster_area
        dataset.createVariable("x_edges", "f8", ("x_edge",))[:] = x_edges
        dataset.createVariable("y_edges", "f8", ("y_edge",))[:] = y_edges
    return {
        "overlap": str(output_file),
        "overlap_count": int(overlap_area.size),
        "maximum_mesh_area_relative_error": maximum_error,
    }


def build_mesh_product(values: dict[str, Any]) -> dict[str, Any]:
    mode, selected = _selected_mesh(values)
    if mode == "regular":
        return build_regular_mesh(RegularMeshConfig.from_mapping(selected))
    selected["mesh_mode"] = "voronoi_fv"
    config = HybridMeshConfig.from_mapping(selected)
    result = build_hybrid_mesh(config)
    if config.stop_after_step is not None:
        return {"schema_version": MESH_PRODUCT_SCHEMA, "mesh_mode": mode, **result}
    overlap_path = config.out_dir / "mesh" / "hydropol_mesh_overlap.nc"
    overlap = write_conservative_overlap(result["ugrid"], result["geopackage"], config.dem, overlap_path)
    resolved_path = config.out_dir / "reports" / "mesh_config_resolved.json"
    resolved_path.write_text(json.dumps({"mesh_mode": mode, **asdict(config)}, indent=2, default=str), encoding="utf-8")
    return {
        "schema_version": MESH_PRODUCT_SCHEMA,
        "mesh_mode": mode,
        **result,
        **overlap,
        "resolved_config": str(resolved_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a regular or Voronoi HydroPol mesh product.")
    parser.add_argument("--config", required=True, type=Path, help="JSON/TOML mesh-product configuration.")
    args = parser.parse_args(argv)
    print(json.dumps(build_mesh_product(load_config_file(args.config)), indent=2))


if __name__ == "__main__":
    main()
