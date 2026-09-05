"""User-facing regular/Voronoi mesh product dispatcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import netCDF4
import numpy as np
import rasterio
from shapely import STRtree, area, box, intersection

from . import __version__
from .config import load_config_file
from .hybrid_mesh import HybridMeshConfig, build_hybrid_mesh
from .mesh_contract import (
    MESH_PRODUCT_SCHEMA,
    OVERLAP_RASTER_ORDER,
    apply_contract_attributes,
)
from .subgrid_product import SubgridProductConfig, build_subgrid_product


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


def _without_subgrid(values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Separate optional product tables before parsing the mesh builder config."""
    cleaned = dict(values)
    if "mesh" not in cleaned:
        subgrid = cleaned.pop("subgrid", None)
        return cleaned, subgrid
    mesh = dict(cleaned["mesh"])
    subgrid = mesh.pop("subgrid", None)
    cleaned["mesh"] = mesh
    if subgrid is not None and not isinstance(subgrid, dict):
        raise ValueError("mesh.subgrid must be a configuration object.")
    return cleaned, subgrid


def build_regular_mesh(config: RegularMeshConfig) -> dict[str, Any]:
    """Package a conditioned raster as the same hydraulic product as Voronoi."""
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
        dem = source.read(1).astype(np.float64)
        valid = (source.read_masks(1) > 0) & np.isfinite(dem)
        if not np.any(valid):
            raise ValueError("The conditioned DEM has no active finite cells.")
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
        crs = source.crs
    mesh_directory = config.out_dir / "mesh"
    reports_directory = config.out_dir / "reports"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    reports_directory.mkdir(parents=True, exist_ok=True)
    grid_path = mesh_directory / "hydropol_regular_grid.json"
    ugrid_path = mesh_directory / "hydropol_regular_mesh.nc"
    geopackage_path = mesh_directory / "hydropol_regular_mesh.gpkg"
    qa_path = reports_directory / "regular_mesh_qa.json"
    resolved_path = reports_directory / "mesh_config_resolved.json"
    grid_path.write_text(json.dumps(product, indent=2), encoding="utf-8")
    topology = _write_regular_ugrid(
        ugrid_path, geopackage_path, dem, valid, transform, crs,
    )
    qa_path.write_text(json.dumps({**product, "passed": True}, indent=2), encoding="utf-8")
    resolved_path.write_text(
        json.dumps({"mesh": {"mode": "regular", "inputs": {"dem": str(config.dem), "out_dir": str(config.out_dir)}, "regular": {"cell_size_m": dx}}}, indent=2),
        encoding="utf-8",
    )
    return {
        **product,
        **topology,
        "grid": str(grid_path),
        "qa": str(qa_path),
        "resolved_config": str(resolved_path),
    }


def _write_regular_ugrid(
    output_path: Path,
    geopackage_path: Path,
    dem: np.ndarray,
    valid: np.ndarray,
    transform: Any,
    crs: Any,
) -> dict[str, Any]:
    """Write active raster cells as a zero-based finite-volume UGRID mesh."""
    rows, columns = np.nonzero(valid)
    n_cells = rows.size
    node_lookup: dict[tuple[int, int], int] = {}
    node_grid: list[tuple[int, int]] = []

    def node_id(row: int, column: int) -> int:
        key = (row, column)
        if key not in node_lookup:
            node_lookup[key] = len(node_grid)
            node_grid.append(key)
        return node_lookup[key]

    connectivity = np.empty((n_cells, 4), dtype=np.int32)
    edge_owners: dict[tuple[int, int], list[int]] = {}
    for face, (row, column) in enumerate(zip(rows, columns, strict=True)):
        # NW, SW, SE, NE is counter-clockwise for a north-up raster.
        ring = (
            node_id(int(row), int(column)),
            node_id(int(row + 1), int(column)),
            node_id(int(row + 1), int(column + 1)),
            node_id(int(row), int(column + 1)),
        )
        connectivity[face] = ring
        for start, end in zip(ring, ring[1:] + ring[:1], strict=True):
            edge_owners.setdefault(tuple(sorted((start, end))), []).append(face)

    node_rows = np.asarray([item[0] for item in node_grid], dtype=np.float64)
    node_columns = np.asarray([item[1] for item in node_grid], dtype=np.float64)
    node_x = transform.c + node_columns * transform.a
    node_y = transform.f + node_rows * transform.e
    center_x = transform.c + (columns.astype(np.float64) + 0.5) * transform.a
    center_y = transform.f + (rows.astype(np.float64) + 0.5) * transform.e

    edge_nodes = np.asarray(list(edge_owners), dtype=np.int32)
    owner = np.asarray([items[0] for items in edge_owners.values()], dtype=np.int32)
    neighbor = np.asarray([
        items[1] if len(items) == 2 else -1 for items in edge_owners.values()
    ], dtype=np.int32)
    if any(len(items) > 2 for items in edge_owners.values()):
        raise ValueError("Regular mesh contains a non-manifold face shared by more than two cells.")
    x0, y0 = node_x[edge_nodes[:, 0]], node_y[edge_nodes[:, 0]]
    x1, y1 = node_x[edge_nodes[:, 1]], node_y[edge_nodes[:, 1]]
    midpoint_x, midpoint_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    length = np.hypot(x1 - x0, y1 - y0)
    internal = neighbor >= 0
    vector_x = midpoint_x - center_x[owner]
    vector_y = midpoint_y - center_y[owner]
    vector_x[internal] = center_x[neighbor[internal]] - center_x[owner[internal]]
    vector_y[internal] = center_y[neighbor[internal]] - center_y[owner[internal]]
    center_distance = np.hypot(vector_x, vector_y)
    normal_x = vector_x / center_distance
    normal_y = vector_y / center_distance
    cfl_width = np.full(n_cells, np.inf, dtype=np.float64)
    for edge, (left, right) in enumerate(zip(owner, neighbor, strict=True)):
        width = center_distance[edge] if right >= 0 else 2.0 * center_distance[edge]
        cfl_width[left] = min(cfl_width[left], width)
        if right >= 0:
            cfl_width[right] = min(cfl_width[right], width)

    dx, dy = abs(transform.a), abs(transform.e)
    areas = np.full(n_cells, dx * dy, dtype=np.float64)
    bed = dem[rows, columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(output_path, "w") as dataset:
        dataset.Conventions = "CF-1.13, UGRID-1.0"
        dataset.title = "HydroBathyDEM regular HydroPol2D mesh"
        dataset.crs_wkt = crs.to_wkt()
        dataset.edge_normal_convention = "unit normal points outward from edge_owner"
        apply_contract_attributes(dataset, product="hydraulic_mesh")
        dataset.createDimension("face", n_cells)
        dataset.createDimension("node", len(node_grid))
        dataset.createDimension("edge", owner.size)
        dataset.createDimension("max_face_nodes", 4)
        dataset.createDimension("two", 2)
        topology = dataset.createVariable("mesh2d", "i4")
        topology.cf_role = "mesh_topology"
        topology.topology_dimension = 2
        topology.node_coordinates = "mesh2d_node_x mesh2d_node_y"
        topology.face_node_connectivity = "mesh2d_face_nodes"
        topology.edge_node_connectivity = "mesh2d_edge_nodes"
        topology.face_coordinates = "mesh2d_face_x mesh2d_face_y"
        face_nodes = dataset.createVariable(
            "mesh2d_face_nodes", "i4", ("face", "max_face_nodes"), fill_value=-1,
        )
        face_nodes.start_index = 0
        face_nodes[:] = connectivity
        saved_edge_nodes = dataset.createVariable("mesh2d_edge_nodes", "i4", ("edge", "two"))
        saved_edge_nodes.start_index = 0
        saved_edge_nodes[:] = edge_nodes
        for name, dimension, values, units in (
            ("mesh2d_node_x", "node", node_x, "m"),
            ("mesh2d_node_y", "node", node_y, "m"),
            ("mesh2d_face_x", "face", center_x, "m"),
            ("mesh2d_face_y", "face", center_y, "m"),
            ("cell_area_m2", "face", areas, "m2"),
            ("cell_bed_elevation_m", "face", bed, "m"),
            ("cell_target_width_m", "face", np.full(n_cells, dx), "m"),
            ("cell_cfl_width_m", "face", cfl_width, "m"),
            ("cell_hydraulic_roughness", "face", np.full(n_cells, np.nan), "s m-1/3"),
            ("edge_length_m", "edge", length, "m"),
            ("edge_center_distance_m", "edge", center_distance, "m"),
            ("edge_midpoint_x", "edge", midpoint_x, "m"),
            ("edge_midpoint_y", "edge", midpoint_y, "m"),
        ):
            variable = dataset.createVariable(name, "f8", (dimension,))
            variable.units = units
            variable[:] = values
        dataset.createVariable("edge_owner", "i4", ("edge",))[:] = owner
        dataset.createVariable("edge_neighbor", "i4", ("edge",))[:] = neighbor
        dataset.createVariable("edge_normal_x", "f8", ("edge",))[:] = normal_x
        dataset.createVariable("edge_normal_y", "f8", ("edge",))[:] = normal_y
        dataset.createVariable("edge_boundary_type", "i2", ("edge",))[:] = 0
        dataset.createVariable("cell_refinement_source", "i2", ("face",))[:] = 0
        dataset.createVariable("cell_river_fraction", "f8", ("face",))[:] = 0.0
        dataset.createVariable("cell_river_channel_fraction", "f8", ("face",))[:] = 0.0
        dataset.createVariable("cell_floodplain_fraction", "f8", ("face",))[:] = 0.0
        dataset.createVariable("cell_waterbody_fraction", "f8", ("face",))[:] = 0.0
        dataset.createVariable("cell_feature_class", str, ("face",))[:] = np.full(
            n_cells, "regular", dtype=object,
        )

    for sidecar in (geopackage_path, Path(f"{geopackage_path}-wal"), Path(f"{geopackage_path}-shm")):
        sidecar.unlink(missing_ok=True)
    polygons = [
        box(
            transform.c + column * transform.a,
            transform.f + (row + 1) * transform.e,
            transform.c + (column + 1) * transform.a,
            transform.f + row * transform.e,
        )
        for row, column in zip(rows, columns, strict=True)
    ]
    gpd.GeoDataFrame(
        {"face_id": np.arange(n_cells), "area_m2": areas, "feature_class": "regular"},
        geometry=polygons,
        crs=crs,
    ).to_file(geopackage_path, layer="mesh", driver="GPKG")
    return {
        "ugrid": str(output_path),
        "geopackage": str(geopackage_path),
        "cell_count": int(n_cells),
        "face_count": int(owner.size),
        "boundary_face_count": int((neighbor < 0).sum()),
        "minimum_exported_cfl_width_m": float(cfl_width.min()),
    }


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
        dataset.crs_wkt = polygons.crs.to_wkt()
        dataset.raster_rows = rows
        dataset.raster_cols = columns
        dataset.raster_index_order = OVERLAP_RASTER_ORDER
        apply_contract_attributes(dataset, product="conservative_raster_overlap")
        dataset.createDimension("overlap", overlap_area.size)
        dataset.createDimension("mesh_cell", mesh_area.size)
        dataset.createDimension("raster_cell", raster_area.size)
        dataset.createDimension("x_edge", x_edges.size)
        dataset.createDimension("y_edge", y_edges.size)
        dataset.createVariable("overlap_mesh_index", "i4", ("overlap",), zlib=True)[:] = mesh_index
        dataset.createVariable("overlap_raster_index", "i8", ("overlap",), zlib=True)[:] = raster_index
        for name, dimensions, values, units in (
            ("overlap_area_m2", ("overlap",), overlap_area, "m2"),
            ("mesh_area_m2", ("mesh_cell",), mesh_area, "m2"),
            ("raster_area_m2", ("raster_cell",), raster_area, "m2"),
            ("x_edges", ("x_edge",), x_edges, "m"),
            ("y_edges", ("y_edge",), y_edges, "m"),
        ):
            variable = dataset.createVariable(name, "f8", dimensions, zlib=True)
            variable.units = units
            variable[:] = values
    return {
        "overlap": str(output_file),
        "overlap_count": int(overlap_area.size),
        "maximum_mesh_area_relative_error": maximum_error,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ("git", "-C", str(repository), *args),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "version": __version__,
        "repository": str(repository),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(status) if status is not None else None,
    }


def _configuration_sha256(values: dict[str, Any]) -> str:
    serialized = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_product_manifest(
    out_dir: Path,
    files: dict[str, str],
    *,
    configuration_sha256: str,
) -> Path:
    """Write a reproducibility manifest for generated mesh-package files."""
    records = []
    for role, value in sorted(files.items()):
        if not isinstance(value, (str, Path)):
            continue
        path = Path(value)
        if path.is_file():
            records.append({
                "role": role,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    manifest = {
        "schema_version": MESH_PRODUCT_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "generator": _source_provenance(),
        "configuration_sha256": configuration_sha256,
        "files": records,
    }
    path = out_dir / "reports" / "mesh_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_mesh_product(values: dict[str, Any]) -> dict[str, Any]:
    configuration_sha256 = _configuration_sha256(values)
    values, subgrid_values = _without_subgrid(values)
    mode, selected = _selected_mesh(values)
    if mode == "regular":
        config = RegularMeshConfig.from_mapping(selected)
        result_files = build_regular_mesh(config)
        overlap_path = config.out_dir / "mesh" / "hydropol_mesh_overlap.nc"
        result_files.update(write_conservative_overlap(
            result_files["ugrid"], result_files["geopackage"], config.dem, overlap_path,
        ))
        if subgrid_values and subgrid_values.get("enabled", True):
            normalized_subgrid = dict(subgrid_values)
            normalized_subgrid.setdefault("fine_dem", str(config.dem))
            subgrid_config = SubgridProductConfig.from_mapping(normalized_subgrid)
            subgrid_path = config.out_dir / "mesh" / subgrid_config.output_name
            result_files.update(build_subgrid_product(
                result_files["ugrid"], result_files["overlap"], subgrid_config, subgrid_path,
            ))
        manifest_path = write_product_manifest(
            config.out_dir,
            result_files,
            configuration_sha256=configuration_sha256,
        )
        result_files["manifest"] = str(manifest_path)
        return result_files
    selected["mesh_mode"] = "voronoi_fv"
    config = HybridMeshConfig.from_mapping(selected)
    result = build_hybrid_mesh(config)
    if config.stop_after_step is not None:
        return {"schema_version": MESH_PRODUCT_SCHEMA, "mesh_mode": mode, **result}
    overlap_path = config.out_dir / "mesh" / "hydropol_mesh_overlap.nc"
    overlap = write_conservative_overlap(result["ugrid"], result["geopackage"], config.dem, overlap_path)
    resolved_path = config.out_dir / "reports" / "mesh_config_resolved.json"
    resolved = {"mesh_mode": mode, **asdict(config)}
    result_files = {
        "schema_version": MESH_PRODUCT_SCHEMA,
        "mesh_mode": mode,
        **result,
        **overlap,
    }
    if subgrid_values and subgrid_values.get("enabled", True):
        normalized_subgrid = dict(subgrid_values)
        normalized_subgrid.setdefault("fine_dem", str(config.dem))
        subgrid_config = SubgridProductConfig.from_mapping(normalized_subgrid)
        subgrid_path = config.out_dir / "mesh" / subgrid_config.output_name
        result_files.update(build_subgrid_product(
            result["ugrid"], overlap["overlap"], subgrid_config, subgrid_path,
        ))
        resolved["subgrid"] = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(subgrid_config).items()}
    resolved_path.write_text(json.dumps(resolved, indent=2, default=str), encoding="utf-8")
    result_files["resolved_config"] = str(resolved_path)
    manifest_path = write_product_manifest(
        config.out_dir,
        result_files,
        configuration_sha256=configuration_sha256,
    )
    result_files["manifest"] = str(manifest_path)
    return result_files


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a regular or Voronoi HydroPol mesh product.")
    parser.add_argument("--config", required=True, type=Path, help="JSON/TOML mesh-product configuration.")
    args = parser.parse_args(argv)
    print(json.dumps(build_mesh_product(load_config_file(args.config)), indent=2))


if __name__ == "__main__":
    main()
