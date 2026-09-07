#!/usr/bin/env python3
"""Stamp a validated legacy bundle with contract-1.0 metadata only."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dem_processing.mesh_contract import (  # noqa: E402
    EDGE_NORMAL_CONVENTION,
    OVERLAP_RASTER_ORDER,
    SUBGRID_CONVEYANCE_CONVENTION,
    apply_contract_attributes,
)


MESH_UNITS = {
    "mesh2d_face_x": "m", "mesh2d_face_y": "m", "cell_area_m2": "m2",
    "cell_bed_elevation_m": "m", "cell_hydraulic_roughness": "s m-1/3",
    "cell_cfl_width_m": "m", "edge_length_m": "m",
    "edge_center_distance_m": "m", "edge_midpoint_x": "m",
    "edge_midpoint_y": "m",
}
OVERLAP_UNITS = {
    "overlap_area_m2": "m2", "mesh_area_m2": "m2",
    "raster_area_m2": "m2", "x_edges": "m", "y_edges": "m",
}
SUBGRID_UNITS = {
    "cell_datum_m": "m", "cell_zeta_m": "m", "cell_volume_m3": "m3",
    "cell_wet_area_m2": "m2", "cell_plan_area_m2": "m2",
    "face_datum_m": "m", "face_zeta_m": "m", "face_flow_area_m2": "m2",
    "face_perimeter_m": "m", "face_conveyance": "m3 s-1",
    "face_length_m": "m",
}


def _array_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        for name, variable in dataset.variables.items():
            values = variable[:]
            array = np.ascontiguousarray(values)
            digest = hashlib.sha256()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            if array.dtype.kind in {"O", "U", "S"}:
                digest.update(repr(array.tolist()).encode("utf-8"))
            else:
                digest.update(array.tobytes())
            hashes[name] = digest.hexdigest()
    return hashes


def _set_units(dataset: Dataset, units: dict[str, str], path: Path) -> None:
    missing = [name for name in units if name not in dataset.variables]
    if missing:
        raise ValueError(f"{path}: cannot migrate; missing {', '.join(missing)}")
    for name, value in units.items():
        dataset[name].units = value


def migrate_bundle(mesh: Path, overlap: Path | None, subgrid: Path | None) -> None:
    paths = [path for path in (mesh, overlap, subgrid) if path is not None]
    before = {path: _array_hashes(path) for path in paths}

    with Dataset(mesh, "a") as dataset:
        crs_wkt = str(getattr(dataset, "crs_wkt", "")).strip()
        if not crs_wkt:
            raise ValueError(f"{mesh}: cannot migrate without crs_wkt")
        apply_contract_attributes(dataset, product="hydraulic_mesh")
        dataset.edge_normal_convention = EDGE_NORMAL_CONVENTION
        _set_units(dataset, MESH_UNITS, mesh)
        if "mesh2d_face_nodes" in dataset.variables:
            dataset["mesh2d_face_nodes"].start_index = 0

    if overlap is not None:
        with Dataset(overlap, "a") as dataset:
            apply_contract_attributes(dataset, product="conservative_raster_overlap")
            dataset.crs_wkt = crs_wkt
            dataset.raster_index_order = OVERLAP_RASTER_ORDER
            _set_units(dataset, OVERLAP_UNITS, overlap)
            if "boundary_neighbor_sentinel" in dataset.ncattrs():
                dataset.delncattr("boundary_neighbor_sentinel")

    if subgrid is not None:
        with Dataset(subgrid, "a") as dataset:
            apply_contract_attributes(dataset, product="hydraulic_subgrid_tables")
            dataset.crs_wkt = crs_wkt
            dataset.conveyance_convention = SUBGRID_CONVEYANCE_CONVENTION
            _set_units(dataset, SUBGRID_UNITS, subgrid)
            if "boundary_neighbor_sentinel" in dataset.ncattrs():
                dataset.delncattr("boundary_neighbor_sentinel")

    after = {path: _array_hashes(path) for path in paths}
    if before != after:
        changed = {
            str(path): sorted(
                name for name in set(before[path]) | set(after[path])
                if before[path].get(name) != after[path].get(name)
            )
            for path in paths if before[path] != after[path]
        }
        raise RuntimeError(f"Migration changed one or more numerical arrays: {changed}")
    for path in paths:
        print(f"Migrated metadata only: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--overlap", type=Path)
    parser.add_argument("--subgrid", type=Path)
    args = parser.parse_args()
    migrate_bundle(args.mesh, args.overlap, args.subgrid)


if __name__ == "__main__":
    main()
