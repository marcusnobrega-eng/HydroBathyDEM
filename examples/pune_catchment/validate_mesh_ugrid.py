"""Check a generated UGRID mesh against the contract HydroPol2D relies on.

The mesh QA report describes how the mesh was built.  This script instead asks
whether the exported topology is usable by a cell-centred finite-volume solver,
which is a different question and the one that was previously unchecked:

* every internal face must have two distinct owners, or the solver finds a wall
  in the middle of the domain and mass stops crossing it;
* a face marked boundary must actually be on the domain edge;
* the outward face normals of a closed cell must sum to zero, otherwise the
  divergence of a uniform flux is non-zero and the scheme is not conservative;
* cell areas, control widths, and centre distances must be strictly positive;
* the cells must tile the domain -- no void to lose mass into, no overlap to
  count it twice.

Usage:
    PYTHONPATH=src python3 examples/pune_catchment/validate_mesh_ugrid.py \
        --mesh <out_dir>/mesh/hydropol_hybrid_mesh.nc \
        --geopackage <out_dir>/mesh/hydropol_hybrid_mesh.gpkg \
        --domain examples/pune_catchment/data/pune_full_valid_domain.geojson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
from shapely import STRtree, points as shapely_points


def _line_parts(geometry) -> list:
    if geometry is None or geometry.is_empty:
        return []
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return [geometry]


def check_ugrid(mesh_path: Path, geopackage: Path | None, domain_path: Path | None) -> dict:
    report: dict[str, object] = {"mesh": str(mesh_path)}
    with netCDF4.Dataset(mesh_path) as dataset:
        owner = np.asarray(dataset["edge_owner"][:], dtype=np.int64)
        neighbour = np.asarray(dataset["edge_neighbor"][:], dtype=np.int64)
        length = np.asarray(dataset["edge_length_m"][:], dtype=np.float64)
        normal_x = np.asarray(dataset["edge_normal_x"][:], dtype=np.float64)
        normal_y = np.asarray(dataset["edge_normal_y"][:], dtype=np.float64)
        midpoint_x = np.asarray(dataset["edge_midpoint_x"][:], dtype=np.float64)
        midpoint_y = np.asarray(dataset["edge_midpoint_y"][:], dtype=np.float64)
        centre_distance = np.asarray(dataset["edge_center_distance_m"][:], dtype=np.float64)
        area = np.asarray(dataset["cell_area_m2"][:], dtype=np.float64)
        control_width = np.asarray(dataset["cell_cfl_width_m"][:], dtype=np.float64)
        feature_class = np.asarray(dataset["cell_feature_class"][:], dtype=object)
        crs_wkt = dataset.crs_wkt

    cells = len(area)
    internal = neighbour >= 0
    report["cells"] = cells
    report["edges"] = int(len(owner))
    report["internal_faces"] = int(internal.sum())
    report["boundary_faces"] = int((~internal).sum())

    # 1. Owners must differ, and no pair may appear twice.
    report["self_paired_faces"] = int((internal & (owner == neighbour)).sum())
    pairs = np.sort(np.column_stack((owner[internal], neighbour[internal])), axis=1)
    report["duplicate_internal_pairs"] = int(len(pairs) - len(np.unique(pairs, axis=0)))

    # 2. Boundary faces must lie on the domain edge.
    if domain_path is not None:
        domain = gpd.read_file(domain_path).to_crs(crs_wkt).geometry.union_all()
        boundary_parts = _line_parts(domain.boundary)
        edge_mid = shapely_points(np.column_stack((midpoint_x[~internal], midpoint_y[~internal])))
        near = np.unique(
            STRtree(boundary_parts).query(edge_mid, predicate="dwithin", distance=1.0)[0]
        )
        interior_boundary = np.setdiff1d(np.arange(int((~internal).sum())), near)
        report["boundary_faces_not_on_domain_edge"] = int(len(interior_boundary))
        report["boundary_faces_not_on_domain_edge_length_m"] = float(
            length[~internal][interior_boundary].sum()
        )
        report["domain_perimeter_m"] = float(domain.length)
        report["domain_area_km2"] = float(domain.area / 1e6)

    # 3. Closed-cell normal sum: sum(L * n) over a cell's faces must vanish.
    closure_x = np.zeros(cells)
    closure_y = np.zeros(cells)
    np.add.at(closure_x, owner, length * normal_x)
    np.add.at(closure_y, owner, length * normal_y)
    np.subtract.at(closure_x, neighbour[internal], length[internal] * normal_x[internal])
    np.subtract.at(closure_y, neighbour[internal], length[internal] * normal_y[internal])
    perimeter_scale = np.zeros(cells)
    np.add.at(perimeter_scale, owner, length)
    np.add.at(perimeter_scale, neighbour[internal], length[internal])
    closure = np.hypot(closure_x, closure_y) / np.maximum(perimeter_scale, 1e-12)
    report["max_relative_normal_closure_error"] = float(closure.max())
    report["cells_with_closure_error_gt_1e_6"] = int((closure > 1e-6).sum())

    # 4. Strictly positive metrics.
    report["nonpositive_cell_area"] = int((area <= 0).sum())
    report["nonpositive_face_length"] = int((length <= 0).sum())
    report["nonpositive_center_distance"] = int((centre_distance <= 0).sum())
    report["nonpositive_control_width"] = int((control_width <= 0).sum())
    report["min_face_length_m"] = float(length.min())
    report["min_center_distance_m"] = float(centre_distance.min())
    report["min_control_width_m"] = float(control_width.min())

    # 5. Partition: the cells must tile the domain.
    if geopackage is not None:
        frame = gpd.read_file(geopackage, layer="mesh")
        report["geopackage_cells"] = int(len(frame))
        report["cell_area_sum_km2"] = float(frame.area.sum() / 1e6)
        report["cells_with_interior_ring"] = int(sum(1 for item in frame.geometry if item.interiors))
        report["invalid_cells"] = int((~frame.geometry.is_valid).sum())
        union = frame.geometry.union_all()
        report["overlap_area_m2"] = float(frame.area.sum() - union.area)
        if domain_path is not None:
            report["void_area_m2"] = float(domain.difference(union).area)
    report["class_counts"] = {
        str(name): int(count) for name, count in zip(*np.unique(feature_class, return_counts=True))
    }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--geopackage", type=Path)
    parser.add_argument("--domain", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = check_ugrid(args.mesh, args.geopackage, args.domain)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
