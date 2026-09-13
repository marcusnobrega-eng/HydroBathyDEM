from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from dem_processing.hybrid_mesh import (
    ComputationPointSet,
    HybridMeshConfig,
    build_hybrid_mesh,
    compile_mesh_from_points,
    generate_computation_points,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tests" / "baselines" / "mesh_editor_phase0.json"
PUNE_DEM = ROOT / "examples" / "pune_catchment" / "data" / "DEM_fabdem_pune.tif"


def _write_dem(path: Path, values: np.ndarray, transform, crs) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(values.astype("float32"), 1)


def _case_dem(case_name: str, path: Path) -> Path:
    if case_name == "synthetic_slope":
        rows, columns = np.mgrid[0:12, 0:12]
        values = 120.0 - 0.8 * columns - 0.4 * rows + 0.05 * np.sin(columns)
        _write_dem(path, values, from_origin(500000.0, 2000000.0, 10.0, 10.0), "EPSG:32643")
        return path

    if case_name != "pune_real_window":
        raise ValueError(f"Unknown mesh baseline case: {case_name}")
    with rasterio.open(PUNE_DEM) as source:
        window = Window(600, 600, 24, 24)
        values = source.read(1, window=window, masked=True)
        assert values.shape == (24, 24)
        assert not np.ma.getmaskarray(values).any()
        transform = source.window_transform(window)
        crs = source.crs
    _write_dem(path, values.filled(np.nan), transform, crs)
    return path


def _build(case_name: str, directory: Path, settings: dict[str, float]) -> dict[str, object]:
    directory.mkdir(parents=True)
    dem = _case_dem(case_name, directory / "dem.tif")
    config = HybridMeshConfig(
        dem=dem,
        out_dir=directory / "product",
        background_width_m=settings["background_width_m"],
        urban_width_m=settings["background_width_m"],
        river_along_river_cell_length_m=settings["background_width_m"],
        river_cross_river_target_width_m=settings["background_width_m"],
        floodplain_enabled=False,
        minimum_hydraulic_width_m=settings["minimum_hydraulic_width_m"],
        minimum_upstream_area_km2=1000.0,
        river_cell_style="voronoi",
        write_diagnostics=False,
        max_seed_count=5000,
    )
    result = build_hybrid_mesh(config)
    return _snapshot(Path(result["ugrid"]), Path(result["qa"]))


def _build_from_points(
    case_name: str, directory: Path, settings: dict[str, float]
) -> tuple[dict[str, object], ComputationPointSet]:
    directory.mkdir(parents=True)
    dem = _case_dem(case_name, directory / "dem.tif")
    config = HybridMeshConfig(
        dem=dem,
        out_dir=directory / "product",
        background_width_m=settings["background_width_m"],
        urban_width_m=settings["background_width_m"],
        river_along_river_cell_length_m=settings["background_width_m"],
        river_cross_river_target_width_m=settings["background_width_m"],
        floodplain_enabled=False,
        minimum_hydraulic_width_m=settings["minimum_hydraulic_width_m"],
        minimum_upstream_area_km2=1000.0,
        river_cell_style="voronoi",
        write_diagnostics=False,
        max_seed_count=5000,
    )
    points = generate_computation_points(config)
    result = compile_mesh_from_points(config, points)
    return _snapshot(Path(result["ugrid"]), Path(result["qa"])), points


def _digest(values: np.ndarray, *, decimals: int | None = None) -> str:
    array = np.asarray(values)
    if decimals is not None:
        array = np.round(array.astype(np.float64), decimals=decimals)
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _snapshot(mesh_path: Path, qa_path: Path) -> dict[str, object]:
    with netCDF4.Dataset(mesh_path) as dataset:
        centers = np.column_stack(
            (dataset["mesh2d_face_x"][:], dataset["mesh2d_face_y"][:])
        )
        center_order = np.lexsort((centers[:, 1], centers[:, 0]))
        centers = centers[center_order]
        area = np.asarray(dataset["cell_area_m2"][:], dtype=np.float64)
        bed = np.asarray(dataset["cell_bed_elevation_m"][:], dtype=np.float64)
        cfl = np.asarray(dataset["cell_cfl_width_m"][:], dtype=np.float64)
        owner = np.asarray(dataset["edge_owner"][:], dtype=np.int64)
        neighbor = np.asarray(dataset["edge_neighbor"][:], dtype=np.int64)
        edge_length = np.asarray(dataset["edge_length_m"][:], dtype=np.float64)
        center_distance = np.asarray(
            dataset["edge_center_distance_m"][:], dtype=np.float64
        )
        feature_class = np.asarray(dataset["cell_feature_class"][:], dtype=str)
        degree = np.bincount(
            np.concatenate((owner, neighbor[neighbor >= 0])), minlength=area.size
        )
        internal = neighbor >= 0
        node_count = len(dataset.dimensions["node"])

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    return {
        "cell_count": int(area.size),
        "edge_count": int(owner.size),
        "node_count": int(node_count),
        "internal_edge_count": int(internal.sum()),
        "boundary_edge_count": int((~internal).sum()),
        "degree_histogram": {
            str(value): int((degree == value).sum()) for value in np.unique(degree)
        },
        "center_extent_m": [
            round(float(centers[:, 0].min()), 6),
            round(float(centers[:, 1].min()), 6),
            round(float(centers[:, 0].max()), 6),
            round(float(centers[:, 1].max()), 6),
        ],
        "center_digest": _digest(centers, decimals=6),
        "area_sum_m2": round(float(area.sum()), 6),
        "area_min_m2": round(float(area.min()), 6),
        "area_max_m2": round(float(area.max()), 6),
        "bed_min_m": round(float(bed.min()), 6),
        "bed_max_m": round(float(bed.max()), 6),
        "bed_sum_m": round(float(bed.sum()), 6),
        "minimum_cfl_width_m": round(float(cfl.min()), 6),
        "minimum_internal_edge_length_m": round(float(edge_length[internal].min()), 6),
        "minimum_internal_center_distance_m": round(
            float(center_distance[internal].min()), 6
        ),
        "feature_class_counts": {
            value: int((feature_class == value).sum()) for value in np.unique(feature_class)
        },
        "qa": {
            "short_internal_face_count": int(qa["short_internal_face_count"]),
            "close_internal_center_count": int(qa["close_internal_center_count"]),
            "interior_unpaired_face_count": int(qa["interior_unpaired_face_count"]),
            "faces_with_more_than_two_owners": int(qa["faces_with_more_than_two_owners"]),
            "cells_with_interior_ring": int(qa["cells_with_interior_ring"]),
            "cell_area_sum_km2": round(float(qa["cell_area_sum_km2"]), 9),
            "domain_area_km2": round(float(qa["domain_area_km2"]), 9),
        },
    }


@pytest.mark.parametrize("case_name", ("synthetic_slope", "pune_real_window"))
def test_mesh_editor_phase0_baseline_is_reproducible(
    case_name: str, tmp_path: Path
) -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    case = baseline["cases"][case_name]
    first = _build(case_name, tmp_path / "first", case["settings"])
    second = _build(case_name, tmp_path / "second", case["settings"])

    assert first == second
    assert first == case["expected"]


@pytest.mark.parametrize("case_name", ("synthetic_slope", "pune_real_window"))
def test_phase1_explicit_points_match_legacy_mesh(
    case_name: str, tmp_path: Path
) -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    case = baseline["cases"][case_name]
    legacy = _build(case_name, tmp_path / "legacy", case["settings"])
    compiled, points = _build_from_points(
        case_name, tmp_path / "explicit", case["settings"]
    )

    assert compiled == legacy == case["expected"]
    assert np.array_equal(points.point_id, np.arange(len(points.coordinates)))
    assert points.active.all()
    assert not points.manually_modified.any()
    assert set(points.origin).issubset(
        {"background", "river", "floodplain", "waterbody", "breakline"}
    )


def test_computation_point_set_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="IDs must be unique"):
        ComputationPointSet(
            point_id=np.asarray([1, 1]),
            coordinates=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
            origin=np.asarray(["background", "background"]),
            source_id=np.asarray(["background:0", "background:1"]),
            active=np.asarray([True, True]),
            locked=np.asarray([False, False]),
            manually_modified=np.asarray([False, False]),
            crs_wkt="",
        )


def test_compile_mesh_uses_active_computation_points(tmp_path: Path) -> None:
    directory = tmp_path / "edited"
    directory.mkdir()
    dem = _case_dem("synthetic_slope", directory / "dem.tif")
    config = HybridMeshConfig(
        dem=dem,
        out_dir=directory / "product",
        background_width_m=30.0,
        urban_width_m=30.0,
        river_along_river_cell_length_m=30.0,
        river_cross_river_target_width_m=30.0,
        floodplain_enabled=False,
        minimum_hydraulic_width_m=10.0,
        minimum_upstream_area_km2=1000.0,
        river_cell_style="voronoi",
        write_diagnostics=False,
    )
    points = generate_computation_points(config)
    active = points.active.copy()
    active[-1] = False

    result = compile_mesh_from_points(config, replace(points, active=active))

    assert result["cells"] < len(points.coordinates)
