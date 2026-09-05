from pathlib import Path

import netCDF4
import numpy as np

from dem_processing.mesh_contract import apply_contract_attributes, OVERLAP_RASTER_ORDER
from dem_processing.mesh_mapping import (
    aggregate_class_fractions,
    aggregate_continuous,
    read_overlap,
    remap_mesh_depth_to_raster,
)


def _overlap(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        apply_contract_attributes(dataset, product="conservative_raster_overlap")
        dataset.raster_rows = 1
        dataset.raster_cols = 2
        dataset.raster_index_order = OVERLAP_RASTER_ORDER
        for name, size in (("overlap", 2), ("mesh_cell", 1), ("raster_cell", 2), ("x_edge", 3), ("y_edge", 2)):
            dataset.createDimension(name, size)
        dataset.createVariable("overlap_mesh_index", "i4", ("overlap",))[:] = [0, 0]
        dataset.createVariable("overlap_raster_index", "i4", ("overlap",))[:] = [0, 1]
        dataset.createVariable("overlap_area_m2", "f8", ("overlap",))[:] = [50, 50]
        dataset.createVariable("mesh_area_m2", "f8", ("mesh_cell",))[:] = [100]
        dataset.createVariable("raster_area_m2", "f8", ("raster_cell",))[:] = [50, 50]
        dataset.createVariable("x_edges", "f8", ("x_edge",))[:] = [0, 5, 10]
        dataset.createVariable("y_edges", "f8", ("y_edge",))[:] = [0, 10]


def test_overlap_supports_both_conservative_transfer_directions(tmp_path: Path) -> None:
    path = tmp_path / "overlap.nc"
    _overlap(path)
    overlap = read_overlap(path)
    assert np.allclose(aggregate_continuous(overlap, np.array([[2.0, 4.0]])), [3.0])
    assert np.allclose(aggregate_class_fractions(overlap, np.array([[1, 2]]), np.array([1, 2])), [[0.5, 0.5]])
    result = remap_mesh_depth_to_raster(overlap, np.array([0.25]))
    assert np.allclose(result, [[0.25, 0.25]])
    assert np.isclose(np.sum(result.reshape(-1) * overlap.raster_area_m2), 25.0)
