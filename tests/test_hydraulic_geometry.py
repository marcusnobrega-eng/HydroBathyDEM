from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from dem_processing.hydraulic_geometry import (
    estimate_power_law_geometry,
    write_power_law_geometry,
)


def test_power_law_geometry_and_writer(tmp_path: Path) -> None:
    area = np.array([[0.0, 1.0], [4.0, 9.0]])
    river = np.array([[False, True], [True, False]])
    width, depth = estimate_power_law_geometry(
        area,
        river,
        width_coefficient=2.0,
        width_exponent=0.5,
        depth_coefficient=0.5,
        depth_exponent=1.0,
    )
    np.testing.assert_allclose(width, [[0.0, 2.0], [4.0, 0.0]])
    np.testing.assert_allclose(depth, [[0.0, 0.5], [2.0, 0.0]])

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float64",
        "crs": "EPSG:32610",
        "transform": from_origin(0.0, 60.0, 30.0, 30.0),
    }
    inputs = []
    for name, values in (("area.tif", area), ("river.tif", river), ("mask.tif", np.ones((2, 2)))):
        path = tmp_path / name
        with rasterio.open(path, "w", **profile) as destination:
            destination.write(np.asarray(values, dtype=np.float64), 1)
        inputs.append(path)
    width_path, depth_path = write_power_law_geometry(
        *inputs,
        tmp_path / "output",
        width_coefficient=20.0,
        width_exponent=1.0,
    )
    with rasterio.open(width_path) as source:
        saved_width = source.read(1)
    with rasterio.open(depth_path) as source:
        saved_depth = source.read(1)
    assert saved_width[1, 0] == 30.0
    assert np.count_nonzero(saved_width) == np.count_nonzero(saved_depth) == 2
