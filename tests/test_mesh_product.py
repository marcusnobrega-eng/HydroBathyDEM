from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from dem_processing.mesh_product import (
    _selected_mesh,
    build_mesh_product,
    write_conservative_overlap,
)


def _write_dem(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=1,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(0, 100, 100, 100),
        nodata=-9999.0,
    ) as target:
        target.write(np.asarray([[2.0, 1.0]], dtype=np.float32), 1)


def test_documented_regular_mode_packages_existing_grid(tmp_path: Path) -> None:
    dem = tmp_path / "dem.tif"
    _write_dem(dem)
    result = build_mesh_product(
        {
            "mesh": {
                "mode": "regular",
                "inputs": {"dem": str(dem), "out_dir": str(tmp_path / "product")},
                "regular": {"cell_size_m": 100},
            }
        }
    )
    assert result["mesh_mode"] == "regular"
    assert result["active_cell_count"] == 2
    assert Path(result["grid"]).is_file()
    assert Path(result["qa"]).is_file()


def test_mode_specific_blocks_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _selected_mesh({"mesh": {"mode": "regular", "voronoi": {}}})


def test_unknown_mesh_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported mesh schema_version"):
        _selected_mesh({"schema_version": "future-schema", "mesh": {"mode": "regular"}})


def test_documented_voronoi_template_selects_existing_builder() -> None:
    mode, values = _selected_mesh(
        {
            "schema_version": "hydrobathydem-mesh-1.0",
            "mesh": {
                "mode": "voronoi_fv",
                "inputs": {"dem": "dem.tif", "out_dir": "product"},
                "voronoi": {
                    "resolution": {"minimum_cell_width_m": 30},
                    "rivers": {"unresolved_policy": "none"},
                },
            },
        }
    )
    assert mode == "voronoi_fv"
    assert values["resolution"]["minimum_cell_width_m"] == 30
    assert values["rivers"]["unresolved_policy"] == "none"


def test_conservative_overlap_covers_each_polygon_exactly(tmp_path: Path) -> None:
    dem = tmp_path / "dem.tif"
    mesh_nc = tmp_path / "mesh.nc"
    mesh_gpkg = tmp_path / "mesh.gpkg"
    overlap = tmp_path / "overlap.nc"
    _write_dem(dem)
    polygons = gpd.GeoDataFrame(
        {"face_id": [0, 1]},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs="EPSG:32643",
    )
    polygons.to_file(mesh_gpkg, layer="mesh", driver="GPKG")
    with netCDF4.Dataset(mesh_nc, "w") as dataset:
        dataset.createDimension("face", 2)
        dataset.createVariable("cell_area_m2", "f8", ("face",))[:] = [10_000, 10_000]

    result = write_conservative_overlap(mesh_nc, mesh_gpkg, dem, overlap, row_chunk_size=1)
    assert result["overlap_count"] == 2
    assert result["maximum_mesh_area_relative_error"] < 1e-12
    with netCDF4.Dataset(overlap) as dataset:
        assert dataset["overlap_mesh_index"][:].tolist() == [0, 1]
        assert dataset["overlap_raster_index"][:].tolist() == [0, 1]
        assert np.allclose(dataset["overlap_area_m2"][:], 10_000)
