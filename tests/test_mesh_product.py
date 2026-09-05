import json
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
    _without_subgrid,
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
    assert Path(result["ugrid"]).is_file()
    assert Path(result["geopackage"]).is_file()
    assert Path(result["overlap"]).is_file()
    assert Path(result["manifest"]).is_file()
    assert Path(result["qa"]).is_file()
    with netCDF4.Dataset(result["ugrid"]) as dataset:
        assert dataset.mesh_contract_version == "1.0"
        assert dataset.product_type == "hydraulic_mesh"
        assert dataset["cell_cfl_width_m"][:].tolist() == [100.0, 100.0]
        assert dataset["edge_neighbor"][:].tolist().count(-1) == 6
        assert np.allclose(
            np.hypot(dataset["edge_normal_x"][:], dataset["edge_normal_y"][:]), 1.0
        )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["generator"]["version"] == "0.3.0rc1"
    assert len(manifest["configuration_sha256"]) == 64
    assert {record["role"] for record in manifest["files"]} >= {
        "ugrid",
        "overlap",
        "geopackage",
    }


def test_regular_mode_can_publish_structured_subgrid_tables(tmp_path: Path) -> None:
    dem = tmp_path / "dem.tif"
    _write_dem(dem)
    result = build_mesh_product(
        {
            "mesh": {
                "mode": "regular",
                "inputs": {"dem": str(dem), "out_dir": str(tmp_path / "product")},
                "regular": {"cell_size_m": 100},
                "subgrid": {"enabled": True, "policy": "all"},
            }
        }
    )
    assert result["subgrid_cell_count"] == 2
    assert Path(result["subgrid"]).is_file()
    with netCDF4.Dataset(result["subgrid"]) as dataset:
        assert dataset.product_type == "hydraulic_subgrid_tables"
        assert dataset.conveyance_convention == "K=sum((A/n)*(A/P)^(2/3))"


def test_manifest_hash_includes_subgrid_configuration(tmp_path: Path) -> None:
    dem = tmp_path / "dem.tif"
    _write_dem(dem)

    def build(policy: str, directory: str) -> str:
        result = build_mesh_product(
            {
                "mesh": {
                    "mode": "regular",
                    "inputs": {"dem": str(dem), "out_dir": str(tmp_path / directory)},
                    "regular": {"cell_size_m": 100},
                    "subgrid": {"enabled": True, "policy": policy},
                }
            }
        )
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        return manifest["configuration_sha256"]

    assert build("all", "all") != build("river", "river")


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


def test_subgrid_block_is_kept_out_of_mesh_builder_configuration() -> None:
    values, subgrid = _without_subgrid(
        {
            "schema_version": "hydrobathydem-mesh-1.0",
            "mesh": {
                "mode": "voronoi_fv",
                "inputs": {"dem": "dem.tif", "out_dir": "product"},
                "voronoi": {},
                "subgrid": {"enabled": True, "policy": "all"},
            },
        }
    )
    assert "subgrid" not in values["mesh"]
    assert subgrid == {"enabled": True, "policy": "all"}


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
        assert dataset.mesh_contract_version == "1.0"
        assert dataset.raster_index_order == "south_up_row_major"
        assert dataset.crs_wkt
        assert dataset["overlap_mesh_index"][:].tolist() == [0, 1]
        assert dataset["overlap_raster_index"][:].tolist() == [0, 1]
        assert np.allclose(dataset["overlap_area_m2"][:], 10_000)
        assert dataset["overlap_area_m2"].units == "m2"
