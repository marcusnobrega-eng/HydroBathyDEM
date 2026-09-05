from pathlib import Path

import netCDF4
import numpy as np
import rasterio
from rasterio.transform import from_origin

from dem_processing.mesh_contract import apply_contract_attributes
from dem_processing.subgrid_product import SubgridProductConfig, build_subgrid_product


def _write_dem(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=1,
        count=1,
        dtype="float64",
        crs="EPSG:32643",
        transform=from_origin(0, 100, 100, 100),
        nodata=-9999.0,
    ) as target:
        target.write(np.asarray([[2.0, 1.0]]), 1)


def _write_mesh(path: Path) -> str:
    crs_wkt = rasterio.crs.CRS.from_epsg(32643).to_wkt()
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.crs_wkt = crs_wkt
        apply_contract_attributes(dataset, product="unstructured_mesh")
        dataset.createDimension("face", 2)
        dataset.createDimension("edge", 3)
        values = {
            "edge_owner": ("i4", [0, 0, 1]),
            "edge_neighbor": ("i4", [1, -1, -1]),
            "edge_length_m": ("f8", [100.0, 100.0, 100.0]),
            "edge_midpoint_x": ("f8", [100.0, 0.0, 200.0]),
            "edge_midpoint_y": ("f8", [50.0, 50.0, 50.0]),
            "edge_normal_x": ("f8", [1.0, -1.0, 1.0]),
            "edge_normal_y": ("f8", [0.0, 0.0, 0.0]),
        }
        for name, (dtype, data) in values.items():
            dataset.createVariable(name, dtype, ("edge",))[:] = data
        dataset.createVariable("cell_area_m2", "f8", ("face",))[:] = [10_000.0, 10_000.0]
        dataset.createVariable("cell_bed_elevation_m", "f8", ("face",))[:] = [2.0, 1.0]
        feature = dataset.createVariable("cell_feature_class", str, ("face",))
        feature[:] = np.asarray(["floodplain", "floodplain"], dtype=object)
    return crs_wkt


def _write_overlap(path: Path, crs_wkt: str) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.crs_wkt = crs_wkt
        dataset.raster_rows = 1
        dataset.raster_cols = 2
        dataset.raster_index_order = "south_up_row_major"
        apply_contract_attributes(dataset, product="conservative_raster_overlap")
        dataset.createDimension("overlap", 2)
        dataset.createVariable("overlap_mesh_index", "i4", ("overlap",))[:] = [0, 1]
        dataset.createVariable("overlap_raster_index", "i8", ("overlap",))[:] = [0, 1]
        dataset.createVariable("overlap_area_m2", "f8", ("overlap",))[:] = [10_000.0, 10_000.0]


def test_build_subgrid_product_writes_shared_contract(tmp_path: Path) -> None:
    dem = tmp_path / "dem.tif"
    mesh = tmp_path / "mesh.nc"
    overlap = tmp_path / "overlap.nc"
    output = tmp_path / "subgrid.nc"
    _write_dem(dem)
    crs_wkt = _write_mesh(mesh)
    _write_overlap(overlap, crs_wkt)

    result = build_subgrid_product(
        mesh,
        overlap,
        SubgridProductConfig(fine_dem=dem, policy="all", default_manning_n=0.05),
        output,
    )

    assert result["subgrid_cell_count"] == 2
    assert result["subgrid_profile_face_count"] == 1
    with netCDF4.Dataset(output) as dataset:
        assert dataset.mesh_contract_version == "1.0"
        assert dataset.conveyance_convention == "K=sum((A/n)*(A/P)^(2/3))"
        assert dataset["cell_is_subgrid"][:].tolist() == [1, 1]
        assert dataset["cell_volume_m3"].units == "m3"
        assert dataset["face_conveyance"].units == "m3 s-1"
        assert np.all(np.isfinite(dataset["face_conveyance"][:]))
