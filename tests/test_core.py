from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import LineString
import geopandas as gpd

from dem_processing.config import config_to_cli_args, load_config_file
from dem_processing.condition_dem import (
    aggregate_channel_surface_minimum,
    build_external_d4_river_network,
    condition_d4_channel_bed,
    compute_d4_channel_bed_sills,
    compute_d4_flow_accumulation,
    compute_equivalent_H_abg,
    enforce_downstream_channel_surface,
)
from dem_processing.fabdem import choose_target_crs, parse_resolution
from dem_processing.geometry_export import export_geometry_only_products
from dem_processing.paths import output_path, output_theme


class CoreToolboxTests(unittest.TestCase):
    def test_habg_wide_formula(self) -> None:
        width = np.array([100.0], dtype="float32")
        depth = np.array([2.0], dtype="float32")
        habg = compute_equivalent_H_abg(width, depth, grid_width=1000.0, mode="wide", max_depth=50.0)
        expected = ((100.0 / 1000.0) * (2.0 ** (5.0 / 3.0))) ** (3.0 / 5.0)
        self.assertAlmostEqual(float(habg[0]), expected, places=6)

    def test_output_theming(self) -> None:
        self.assertEqual(output_theme("DEM_hydraulic_conditioned.tif"), "dem")
        self.assertEqual(output_theme("D4_idx_facc.tif"), "d4")
        self.assertEqual(output_theme("diagnostic_final_modifications.png"), "diagnostics")
        self.assertEqual(output_theme("qa_scorecard.csv"), "reports")
        self.assertEqual(output_theme("D4_external_river_profile_QA_summary.csv"), "reports")
        self.assertEqual(output_theme("geometry_only_export_summary.json"), "reports")
        self.assertTrue(str(output_path("/tmp/out", "D4_idx_facc.tif")).endswith("/tmp/out/d4/D4_idx_facc.tif"))

    def test_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(json.dumps({"target-resolution-m": 1000, "auto-rivers-d4": True}), encoding="utf-8")
            config = load_config_file(path)
        self.assertEqual(config["target_resolution_m"], 1000)
        self.assertTrue(config["auto_rivers_d4"])
        self.assertEqual(config_to_cli_args(config), ["--target-resolution-m", "1000", "--auto-rivers-d4"])

    def test_d4_flow_accumulation_simple_slope(self) -> None:
        dem = np.array(
            [
                [5, 4, 3],
                [6, 5, 2],
                [7, 6, 1],
            ],
            dtype="float32",
        )
        profile = {"transform": Affine.translation(0, 0) * Affine.scale(1, -1)}
        acc, receiver = compute_d4_flow_accumulation(dem, profile, nodata=-9999.0)
        self.assertEqual(acc.shape, dem.shape)
        self.assertEqual(receiver.shape, dem.shape)
        self.assertGreaterEqual(float(np.nanmax(acc)), 3.0)

    def test_d4_channel_bed_sill_flags_uphill_receiver(self) -> None:
        dem = np.array([[10.0, 9.0, 8.0]], dtype="float32")
        depth = np.array([[3.0, 0.1, 1.0]], dtype="float32")
        river = np.array([[1, 1, 1]], dtype=bool)
        receiver = np.array([[1, 2, -1]], dtype=np.int64)
        bed, sill = compute_d4_channel_bed_sills(dem, depth, river, receiver)
        self.assertAlmostEqual(float(bed[0, 0]), 7.0)
        self.assertAlmostEqual(float(bed[0, 1]), 8.9, places=5)
        self.assertAlmostEqual(float(sill[0, 0]), 1.9, places=5)
        self.assertAlmostEqual(float(sill[0, 1]), 0.0)

    def test_d4_channel_bed_conditioning_deepens_only_downstream_cells(self) -> None:
        dem = np.array([[10.0, 9.0, 8.0]], dtype="float32")
        depth = np.array([[3.0, 0.1, 1.0]], dtype="float32")
        river = np.array([[1, 1, 1]], dtype=bool)
        receiver = np.array([[1, 2, -1]], dtype=np.int64)
        route = np.array([[10.0, 9.0, 8.0]], dtype="float32")
        profile = {"transform": Affine.translation(0, 0) * Affine.scale(1, -1)}
        conditioned, adjustment = condition_d4_channel_bed(
            dem, depth, river, receiver, route, profile, min_slope=0.1, depth_cap_m=3.0,
        )
        self.assertAlmostEqual(float(conditioned[0, 0]), 3.0)
        self.assertAlmostEqual(float(conditioned[0, 1]), 2.1, places=5)
        self.assertAlmostEqual(float(conditioned[0, 2]), 1.2, places=5)
        self.assertAlmostEqual(float(adjustment[0, 0]), 0.0)
        self.assertAlmostEqual(float(adjustment[0, 1]), 2.0, places=5)

    def test_external_network_builds_directed_d4_receivers(self) -> None:
        profile = {
            "height": 3,
            "width": 5,
            "crs": "EPSG:32643",
            "transform": Affine.translation(0, 3) * Affine.scale(1, -1),
        }
        reaches = gpd.GeoDataFrame(
            {
                "HYRIV_ID": [1, 2],
                "NEXT_DOWN": [2, 0],
                "UPLAND_SKM": [100.0, 200.0],
                "geometry": [
                    LineString([(1.2, 2.5), (0.2, 2.5)]),
                    LineString([(1.2, 2.5), (3.2, 2.5)]),
                ],
            },
            crs=profile["crs"],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "reaches.geojson"
            reaches.to_file(path, driver="GeoJSON")
            mask, receiver, area = build_external_d4_river_network(
                str(path), profile, 50.0, "HYRIV_ID", "NEXT_DOWN", "UPLAND_SKM",
            )
        self.assertTrue(np.all(mask[0, :4]))
        self.assertEqual(int(receiver[0, 0]), 1)
        self.assertEqual(int(receiver[0, 1]), 2)
        self.assertEqual(int(receiver[0, 2]), 3)
        self.assertAlmostEqual(float(area[0, 0]), 100.0)
        self.assertAlmostEqual(float(area[0, 1]), 200.0)

    def test_external_network_can_snap_to_local_dem_low(self) -> None:
        profile = {
            "height": 3,
            "width": 4,
            "crs": "EPSG:32643",
            "transform": Affine.translation(0, 3) * Affine.scale(1, -1),
        }
        reaches = gpd.GeoDataFrame(
            {
                "HYRIV_ID": [1], "NEXT_DOWN": [0], "UPLAND_SKM": [100.0],
                "geometry": [LineString([(0.2, 2.5), (3.2, 2.5)])],
            },
            crs=profile["crs"],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "reach.geojson"
            reaches.to_file(path, driver="GeoJSON")
            mask, receiver, _ = build_external_d4_river_network(
                str(path), profile, 50.0, "HYRIV_ID", "NEXT_DOWN", "UPLAND_SKM",
                elevation=np.array([[10, 10, 10, 10], [0, 0, 0, 0], [10, 10, 10, 10]], dtype=float),
                snap_radius_cells=1,
            )
        self.assertTrue(np.all(mask[1, :3]))
        self.assertEqual(int(receiver[1, 0]), 5)

    def test_channel_surface_profile_descends_and_aggregates_by_minimum(self) -> None:
        profile = {"height": 1, "width": 3, "transform": Affine.scale(1, -1), "nodata": -9999.0}
        dem = np.array([[10.0, 20.0, 1.0]], dtype="float32")
        river = np.array([[1, 1, 1]], dtype=bool)
        receiver = np.array([[1, 2, -1]], dtype=np.int64)
        surface, lowering = enforce_downstream_channel_surface(dem, river, receiver, profile, min_slope=1.0)
        self.assertEqual(surface.tolist(), [[10.0, 9.0, 1.0]])
        self.assertEqual(lowering.tolist(), [[0.0, 11.0, 0.0]])

        source_profile = {"height": 1, "width": 4, "transform": Affine.translation(500000, 1000) * Affine.scale(1, -1), "crs": "EPSG:32643"}
        target_profile = {"height": 1, "width": 2, "transform": Affine.translation(500000, 1000) * Affine.scale(2, -1), "crs": "EPSG:32643"}
        aggregated = aggregate_channel_surface_minimum(
            np.array([[10.0, np.nan, 8.0, 6.0]], dtype="float32"), source_profile, target_profile,
        )
        self.assertEqual(aggregated.tolist(), [[10.0, 6.0]])

    def test_channel_surface_profile_handles_raster_induced_cycle(self) -> None:
        profile = {"height": 1, "width": 2, "transform": Affine.scale(1, -1), "nodata": -9999.0}
        with self.assertWarnsRegex(UserWarning, "raster-induced"):
            surface, _ = enforce_downstream_channel_surface(
                np.array([[10.0, 20.0]], dtype="float32"),
                np.array([[1, 1]], dtype=bool),
                np.array([[1, 0]], dtype=np.int64),
                profile,
                min_slope=1.0,
            )
        self.assertTrue(np.all(np.isfinite(surface)))

    def test_parse_fabdem_resolution(self) -> None:
        self.assertEqual(parse_resolution("30"), 30.0)
        self.assertEqual(parse_resolution("30,60"), (30.0, 60.0))
        self.assertEqual(parse_resolution("30x60"), (30.0, 60.0))

    def test_choose_target_crs_prefers_explicit(self) -> None:
        class Aoi:
            crs = "EPSG:4326"

        crs = choose_target_crs(Aoi(), target_crs="EPSG:32643")
        self.assertEqual(crs.to_epsg(), 32643)

    def test_geometry_export_masks_non_river_cells(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 3,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:32643",
                "transform": Affine.translation(0, 0) * Affine.scale(30, -30),
                "nodata": -9999.0,
            }

            def write(name: str, arr: np.ndarray) -> None:
                path = output_path(out_dir, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(arr.astype("float32"), 1)

            write("DEM_hydrologically_conditioned_pre_bathymetry.tif", np.arange(6).reshape(2, 3))
            write("D4_idx_facc.tif", np.array([[0, 1, 0], [1, 0, 1]], dtype="float32"))
            write("D4_Wshed_Properties_River_Width_m.tif", np.full((2, 3), 50.0, dtype="float32"))
            write("D4_Wshed_Properties_River_Depth_m.tif", np.full((2, 3), 2.0, dtype="float32"))

            summary = export_geometry_only_products(out_dir, overwrite=True)

            self.assertTrue(Path(summary["no_bathymetry_dem"]).exists())
            self.assertTrue(Path(summary["summary_path"]).exists())
            with rasterio.open(summary["river_width"]["path"]) as src:
                width = src.read(1, masked=False)
            self.assertTrue(np.isnan(width[0, 0]))
            self.assertEqual(float(width[0, 1]), 50.0)
            self.assertTrue(np.isnan(width[1, 1]))


if __name__ == "__main__":
    unittest.main()
