from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

from dem_processing.condition_dem import (
    DEMConditioningConfig,
    build_d4_monotonic_routing_surface,
    compute_d4_flow_accumulation,
    compute_flow_accumulation,
    condition_d4_channel_bed,
    read_entry_upstream_cells,
    receiver_link_length_m,
    receiver_to_d8_direction,
    river_connectivity,
)
from dem_processing.geometry_export import main as export_geometry
from dem_processing.paths import output_path, output_theme

PROFILE = {"transform": Affine.translation(0.0, 50.0) * Affine.scale(10.0, -10.0)}


def _diagonal_valley(size: int = 5) -> np.ndarray:
    """A plane falling to the south-east: the steepest neighbour is the diagonal."""
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    return (100.0 - rows - cols).astype("float32")


class D8RiverTests(unittest.TestCase):
    def test_d8_receivers_follow_the_diagonal_and_d4_steps_around_it(self) -> None:
        dem = _diagonal_valley()
        _, d8 = compute_flow_accumulation(dem, PROFILE, -9999.0, connectivity="d8")
        _, d4 = compute_flow_accumulation(dem, PROFILE, -9999.0, connectivity="d4")
        cols = dem.shape[1]
        self.assertEqual(int(d8[0, 0]), 1 * cols + 1)  # south-east
        self.assertIn(int(d4[0, 0]), {1, cols})  # east or south, never diagonal
        codes = receiver_to_d8_direction(d8)
        self.assertEqual(float(codes[0, 0]), 2.0)  # ESRI south-east
        self.assertEqual(float(codes[-1, -1]), 0.0)  # the outlet has no receiver

    def test_d8_cardinal_neighbour_wins_a_slope_tie(self) -> None:
        # Falling east only: the east drop over 10 m beats the south-east drop over 14 m.
        dem = np.tile(np.arange(5, 0, -1, dtype="float32"), (3, 1))
        _, receiver = compute_flow_accumulation(dem, PROFILE, -9999.0, connectivity="d8")
        self.assertEqual(int(receiver[1, 0]), 1 * 5 + 1)

    def test_d4_wrapper_is_unchanged(self) -> None:
        dem = _diagonal_valley()
        acc_a, rec_a = compute_d4_flow_accumulation(dem, PROFILE, -9999.0)
        acc_b, rec_b = compute_flow_accumulation(dem, PROFILE, -9999.0, connectivity="d4")
        np.testing.assert_array_equal(acc_a, acc_b)
        np.testing.assert_array_equal(rec_a, rec_b)

    def test_direction_codes_cover_all_eight_neighbours(self) -> None:
        receiver = -np.ones((3, 3), dtype=np.int64)
        expected = {}
        for (dr, dc), code in {
            (0, 1): 1, (1, 1): 2, (1, 0): 4, (1, -1): 8,
            (0, -1): 16, (-1, -1): 32, (-1, 0): 64, (-1, 1): 128,
        }.items():
            grid = -np.ones((3, 3), dtype=np.int64)
            grid[1, 1] = (1 + dr) * 3 + (1 + dc)
            expected[code] = float(receiver_to_d8_direction(grid)[1, 1])
        self.assertEqual(expected, {code: float(code) for code in expected})
        self.assertTrue(np.all(receiver_to_d8_direction(receiver) == 0.0))

    def test_entry_area_is_carried_downstream_in_d4_and_d8(self) -> None:
        dem = _diagonal_valley()
        extra = np.zeros(dem.shape)
        extra[0, 0] = 1000.0
        for connectivity in ("d4", "d8"):
            plain, receiver = compute_flow_accumulation(dem, PROFILE, -9999.0, connectivity=connectivity)
            boosted, _ = compute_flow_accumulation(
                dem, PROFILE, -9999.0, connectivity=connectivity, extra_cells=extra
            )
            gained = boosted - plain
            # Every cell on the entry's downstream path gains it; no other cell changes.
            path = [0]
            while receiver.ravel()[path[-1]] >= 0:
                path.append(int(receiver.ravel()[path[-1]]))
            on_path = np.zeros(dem.size, dtype=bool)
            on_path[path] = True
            np.testing.assert_allclose(gained.ravel()[on_path], 1000.0)
            np.testing.assert_allclose(gained.ravel()[~on_path], 0.0)

    def test_entry_table_maps_coordinates_to_cells(self) -> None:
        valid = np.ones((5, 5), dtype=bool)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "entries.csv"
            pd.DataFrame({"entry_id": ["a"], "x": [25.0], "y": [45.0], "upstream_area_km2": [2.0]}).to_csv(path)
            extra, used = read_entry_upstream_cells(str(path), PROFILE, valid)
            self.assertEqual((int(used.loc[0, "row"]), int(used.loc[0, "column"])), (0, 2))
            # 10 m cells hold 1e-4 km2, so 2 km2 is 20,000 cells.
            self.assertAlmostEqual(float(extra[0, 2]), 20_000.0)
            pd.DataFrame({"x": [500.0], "y": [45.0], "upstream_area_km2": [2.0]}).to_csv(path)
            with self.assertRaises(ValueError):
                read_entry_upstream_cells(str(path), PROFILE, valid)

    def test_d8_bed_conditioning_drop_uses_the_diagonal_length(self) -> None:
        dem = np.array([[10.0, 10.0], [10.0, 10.0]], dtype="float32")
        depth = np.array([[1.0, 0.0], [0.0, 0.5]], dtype="float32")
        river = np.array([[1, 0], [0, 1]], dtype=bool)
        receiver = np.array([[3, -1], [-1, -1]], dtype=np.int64)  # (0, 0) to (1, 1)
        conditioned, _ = condition_d4_channel_bed(
            dem, depth, river, receiver, dem, PROFILE, min_slope=0.01, depth_cap_m=None,
        )
        self.assertAlmostEqual(float(conditioned[1, 1]), 1.0 + 0.01 * np.hypot(10.0, 10.0), places=5)
        lengths = receiver_link_length_m(receiver, PROFILE)
        self.assertAlmostEqual(float(lengths[0, 0]), np.hypot(10.0, 10.0))

    def test_d8_routing_surface_is_written_under_its_own_name_and_fills_pits(self) -> None:
        dem = np.full((4, 4), 5.0, dtype="float32")
        dem[1, 1] = 1.0
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            source = out_dir / "dem.tif"
            with rasterio.open(
                source, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
                crs="EPSG:32643", transform=PROFILE["transform"], nodata=-9999.0,
            ) as dst:
                dst.write(dem, 1)
            surface = build_d4_monotonic_routing_surface(source, out_dir, 0.01, None, connectivity="d8")
            self.assertEqual(surface.name, "DEM_D8_monotonic_routing_surface.tif")
            with rasterio.open(surface) as src:
                filled = src.read(1)
        self.assertTrue(np.all(np.isfinite(filled)))
        self.assertGreater(float(filled[1, 1]), 1.0)  # the pit drains out through the flood

    def test_d8_option_rules(self) -> None:
        self.assertEqual(river_connectivity(DEMConditioningConfig(auto_rivers_d8=True)), "d8")
        self.assertEqual(river_connectivity(DEMConditioningConfig(auto_rivers_d4=True)), "d4")
        with self.assertRaises(ValueError):
            river_connectivity(DEMConditioningConfig(auto_rivers_d4=True, auto_rivers_d8=True))
        with self.assertRaises(ValueError):
            river_connectivity(DEMConditioningConfig(auto_rivers_d8=True, external_river_network="x.gpkg"))
        self.assertEqual(output_theme("D8_idx_facc.tif"), "d8")
        self.assertEqual(output_theme("D8_HydroPol2D_creek_reduction_summary.csv"), "reports")

    def test_geometry_export_reads_the_d8_rasters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            profile = {
                "driver": "GTiff", "height": 2, "width": 3, "count": 1, "dtype": "float32",
                "crs": "EPSG:32643", "transform": Affine.translation(0, 0) * Affine.scale(30, -30),
                "nodata": -9999.0,
            }

            def write(name: str, arr: np.ndarray) -> None:
                path = output_path(out_dir, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(arr.astype("float32"), 1)

            write("DEM_hydrologically_conditioned_pre_bathymetry.tif", np.arange(6).reshape(2, 3))
            write("D8_idx_facc.tif", np.array([[1, 0, 0], [0, 1, 0]], dtype="float32"))
            write("D8_Wshed_Properties_River_Width_m.tif", np.full((2, 3), 40.0, dtype="float32"))
            write("D8_Wshed_Properties_River_Depth_m.tif", np.full((2, 3), 1.5, dtype="float32"))

            export_geometry(["--out-dir", str(out_dir), "--overwrite", "--connectivity", "d8"])

            with rasterio.open(output_path(out_dir, "D8_River_Width_river_cells_m.tif")) as src:
                width = src.read(1)
        self.assertEqual(float(width[0, 0]), 40.0)
        self.assertTrue(np.isnan(width[0, 1]))


if __name__ == "__main__":
    unittest.main()
