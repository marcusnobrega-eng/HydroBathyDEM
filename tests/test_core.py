from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

from dem_processing.config import config_to_cli_args, load_config_file
from dem_processing.condition_dem import compute_equivalent_H_abg, compute_d4_flow_accumulation
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


if __name__ == "__main__":
    unittest.main()
