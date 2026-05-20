#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_conditioning_1000m.py

Python runner for the HydroPol2D-style DEM conditioning workflow.

This runner does three things:

1) Optionally installs missing Python libraries.
2) Resamples the DEM to the selected model resolution first.
3) Runs the DEM-conditioning workflow with your HydroPol2D parameters.

Folder expected on your Mac:

    /Users/mngomes/Documents/GitHub/DEM_Processing/
        DEM_fabdem.tif
        src/dem_processing/condition_dem.py
        src/dem_processing/run_conditioning_1000m.py

Run:

    cd /Users/mngomes/Documents/GitHub/DEM_Processing
    PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m --install-deps

After dependencies are installed, run:

    PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m

To only print the command without running:

    PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from .config import config_to_cli_args, load_config_file
    from .paths import PROJECT_ROOT
except ImportError:  # Allows direct execution from the source directory.
    from config import config_to_cli_args, load_config_file
    from paths import PROJECT_ROOT


# =============================================================================
# USER SETTINGS
# =============================================================================
# Edit values here. The comments beside each entry explain the parameter.
# Boolean True means: include this flag.
# Boolean False or None means: skip this flag.
# =============================================================================

CONFIG: Dict[str, Any] = {

    # -------------------------------------------------------------------------
    # Main input/output
    # -------------------------------------------------------------------------
    "dem": str(PROJECT_ROOT / "DEM_fabdem.tif"),
    # Input DEM. It must be projected in meters, not latitude/longitude degrees.

    "out-dir": str(PROJECT_ROOT / "Outputs"),
    # Output folder. All rasters, diagnostics, plots, summaries, and documentation go here.


    # -------------------------------------------------------------------------
    # Step 0: resample DEM before all hydrologic/hydraulic analysis
    # -------------------------------------------------------------------------
    "resample-dem": True,
    # Activate first-step DEM resampling.
    # This ensures D4 flow accumulation, H_abg creek carving, smoothing, and breaching
    # are all applied at the same grid resolution used by the flood model.

    "target-resolution-m": 1000.0,
    # Target DEM/model resolution after resampling [m].
    # For your current setup, this matches channel-cell-width-m below.

    "resampling-method": "bilinear",
    # DEM resampling method.
    # Current testing value from the handoff is "bilinear".
    # Other reasonable options: "average", "cubic".


    # -------------------------------------------------------------------------
    # Automatic D4 river / creek extraction
    # -------------------------------------------------------------------------
    "auto-rivers-d4": True,
    # Activate automatic river detection using D4 flow accumulation.
    # D4 means flow can move only north, south, east, or west.

    "min-area": 100.0,
    # GIS_data.min_area [km2].
    # Cells with Wshed_Properties.fac_area >= min_area become river/creek cells.
    # With 1000 m grid cells, each cell is about 1 km2, so this threshold is large.


    # -------------------------------------------------------------------------
    # River width/depth source
    # -------------------------------------------------------------------------
    "river-geometry-source": "spatial_coefficients_or_power_law",
    # Options:
    #   "power_law"              -> use HydroPol2D beta/alfa equations only.
    #   "external"               -> use only external width/depth rasters.
    #   "external_or_power_law"  -> use external rasters where available,
    #                               with power-law values filling gaps.

    "external-river-width-raster": None,
    # Lin et al. 2020 machine-learning bankfull width rasterized to the model grid.

    "external-river-depth-raster": None,
    # Depth solved from Lin Q2, width_m, and slope using Manning's equation.

    "spatial-beta-1-raster": str(PROJECT_ROOT / "Data" / "Lin2020_bankfull_width" / "calibration" / "D4_beta_1_width_5000km2.tif"),
    # Spatially calibrated beta_1 map from FABDEM-D4 subcatchments.

    "spatial-beta-2-raster": str(PROJECT_ROOT / "Data" / "Lin2020_bankfull_width" / "calibration" / "D4_beta_2_width_5000km2.tif"),
    # Spatially calibrated beta_2 map from FABDEM-D4 subcatchments.

    "spatial-alfa-1-raster": str(PROJECT_ROOT / "Data" / "Lin2020_bankfull_width" / "calibration" / "D4_alfa_1_depth_5000km2.tif"),
    # Spatially calibrated alfa_1 map from FABDEM-D4 subcatchments.

    "spatial-alfa-2-raster": str(PROJECT_ROOT / "Data" / "Lin2020_bankfull_width" / "calibration" / "D4_alfa_2_depth_5000km2.tif"),
    # Spatially calibrated alfa_2 map from FABDEM-D4 subcatchments.


    # -------------------------------------------------------------------------
    # HydroPol2D hydraulic geometry parameters
    # -------------------------------------------------------------------------
    "beta-1": 2.2695,
    # GIS_data.beta_1.
    # Coefficient in: River_Width = beta_1 * fac_area^beta_2.

    "beta-2": 0.4942,
    # GIS_data.beta_2.
    # Exponent in: River_Width = beta_1 * fac_area^beta_2.

    "alfa-1": 0.1097,
    # GIS_data.alfa_1.
    # Coefficient in: River_Depth = alfa_1 * fac_area^alfa_2.

    "alfa-2": 0.3856,
    # GIS_data.alfa_2.
    # Exponent in: River_Depth = alfa_1 * fac_area^alfa_2.


    # -------------------------------------------------------------------------
    # HydroPol2D creek-carving method
    # -------------------------------------------------------------------------
    "carve-mode": "wide",
    # Uses HydroPol2D-style H_abg formula based on Manning conveyance:
    # H_abg = ((River_Width / Resolution) * River_Depth^(5/3))^(3/5).

    "channel-cell-width-m": 1000.0,
    # Effective grid-cell width used in the H_abg calculation [m].
    # Usually this should match target-resolution-m.

    "river-width-cap-m": 10000.0,
    # Maximum allowed estimated River_Width [m].
    # This is a safety cap on the power-law width equation.

    "river-depth-cap-m": 30.0,
    # Maximum allowed estimated River_Depth [m].
    # This is a safety cap on the power-law depth equation.

    "max-H-abg-m": 50.0,
    # Maximum allowed DEM lowering from H_abg [m].
    # If H_abg is larger than this, the main script should stop for safety.


    # -------------------------------------------------------------------------
    # NoData cleaning
    # -------------------------------------------------------------------------
    "max-nodata-fill-pixels": 10,
    # Maximum search distance for filling small NoData holes [pixels].


    # -------------------------------------------------------------------------
    # Selective slope-artifact smoothing
    # -------------------------------------------------------------------------
    "slope-percentile": 95,
    # Smooth only cells above this slope percentile.
    # 95 means the steepest 5% of valid DEM cells are considered for smoothing.

    "smooth-filter-cells": 5,
    # Size of the Whitebox feature-preserving smoothing filter [cells].

    "protect-stream-buffer-m": 2000.0,
    # Buffer around provided stream vectors protected from generic smoothing [m].
    # This only matters if a stream vector is provided.


    # -------------------------------------------------------------------------
    # Depression breaching
    # -------------------------------------------------------------------------
    "breach-dist-cells": 10000,
    # Maximum search distance for least-cost depression breaching [cells].
    # At 1000 m resolution, this can search very far. Use carefully.

    "breach-flat-increment": 0.01,
    # Tiny gradient added to avoid perfectly flat flow paths.

    "fill-max-depth-m": 0.25,
    # Maximum depression-fill depth if residual filling is activated.
    # Residual filling is not activated unless fill-residual-depressions=True.

    # "fill-residual-depressions": True,
    # Optional. Use carefully because filling can destroy real floodplain storage.
}


# =============================================================================
# DEPENDENCIES
# =============================================================================

REQUIRED_PACKAGES = [
    ("whitebox", "whitebox"),
    ("rasterio", "rasterio"),
    ("geopandas", "geopandas"),
    ("pyogrio", "pyogrio"),
    ("shapely", "shapely"),
    ("pyproj", "pyproj"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
]


def is_installed(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def missing_packages() -> List[str]:
    return [pip_name for import_name, pip_name in REQUIRED_PACKAGES if not is_installed(import_name)]


def install_missing_packages() -> None:
    missing = missing_packages()
    if not missing:
        print("[OK] All required packages are already installed.")
        return

    print("[INFO] Installing missing packages into the current Python environment:")
    for package in missing:
        print(f"  - {package}")

    cmd = [sys.executable, "-m", "pip", "install", *missing]
    print("\n[RUNNING]", " ".join(cmd))
    subprocess.check_call(cmd)

    still_missing = missing_packages()
    if still_missing:
        raise RuntimeError(
            "Some packages are still missing after pip installation: "
            + ", ".join(still_missing)
            + "\nIf rasterio/geopandas failed, install them with conda-forge."
        )


# =============================================================================
# RUNNER UTILITIES
# =============================================================================

def merge_config_file(config_path: Path | None) -> Dict[str, Any]:
    config = dict(CONFIG)
    loaded = load_config_file(config_path)
    config.update({key.replace("_", "-"): value for key, value in loaded.items()})
    return config


def validate_paths(config: Dict[str, Any]) -> None:
    dem_path = Path(str(config["dem"])).expanduser()
    if not dem_path.exists():
        raise FileNotFoundError(f"Input DEM not found:\n{dem_path}")

    out_dir = Path(str(config["out-dir"])).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)


def print_summary(config: Dict[str, Any], cli_args: List[str]) -> None:
    print("\n" + "=" * 78)
    print("HydroPol2D DEM conditioning run")
    print("=" * 78)
    print(f"Python executable : {sys.executable}")
    print("Main module       : dem_processing.condition_dem")
    print(f"Input DEM         : {config['dem']}")
    print(f"Output directory  : {config['out-dir']}")
    print(f"Target resolution : {config.get('target-resolution-m')} m")
    print(f"D4 min area       : {config.get('min-area')} km2")
    print("=" * 78)
    print("Arguments:")

    i = 0
    while i < len(cli_args):
        if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("--"):
            print(f"  {cli_args[i]:32s} {cli_args[i + 1]}")
            i += 2
        else:
            print(f"  {cli_args[i]}")
            i += 1
    print("=" * 78 + "\n")


def run_workflow(dry_run: bool = False, config_path: Path | None = None) -> None:
    config = merge_config_file(config_path)
    validate_paths(config)

    cli_args = config_to_cli_args(config)
    print_summary(config, cli_args)

    cmd = [sys.executable, "-m", "dem_processing.condition_dem", *cli_args]
    if dry_run:
        print("[DRY RUN] Command not executed:")
        print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
        return

    subprocess.check_call(cmd)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the HydroPol2D-style DEM conditioning workflow with predefined parameters."
    )
    parser.add_argument("--install-deps", action="store_true", help="Install missing Python packages first.")
    parser.add_argument("--setup-only", action="store_true", help="Only install/check dependencies; do not run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command but do not execute it.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/TOML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.install_deps:
        install_missing_packages()

    missing = missing_packages()
    if missing:
        print("[ERROR] Missing packages:")
        for package in missing:
            print(f"  - {package}")
        print("\nRun:")
        print("  PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m --install-deps")
        sys.exit(1)

    if args.setup_only:
        print("[OK] Setup check completed. Workflow was not run.")
        return

    run_workflow(dry_run=args.dry_run, config_path=args.config)


if __name__ == "__main__":
    main()
