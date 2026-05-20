"""Command-line preflight checker."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config_file
from .paths import PROJECT_ROOT
from .qa import preflight_checks
from .run_conditioning_1000m import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that inputs and generated dependencies are ready for the DEM pipeline.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/TOML config file.")
    parser.add_argument("--dem", default=None, help="Input DEM path. Overrides config/default runner value.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Overrides config/default runner value.")
    parser.add_argument("--require-lin", action="store_true", help="Require processed Lin2020 GeoPackage.")
    parser.add_argument("--require-spatial-coefficients", action="store_true", help="Require current 5000 km2 spatial coefficient rasters.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for the preflight report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(CONFIG)
    config.update({key.replace("_", "-"): value for key, value in load_config_file(args.config).items()})
    if args.dem:
        config["dem"] = args.dem
    if args.out_dir:
        config["out-dir"] = args.out_dir

    required = []
    if args.require_lin:
        required.append(PROJECT_ROOT / "Data" / "Lin2020_bankfull_width" / "processed" / "lin2020_dem_domain_width_depth.gpkg")
    if args.require_spatial_coefficients:
        cal = PROJECT_ROOT / "Data" / "Lin2020_bankfull_width" / "calibration"
        required.extend(
            [
                cal / "D4_beta_1_width_5000km2.tif",
                cal / "D4_beta_2_width_5000km2.tif",
                cal / "D4_alfa_1_depth_5000km2.tif",
                cal / "D4_alfa_2_depth_5000km2.tif",
            ]
        )

    df = preflight_checks(config["dem"], config["out-dir"], required)
    print(df.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"[SAVED] {args.output}")
    if (df["status"] == "fail").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
