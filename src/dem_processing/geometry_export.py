"""Export no-bathymetry DEM and river-only hydraulic-geometry rasters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio

from .paths import ensure_output_layout, output_path


DEFAULT_DEM_NAME = "DEM_hydrologically_conditioned_pre_bathymetry.tif"
DEFAULT_MASK_NAME = "D4_idx_facc.tif"
DEFAULT_WIDTH_NAME = "D4_Wshed_Properties_River_Width_m.tif"
DEFAULT_DEPTH_NAME = "D4_Wshed_Properties_River_Depth_m.tif"


def copy_raster(src_path: Path, dst_path: Path, *, overwrite: bool = False) -> Path:
    """Copy a raster while keeping metadata and applying lightweight compression."""

    if dst_path.exists() and not overwrite:
        return dst_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(compress="deflate", BIGTIFF="IF_SAFER")
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                dst.write(src.read(band_idx), band_idx)
            dst.update_tags(**src.tags())

    return dst_path


def write_river_only_raster(
    values_path: Path,
    mask_path: Path,
    output_path_: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Write values where the D4 river mask is positive and NaN elsewhere."""

    if output_path_.exists() and not overwrite:
        with rasterio.open(output_path_) as src:
            arr = src.read(1, masked=False)
        valid = np.isfinite(arr)
        return {
            "path": str(output_path_),
            "river_value_cells": int(valid.sum()),
            "min": float(np.nanmin(arr)) if valid.any() else None,
            "mean": float(np.nanmean(arr)) if valid.any() else None,
            "max": float(np.nanmax(arr)) if valid.any() else None,
            "reused": True,
        }

    output_path_.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(values_path) as values_src, rasterio.open(mask_path) as mask_src:
        if values_src.shape != mask_src.shape:
            raise ValueError(
                "Value raster and river-mask raster must have the same shape: "
                f"{values_path} has {values_src.shape}, {mask_path} has {mask_src.shape}"
            )
        if values_src.transform != mask_src.transform:
            raise ValueError("Value raster and river-mask raster transforms do not match.")

        values = values_src.read(1, masked=True).filled(np.nan).astype("float32")
        mask = mask_src.read(1, masked=True).filled(0) > 0
        out = np.full(values.shape, np.nan, dtype="float32")
        valid = mask & np.isfinite(values) & (values > 0)
        out[valid] = values[valid]

        profile = values_src.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            nodata=np.nan,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(output_path_, "w", **profile) as dst:
            dst.write(out, 1)

    return {
        "path": str(output_path_),
        "river_value_cells": int(valid.sum()),
        "min": float(np.nanmin(out)) if valid.any() else None,
        "mean": float(np.nanmean(out)) if valid.any() else None,
        "max": float(np.nanmax(out)) if valid.any() else None,
        "reused": False,
    }


def export_geometry_only_products(
    out_dir: Path | str,
    *,
    dem_path: Optional[Path | str] = None,
    mask_path: Optional[Path | str] = None,
    width_path: Optional[Path | str] = None,
    depth_path: Optional[Path | str] = None,
    output_dem_name: str = "DEM_conditioned_no_bathymetry.tif",
    output_width_name: str = "D4_River_Width_river_cells_m.tif",
    output_depth_name: str = "D4_River_Depth_river_cells_m.tif",
    overwrite: bool = False,
) -> dict:
    """Export the conditioned DEM before bathymetry and river-only width/depth rasters."""

    out_dir = Path(out_dir)
    ensure_output_layout(out_dir)

    pre_bathy_dem = Path(dem_path) if dem_path else output_path(out_dir, DEFAULT_DEM_NAME)
    river_mask = Path(mask_path) if mask_path else output_path(out_dir, DEFAULT_MASK_NAME)
    width = Path(width_path) if width_path else output_path(out_dir, DEFAULT_WIDTH_NAME)
    depth = Path(depth_path) if depth_path else output_path(out_dir, DEFAULT_DEPTH_NAME)

    required = [pre_bathy_dem, river_mask, width, depth]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required conditioning products:\n" + "\n".join(missing))

    no_bathy_dem = output_path(out_dir, output_dem_name)
    river_width = output_path(out_dir, output_width_name)
    river_depth = output_path(out_dir, output_depth_name)

    copy_raster(pre_bathy_dem, no_bathy_dem, overwrite=overwrite)
    width_summary = write_river_only_raster(width, river_mask, river_width, overwrite=overwrite)
    depth_summary = write_river_only_raster(depth, river_mask, river_depth, overwrite=overwrite)

    summary = {
        "no_bathymetry_dem": str(no_bathy_dem),
        "river_mask": str(river_mask),
        "river_width": width_summary,
        "river_depth": depth_summary,
    }
    summary_path = output_path(out_dir, "geometry_only_export_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a conditioned DEM without bathymetry lowering plus river-only "
            "width/depth rasters with NaN outside the D4 river mask."
        )
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="HydroBathyDEM output directory from a conditioning run.")
    parser.add_argument("--dem", type=Path, default=None, help="Override pre-bathymetry DEM path.")
    parser.add_argument("--river-mask", type=Path, default=None, help="Override D4 river-mask path.")
    parser.add_argument("--width", type=Path, default=None, help="Override river-width raster path.")
    parser.add_argument("--depth", type=Path, default=None, help="Override river-depth raster path.")
    parser.add_argument("--output-dem-name", default="DEM_conditioned_no_bathymetry.tif")
    parser.add_argument("--output-width-name", default="D4_River_Width_river_cells_m.tif")
    parser.add_argument("--output-depth-name", default="D4_River_Depth_river_cells_m.tif")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing export rasters.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = export_geometry_only_products(
        args.out_dir,
        dem_path=args.dem,
        mask_path=args.river_mask,
        width_path=args.width,
        depth_path=args.depth,
        output_dem_name=args.output_dem_name,
        output_width_name=args.output_width_name,
        output_depth_name=args.output_depth_name,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
