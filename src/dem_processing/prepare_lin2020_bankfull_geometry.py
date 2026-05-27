#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Lin et al. 2020 bankfull river geometry for the DEM-conditioning model.

The source dataset is the Zenodo release:

    Global estimates of reach-level bankfull river width leveraging big-data
    geospatial analysis
    https://zenodo.org/records/3552776

The raw shapefile has no .prj sidecar in the Zenodo record, but its coordinate
bounds are global lon/lat degrees. This script treats it as EPSG:4326, subsets
it to the DEM grid domain, computes a Q2 bankfull depth from Manning's equation,
and rasterizes width/depth products to the model grid.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from shapely.geometry import box

try:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dem_processing_matplotlib")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
except Exception:
    plt = None

try:
    from .condition_dem import compute_equivalent_H_abg, rectangular_conveyance_term, write_raster
    from .paths import PROJECT_ROOT, themed_output_path
except ImportError:  # Allows direct execution from the source directory.
    from condition_dem import compute_equivalent_H_abg, rectangular_conveyance_term, write_raster
    from paths import PROJECT_ROOT, themed_output_path


PROJECT_DIR = PROJECT_ROOT
DEFAULT_RAW_DIR = PROJECT_DIR / "Data" / "Lin2020_bankfull_width" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_DIR / "Data" / "Lin2020_bankfull_width" / "processed"
DEFAULT_DEM_GRID = themed_output_path(PROJECT_DIR / "Outputs", "DEM_resampled_1000m.tif")
DEFAULT_D4_MASK = themed_output_path(PROJECT_DIR / "Outputs", "D4_idx_facc.tif")

SOURCE_RECORD_URL = "https://zenodo.org/records/3552776"
ZENODO_FILES: Dict[str, str] = {
    "rivers_ge30m.cpg": "https://zenodo.org/records/3552776/files/rivers_ge30m.cpg?download=1",
    "rivers_ge30m.shx": "https://zenodo.org/records/3552776/files/rivers_ge30m.shx?download=1",
    "rivers_ge30m.dbf": "https://zenodo.org/records/3552776/files/rivers_ge30m.dbf?download=1",
    "rivers_ge30m.shp": "https://zenodo.org/records/3552776/files/rivers_ge30m.shp?download=1",
}
RAW_STEM = "rivers_ge30m"
HYDRAULIC_COLUMNS = ["COMID", "area", "Slp", "QMEAN", "Q2", "width_m", "width_DHG"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_missing_files(raw_dir: Path) -> None:
    ensure_dir(raw_dir)
    for filename, url in ZENODO_FILES.items():
        out_path = raw_dir / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[OK] already downloaded: {out_path} ({out_path.stat().st_size:,} bytes)")
            continue

        cmd = [
            "curl",
            "-L",
            "--fail",
            "-C",
            "-",
            "--retry",
            "10",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "-o",
            str(out_path),
            url,
        ]
        print("[DOWNLOAD]", " ".join(cmd))
        subprocess.check_call(cmd)


def validate_raw_files(raw_dir: Path) -> Path:
    missing = []
    for filename in ZENODO_FILES:
        path = raw_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(filename)
    if missing:
        raise FileNotFoundError(
            "Missing Lin et al. 2020 raw shapefile parts:\n"
            + "\n".join(f"  - {raw_dir / name}" for name in missing)
            + "\nRun this script with --download to fetch the missing files."
        )
    return raw_dir / f"{RAW_STEM}.shp"


def looks_like_lonlat(bounds: Tuple[float, float, float, float]) -> bool:
    left, bottom, right, top = bounds
    return -181.0 <= left <= 181.0 and -91.0 <= bottom <= 91.0 and -181.0 <= right <= 181.0 and -91.0 <= top <= 91.0


def padded_bounds(bounds: Tuple[float, float, float, float], pad: float) -> Tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    return left - pad, bottom - pad, right + pad, top + pad


def dem_grid_context(dem_grid: Path) -> Dict[str, object]:
    with rasterio.open(dem_grid) as src:
        profile = src.profile.copy()
        if src.crs is None:
            raise ValueError(f"DEM grid has no CRS: {dem_grid}")
        bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        dx = abs(src.transform.a)
        dy = abs(src.transform.e)
        return {
            "profile": profile,
            "crs": src.crs,
            "bounds": src.bounds,
            "bounds_4326": bounds_4326,
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "cell_width_m": float(np.sqrt(dx * dy)),
            "dx_m": float(dx),
            "dy_m": float(dy),
        }


def read_domain_subset(raw_shp: Path, dem_info: Dict[str, object], bbox_pad_deg: float) -> gpd.GeoDataFrame:
    info = pyogrio.read_info(raw_shp)
    raw_bounds = tuple(info["total_bounds"])
    raw_crs = info.get("crs")
    if raw_crs is None and not looks_like_lonlat(raw_bounds):
        raise ValueError(
            f"The Lin shapefile has no CRS and its bounds do not look like lon/lat degrees: {raw_bounds}"
        )

    bbox = padded_bounds(tuple(dem_info["bounds_4326"]), bbox_pad_deg)
    print(f"[INFO] raw feature count: {info.get('features'):,}")
    print(f"[INFO] raw CRS: {raw_crs or 'missing; assuming EPSG:4326'}")
    print(f"[INFO] raw bounds: {raw_bounds}")
    print(f"[INFO] DEM bbox EPSG:4326 plus {bbox_pad_deg:g} deg pad: {bbox}")

    gdf = pyogrio.read_dataframe(raw_shp, bbox=bbox, columns=HYDRAULIC_COLUMNS)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.to_crs(dem_info["crs"])

    domain = box(*dem_info["bounds"])
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.intersects(domain)].copy()
    print(f"[INFO] features intersecting DEM grid domain: {len(gdf):,}")
    return gdf


def manning_depth_rectangular(
    discharge_m3s: np.ndarray,
    width_m: np.ndarray,
    slope: np.ndarray,
    manning_n: float,
    max_depth_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Q = (1/n) A R^(2/3) S^(1/2) for depth in a rectangular channel.
    """
    q = np.asarray(discharge_m3s, dtype="float64")
    w = np.asarray(width_m, dtype="float64")
    s = np.asarray(slope, dtype="float64")
    depth = np.zeros_like(q, dtype="float64")
    clipped = np.zeros_like(q, dtype=bool)

    valid = np.isfinite(q) & np.isfinite(w) & np.isfinite(s) & (q > 0.0) & (w > 0.0) & (s > 0.0)
    if not np.any(valid):
        return depth.astype("float32"), clipped

    target = q[valid] * manning_n / np.sqrt(s[valid])
    width_valid = w[valid]
    cap_term = rectangular_conveyance_term(width_valid, np.full_like(width_valid, max_depth_m))
    clipped_valid = target > cap_term

    lo = np.zeros_like(target)
    hi = np.full_like(target, max_depth_m)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        mid_term = rectangular_conveyance_term(width_valid, mid)
        too_low = mid_term < target
        lo[too_low] = mid[too_low]
        hi[~too_low] = mid[~too_low]

    depth_valid = 0.5 * (lo + hi)
    depth_valid = np.where(np.isfinite(depth_valid), depth_valid, 0.0)
    depth_valid = np.clip(depth_valid, 0.0, max_depth_m)
    depth[valid] = depth_valid
    clipped[valid] = clipped_valid
    return depth.astype("float32"), clipped


def compute_reach_geometry(
    gdf: gpd.GeoDataFrame,
    dem_info: Dict[str, object],
    manning_n: float,
    min_slope: float,
    max_slope: float,
    depth_cap_m: float,
    max_h_abg_m: float,
    carve_mode: str,
) -> gpd.GeoDataFrame:
    out = gdf.copy()
    for col in ["area", "Slp", "QMEAN", "Q2", "width_m", "width_DHG"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out[np.isfinite(out["width_m"]) & np.isfinite(out["Q2"]) & (out["width_m"] > 0.0) & (out["Q2"] > 0.0)].copy()
    print(f"[INFO] features with positive width_m and Q2: {len(out):,} of {before:,}")

    slope_raw = out["Slp"].to_numpy(dtype="float64")
    slope_used = np.clip(np.where(np.isfinite(slope_raw), slope_raw, min_slope), min_slope, max_slope)
    out["Slope_raw"] = slope_raw.astype("float32")
    out["Slope_used"] = slope_used.astype("float32")
    out["Slope_was_clamped"] = (slope_used != slope_raw)
    out["Manning_n"] = float(manning_n)

    depth, clipped = manning_depth_rectangular(
        out["Q2"].to_numpy(dtype="float64"),
        out["width_m"].to_numpy(dtype="float64"),
        slope_used,
        manning_n=manning_n,
        max_depth_m=depth_cap_m,
    )
    out["Depth_Q2_m"] = depth
    out["Depth_Q2_clipped"] = clipped

    out["H_abg_m"] = compute_equivalent_H_abg(
        out["width_m"].to_numpy(dtype="float64"),
        out["Depth_Q2_m"].to_numpy(dtype="float64"),
        grid_width=float(dem_info["cell_width_m"]),
        mode=carve_mode,
        max_depth=max_h_abg_m,
    )
    out = out[np.isfinite(out["Depth_Q2_m"]) & (out["Depth_Q2_m"] > 0.0)].copy()
    print(f"[INFO] features with computed positive Manning depth: {len(out):,}")
    return out


def summary_rows(gdf: gpd.GeoDataFrame, settings: Dict[str, object]) -> pd.DataFrame:
    rows = []
    rows.append({"metric": "feature_count", "value": float(len(gdf)), "units": "count"})
    rows.append({"metric": "manning_n", "value": float(settings["manning_n"]), "units": ""})
    rows.append({"metric": "min_slope", "value": float(settings["min_slope"]), "units": "m/m"})
    rows.append({"metric": "max_slope", "value": float(settings["max_slope"]), "units": "m/m"})
    rows.append({"metric": "depth_cap_m", "value": float(settings["depth_cap_m"]), "units": "m"})
    rows.append({"metric": "slope_clamped_features", "value": float(gdf["Slope_was_clamped"].sum()), "units": "count"})
    rows.append({"metric": "depth_cap_hit_features", "value": float(gdf["Depth_Q2_clipped"].sum()), "units": "count"})

    for col, units in [
        ("area", "km2"),
        ("QMEAN", "m3/s"),
        ("Q2", "m3/s"),
        ("width_m", "m"),
        ("width_DHG", "m"),
        ("Slope_raw", "m/m"),
        ("Slope_used", "m/m"),
        ("Depth_Q2_m", "m"),
        ("H_abg_m", "m"),
    ]:
        vals = gdf[col].to_numpy(dtype="float64")
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        for stat, value in [
            ("min", np.nanmin(vals)),
            ("p05", np.nanpercentile(vals, 5)),
            ("p50", np.nanpercentile(vals, 50)),
            ("mean", np.nanmean(vals)),
            ("p95", np.nanpercentile(vals, 95)),
            ("max", np.nanmax(vals)),
        ]:
            rows.append({"metric": f"{col}_{stat}", "value": float(value), "units": units})
    return pd.DataFrame(rows)


def rasterize_attribute(
    geometries: Iterable[object],
    values: Iterable[float],
    dem_info: Dict[str, object],
    all_touched: bool,
) -> np.ndarray:
    shapes = ((geom, float(value)) for geom, value in zip(geometries, values) if geom is not None and np.isfinite(value))
    return rasterize(
        shapes=shapes,
        out_shape=(int(dem_info["height"]), int(dem_info["width"])),
        transform=dem_info["transform"],
        fill=0.0,
        dtype="float32",
        all_touched=all_touched,
    )


def rasterize_products(
    gdf: gpd.GeoDataFrame,
    dem_info: Dict[str, object],
    processed_dir: Path,
    buffer_m: float,
    all_touched: bool,
) -> Dict[str, Path]:
    res_tag = f"{int(round(float(dem_info['cell_width_m'])))}m"
    sorted_gdf = gdf.sort_values("H_abg_m").copy()
    if buffer_m > 0.0:
        raster_geoms = sorted_gdf.geometry.buffer(buffer_m, cap_style=2, join_style=2)
    else:
        raster_geoms = sorted_gdf.geometry

    products = {
        "width_m": ("Lin2020_width_m", "width_m"),
        "depth_Q2_m": ("Lin2020_depth_Q2_m", "Depth_Q2_m"),
        "Q2_m3s": ("Lin2020_Q2_m3s", "Q2"),
        "slope_used": ("Lin2020_slope_used", "Slope_used"),
        "H_abg_m": ("Lin2020_H_abg_m", "H_abg_m"),
    }
    paths: Dict[str, Path] = {}
    arrays: Dict[str, np.ndarray] = {}
    for key, (stem, col) in products.items():
        arr = rasterize_attribute(raster_geoms, sorted_gdf[col].to_numpy(dtype="float64"), dem_info, all_touched)
        out_path = processed_dir / f"{stem}_{res_tag}.tif"
        write_raster(out_path, arr, dem_info["profile"], nodata=-9999.0)
        paths[key] = out_path
        arrays[key] = arr

    mask = ((arrays["width_m"] > 0.0) & (arrays["depth_Q2_m"] > 0.0)).astype("float32")
    mask_path = processed_dir / f"Lin2020_mask_{res_tag}.tif"
    write_raster(mask_path, mask, dem_info["profile"], nodata=-9999.0)
    paths["mask"] = mask_path
    print(f"[INFO] Lin2020 raster cells with width/depth: {int(mask.sum()):,}")
    return paths


def d4_overlap_summary(processed_dir: Path, raster_paths: Dict[str, Path], d4_mask_path: Path) -> pd.DataFrame:
    with rasterio.open(raster_paths["mask"]) as src:
        external_mask = src.read(1) > 0
        shape = src.shape
    if not d4_mask_path.exists():
        return pd.DataFrame([{"metric": "d4_mask_available", "value": 0.0, "units": "boolean"}])

    with rasterio.open(d4_mask_path) as src:
        if src.shape != shape:
            return pd.DataFrame([{"metric": "d4_mask_same_shape", "value": 0.0, "units": "boolean"}])
        d4_mask = src.read(1) > 0

    both = external_mask & d4_mask
    rows = [
        {"metric": "d4_mask_available", "value": 1.0, "units": "boolean"},
        {"metric": "lin2020_cells", "value": float(external_mask.sum()), "units": "cells"},
        {"metric": "d4_river_cells", "value": float(d4_mask.sum()), "units": "cells"},
        {"metric": "overlap_cells", "value": float(both.sum()), "units": "cells"},
        {
            "metric": "lin2020_percent_overlapping_d4",
            "value": float(100.0 * both.sum() / max(external_mask.sum(), 1)),
            "units": "percent",
        },
        {
            "metric": "d4_percent_with_lin2020_geometry",
            "value": float(100.0 * both.sum() / max(d4_mask.sum(), 1)),
            "units": "percent",
        },
    ]
    df = pd.DataFrame(rows)
    out_csv = processed_dir / "lin2020_overlap_with_current_d4_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")
    print(df.to_string(index=False))
    return df


def plot_diagnostics(processed_dir: Path, raster_paths: Dict[str, Path], d4_mask_path: Path | None) -> None:
    if plt is None:
        print("[WARN] matplotlib unavailable; skipping Lin2020 diagnostic plot.")
        return

    def read(path_key: str) -> np.ndarray:
        with rasterio.open(raster_paths[path_key]) as src:
            arr = src.read(1).astype("float64")
        arr[arr <= 0.0] = np.nan
        return arr

    width = read("width_m")
    depth = read("depth_Q2_m")
    habg = read("H_abg_m")

    overlap = np.zeros_like(width, dtype="float32")
    external_mask = np.isfinite(width) & np.isfinite(depth)
    overlap[external_mask] = 2.0
    if d4_mask_path and d4_mask_path.exists():
        with rasterio.open(d4_mask_path) as src:
            d4 = src.read(1) > 0
        if d4.shape == overlap.shape:
            overlap[d4] = 1.0
            overlap[external_mask] = 2.0
            overlap[external_mask & d4] = 3.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    def show(ax, arr, title, cmap, label):
        vals = arr[np.isfinite(arr)]
        if vals.size:
            vmax = float(np.nanpercentile(vals, 98))
            vmin = float(np.nanpercentile(vals, 2))
            if vmax <= vmin:
                vmax = float(np.nanmax(vals))
                vmin = float(np.nanmin(vals))
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, shrink=0.82)
        cb.set_label(label)

    show(axes[0, 0], width, "Lin2020 bankfull width", "viridis", "m")
    show(axes[0, 1], depth, "Manning depth from Q2", "magma", "m")
    show(axes[1, 0], habg, "Equivalent H_abg lowering", "inferno", "m")

    cmap = ListedColormap(["white", "#4d4d4d", "#2b8cbe", "#31a354"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = axes[1, 1].imshow(overlap, cmap=cmap, norm=norm)
    axes[1, 1].set_title("Current D4 mask vs Lin2020 geometry")
    axes[1, 1].axis("off")
    cb = fig.colorbar(im, ax=axes[1, 1], shrink=0.82, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["neither", "D4 only", "Lin only", "both"])

    out_png = processed_dir / "diagnostic_lin2020_width_depth.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def write_data_readme(raw_dir: Path, processed_dir: Path, settings: Dict[str, object], raster_paths: Dict[str, Path]) -> None:
    readme = processed_dir.parent / "README.md"
    base_dir = processed_dir.parent

    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            return str(path)

    lines = [
        "# Lin et al. 2020 Bankfull Geometry",
        "",
        "Source: https://zenodo.org/records/3552776",
        "",
        "This folder stores the project-local copy and processed subset of the Lin et al. global reach-level bankfull-width dataset.",
        "The Zenodo shapefile does not include a `.prj` file, but its bounds are global longitude/latitude degrees, so the prep script assigns EPSG:4326 before projecting to the DEM grid CRS.",
        "",
        "## Raw Files",
        "",
        f"The raw `{RAW_STEM}` shapefile parts were read from `{display_path(raw_dir)}`.",
        "They are not copied into this processed-data folder, which keeps small examples lightweight.",
        "",
    ]
    for filename in ZENODO_FILES:
        path = raw_dir / filename
        size = path.stat().st_size if path.exists() else 0
        lines.append(f"- `{display_path(path)}` ({size:,} bytes)")
    lines.extend(
        [
            "",
            "## Processed Files",
            "",
            "- `processed/lin2020_dem_domain_width_depth.gpkg`: DEM-domain reaches with Manning Q2 depth.",
            "- `processed/lin2020_width_depth_summary.csv`: field statistics and processing counts.",
            "- `processed/lin2020_processing_metadata.json`: source, grid, and parameter metadata.",
            "- `processed/diagnostic_lin2020_width_depth.png`: width/depth/H_abg diagnostics and current D4 overlap.",
            "",
            "Raster products on the model grid:",
            "",
        ]
    )
    for key, path in raster_paths.items():
        lines.append(f"- `{display_path(path)}`")
    lines.extend(
        [
            "",
            "## Manning Depth Calculation",
            "",
            "Depth is solved from the rectangular Manning equation:",
            "",
            "```text",
            "Q2 = (1/n) * A * R^(2/3) * S^(1/2)",
            "A = width_m * depth",
            "R = A / (width_m + 2 * depth)",
            "```",
            "",
            f"Current settings: `n={settings['manning_n']}`, `min_slope={settings['min_slope']}`, `max_slope={settings['max_slope']}`, `depth_cap_m={settings['depth_cap_m']}`.",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {readme}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, subset, and rasterize Lin et al. 2020 bankfull river width/depth data."
    )
    parser.add_argument("--download", action="store_true", help="Download missing Zenodo shapefile parts with curl.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--dem-grid", type=Path, default=DEFAULT_DEM_GRID)
    parser.add_argument("--d4-mask", type=Path, default=DEFAULT_D4_MASK)
    parser.add_argument("--bbox-pad-deg", type=float, default=0.25)
    parser.add_argument("--manning-n", type=float, default=0.035)
    parser.add_argument("--min-slope", type=float, default=1.0e-5)
    parser.add_argument("--max-slope", type=float, default=0.05)
    parser.add_argument("--depth-cap-m", type=float, default=60.0)
    parser.add_argument("--max-H-abg-m", type=float, default=50.0)
    parser.add_argument("--carve-mode", choices=["wide", "manning_exact"], default="wide")
    parser.add_argument(
        "--rasterize-buffer-m",
        type=float,
        default=None,
        help="Buffer reaches before rasterization. Default is half the DEM cell width.",
    )
    parser.add_argument("--all-touched", action="store_true", help="Use rasterio all_touched during rasterization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.raw_dir)
    ensure_dir(args.processed_dir)

    if args.download:
        download_missing_files(args.raw_dir)

    raw_shp = validate_raw_files(args.raw_dir)
    dem_info = dem_grid_context(args.dem_grid)
    buffer_m = args.rasterize_buffer_m
    if buffer_m is None:
        buffer_m = 0.5 * float(dem_info["cell_width_m"])

    subset = read_domain_subset(raw_shp, dem_info, args.bbox_pad_deg)
    settings = {
        "source_record_url": SOURCE_RECORD_URL,
        "manning_n": args.manning_n,
        "min_slope": args.min_slope,
        "max_slope": args.max_slope,
        "depth_cap_m": args.depth_cap_m,
        "max_H_abg_m": args.max_H_abg_m,
        "carve_mode": args.carve_mode,
        "rasterize_buffer_m": buffer_m,
        "all_touched": args.all_touched,
    }
    reaches = compute_reach_geometry(
        subset,
        dem_info,
        manning_n=args.manning_n,
        min_slope=args.min_slope,
        max_slope=args.max_slope,
        depth_cap_m=args.depth_cap_m,
        max_h_abg_m=args.max_H_abg_m,
        carve_mode=args.carve_mode,
    )

    gpkg_path = args.processed_dir / "lin2020_dem_domain_width_depth.gpkg"
    reaches.to_file(gpkg_path, layer="width_depth", driver="GPKG")
    print(f"[SAVED] {gpkg_path}")

    summary = summary_rows(reaches, settings)
    summary_csv = args.processed_dir / "lin2020_width_depth_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[SAVED] {summary_csv}")
    print(summary.head(20).to_string(index=False))

    metadata = {
        "source_record_url": SOURCE_RECORD_URL,
        "raw_dir": str(args.raw_dir),
        "processed_dir": str(args.processed_dir),
        "raw_files": {name: (args.raw_dir / name).stat().st_size for name in ZENODO_FILES},
        "raw_crs_assumption": "EPSG:4326 because the Zenodo shapefile has no .prj and its bounds are lon/lat degrees",
        "dem_grid": str(args.dem_grid),
        "dem_grid_crs": str(dem_info["crs"]),
        "dem_grid_bounds": tuple(float(x) for x in dem_info["bounds"]),
        "dem_grid_bounds_epsg4326": tuple(float(x) for x in dem_info["bounds_4326"]),
        "settings": settings,
        "feature_count_processed": int(len(reaches)),
    }
    metadata_json = args.processed_dir / "lin2020_processing_metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[SAVED] {metadata_json}")

    raster_paths = rasterize_products(reaches, dem_info, args.processed_dir, buffer_m, args.all_touched)
    d4_overlap_summary(args.processed_dir, raster_paths, args.d4_mask)
    plot_diagnostics(args.processed_dir, raster_paths, args.d4_mask)
    write_data_readme(args.raw_dir, args.processed_dir, settings, raster_paths)


if __name__ == "__main__":
    main()
