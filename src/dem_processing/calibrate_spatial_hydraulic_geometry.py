#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatially calibrate HydroPol2D hydraulic-geometry coefficients with Lin et al. data.

This script keeps the river network tied to the FABDEM-derived D4 routing grid,
but uses the Lin et al. 2020 reach data to estimate spatially variable power-law
coefficients:

    River_Width = beta_1 * A^beta_2
    River_Depth = alfa_1 * A^alfa_2

where A is drainage area in km2. The calibration zones are D4 subcatchments
defined from a flow-accumulation threshold, e.g. 1000 km2.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy import ndimage

try:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dem_processing_matplotlib")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from .condition_dem import compute_equivalent_H_abg, write_raster
    from .paths import PROJECT_ROOT, themed_output_path
except ImportError:  # Allows direct execution from the source directory.
    from condition_dem import compute_equivalent_H_abg, write_raster
    from paths import PROJECT_ROOT, themed_output_path


PROJECT_DIR = PROJECT_ROOT
DEFAULT_FAC_AREA = themed_output_path(PROJECT_DIR / "Outputs", "D4_Wshed_Properties_fac_area_km2.tif")
DEFAULT_D4_DIRECTION = themed_output_path(PROJECT_DIR / "Outputs", "D4_flow_direction.tif")
DEFAULT_LIN_GPKG = PROJECT_DIR / "Data" / "Lin2020_bankfull_width" / "processed" / "lin2020_dem_domain_width_depth.gpkg"
DEFAULT_OUT_DIR = PROJECT_DIR / "Data" / "Lin2020_bankfull_width" / "calibration"


@dataclass
class PowerLawFit:
    coefficient: float
    exponent: float
    n_total: int
    n_used: int
    r2_log: float
    rmse_log: float
    mae_log: float
    valid: bool


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_thresholds(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def read_grid(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).filled(np.nan).astype("float32")
        profile = src.profile.copy()
    return arr, profile


def reconstruct_receiver(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = direction.shape
    valid = np.isfinite(direction) & (direction > 0)
    lin = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    receiver = -np.ones((rows, cols), dtype=np.int64)

    code = np.where(valid, direction, 0).astype("int16", copy=False)

    mask = valid & (code == 1)
    receiver[1:, :][mask[1:, :]] = lin[:-1, :][mask[1:, :]]

    mask = valid & (code == 2)
    receiver[:, :-1][mask[:, :-1]] = lin[:, 1:][mask[:, :-1]]

    mask = valid & (code == 3)
    receiver[:-1, :][mask[:-1, :]] = lin[1:, :][mask[:-1, :]]

    mask = valid & (code == 4)
    receiver[:, 1:][mask[:, 1:]] = lin[:, :-1][mask[:, 1:]]

    return receiver, valid


def delineate_stream_links_and_zones(
    fac_area: np.ndarray,
    direction: np.ndarray,
    threshold_km2: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    receiver, valid_direction = reconstruct_receiver(direction)
    valid = valid_direction & np.isfinite(fac_area) & (fac_area > 0)
    stream = valid & (fac_area >= threshold_km2)

    flat_stream = stream.ravel()
    flat_receiver = receiver.ravel()
    n_cells = flat_stream.size

    stream_idx = np.flatnonzero(flat_stream)
    stream_receiver = flat_receiver[stream_idx]
    rec_is_stream = (stream_receiver >= 0) & flat_stream[np.clip(stream_receiver, 0, n_cells - 1)]

    upstream_count = np.zeros(n_cells, dtype="int16")
    np.add.at(upstream_count, stream_receiver[rec_is_stream], 1)

    link_id = np.zeros(n_cells, dtype="int32")
    starts = stream_idx[upstream_count[stream_idx] != 1]
    next_link = 1
    for start in starts:
        if link_id[start] != 0:
            continue
        cur = int(start)
        while cur >= 0 and flat_stream[cur] and link_id[cur] == 0:
            link_id[cur] = next_link
            rec = int(flat_receiver[cur])
            if rec < 0 or not flat_stream[rec] or upstream_count[rec] != 1:
                break
            cur = rec
        next_link += 1

    # Defensive pass for any stream cells missed by unusual local topology.
    missed = stream_idx[link_id[stream_idx] == 0]
    for start in missed:
        if link_id[start] != 0:
            continue
        cur = int(start)
        while cur >= 0 and flat_stream[cur] and link_id[cur] == 0:
            link_id[cur] = next_link
            rec = int(flat_receiver[cur])
            if rec < 0 or not flat_stream[rec] or upstream_count[rec] != 1:
                break
            cur = rec
        next_link += 1

    zone = link_id.copy()
    flat_valid = valid.ravel()
    flat_fac = np.where(np.isfinite(fac_area.ravel()), fac_area.ravel(), -1.0)
    order = np.flatnonzero(flat_valid)
    order = order[np.argsort(flat_fac[order])[::-1]]
    for idx in order:
        if zone[idx] != 0:
            continue
        rec = flat_receiver[idx]
        if rec >= 0:
            zone[idx] = zone[rec]

    zone_arr = zone.reshape(fac_area.shape)
    link_arr = link_id.reshape(fac_area.shape)
    stats = {
        "threshold_km2": float(threshold_km2),
        "stream_cells": int(stream.sum()),
        "stream_links": int(next_link - 1),
        "zoned_cells": int((zone_arr > 0).sum()),
    }
    return zone_arr, link_arr, stats


def zone_boundary_distance(zone: np.ndarray, profile: dict) -> np.ndarray:
    active = zone > 0
    boundary = np.zeros_like(active, dtype=bool)
    boundary[0, :] |= active[0, :]
    boundary[-1, :] |= active[-1, :]
    boundary[:, 0] |= active[:, 0]
    boundary[:, -1] |= active[:, -1]
    boundary[1:, :] |= active[1:, :] & (zone[1:, :] != zone[:-1, :])
    boundary[:-1, :] |= active[:-1, :] & (zone[:-1, :] != zone[1:, :])
    boundary[:, 1:] |= active[:, 1:] & (zone[:, 1:] != zone[:, :-1])
    boundary[:, :-1] |= active[:, :-1] & (zone[:, :-1] != zone[:, 1:])
    dx = abs(profile["transform"].a)
    dy = abs(profile["transform"].e)
    return ndimage.distance_transform_edt(~boundary, sampling=(dy, dx)).astype("float32")


def nearest_candidate_context(
    fac_area: np.ndarray,
    valid: np.ndarray,
    threshold_km2: float,
    profile: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    candidate = valid & np.isfinite(fac_area) & (fac_area >= threshold_km2)
    dx = abs(profile["transform"].a)
    dy = abs(profile["transform"].e)
    distance, indices = ndimage.distance_transform_edt(
        ~candidate,
        sampling=(dy, dx),
        return_indices=True,
    )
    nearest_area = fac_area[indices[0], indices[1]].astype("float32")
    nearest_area[~np.isfinite(nearest_area)] = np.nan
    return distance.astype("float32"), nearest_area


def lin_midpoints(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    try:
        return gdf.geometry.interpolate(0.5, normalized=True)
    except Exception:
        return gdf.geometry.representative_point()


def build_sample_table(
    lin_gpkg: Path,
    grid_profile: dict,
    fac_area: np.ndarray,
    zone: np.ndarray,
    nearest_d4_area: np.ndarray,
    nearest_d4_distance_m: np.ndarray,
    boundary_distance_m: np.ndarray,
) -> pd.DataFrame:
    gdf = gpd.read_file(lin_gpkg, layer="width_depth")
    if gdf.crs is None:
        raise ValueError(f"Lin GeoPackage has no CRS: {lin_gpkg}")
    if str(gdf.crs) != str(grid_profile["crs"]):
        gdf = gdf.to_crs(grid_profile["crs"])

    pts = lin_midpoints(gdf)
    xs = pts.x.to_numpy()
    ys = pts.y.to_numpy()
    rows, cols = rowcol(grid_profile["transform"], xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    in_grid = (rows >= 0) & (cols >= 0) & (rows < zone.shape[0]) & (cols < zone.shape[1])

    def sample_grid(arr: np.ndarray, default=np.nan) -> np.ndarray:
        out = np.full(len(gdf), default, dtype="float64")
        out[in_grid] = arr[rows[in_grid], cols[in_grid]]
        return out

    table = pd.DataFrame(
        {
            "COMID": pd.to_numeric(gdf["COMID"], errors="coerce").astype("Int64"),
            "x": xs,
            "y": ys,
            "row": np.where(in_grid, rows, -1),
            "col": np.where(in_grid, cols, -1),
            "zone_id": sample_grid(zone, default=0).astype("int64"),
            "lin_area_km2": pd.to_numeric(gdf["area"], errors="coerce").to_numpy(dtype="float64"),
            "lin_width_m": pd.to_numeric(gdf["width_m"], errors="coerce").to_numpy(dtype="float64"),
            "lin_depth_m": pd.to_numeric(gdf["Depth_Q2_m"], errors="coerce").to_numpy(dtype="float64"),
            "lin_Q2_m3s": pd.to_numeric(gdf["Q2"], errors="coerce").to_numpy(dtype="float64"),
            "d4_area_at_point_km2": sample_grid(fac_area),
            "nearest_d4_area_km2": sample_grid(nearest_d4_area),
            "nearest_d4_distance_m": sample_grid(nearest_d4_distance_m),
            "zone_boundary_distance_m": sample_grid(boundary_distance_m),
            "in_grid": in_grid,
        }
    )
    table["area_ratio_lin_to_nearest_d4"] = table["lin_area_km2"] / table["nearest_d4_area_km2"]
    return table


def add_quality_flags(
    samples: pd.DataFrame,
    max_nearest_distance_m: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_boundary_distance_m: float,
) -> pd.DataFrame:
    out = samples.copy()
    out["basic_valid"] = (
        out["in_grid"]
        & (out["zone_id"] > 0)
        & np.isfinite(out["lin_area_km2"])
        & np.isfinite(out["lin_width_m"])
        & np.isfinite(out["lin_depth_m"])
        & (out["lin_area_km2"] > 0)
        & (out["lin_width_m"] > 0)
        & (out["lin_depth_m"] > 0)
    )
    out["passes_distance"] = out["nearest_d4_distance_m"] <= max_nearest_distance_m
    out["passes_area_ratio"] = (
        np.isfinite(out["area_ratio_lin_to_nearest_d4"])
        & (out["area_ratio_lin_to_nearest_d4"] >= min_area_ratio)
        & (out["area_ratio_lin_to_nearest_d4"] <= max_area_ratio)
    )
    out["passes_boundary"] = out["zone_boundary_distance_m"] >= min_boundary_distance_m
    out["calibration_valid"] = (
        out["basic_valid"]
        & out["passes_distance"]
        & out["passes_area_ratio"]
        & out["passes_boundary"]
    )
    return out


def add_fit_area(samples: pd.DataFrame, fit_area_source: str) -> pd.DataFrame:
    """Choose the drainage-area predictor used in width/depth power-law fitting."""
    out = samples.copy()
    if fit_area_source == "lin":
        out["fit_area_km2"] = out["lin_area_km2"]
    elif fit_area_source == "d4":
        out["fit_area_km2"] = out["nearest_d4_area_km2"]
    else:
        raise ValueError("fit_area_source must be either 'lin' or 'd4'.")
    out["calibration_valid"] = (
        out["calibration_valid"]
        & np.isfinite(out["fit_area_km2"])
        & (out["fit_area_km2"] > 0)
    )
    return out


def deterministic_train_mask(samples: pd.DataFrame) -> np.ndarray:
    comid = samples["COMID"].fillna(0).astype("int64").to_numpy()
    return (np.abs(comid) % 5) != 0


def robust_power_law_fit(
    area_km2: np.ndarray,
    target: np.ndarray,
    min_samples: int,
    min_log10_area_range: float,
    exponent_bounds: Tuple[float, float],
    max_mad: float,
    min_abs_log_residual: float,
) -> PowerLawFit:
    area = np.asarray(area_km2, dtype="float64")
    yval = np.asarray(target, dtype="float64")
    valid = np.isfinite(area) & np.isfinite(yval) & (area > 0) & (yval > 0)
    if int(valid.sum()) < min_samples:
        return PowerLawFit(np.nan, np.nan, int(valid.sum()), 0, np.nan, np.nan, np.nan, False)

    x = np.log(area[valid])
    y = np.log(yval[valid])
    if (np.nanmax(x) - np.nanmin(x)) / np.log(10.0) < min_log10_area_range:
        return PowerLawFit(np.nan, np.nan, int(valid.sum()), 0, np.nan, np.nan, np.nan, False)

    keep = np.ones_like(x, dtype=bool)
    slope = np.nan
    intercept = np.nan
    for _ in range(8):
        if int(keep.sum()) < min_samples:
            return PowerLawFit(np.nan, np.nan, int(valid.sum()), int(keep.sum()), np.nan, np.nan, np.nan, False)
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (intercept + slope * x)
        med = np.nanmedian(residual[keep])
        mad = np.nanmedian(np.abs(residual[keep] - med))
        robust_sigma = 1.4826 * mad
        cutoff = max(max_mad * robust_sigma, min_abs_log_residual)
        new_keep = np.abs(residual - med) <= cutoff
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    if not (exponent_bounds[0] <= slope <= exponent_bounds[1]):
        return PowerLawFit(np.nan, np.nan, int(valid.sum()), int(keep.sum()), np.nan, np.nan, np.nan, False)

    pred = intercept + slope * x[keep]
    resid = y[keep] - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y[keep] - np.mean(y[keep])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    coefficient = float(np.exp(intercept))
    return PowerLawFit(coefficient, float(slope), int(valid.sum()), int(keep.sum()), float(r2), rmse, mae, True)


def predict_power_law(area_km2: np.ndarray, coefficient: float, exponent: float) -> np.ndarray:
    area = np.asarray(area_km2, dtype="float64")
    out = coefficient * np.power(np.maximum(area, 0.0), exponent)
    out[~np.isfinite(out)] = np.nan
    return out


def log_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    obs = np.asarray(observed, dtype="float64")
    pred = np.asarray(predicted, dtype="float64")
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    if int(valid.sum()) < 3:
        return {"n": int(valid.sum()), "rmse_log": np.nan, "mae_log": np.nan, "r2_log": np.nan}
    y = np.log(obs[valid])
    p = np.log(pred[valid])
    resid = y - p
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(valid.sum()),
        "rmse_log": float(np.sqrt(np.mean(resid**2))),
        "mae_log": float(np.mean(np.abs(resid))),
        "r2_log": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def fit_zone_coefficients(
    samples: pd.DataFrame,
    train_mask: np.ndarray,
    min_samples: int,
    min_log10_area_range: float,
    max_mad: float,
    min_abs_log_residual: float,
) -> Tuple[pd.DataFrame, Dict[str, PowerLawFit]]:
    train = samples["calibration_valid"].to_numpy() & train_mask
    train_count = int(train.sum())
    global_min_samples = max(min_samples, 50) if train_count >= 50 else min_samples
    global_width = robust_power_law_fit(
        samples.loc[train, "fit_area_km2"].to_numpy(),
        samples.loc[train, "lin_width_m"].to_numpy(),
        min_samples=global_min_samples,
        min_log10_area_range=min_log10_area_range,
        exponent_bounds=(0.05, 1.20),
        max_mad=max_mad,
        min_abs_log_residual=min_abs_log_residual,
    )
    global_depth = robust_power_law_fit(
        samples.loc[train, "fit_area_km2"].to_numpy(),
        samples.loc[train, "lin_depth_m"].to_numpy(),
        min_samples=global_min_samples,
        min_log10_area_range=min_log10_area_range,
        exponent_bounds=(0.02, 1.20),
        max_mad=max_mad,
        min_abs_log_residual=min_abs_log_residual,
    )
    if not global_width.valid or not global_depth.valid:
        raise RuntimeError("Global Lin hydraulic-geometry fit failed. Relax filters or check the Lin input data.")

    rows = []
    good = samples[samples["calibration_valid"]].copy()
    train_good = samples[train].copy()
    zone_ids = sorted(int(z) for z in good["zone_id"].dropna().unique() if int(z) > 0)
    train_by_zone = {int(k): v for k, v in train_good.groupby("zone_id", sort=False)}
    good_counts = good.groupby("zone_id").size().to_dict()

    for zone_id in zone_ids:
        zone_train = train_by_zone.get(zone_id)
        if zone_train is None:
            zone_train = train_good.iloc[0:0]
        width_fit = robust_power_law_fit(
            zone_train["fit_area_km2"].to_numpy(),
            zone_train["lin_width_m"].to_numpy(),
            min_samples=min_samples,
            min_log10_area_range=min_log10_area_range,
            exponent_bounds=(0.05, 1.20),
            max_mad=max_mad,
            min_abs_log_residual=min_abs_log_residual,
        )
        depth_fit = robust_power_law_fit(
            zone_train["fit_area_km2"].to_numpy(),
            zone_train["lin_depth_m"].to_numpy(),
            min_samples=min_samples,
            min_log10_area_range=min_log10_area_range,
            exponent_bounds=(0.02, 1.20),
            max_mad=max_mad,
            min_abs_log_residual=min_abs_log_residual,
        )
        local_valid = width_fit.valid and depth_fit.valid
        rows.append(
            {
                "zone_id": zone_id,
                "calibration_samples": int(good_counts.get(zone_id, 0)),
                "train_samples": int(len(zone_train)),
                "local_fit_valid": bool(local_valid),
                "fallback_level": 0 if local_valid else 1,
                "beta_1": width_fit.coefficient if local_valid else global_width.coefficient,
                "beta_2": width_fit.exponent if local_valid else global_width.exponent,
                "alfa_1": depth_fit.coefficient if local_valid else global_depth.coefficient,
                "alfa_2": depth_fit.exponent if local_valid else global_depth.exponent,
                "width_n_total": width_fit.n_total,
                "width_n_used": width_fit.n_used,
                "width_r2_log": width_fit.r2_log,
                "width_rmse_log": width_fit.rmse_log,
                "depth_n_total": depth_fit.n_total,
                "depth_n_used": depth_fit.n_used,
                "depth_r2_log": depth_fit.r2_log,
                "depth_rmse_log": depth_fit.rmse_log,
            }
        )

    zone_df = pd.DataFrame(rows)
    globals_ = {"width": global_width, "depth": global_depth}
    return zone_df, globals_


def evaluate_coefficients(
    samples: pd.DataFrame,
    zone_df: pd.DataFrame,
    globals_: Dict[str, PowerLawFit],
    validation_mask: np.ndarray,
) -> Dict[str, float]:
    valid_eval = samples["calibration_valid"].to_numpy() & validation_mask
    eval_df = samples.loc[valid_eval].copy()
    if eval_df.empty:
        return {
            "validation_samples": 0,
            "width_rmse_log": np.nan,
            "depth_rmse_log": np.nan,
            "width_r2_log": np.nan,
            "depth_r2_log": np.nan,
        }

    coeffs = zone_df.set_index("zone_id")[["beta_1", "beta_2", "alfa_1", "alfa_2"]]
    joined = eval_df.join(coeffs, on="zone_id")
    for col, fit_key, attr in [
        ("beta_1", "width", "coefficient"),
        ("beta_2", "width", "exponent"),
        ("alfa_1", "depth", "coefficient"),
        ("alfa_2", "depth", "exponent"),
    ]:
        fallback = getattr(globals_[fit_key], attr)
        joined[col] = joined[col].fillna(fallback)

    pred_width = predict_power_law(joined["fit_area_km2"].to_numpy(), joined["beta_1"].to_numpy(), joined["beta_2"].to_numpy())
    pred_depth = predict_power_law(joined["fit_area_km2"].to_numpy(), joined["alfa_1"].to_numpy(), joined["alfa_2"].to_numpy())
    wm = log_metrics(joined["lin_width_m"].to_numpy(), pred_width)
    dm = log_metrics(joined["lin_depth_m"].to_numpy(), pred_depth)
    return {
        "validation_samples": int(len(joined)),
        "width_rmse_log": wm["rmse_log"],
        "width_mae_log": wm["mae_log"],
        "width_r2_log": wm["r2_log"],
        "depth_rmse_log": dm["rmse_log"],
        "depth_mae_log": dm["mae_log"],
        "depth_r2_log": dm["r2_log"],
    }


def coefficient_maps(
    zone: np.ndarray,
    valid_grid: np.ndarray,
    zone_df: pd.DataFrame,
    globals_: Dict[str, PowerLawFit],
) -> Dict[str, np.ndarray]:
    maps = {
        "beta_1": np.full(zone.shape, globals_["width"].coefficient, dtype="float32"),
        "beta_2": np.full(zone.shape, globals_["width"].exponent, dtype="float32"),
        "alfa_1": np.full(zone.shape, globals_["depth"].coefficient, dtype="float32"),
        "alfa_2": np.full(zone.shape, globals_["depth"].exponent, dtype="float32"),
        "sample_count": np.zeros(zone.shape, dtype="float32"),
        "fit_quality": np.full(zone.shape, np.nan, dtype="float32"),
        "fallback_level": np.ones(zone.shape, dtype="float32"),
    }
    for _, row in zone_df.iterrows():
        mask = zone == int(row["zone_id"])
        if not np.any(mask):
            continue
        maps["beta_1"][mask] = float(row["beta_1"])
        maps["beta_2"][mask] = float(row["beta_2"])
        maps["alfa_1"][mask] = float(row["alfa_1"])
        maps["alfa_2"][mask] = float(row["alfa_2"])
        maps["sample_count"][mask] = float(row["calibration_samples"])
        if bool(row["local_fit_valid"]):
            quality = np.nanmean([row["width_r2_log"], row["depth_r2_log"]])
            maps["fit_quality"][mask] = quality
        maps["fallback_level"][mask] = float(row["fallback_level"])

    for key in ["beta_1", "beta_2", "alfa_1", "alfa_2", "sample_count", "fallback_level"]:
        maps[key][~valid_grid] = np.nan
    maps["fit_quality"][~valid_grid] = np.nan
    return maps


def write_calibration_rasters(
    out_dir: Path,
    threshold_km2: float,
    profile: dict,
    zone: np.ndarray,
    maps: Dict[str, np.ndarray],
    fac_area: np.ndarray,
    valid_grid: np.ndarray,
    application_min_area_km2: float,
    max_h_abg_m: float,
) -> Dict[str, Path]:
    tag = f"{threshold_km2:g}km2".replace(".", "p")
    paths: Dict[str, Path] = {}

    zone_path = out_dir / f"D4_subcatchments_{tag}.tif"
    write_raster(zone_path, zone.astype("int32"), profile, nodata=0, dtype="int32")
    paths["subcatchments"] = zone_path

    for key, stem in [
        ("beta_1", "D4_beta_1_width"),
        ("beta_2", "D4_beta_2_width"),
        ("alfa_1", "D4_alfa_1_depth"),
        ("alfa_2", "D4_alfa_2_depth"),
        ("sample_count", "D4_coeff_sample_count"),
        ("fit_quality", "D4_coeff_fit_quality"),
        ("fallback_level", "D4_coeff_fallback_level"),
    ]:
        path = out_dir / f"{stem}_{tag}.tif"
        write_raster(path, maps[key].astype("float32"), profile, nodata=-9999.0)
        paths[key] = path

    river_mask = valid_grid & np.isfinite(fac_area) & (fac_area >= application_min_area_km2)
    width = np.zeros(fac_area.shape, dtype="float32")
    depth = np.zeros(fac_area.shape, dtype="float32")
    width[river_mask] = maps["beta_1"][river_mask] * np.power(fac_area[river_mask], maps["beta_2"][river_mask])
    depth[river_mask] = maps["alfa_1"][river_mask] * np.power(fac_area[river_mask], maps["alfa_2"][river_mask])
    width[~np.isfinite(width)] = 0.0
    depth[~np.isfinite(depth)] = 0.0
    h_abg = compute_equivalent_H_abg(width, depth, grid_width=abs(profile["transform"].a), mode="wide", max_depth=max_h_abg_m)

    for key, arr, stem in [
        ("calibrated_width", width, "D4_calibrated_River_Width_m"),
        ("calibrated_depth", depth, "D4_calibrated_River_Depth_m"),
        ("calibrated_h_abg", h_abg, "D4_calibrated_H_abg_m"),
    ]:
        path = out_dir / f"{stem}_{tag}.tif"
        write_raster(path, arr.astype("float32"), profile, nodata=-9999.0)
        paths[key] = path

    return paths


def plot_threshold_summary(out_dir: Path, summary: pd.DataFrame) -> None:
    if plt is None or summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(summary["threshold_km2"], summary["width_rmse_log"], marker="o", label="width")
    axes[0].plot(summary["threshold_km2"], summary["depth_rmse_log"], marker="o", label="depth")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("subcatchment threshold (km2)")
    axes[0].set_ylabel("validation RMSE in log space")
    axes[0].legend()

    axes[1].plot(summary["threshold_km2"], summary["local_fit_valid_zones"], marker="o")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("subcatchment threshold (km2)")
    axes[1].set_ylabel("zones with local fits")

    axes[2].plot(summary["threshold_km2"], summary["local_fit_valid_sample_percent"], marker="o")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("subcatchment threshold (km2)")
    axes[2].set_ylabel("samples in locally fitted zones (%)")

    out_png = out_dir / "diagnostic_spatial_calibration_thresholds.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def plot_selected_diagnostics(
    out_dir: Path,
    threshold_km2: float,
    zone_df: pd.DataFrame,
    samples: pd.DataFrame,
    globals_: Dict[str, PowerLawFit],
    maps: Dict[str, np.ndarray],
    fit_area_source: str,
) -> None:
    if plt is None:
        return
    good = samples[samples["calibration_valid"]].copy()
    tag = f"{threshold_km2:g}km2".replace(".", "p")
    area_label = "Lin area" if fit_area_source == "lin" else "matched FABDEM-D4 area"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    sample = good.sample(min(len(good), 8000), random_state=42) if len(good) else good
    x = np.linspace(
        max(1.0, float(np.nanpercentile(good["fit_area_km2"], 1))) if len(good) else 1.0,
        float(np.nanpercentile(good["fit_area_km2"], 99)) if len(good) else 1000.0,
        200,
    )
    axes[0, 0].scatter(sample["fit_area_km2"], sample["lin_width_m"], s=4, alpha=0.25)
    axes[0, 0].plot(x, predict_power_law(x, globals_["width"].coefficient, globals_["width"].exponent), color="red")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(f"Lin width samples and global fit ({fit_area_source})")
    axes[0, 0].set_xlabel(f"{area_label} (km2)")
    axes[0, 0].set_ylabel("width_m")

    axes[0, 1].scatter(sample["fit_area_km2"], sample["lin_depth_m"], s=4, alpha=0.25)
    axes[0, 1].plot(x, predict_power_law(x, globals_["depth"].coefficient, globals_["depth"].exponent), color="red")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title(f"Manning depth samples and global fit ({fit_area_source})")
    axes[0, 1].set_xlabel(f"{area_label} (km2)")
    axes[0, 1].set_ylabel("depth_Q2_m")

    im = axes[1, 0].imshow(maps["sample_count"], cmap="viridis")
    axes[1, 0].set_title("Calibration sample count by D4 subcatchment")
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    im = axes[1, 1].imshow(maps["fallback_level"], cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("Fallback level (0 local, 1 global)")
    axes[1, 1].axis("off")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

    out_png = out_dir / f"diagnostic_spatial_calibration_{tag}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    if good.empty:
        return

    d4_sample = sample[
        np.isfinite(sample["nearest_d4_area_km2"])
        & (sample["nearest_d4_area_km2"] > 0.0)
        & np.isfinite(sample["lin_area_km2"])
        & (sample["lin_area_km2"] > 0.0)
    ].copy()
    if d4_sample.empty:
        return

    d4_good = good[
        np.isfinite(good["nearest_d4_area_km2"])
        & (good["nearest_d4_area_km2"] > 0.0)
        & np.isfinite(good["lin_area_km2"])
        & (good["lin_area_km2"] > 0.0)
    ].copy()
    x_d4 = np.linspace(
        max(1.0, float(np.nanpercentile(d4_good["nearest_d4_area_km2"], 1))),
        float(np.nanpercentile(d4_good["nearest_d4_area_km2"], 99)),
        200,
    )

    width_pred_d4 = predict_power_law(
        d4_good["nearest_d4_area_km2"].to_numpy(),
        globals_["width"].coefficient,
        globals_["width"].exponent,
    )
    depth_pred_d4 = predict_power_law(
        d4_good["nearest_d4_area_km2"].to_numpy(),
        globals_["depth"].coefficient,
        globals_["depth"].exponent,
    )
    width_metrics_d4 = log_metrics(d4_good["lin_width_m"].to_numpy(), width_pred_d4)
    depth_metrics_d4 = log_metrics(d4_good["lin_depth_m"].to_numpy(), depth_pred_d4)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    area_min = max(
        1.0,
        float(
            np.nanpercentile(
                np.concatenate(
                    [
                        d4_good["lin_area_km2"].to_numpy(dtype="float64"),
                        d4_good["nearest_d4_area_km2"].to_numpy(dtype="float64"),
                    ]
                ),
                1,
            )
        ),
    )
    area_max = float(
        np.nanpercentile(
            np.concatenate(
                [
                    d4_good["lin_area_km2"].to_numpy(dtype="float64"),
                    d4_good["nearest_d4_area_km2"].to_numpy(dtype="float64"),
                ]
            ),
            99,
        )
    )
    area_line = np.linspace(area_min, area_max, 200)

    axes[0].scatter(d4_sample["nearest_d4_area_km2"], d4_sample["lin_area_km2"], s=4, alpha=0.25)
    axes[0].plot(area_line, area_line, color="red", linewidth=1.5)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Lin area vs matched FABDEM-D4 area")
    axes[0].set_xlabel("nearest FABDEM-D4 area (km2)")
    axes[0].set_ylabel("Lin area (km2)")

    axes[1].scatter(d4_sample["nearest_d4_area_km2"], d4_sample["lin_width_m"], s=4, alpha=0.25)
    axes[1].plot(
        x_d4,
        predict_power_law(x_d4, globals_["width"].coefficient, globals_["width"].exponent),
        color="red",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Width vs FABDEM-D4 area (RMSE log={width_metrics_d4['rmse_log']:.3f})")
    axes[1].set_xlabel("nearest FABDEM-D4 area (km2)")
    axes[1].set_ylabel("width_m")

    axes[2].scatter(d4_sample["nearest_d4_area_km2"], d4_sample["lin_depth_m"], s=4, alpha=0.25)
    axes[2].plot(
        x_d4,
        predict_power_law(x_d4, globals_["depth"].coefficient, globals_["depth"].exponent),
        color="red",
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title(f"Manning depth vs FABDEM-D4 area (RMSE log={depth_metrics_d4['rmse_log']:.3f})")
    axes[2].set_xlabel("nearest FABDEM-D4 area (km2)")
    axes[2].set_ylabel("depth_Q2_m")

    out_png = out_dir / f"diagnostic_spatial_calibration_d4_area_{tag}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def run_for_threshold(
    threshold_km2: float,
    fac_area: np.ndarray,
    direction: np.ndarray,
    profile: dict,
    lin_gpkg: Path,
    args: argparse.Namespace,
    final_outputs: bool,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, PowerLawFit]], Optional[Dict[str, np.ndarray]]]:
    receiver, valid_grid = reconstruct_receiver(direction)
    del receiver
    zone, link_id, zone_stats = delineate_stream_links_and_zones(fac_area, direction, threshold_km2)
    print(f"[INFO] threshold {threshold_km2:g} km2: {zone_stats}")

    nearest_dist, nearest_area = nearest_candidate_context(
        fac_area,
        valid_grid,
        args.match_network_threshold_km2,
        profile,
    )
    boundary_dist = zone_boundary_distance(zone, profile)
    samples = build_sample_table(lin_gpkg, profile, fac_area, zone, nearest_area, nearest_dist, boundary_dist)
    samples = add_quality_flags(
        samples,
        max_nearest_distance_m=args.max_nearest_distance_m,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_boundary_distance_m=args.min_boundary_distance_m,
    )
    samples = add_fit_area(samples, args.fit_area_source)

    train_mask = deterministic_train_mask(samples)
    zone_train_df, train_globals = fit_zone_coefficients(
        samples,
        train_mask=train_mask,
        min_samples=args.min_samples_per_zone,
        min_log10_area_range=args.min_log10_area_range,
        max_mad=args.max_mad,
        min_abs_log_residual=args.min_abs_log_residual,
    )
    validation = evaluate_coefficients(samples, zone_train_df, train_globals, validation_mask=~train_mask)
    local_valid_zones = int(zone_train_df["local_fit_valid"].sum()) if len(zone_train_df) else 0
    locally_fit_zone_ids = set(zone_train_df.loc[zone_train_df["local_fit_valid"], "zone_id"].astype(int))
    good = samples[samples["calibration_valid"]].copy()
    local_samples = int(good["zone_id"].astype(int).isin(locally_fit_zone_ids).sum()) if len(good) else 0

    row = {
        **zone_stats,
        "fit_area_source": args.fit_area_source,
        "lin_samples_total": int(len(samples)),
        "lin_samples_calibration_valid": int(samples["calibration_valid"].sum()),
        "lin_samples_distance_rejected": int((samples["basic_valid"] & ~samples["passes_distance"]).sum()),
        "lin_samples_area_ratio_rejected": int((samples["basic_valid"] & ~samples["passes_area_ratio"]).sum()),
        "lin_samples_boundary_rejected": int((samples["basic_valid"] & ~samples["passes_boundary"]).sum()),
        "local_fit_valid_zones": local_valid_zones,
        "total_sampled_zones": int(samples.loc[samples["calibration_valid"], "zone_id"].nunique()),
        "local_fit_valid_sample_percent": 100.0 * local_samples / max(int(samples["calibration_valid"].sum()), 1),
        **validation,
    }

    if not final_outputs:
        return row, None, None, None, None

    final_zone_df, final_globals = fit_zone_coefficients(
        samples,
        train_mask=np.ones(len(samples), dtype=bool),
        min_samples=args.min_samples_per_zone,
        min_log10_area_range=args.min_log10_area_range,
        max_mad=args.max_mad,
        min_abs_log_residual=args.min_abs_log_residual,
    )
    maps = coefficient_maps(zone, valid_grid, final_zone_df, final_globals)
    raster_paths = write_calibration_rasters(
        args.out_dir,
        threshold_km2,
        profile,
        zone,
        maps,
        fac_area,
        valid_grid,
        application_min_area_km2=args.application_min_area_km2,
        max_h_abg_m=args.max_H_abg_m,
    )

    tag = f"{threshold_km2:g}km2".replace(".", "p")
    samples_csv = args.out_dir / f"lin_spatial_calibration_samples_{tag}.csv"
    samples.to_csv(samples_csv, index=False)
    print(f"[SAVED] {samples_csv}")

    coeff_csv = args.out_dir / f"spatial_hydraulic_coefficients_{tag}.csv"
    final_zone_df.to_csv(coeff_csv, index=False)
    print(f"[SAVED] {coeff_csv}")

    global_json = args.out_dir / f"spatial_hydraulic_global_fit_{tag}.json"
    global_json.write_text(
        json.dumps(
            {
                "threshold_km2": threshold_km2,
                "global_width": final_globals["width"].__dict__,
                "global_depth": final_globals["depth"].__dict__,
                "raster_paths": {k: str(v) for k, v in raster_paths.items()},
                "filters": {
                    "match_network_threshold_km2": args.match_network_threshold_km2,
                    "max_nearest_distance_m": args.max_nearest_distance_m,
                    "min_area_ratio": args.min_area_ratio,
                    "max_area_ratio": args.max_area_ratio,
                    "min_boundary_distance_m": args.min_boundary_distance_m,
                    "min_samples_per_zone": args.min_samples_per_zone,
                    "min_log10_area_range": args.min_log10_area_range,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[SAVED] {global_json}")
    return row, samples, final_zone_df, final_globals, maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate spatially variable HydroPol2D width/depth coefficients from Lin et al. 2020 data."
    )
    parser.add_argument("--fac-area", type=Path, default=DEFAULT_FAC_AREA)
    parser.add_argument("--d4-direction", type=Path, default=DEFAULT_D4_DIRECTION)
    parser.add_argument("--lin-gpkg", type=Path, default=DEFAULT_LIN_GPKG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selected-threshold-km2", type=float, default=1000.0)
    parser.add_argument("--thresholds-km2", default="500,1000,2500,5000")
    parser.add_argument(
        "--fit-area-source",
        choices=["lin", "d4"],
        default="d4",
        help="Drainage-area predictor used for fitting. 'd4' fits Lin width/depth against matched FABDEM-D4 area.",
    )
    parser.add_argument("--match-network-threshold-km2", type=float, default=25.0)
    parser.add_argument("--application-min-area-km2", type=float, default=100.0)
    parser.add_argument("--max-nearest-distance-m", type=float, default=5000.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.10)
    parser.add_argument("--max-area-ratio", type=float, default=10.0)
    parser.add_argument("--min-boundary-distance-m", type=float, default=0.0)
    parser.add_argument("--min-samples-per-zone", type=int, default=30)
    parser.add_argument("--min-log10-area-range", type=float, default=0.35)
    parser.add_argument("--max-mad", type=float, default=3.5)
    parser.add_argument("--min-abs-log-residual", type=float, default=0.35)
    parser.add_argument("--max-H-abg-m", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    fac_area, profile = read_grid(args.fac_area)
    direction, direction_profile = read_grid(args.d4_direction)
    if fac_area.shape != direction.shape:
        raise ValueError("Flow accumulation and D4 direction rasters have different shapes.")
    if str(profile["crs"]) != str(direction_profile["crs"]):
        raise ValueError("Flow accumulation and D4 direction rasters have different CRS.")

    thresholds = sorted(set(parse_thresholds(args.thresholds_km2) + [float(args.selected_threshold_km2)]))
    rows = []
    selected_outputs = None
    for threshold in thresholds:
        final_outputs = np.isclose(threshold, args.selected_threshold_km2)
        result = run_for_threshold(threshold, fac_area, direction, profile, args.lin_gpkg, args, final_outputs)
        row, samples, zone_df, globals_, maps = result
        rows.append(row)
        if final_outputs:
            selected_outputs = (samples, zone_df, globals_, maps)

    summary = pd.DataFrame(rows).sort_values("threshold_km2")
    summary_csv = args.out_dir / "spatial_calibration_threshold_optimization_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[SAVED] {summary_csv}")
    print(summary.to_string(index=False))
    plot_threshold_summary(args.out_dir, summary)

    if selected_outputs is not None:
        samples, zone_df, globals_, maps = selected_outputs
        plot_selected_diagnostics(args.out_dir, args.selected_threshold_km2, zone_df, samples, globals_, maps, args.fit_area_source)

    readme = args.out_dir / "README.md"
    tag = f"{args.selected_threshold_km2:g}km2".replace(".", "p")
    readme.write_text(
        "\n".join(
            [
                "# Spatial Hydraulic Geometry Calibration",
                "",
                "This folder contains FABDEM-D4 subcatchment coefficient rasters calibrated from Lin et al. 2020 data.",
                "",
                f"Selected threshold: `{args.selected_threshold_km2:g} km2`",
                f"Thresholds tested: `{', '.join(str(t) for t in thresholds)} km2`",
                f"Fit area source: `{args.fit_area_source}`",
                "",
                "The coefficient rasters can be used by the DEM-conditioning workflow with:",
                "",
                "```bash",
                "--river-geometry-source spatial_coefficients_or_power_law \\",
                f"--spatial-beta-1-raster {args.out_dir / ('D4_beta_1_width_' + tag + '.tif')} \\",
                f"--spatial-beta-2-raster {args.out_dir / ('D4_beta_2_width_' + tag + '.tif')} \\",
                f"--spatial-alfa-1-raster {args.out_dir / ('D4_alfa_1_depth_' + tag + '.tif')} \\",
                f"--spatial-alfa-2-raster {args.out_dir / ('D4_alfa_2_depth_' + tag + '.tif')}",
                "```",
                "",
                "Diagnostics:",
                "",
                f"- `diagnostic_spatial_calibration_thresholds.png`: validation metrics across tested subcatchment thresholds.",
                f"- `diagnostic_spatial_calibration_{tag}.png`: Lin-area width/depth fits, sample counts, and local/global fallback map.",
                f"- `diagnostic_spatial_calibration_d4_area_{tag}.png`: companion plots using matched FABDEM-D4 drainage area.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[SAVED] {readme}")


if __name__ == "__main__":
    main()
