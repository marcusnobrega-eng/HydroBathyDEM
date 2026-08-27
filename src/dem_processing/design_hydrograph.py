"""Design runoff from the ready-made GCN250 Curve Number product.

This intentionally does *not* rebuild CN from local land cover or soil data.
It crops/resamples the published AMC-II GCN250 raster, then calculates the
area-weighted composite CN, SCS event runoff, and the SCS dimensionless-unit-
hydrograph peak at every D4 river cell.  The resulting design discharge is an
input to mesh-corridor design; it is not a flood map or a HydroPol boundary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt

from .condition_dem import compute_d4_flow_accumulation
from .config import load_config_file


GCN250_URLS = {
    "I": "https://ndownloader.figshare.com/files/15377357",
    "II": "https://ndownloader.figshare.com/files/15377363",
    "III": "https://ndownloader.figshare.com/files/15377342",
}


@dataclass(frozen=True)
class DesignHydrographConfig:
    dem: Path
    cn: Path
    river_mask: Path
    out_dir: Path
    idf_k: float
    idf_a: float
    idf_b_min: float
    idf_c: float
    river_direction: Path | None = None
    contributing_area: Path | None = None
    return_period_yr: float = 100.0
    initial_abstraction_ratio: float = 0.20
    storm_duration_ratio_to_tc: float = 0.133
    minimum_tc_min: float = 5.0
    minimum_storm_duration_min: float = 5.0
    hut_lag_ratio_to_tc: float = 0.60
    minimum_cn_coverage_fraction: float = 0.90

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DesignHydrographConfig":
        values = dict(values)
        grouped = {name: values.pop(name, {}) for name in ("inputs", "idf", "scs", "hut")}
        for name, group in grouped.items():
            if group and not isinstance(group, dict):
                raise ValueError(f"{name} must be a configuration object.")
        values = {**grouped["inputs"], **grouped["idf"], **grouped["scs"], **grouped["hut"], **values}
        aliases = {
            "cn_path": "cn", "dem_path": "dem", "river_mask_path": "river_mask", "output_dir": "out_dir",
            "k": "idf_k", "a": "idf_a", "b": "idf_b_min", "c": "idf_c",
            "lambda": "initial_abstraction_ratio", "lag_ratio": "hut_lag_ratio_to_tc",
        }
        values = {aliases.get(key, key): value for key, value in values.items()}
        for key in ("dem", "cn", "river_mask", "out_dir", "river_direction", "contributing_area"):
            if values.get(key) is None:
                continue
            values[key] = Path(values[key])
        config = cls(**values)
        if not (0.0 < config.initial_abstraction_ratio <= 1.0):
            raise ValueError("scs.initial_abstraction_ratio must be in (0, 1].")
        if min(config.idf_k, config.idf_b_min, config.idf_c, config.return_period_yr, config.minimum_tc_min) <= 0:
            raise ValueError("IDF and time-of-concentration parameters must be positive.")
        if not 0.0 < config.minimum_cn_coverage_fraction <= 1.0:
            raise ValueError("minimum_cn_coverage_fraction must be in (0, 1].")
        return config


def download_gcn250(amc: str, output: str | Path) -> Path:
    """Download one published GCN250 GeoTIFF atomically, unless already present."""
    amc = amc.upper().replace("ARC", "")
    if amc not in GCN250_URLS:
        raise ValueError("AMC must be I, II, or III.")
    output = Path(output).expanduser().resolve()
    if output.exists() and output.stat().st_size > 100_000_000:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(GCN250_URLS[amc], timeout=120) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(output)
    return output


def _aligned_cn(path: Path, reference: rasterio.DatasetReader) -> np.ndarray:
    """Area-average ready-made CN onto a model grid and reject invalid values."""
    result = np.full(reference.shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as source:
        reproject(
            rasterio.band(source, 1), result,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=reference.transform, dst_crs=reference.crs, dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return np.where((result >= 1.0) & (result <= 100.0), result, np.nan).astype(np.float64)


def _fill_missing_cn_from_nearest(values: np.ndarray, domain: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill GCN250 gaps from the nearest valid published CN within the domain."""
    result = values.astype(np.float64, copy=True)
    source = domain & np.isfinite(result)
    missing = domain & ~np.isfinite(result)
    if missing.any():
        if not source.any():
            raise ValueError("GCN250 has no valid Curve Number inside the model domain.")
        _, nearest = distance_transform_edt(~source, return_indices=True)
        result[missing] = result[tuple(indices[missing] for indices in nearest)]
    result[~domain] = np.nan
    return result, int(missing.sum())


def _aligned_mask(path: Path, reference: rasterio.DatasetReader) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.shape != reference.shape or source.transform != reference.transform or source.crs != reference.crs:
            raise ValueError(f"River mask is not aligned with the DEM: {path}")
        return source.read(1, masked=True).filled(0) > 0


def _aligned_float(path: Path, reference: rasterio.DatasetReader, label: str) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.shape != reference.shape or source.transform != reference.transform or source.crs != reference.crs:
            raise ValueError(f"{label} is not aligned with the DEM: {path}")
        return source.read(1, masked=True).astype(np.float64).filled(np.nan)


def receiver_from_d4_direction(direction: np.ndarray) -> np.ndarray:
    """Convert HydroBathyDEM's 1=N, 2=E, 3=S, 4=W codes to receivers."""
    rows, cols = direction.shape
    receiver = np.full(direction.shape, -1, dtype=np.int64)
    index = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    receiver[1:, :][direction[1:, :] == 1] = index[:-1, :][direction[1:, :] == 1]
    receiver[:-1, :][direction[:-1, :] == 3] = index[1:, :][direction[:-1, :] == 3]
    receiver[:, :-1][direction[:, :-1] == 2] = index[:, 1:][direction[:, :-1] == 2]
    receiver[:, 1:][direction[:, 1:] == 4] = index[:, :-1][direction[:, 1:] == 4]
    return receiver


def _d4_topological_order(receiver: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Order a D4 DAG upstream-to-downstream without assuming surface elevations."""
    flat_receiver = receiver.ravel().copy()
    flat_valid = valid.ravel()
    flat_receiver[~flat_valid] = -1
    good_receiver = (flat_receiver >= 0) & flat_valid[flat_receiver.clip(min=0)]
    flat_receiver[~good_receiver] = -1
    indegree = np.bincount(flat_receiver[flat_receiver >= 0], minlength=flat_receiver.size).astype(np.int32)
    queue = list(np.flatnonzero(flat_valid & (indegree == 0)))
    order = np.empty(int(flat_valid.sum()), dtype=np.int64)
    n = 0
    while queue:
        item = queue.pop()
        order[n] = item; n += 1
        downstream = int(flat_receiver[item])
        if downstream >= 0:
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                queue.append(downstream)
    if n != len(order):
        raise ValueError("D4 direction contains a cycle; cannot accumulate CN or travel time.")
    return order


def _upstream_composite_cn(cn: np.ndarray, receiver: np.ndarray, order: np.ndarray, cell_area_km2: float) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate CN-area and area once through an authoritative D4 DAG."""
    flat_cn, flat_receiver = cn.ravel(), receiver.ravel()
    area = np.where(np.isfinite(flat_cn), cell_area_km2, 0.0)
    cn_area = np.where(np.isfinite(flat_cn), flat_cn * cell_area_km2, 0.0)
    for item in order:
        downstream = int(flat_receiver[item])
        if downstream >= 0:
            area[downstream] += area[item]
            cn_area[downstream] += cn_area[item]
    composite = np.divide(cn_area, area, out=np.full_like(cn_area, np.nan), where=area > 0)
    return composite.reshape(cn.shape), area.reshape(cn.shape)


def _longest_d4_paths(dem: np.ndarray, receiver: np.ndarray, order: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Longest contributing D4 path and its drop to each cell, in metres."""
    flat_dem, flat_receiver = dem.ravel(), receiver.ravel()
    ncols = dem.shape[1]
    length = np.zeros(flat_dem.size, dtype=np.float64)
    head = flat_dem.copy()
    for item in order:
        downstream = int(flat_receiver[item])
        if downstream < 0:
            continue
        distance = dy_m if item // ncols != downstream // ncols else dx_m
        candidate = length[item] + distance
        if candidate > length[downstream]:
            length[downstream] = candidate
            head[downstream] = head[item]
    drop = np.maximum(head - flat_dem, 0.0)
    return length.reshape(dem.shape), drop.reshape(dem.shape)


def kirpich_tc_minutes(length_m: np.ndarray, drop_m: np.ndarray, minimum_minutes: float) -> np.ndarray:
    """Classic Kirpich Tc, with an explicit numerical floor for tiny headwaters."""
    slope = np.divide(drop_m, length_m, out=np.zeros_like(drop_m), where=length_m > 0)
    tc = 0.01947 * np.power(np.maximum(length_m, 1.0), 0.77) * np.power(np.maximum(slope, 1e-5), -0.385)
    return np.maximum(tc, minimum_minutes)


def scs_runoff_depth_mm(precipitation_mm: np.ndarray, cn: np.ndarray, initial_abstraction_ratio: float) -> np.ndarray:
    """SCS-CN event excess rainfall in mm."""
    storage = 25400.0 / cn - 254.0
    initial = initial_abstraction_ratio * storage
    available = precipitation_mm - initial
    return np.where(available > 0.0, available * available / (available + storage), 0.0)


def build_design_hydrograph(config: DesignHydrographConfig) -> dict[str, Any]:
    """Write ready-CN, composite-CN, and SCS-HUT design-raster diagnostics."""
    with rasterio.open(config.dem) as source:
        dem = source.read(1, masked=True).filled(np.nan).astype(np.float64)
        profile = source.profile.copy()
        cn_raw = _aligned_cn(config.cn, source)
        river = _aligned_mask(config.river_mask, source)
        supplied_direction = _aligned_float(config.river_direction, source, "D4 flow direction") if config.river_direction else None
        supplied_area = _aligned_float(config.contributing_area, source, "D4 contributing area") if config.contributing_area else None
    valid_dem = np.isfinite(dem)
    cn, cn_fallback_cells = _fill_missing_cn_from_nearest(cn_raw, valid_dem)
    if supplied_direction is None:
        accumulation, receiver = compute_d4_flow_accumulation(dem.astype(np.float32), profile, nodata=np.nan)
        contributing_area_km2 = accumulation * abs(profile["transform"].a * profile["transform"].e) / 1e6
        routing_source = "recomputed D4 from supplied DEM"
    else:
        receiver = receiver_from_d4_direction(supplied_direction)
        if supplied_area is None:
            raise ValueError("inputs.contributing_area is required when inputs.river_direction is supplied.")
        contributing_area_km2 = supplied_area
        routing_source = "HydroBathyDEM D4 direction and contributing-area products"
    cell_area_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
    order = _d4_topological_order(receiver, np.isfinite(dem))
    composite_cn, cn_covered_area_km2 = _upstream_composite_cn(cn, receiver, order, cell_area_km2)
    cn_coverage_fraction = np.divide(
        cn_covered_area_km2, contributing_area_km2,
        out=np.full_like(cn_covered_area_km2, np.nan), where=contributing_area_km2 > 0,
    )
    length_m, drop_m = _longest_d4_paths(dem, receiver, order, abs(profile["transform"].a), abs(profile["transform"].e))
    tc_min = kirpich_tc_minutes(length_m, drop_m, config.minimum_tc_min)
    duration_min = np.maximum(config.minimum_storm_duration_min, config.storm_duration_ratio_to_tc * tc_min)
    intensity_mm_h = config.idf_k * config.return_period_yr ** config.idf_a / np.power(duration_min + config.idf_b_min, config.idf_c)
    precipitation_mm = intensity_mm_h * duration_min / 60.0
    runoff_mm = scs_runoff_depth_mm(precipitation_mm, composite_cn, config.initial_abstraction_ratio)
    tp_h = 0.5 * duration_min / 60.0 + config.hut_lag_ratio_to_tc * tc_min / 60.0
    peak_m3_s = 0.208 * contributing_area_km2 * runoff_mm / tp_h
    failed_coverage = river & (
        ~np.isfinite(cn_coverage_fraction)
        | (cn_coverage_fraction < config.minimum_cn_coverage_fraction)
    )
    if failed_coverage.any():
        raise ValueError(
            f"CN fallback did not cover {int(failed_coverage.sum()):,} mapped river cells; "
            "the design hydrograph cannot omit them."
        )
    missing_peak = river & ~np.isfinite(peak_m3_s)
    if missing_peak.any():
        raise ValueError(
            f"Design peak is not finite at {int(missing_peak.sum()):,} mapped river cells."
        )
    outputs = {
        "GCN250_CN_AMCII_model_grid.tif": cn,
        "D4_composite_CN_AMCII.tif": composite_cn,
        "D4_contributing_area_km2.tif": contributing_area_km2,
        "D4_CN_coverage_fraction.tif": cn_coverage_fraction,
        "D4_kirpich_tc_min.tif": tc_min,
        "D4_design_storm_depth_mm.tif": precipitation_mm,
        "D4_SCS_runoff_depth_mm.tif": runoff_mm,
        "D4_SCS_HUT_Qp_100yr_m3s.tif": np.where(river, peak_m3_s, np.nan),
    }
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.update(driver="GTiff", dtype="float32", nodata=np.nan, count=1, compress="deflate")
    paths: dict[str, str] = {}
    for name, data in outputs.items():
        path = out_dir / name
        with rasterio.open(path, "w", **profile) as target:
            target.write(data.astype(np.float32), 1)
        paths[name] = str(path)
    active = river & np.isfinite(peak_m3_s)
    report = {
        "method": "published GCN250 AMC-II -> D4 area-weighted CN -> SCS-CN -> SCS dimensionless UH peak",
        "routing_source": routing_source,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "mapped_river_cells": int(river.sum()),
        "cn_fallback_cells": cn_fallback_cells,
        "river_cn_fallback_cells": int((river & valid_dem & ~np.isfinite(cn_raw)).sum()),
        "raw_cn_coverage_fraction_of_valid_dem": float(np.mean(np.isfinite(cn_raw)[valid_dem])),
        "cn_coverage_fraction_of_valid_dem": float(np.mean(np.isfinite(cn)[valid_dem])),
        "river_cells_below_cn_coverage_gate": int(failed_coverage.sum()),
        "river_cells_with_design_peak": int(active.sum()),
        "composite_cn_range": [float(np.nanmin(composite_cn)), float(np.nanmax(composite_cn))],
        "river_qp_100yr_m3s_quantiles": [float(np.nanquantile(peak_m3_s[active], value)) for value in (0.0, 0.5, 0.95, 1.0)] if active.any() else [],
        "outputs": paths,
        "scope": "Design-flow screening for mesh refinement only; not a calibrated flood-frequency estimate or inundation map.",
    }
    report_path = out_dir / "design_hydrograph_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "report": str(report_path)}


def main_download(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download a published ready-made GCN250 Curve Number GeoTIFF.")
    parser.add_argument("--amc", default="II", choices=("I", "II", "III"), help="Antecedent runoff condition product.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(download_gcn250(args.amc, args.output))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute D4 composite CN and SCS-HUT design peak from ready-made GCN250.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_design_hydrograph(DesignHydrographConfig.from_mapping(load_config_file(args.config))), indent=2))


if __name__ == "__main__":
    main()
