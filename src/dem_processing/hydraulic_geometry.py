"""Public hydraulic-geometry helpers shared by HydroBathyDEM consumers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def estimate_power_law_geometry(
    contributing_area_km2: np.ndarray,
    river_mask: np.ndarray,
    *,
    width_coefficient: float = 2.2695,
    width_exponent: float = 0.4942,
    depth_coefficient: float = 0.1097,
    depth_exponent: float = 0.3856,
    width_cap_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rectangular river width and depth from contributing area."""
    area = np.asarray(contributing_area_km2, dtype=np.float64)
    river = np.asarray(river_mask, dtype=bool)
    if area.shape != river.shape:
        raise ValueError("Contributing area and river mask must have the same shape.")
    if min(width_coefficient, depth_coefficient) <= 0 or min(width_exponent, depth_exponent) < 0:
        raise ValueError("Hydraulic-geometry coefficients must be physically positive.")
    channel = river & np.isfinite(area) & (area > 0.0)
    width = np.zeros(area.shape, dtype=np.float64)
    depth = np.zeros(area.shape, dtype=np.float64)
    width[channel] = width_coefficient * np.power(area[channel], width_exponent)
    depth[channel] = depth_coefficient * np.power(area[channel], depth_exponent)
    if width_cap_m is not None:
        if width_cap_m <= 0:
            raise ValueError("width_cap_m must be positive when provided.")
        width[channel] = np.minimum(width[channel], width_cap_m)
    return width, depth


def write_power_law_geometry(
    contributing_area_path: Path | str,
    river_mask_path: Path | str,
    active_mask_path: Path | str,
    output_directory: Path | str,
    *,
    width_coefficient: float = 2.2695,
    width_exponent: float = 0.4942,
    depth_coefficient: float = 0.1097,
    depth_exponent: float = 0.3856,
) -> tuple[Path, Path]:
    """Write river width/depth rasters on an existing projected model grid."""
    paths = tuple(map(Path, (contributing_area_path, river_mask_path, active_mask_path)))
    with rasterio.open(paths[0]) as area_source:
        area = area_source.read(1).astype(np.float64)
        profile = area_source.profile.copy()
        shape = area_source.shape
        transform = area_source.transform
        crs = area_source.crs
        cell_width_m = abs(float(transform.a))
    with rasterio.open(paths[1]) as river_source, rasterio.open(paths[2]) as mask_source:
        for label, source in (("river mask", river_source), ("active mask", mask_source)):
            if source.shape != shape or source.transform != transform or source.crs != crs:
                raise ValueError(f"The {label} is not aligned with the contributing-area raster.")
        river = river_source.read(1).astype(bool)
        active = mask_source.read(1).astype(bool)
    width, depth = estimate_power_law_geometry(
        area,
        river & active,
        width_coefficient=width_coefficient,
        width_exponent=width_exponent,
        depth_coefficient=depth_coefficient,
        depth_exponent=depth_exponent,
        width_cap_m=cell_width_m,
    )
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float64", count=1, nodata=0.0, compress="deflate")
    width_path = output / "river_width_m.tif"
    depth_path = output / "river_depth_m.tif"
    for path, values in ((width_path, width), (depth_path, depth)):
        with rasterio.open(path, "w", **profile) as destination:
            destination.write(values, 1)
    return width_path, depth_path
