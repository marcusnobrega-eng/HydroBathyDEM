"""Convert screened SCS-HUT peak flows into a terrain-connected mesh corridor.

The calculation is deliberately a screening cross-section, not a flood map:
it solves compound Manning conveyance at each routed river cell using the mapped
bankfull width/depth and the uncarved terrain on either side of the channel.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from .config import load_config_file
from .design_hydrograph import receiver_from_d4_direction, receiver_from_d8_direction


@dataclass(frozen=True)
class DesignCorridorConfig:
    dem: Path
    river_mask: Path
    river_direction: Path
    river_width: Path
    river_depth: Path
    river_bed: Path
    design_peak: Path
    out_dir: Path
    fallback_dem: Path | None = None
    routing_scheme: str = "d4"
    channel_manning_n: float = 0.035
    floodplain_manning_n: float = 0.05
    minimum_slope: float = 1e-4
    max_half_width_m: float = 2_000.0
    max_stage_above_bank_m: float = 20.0
    cross_section_spacing_m: float = 150.0
    slope_window_m: float = 600.0
    bank_sampling_width_m: float = 90.0
    bank_smoothing_window_m: float = 600.0
    centerline_simplification_m: float = 45.0
    floodplain_axis_min_upstream_area_km2: float = 0.02
    maximum_slope: float = 0.05

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DesignCorridorConfig":
        values = dict(values)
        grouped = {name: values.pop(name, {}) for name in ("inputs", "routing", "hydraulics")}
        for name, group in grouped.items():
            if group and not isinstance(group, dict):
                raise ValueError(f"{name} must be a configuration object.")
        values = {**grouped["inputs"], **grouped["routing"], **grouped["hydraulics"], **values}
        aliases = {
            "output_dir": "out_dir", "design_peak_path": "design_peak",
            "fallback_dem_path": "fallback_dem",
        }
        values = {aliases.get(key, key): value for key, value in values.items()}
        for key in ("dem", "river_mask", "river_direction", "river_width", "river_depth", "river_bed", "design_peak", "out_dir", "fallback_dem"):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        config = cls(**values)
        if min(config.channel_manning_n, config.floodplain_manning_n, config.minimum_slope, config.max_half_width_m, config.max_stage_above_bank_m, config.cross_section_spacing_m, config.slope_window_m, config.bank_sampling_width_m, config.bank_smoothing_window_m, config.maximum_slope, config.floodplain_axis_min_upstream_area_km2) <= 0 or config.centerline_simplification_m < 0:
            raise ValueError("hydraulic corridor parameters must be positive.")
        if config.maximum_slope < config.minimum_slope:
            raise ValueError("maximum_slope must be at least minimum_slope.")
        if config.routing_scheme not in {"d4", "d8"}:
            raise ValueError("routing.routing_scheme must be 'd4' or 'd8'.")
        return config


def _aligned(path: Path, reference: rasterio.DatasetReader, label: str) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.shape != reference.shape or source.transform != reference.transform or source.crs != reference.crs:
            raise ValueError(f"{label} is not aligned with the uncarved DEM: {path}")
        return source.read(1, masked=True).astype(np.float64).filled(np.nan)


def _fill_terminal_river_directions(
    direction: np.ndarray, river: np.ndarray, width: np.ndarray, routing_scheme: str = "d4",
) -> tuple[np.ndarray, int]:
    """Give terminal river cells a cross-section axis from their upstream link."""
    result = direction.astype(np.float64, copy=True)
    missing = river & ~np.isfinite(result)
    receiver = (
        receiver_from_d8_direction(result)
        if routing_scheme == "d8"
        else receiver_from_d4_direction(result)
    )
    rows, cols = result.shape
    for row, col in np.argwhere(missing):
        target = int(row * cols + col)
        upstream: list[tuple[float, float]] = []
        neighbours: list[tuple[int, int]] = []
        neighbours_to_check = (
            ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
            if routing_scheme == "d8" else ((-1, 0), (0, 1), (1, 0), (0, -1))
        )
        for drow, dcol in neighbours_to_check:
            nr, nc = int(row + drow), int(col + dcol)
            if not (0 <= nr < rows and 0 <= nc < cols) or not river[nr, nc]:
                continue
            neighbours.append((nr, nc))
            if receiver[nr, nc] == target and np.isfinite(result[nr, nc]):
                upstream.append((float(width[nr, nc]), float(result[nr, nc])))
        if upstream:
            result[row, col] = max(upstream, key=lambda item: item[0])[1]
        elif neighbours:
            nr, nc = max(neighbours, key=lambda item: float(width[item]))
            if routing_scheme == "d8":
                result[row, col] = {
                    (-1, 0): 1, (-1, 1): 2, (0, 1): 3, (1, 1): 4,
                    (1, 0): 5, (1, -1): 6, (0, -1): 7, (-1, -1): 8,
                }[(nr - row, nc - col)]
            else:
                result[row, col] = 1.0 if nr != row else 2.0
    unresolved = river & ~np.isfinite(result)
    if unresolved.any():
        raise ValueError(
            f"Cannot infer a cross-section orientation for {int(unresolved.sum()):,} terminal river cells."
        )
    return result, int(missing.sum())


def _rectangular_conveyance(width_m: float, depth_m: float, manning_n: float) -> float:
    if depth_m <= 0 or width_m <= 0:
        return 0.0
    area = width_m * depth_m
    radius = area / (width_m + 2.0 * depth_m)
    return area * radius ** (2.0 / 3.0) / manning_n


def _connected_side_depths(ground: np.ndarray, stage: float) -> np.ndarray:
    """Wet only the terrain connected laterally to the channel bank."""
    wet = np.where(np.isfinite(ground), np.maximum(stage - ground, 0.0), 0.0)
    dry = np.flatnonzero(wet <= 0.0)
    if len(dry):
        wet[dry[0]:] = 0.0
    return wet


def _compound_discharge(
    stage: float, bed: float, bank_depth: float, width: float, slope: float,
    left_ground: np.ndarray, right_ground: np.ndarray, cell_width: float,
    channel_n: float, floodplain_n: float,
) -> float:
    channel_depth = min(max(stage - bed, 0.0), bank_depth)
    conveyance = _rectangular_conveyance(width, channel_depth, channel_n)
    for ground in (left_ground, right_ground):
        for depth in _connected_side_depths(ground, stage):
            conveyance += _rectangular_conveyance(cell_width, float(depth), floodplain_n)
    return np.sqrt(slope) * conveyance


def _solve_stage(
    q_target: float, bed: float, bank_depth: float, width: float, slope: float,
    left_ground: np.ndarray, right_ground: np.ndarray, cell_width: float,
    channel_n: float, floodplain_n: float, max_stage_above_bank_m: float,
) -> tuple[float, bool]:
    bank = bed + bank_depth
    upper = min(max(np.nanmax(left_ground), np.nanmax(right_ground), bank), bank + max_stage_above_bank_m)
    if _compound_discharge(upper, bed, bank_depth, width, slope, left_ground, right_ground, cell_width, channel_n, floodplain_n) < q_target:
        return np.nan, False
    lower = bed
    for _ in range(36):
        middle = 0.5 * (lower + upper)
        if _compound_discharge(middle, bed, bank_depth, width, slope, left_ground, right_ground, cell_width, channel_n, floodplain_n) >= q_target:
            upper = middle
        else:
            lower = middle
    return upper, True


def _cross_section_indices(row: int, col: int, direction: int, first_offset: int, reach_cells: int, shape: tuple[int, int], routing_scheme: str = "d4") -> tuple[np.ndarray, np.ndarray]:
    """Return near-bank-to-outside cells normal to a D4 or D8 link."""
    # Coordinates are image rows/columns.  The convention is only used for
    # connected cross-sectional sampling; left/right labels have no output role.
    normal = (
        {
            1: (0, -1), 2: (-1, -1), 3: (-1, 0), 4: (-1, 1),
            5: (0, 1), 6: (1, 1), 7: (1, 0), 8: (1, -1),
        }.get(direction)
        if routing_scheme == "d8"
        else {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}.get(direction)
    )
    if normal is None:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)
    dr, dc = normal
    steps = np.arange(first_offset, first_offset + reach_cells)
    left = np.column_stack((row + dr * steps, col + dc * steps))
    right = np.column_stack((row - dr * steps, col - dc * steps))
    inside_left = (left[:, 0] >= 0) & (left[:, 0] < shape[0]) & (left[:, 1] >= 0) & (left[:, 1] < shape[1])
    inside_right = (right[:, 0] >= 0) & (right[:, 0] < shape[0]) & (right[:, 1] >= 0) & (right[:, 1] < shape[1])
    return left[inside_left], right[inside_right]


def _cross_section_cell_width(direction: int, dx_m: float, dy_m: float, routing_scheme: str = "d4") -> float:
    """Physical width represented by one lateral raster step."""
    normal = (
        {
            1: (0, -1), 2: (-1, -1), 3: (-1, 0), 4: (-1, 1),
            5: (0, 1), 6: (1, 1), 7: (1, 0), 8: (1, -1),
        }.get(direction)
        if routing_scheme == "d8"
        else {1: (0, -1), 2: (-1, 0), 3: (0, 1), 4: (1, 0)}.get(direction)
    )
    if normal is None:
        return min(dx_m, dy_m)
    return float(np.hypot(normal[0] * dy_m, normal[1] * dx_m))


def _river_reaches(receiver: np.ndarray, active: np.ndarray) -> list[list[int]]:
    """Directed, junction-to-junction reaches for an active river graph."""
    nodes = np.flatnonzero(active.ravel())
    downstream: dict[int, int] = {}
    indegree = {int(node): 0 for node in nodes}
    for node in nodes:
        target = int(receiver.ravel()[node])
        if target in indegree:
            downstream[int(node)] = target
            indegree[target] += 1
    starts = [node for node in nodes if indegree[int(node)] != 1]
    visited: set[tuple[int, int]] = set()
    reaches: list[list[int]] = []
    for start in starts:
        path = [int(start)]; current = int(start)
        while current in downstream:
            target = downstream[current]
            if (current, target) in visited: break
            visited.add((current, target)); path.append(target); current = target
            if indegree[current] != 1: break
        if len(path) > 1: reaches.append(path)
    for start, target in downstream.items():
        if (start, target) not in visited:
            reaches.append([start, target])
    return reaches


def _refinement_flow_envelope(peak: np.ndarray, receiver: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Conservative mesh-only flow envelope: no abrupt loss after a confluence."""
    result = np.where(active, peak, np.nan).ravel().copy()
    nodes = np.flatnonzero(active.ravel())
    indegree = np.zeros(result.size, dtype=np.int32)
    for node in nodes:
        target = int(receiver.ravel()[node])
        if target >= 0 and active.ravel()[target]: indegree[target] += 1
    queue = list(nodes[indegree[nodes] == 0])
    while queue:
        node = int(queue.pop())
        target = int(receiver.ravel()[node])
        if target >= 0 and active.ravel()[target]:
            result[target] = max(result[target], result[node])
            indegree[target] -= 1
            if indegree[target] == 0: queue.append(target)
    return result.reshape(peak.shape)


def _sample_positions(path: list[int], shape: tuple[int, int], dx_m: float, dy_m: float, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    distance = np.zeros(len(path), dtype=np.float64)
    for index, (left, right) in enumerate(zip(path[:-1], path[1:], strict=True), start=1):
        lr, lc = divmod(left, shape[1]); rr, rc = divmod(right, shape[1])
        distance[index] = distance[index - 1] + np.hypot((lr - rr) * dy_m, (lc - rc) * dx_m)
    picks = np.unique(np.r_[0, np.flatnonzero(np.diff(np.floor(distance / spacing_m)) > 0) + 1, len(path) - 1])
    return distance, picks


def _smoothed_slopes(bed: np.ndarray, distance: np.ndarray, window_m: float, min_slope: float, max_slope: float) -> np.ndarray:
    result = np.empty(len(bed), dtype=np.float64)
    for index, centre in enumerate(distance):
        left = np.searchsorted(distance, centre - 0.5 * window_m, side="left")
        right = np.searchsorted(distance, centre + 0.5 * window_m, side="right") - 1
        if right <= left:
            left, right = max(0, index - 1), min(len(bed) - 1, index + 1)
        slope = (bed[left] - bed[right]) / max(distance[right] - distance[left], 1.0)
        result[index] = np.clip(slope, min_slope, max_slope)
    return result


def _smoothed_profile(values: np.ndarray, distance: np.ndarray, window_m: float) -> np.ndarray:
    """Robust longitudinal profile: a moving median over valid stations."""
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    for index, centre in enumerate(distance):
        within = valid & (np.abs(distance - centre) <= 0.5 * window_m)
        if within.any():
            result[index] = np.median(values[within])
    return result


def _bank_from_lateral_terrain(left: np.ndarray, right: np.ndarray, bank_cells: int) -> float:
    """Use robust near-bank medians; the lower bank controls first overbank flow."""
    left_bank = np.nanmedian(left[:bank_cells])
    right_bank = np.nanmedian(right[:bank_cells])
    return float(min(left_bank, right_bank))


def _mask_geometry(mask: np.ndarray, transform: Any):
    """Polygonise only the reachable raster corridor exported to the mesh."""
    parts = [shape(geometry) for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform) if value]
    return unary_union(parts) if parts else Polygon()


def _conditioned_floodplain_receivers(
    terrain: np.ndarray, active: np.ndarray, river: np.ndarray, river_bed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill each connected floodplain minimally toward its mapped river sinks."""
    rows, cols = terrain.shape
    potential = np.full(terrain.shape, np.nan, dtype=np.float64)
    receiver = np.full(terrain.shape, -1, dtype=np.int64)
    queue: list[tuple[float, int]] = []
    for index in np.flatnonzero(active & river):
        row, col = divmod(int(index), cols)
        elevation = river_bed[row, col] if np.isfinite(river_bed[row, col]) else terrain[row, col]
        potential[row, col] = elevation
        heapq.heappush(queue, (float(elevation), int(index)))
    epsilon = 1e-4
    while queue:
        elevation, index = heapq.heappop(queue)
        row, col = divmod(index, cols)
        if elevation != potential[row, col]:
            continue
        for drow, dcol in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr, nc = row + drow, col + dcol
            if not (0 <= nr < rows and 0 <= nc < cols) or not active[nr, nc] or np.isfinite(potential[nr, nc]):
                continue
            filled = max(float(terrain[nr, nc]), elevation + epsilon)
            potential[nr, nc] = filled
            receiver[nr, nc] = index
            heapq.heappush(queue, (filled, nr * cols + nc))
    return potential, receiver


def _receiver_to_d4_direction(receiver: np.ndarray) -> np.ndarray:
    """Inverse of HydroBathyDEM's 1=N, 2=E, 3=S, 4=W convention."""
    rows, cols = receiver.shape
    index = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    delta = receiver - index
    result = np.zeros(receiver.shape, dtype=np.uint8)
    linked = receiver >= 0
    result[linked & (delta == -cols)] = 1
    result[linked & (delta == 1)] = 2
    result[linked & (delta == cols)] = 3
    result[linked & (delta == -1)] = 4
    return result


def _floodplain_flow_accumulation(receiver: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Area-free local accumulation used only to identify floodplain axes."""
    flat_receiver, flat_active = receiver.ravel(), active.ravel()
    accumulation = flat_active.astype(np.float64)
    indegree = np.zeros(flat_active.size, dtype=np.int32)
    valid_link = flat_active & (flat_receiver >= 0)
    indegree += np.bincount(flat_receiver[valid_link], minlength=flat_active.size).astype(np.int32)
    queue = list(np.flatnonzero(flat_active & (indegree == 0)))
    while queue:
        index = int(queue.pop())
        target = int(flat_receiver[index])
        if target >= 0:
            accumulation[target] += accumulation[index]
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return accumulation.reshape(receiver.shape)


def build_design_corridor(config: DesignCorridorConfig) -> dict[str, Any]:
    with rasterio.open(config.dem) as source:
        terrain = source.read(1, masked=True).astype(np.float64).filled(np.nan)
        profile = source.profile.copy()
        fallback_terrain = _aligned(config.fallback_dem, source, "Fallback DEM") if config.fallback_dem else None
        river = _aligned(config.river_mask, source, "River mask") > 0
        direction = _aligned(config.river_direction, source, "flow direction")
        width = _aligned(config.river_width, source, "River width")
        depth = _aligned(config.river_depth, source, "River depth")
        bed = _aligned(config.river_bed, source, "River bed")
        peak = _aligned(config.design_peak, source, "Design peak discharge")
    terrain_fallback = np.zeros(terrain.shape, dtype=bool)
    if fallback_terrain is not None:
        terrain_fallback = ~np.isfinite(terrain) & np.isfinite(fallback_terrain)
        terrain[terrain_fallback] = fallback_terrain[terrain_fallback]
    direction, direction_fallback_cells = _fill_terminal_river_directions(
        direction, river, width, config.routing_scheme,
    )
    valid = np.isfinite(terrain)
    invalid_inputs = {
        "terrain": river & ~valid,
        "design_peak": river & (~np.isfinite(peak) | (peak < 0)),
        "river_width": river & (~np.isfinite(width) | (width <= 0)),
        "river_depth": river & (~np.isfinite(depth) | (depth <= 0)),
        "river_bed": river & ~np.isfinite(bed),
        "flow_direction": river & ~np.isfinite(direction),
    }
    failed = {name: int(mask.sum()) for name, mask in invalid_inputs.items() if mask.any()}
    if failed:
        details = ", ".join(f"{name}={count:,}" for name, count in failed.items())
        raise ValueError(f"Cannot evaluate every mapped river cell; invalid inputs: {details}.")
    receiver = (
        receiver_from_d8_direction(direction)
        if config.routing_scheme == "d8"
        else receiver_from_d4_direction(direction)
    )
    dx_m = float(abs(profile["transform"].a))
    dy_m = float(abs(profile["transform"].e))
    reach_cells = max(1, int(np.ceil(config.max_half_width_m / min(dx_m, dy_m))))
    # ``peak`` remains the raw SCS-HUT result.  Only the resolution envelope
    # is propagated through confluences; this avoids presenting it as a routed
    # hydrograph or adding non-coincident peaks.
    refinement_peak = _refinement_flow_envelope(peak, receiver, river)
    stage = np.full(terrain.shape, np.nan, dtype=np.float64)
    half_width = np.full(terrain.shape, np.nan, dtype=np.float64)
    left_half_width = np.full(terrain.shape, np.nan, dtype=np.float64)
    right_half_width = np.full(terrain.shape, np.nan, dtype=np.float64)
    lateral_search_cap = np.zeros(terrain.shape, dtype=np.uint8)  # 1=left, 2=right
    status = np.zeros(terrain.shape, dtype=np.uint8)  # 0=not evaluated, 1=solved, 2=capacity cap
    floodplain = np.zeros(terrain.shape, dtype=bool)
    bank_mismatch: list[float] = []
    for path in _river_reaches(receiver, river):
        distance, samples = _sample_positions(
            path, terrain.shape, dx_m, dy_m, config.cross_section_spacing_m,
        )
        sample_stage = np.full(len(samples), np.nan)
        sample_bank = np.full(len(samples), np.nan)
        sample_depth = np.full(len(samples), np.nan)
        for sample_index, path_index in enumerate(samples):
            row, col = divmod(path[path_index], terrain.shape[1])
            cross_width = _cross_section_cell_width(
                int(direction[row, col]), dx_m, dy_m, config.routing_scheme,
            )
            bank_cells = max(1, int(np.ceil(config.bank_sampling_width_m / cross_width)))
            first_offset = max(1, int(np.ceil(0.5 * width[row, col] / cross_width)))
            left, right = _cross_section_indices(
                row, col, int(direction[row, col]), first_offset, reach_cells,
                terrain.shape, config.routing_scheme,
            )
            if not len(left) or not len(right): continue
            left_ground, right_ground = terrain[left[:, 0], left[:, 1]], terrain[right[:, 0], right[:, 1]]
            if not (np.isfinite(left_ground).all() and np.isfinite(right_ground).all()): continue
            # The raw lateral terrain is the bank datum.  Use a median on each
            # margin rather than a single first pixel; the lower robust bank
            # still controls first overbank flow.
            bank = _bank_from_lateral_terrain(left_ground, right_ground, bank_cells)
            sample_bank[sample_index] = bank
            sample_depth[sample_index] = depth[row, col]
            bank_mismatch.append((bed[row, col] + depth[row, col]) - bank)
        good = np.isfinite(sample_bank) & np.isfinite(sample_depth)
        if good.sum() < 2: continue
        smooth_bank = _smoothed_profile(sample_bank, distance[samples], config.bank_smoothing_window_m)
        good_bank = np.isfinite(smooth_bank) & np.isfinite(sample_depth)
        if good_bank.sum() < 2: continue
        interpolated_bank = np.interp(distance, distance[samples][good_bank], smooth_bank[good_bank])
        interpolated_depth = np.interp(distance, distance[samples][good_bank], sample_depth[good_bank])
        interpolated_bed = interpolated_bank - interpolated_depth
        slope = _smoothed_slopes(interpolated_bed, distance, config.slope_window_m, config.minimum_slope, config.maximum_slope)
        for sample_index, path_index in enumerate(samples):
            if not good_bank[sample_index]: continue
            row, col = divmod(path[path_index], terrain.shape[1])
            cross_width = _cross_section_cell_width(
                int(direction[row, col]), dx_m, dy_m, config.routing_scheme,
            )
            first_offset = max(1, int(np.ceil(0.5 * width[row, col] / cross_width)))
            left, right = _cross_section_indices(
                row, col, int(direction[row, col]), first_offset, reach_cells,
                terrain.shape, config.routing_scheme,
            )
            left_ground, right_ground = terrain[left[:, 0], left[:, 1]], terrain[right[:, 0], right[:, 1]]
            water_level, solved = _solve_stage(
                float(refinement_peak[row, col]), float(interpolated_bed[path_index]), float(interpolated_depth[path_index]), float(width[row, col]), float(slope[path_index]),
                left_ground, right_ground, cross_width, config.channel_manning_n, config.floodplain_manning_n, config.max_stage_above_bank_m,
            )
            if solved:
                sample_stage[sample_index] = water_level
            else: status[row, col] = 2
        solved = np.isfinite(sample_stage)
        if solved.sum() < 2: continue
        smooth_stage = np.interp(distance, distance[samples][solved], sample_stage[solved])
        for path_index, node in enumerate(path):
            row, col = divmod(node, terrain.shape[1])
            cross_width = _cross_section_cell_width(
                int(direction[row, col]), dx_m, dy_m, config.routing_scheme,
            )
            first_offset = max(1, int(np.ceil(0.5 * width[row, col] / cross_width)))
            left, right = _cross_section_indices(
                row, col, int(direction[row, col]), first_offset, reach_cells,
                terrain.shape, config.routing_scheme,
            )
            if not len(left) or not len(right): continue
            left_ground, right_ground = terrain[left[:, 0], left[:, 1]], terrain[right[:, 0], right[:, 1]]
            water_level = smooth_stage[path_index]
            wet_left = _connected_side_depths(left_ground, water_level) > 0
            wet_right = _connected_side_depths(right_ground, water_level) > 0
            if wet_left.any():
                floodplain[left[wet_left, 0], left[wet_left, 1]] = True
            if wet_right.any():
                floodplain[right[wet_right, 0], right[wet_right, 1]] = True
            left_half_width[row, col] = wet_left.sum() * cross_width + 0.5 * width[row, col]
            right_half_width[row, col] = wet_right.sum() * cross_width + 0.5 * width[row, col]
            half_width[row, col] = max(left_half_width[row, col], right_half_width[row, col])
            if len(left) == reach_cells and wet_left[-1]:
                lateral_search_cap[row, col] |= 1
            if len(right) == reach_cells and wet_right[-1]:
                lateral_search_cap[row, col] |= 2
            stage[row, col] = water_level; status[row, col] = 1
    # Junction-to-junction smoothing needs at least two usable cross-sections.
    # Very short reaches and boundary-clipped diagonals may not satisfy that
    # requirement.  Evaluate those cells locally instead of omitting them.
    local_fallback_cells = 0
    for row, col in np.argwhere(river & (status == 0)):
        direction_code = int(direction[row, col])
        cross_width = _cross_section_cell_width(
            direction_code, dx_m, dy_m, config.routing_scheme,
        )
        first_offset = max(1, int(np.ceil(0.5 * width[row, col] / cross_width)))
        left, right = _cross_section_indices(
            int(row), int(col), direction_code, first_offset, reach_cells,
            terrain.shape, config.routing_scheme,
        )

        def finite_prefix(indices: np.ndarray) -> np.ndarray:
            values = terrain[indices[:, 0], indices[:, 1]] if len(indices) else np.empty(0)
            invalid = np.flatnonzero(~np.isfinite(values))
            if len(invalid):
                values = values[: invalid[0]]
            return values if len(values) else np.asarray([np.inf])

        left_ground, right_ground = finite_prefix(left), finite_prefix(right)
        node = int(row * terrain.shape[1] + col)
        target = int(receiver[row, col])
        local_slope = config.minimum_slope
        if target >= 0 and river.ravel()[target]:
            target_row, target_col = divmod(target, terrain.shape[1])
            distance = np.hypot((row - target_row) * dy_m, (col - target_col) * dx_m)
            local_slope = (bed[row, col] - bed[target_row, target_col]) / max(distance, 1.0)
        else:
            upstream = np.flatnonzero(river.ravel() & (receiver.ravel() == node))
            if len(upstream):
                slopes = []
                for source in upstream:
                    source_row, source_col = divmod(int(source), terrain.shape[1])
                    distance = np.hypot((row - source_row) * dy_m, (col - source_col) * dx_m)
                    slopes.append((bed[source_row, source_col] - bed[row, col]) / max(distance, 1.0))
                local_slope = float(np.nanmedian(slopes))
        local_slope = float(np.clip(local_slope, config.minimum_slope, config.maximum_slope))
        water_level, solved = _solve_stage(
            float(refinement_peak[row, col]), float(bed[row, col]), float(depth[row, col]),
            float(width[row, col]), local_slope, left_ground, right_ground, cross_width,
            config.channel_manning_n, config.floodplain_manning_n,
            config.max_stage_above_bank_m,
        )
        if not solved:
            status[row, col] = 2
            continue
        wet_left = _connected_side_depths(left_ground, water_level) > 0
        wet_right = _connected_side_depths(right_ground, water_level) > 0
        if len(left) and wet_left.any():
            floodplain[left[:len(wet_left)][wet_left, 0], left[:len(wet_left)][wet_left, 1]] = True
        if len(right) and wet_right.any():
            floodplain[right[:len(wet_right)][wet_right, 0], right[:len(wet_right)][wet_right, 1]] = True
        left_half_width[row, col] = max(0, int(wet_left.sum())) * cross_width + 0.5 * width[row, col]
        right_half_width[row, col] = max(0, int(wet_right.sum())) * cross_width + 0.5 * width[row, col]
        half_width[row, col] = max(left_half_width[row, col], right_half_width[row, col])
        stage[row, col] = water_level
        status[row, col] = 1
        local_fallback_cells += 1
    unresolved_river = river & (status != 1)
    if unresolved_river.any():
        unevaluated = int((river & (status == 0)).sum())
        capacity_capped = int((river & (status == 2)).sum())
        raise ValueError(
            "Hydraulic corridor did not solve every mapped river cell: "
            f"unevaluated={unevaluated:,}, capacity_capped={capacity_capped:,}."
        )
    floodplain &= valid & ~river
    # A design corridor is usable for mesh refinement only when every selected
    # floodplain cell reaches the same river network used by the hydraulic
    # cross-sections.  This removes detached geometry-repair artefacts.
    hydraulic_domain = floodplain | river
    hydraulic_potential, floodplain_receiver = _conditioned_floodplain_receivers(terrain, hydraulic_domain, river, bed)
    disconnected = floodplain & ~np.isfinite(hydraulic_potential)
    disconnected_cells = int(disconnected.sum())
    floodplain &= np.isfinite(hydraulic_potential)
    corridor_geometry = _mask_geometry(floodplain, profile["transform"])
    hydraulic_domain = floodplain | river
    hydraulic_potential, floodplain_receiver = _conditioned_floodplain_receivers(terrain, hydraulic_domain, river, bed)
    floodplain_direction = _receiver_to_d4_direction(floodplain_receiver)
    floodplain_accumulation = _floodplain_flow_accumulation(floodplain_receiver, hydraulic_domain)
    cell_area_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
    floodplain_axis = floodplain & (floodplain_accumulation * cell_area_km2 >= config.floodplain_axis_min_upstream_area_km2)
    prefix = config.routing_scheme.upper()
    outputs = {
        f"{prefix}_mesh_refinement_Qp_envelope_100yr_m3s.tif": refinement_peak,
        f"{prefix}_design_water_level_100yr_m.tif": stage,
        f"{prefix}_design_overbank_half_width_100yr_m.tif": half_width,
        f"{prefix}_design_left_half_width_100yr_m.tif": left_half_width,
        f"{prefix}_design_right_half_width_100yr_m.tif": right_half_width,
        f"{prefix}_design_lateral_search_cap.tif": lateral_search_cap,
        f"{prefix}_design_floodplain_mask_100yr.tif": floodplain.astype(np.uint8),
        f"{prefix}_design_corridor_status.tif": status,
        f"{prefix}_design_floodplain_hydraulic_potential_m.tif": hydraulic_potential,
        f"{prefix}_design_floodplain_flow_direction.tif": floodplain_direction,
        f"{prefix}_design_floodplain_flow_accumulation_km2.tif": floodplain_accumulation * cell_area_km2,
        f"{prefix}_design_floodplain_axis_mask.tif": floodplain_axis.astype(np.uint8),
    }
    config.out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, data in outputs.items():
        path = config.out_dir / name
        out_profile = profile.copy(); out_profile.update(driver="GTiff", count=1, dtype=str(data.dtype), nodata=0 if data.dtype == np.uint8 else np.nan, compress="deflate")
        with rasterio.open(path, "w", **out_profile) as target:
            target.write(data, 1)
        paths[name] = str(path)
    vector_path = config.out_dir / f"{prefix}_design_floodplain_corridor_100yr.gpkg"
    gpd.GeoDataFrame({"feature": ["design_flow_corridor"]}, geometry=[corridor_geometry], crs=profile["crs"]).to_file(vector_path, layer="corridor", driver="GPKG")
    paths[vector_path.name] = str(vector_path)
    left_overbank = river & (left_half_width > 0.5 * width + 1e-6)
    right_overbank = river & (right_half_width > 0.5 * width + 1e-6)
    one_sided_overbank = left_overbank ^ right_overbank
    left_search_cap = river & ((lateral_search_cap & 1) > 0)
    right_search_cap = river & ((lateral_search_cap & 2) > 0)
    report = {
        "method": "cross-section compound Manning screening from SCS-HUT Qp; direct raster wetting at every river cell",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "mapped_river_cells": int(river.sum()),
        "terrain_fallback_cells": int(terrain_fallback.sum()),
        "river_terrain_fallback_cells": int((river & terrain_fallback).sum()),
        "terminal_direction_fallback_cells": direction_fallback_cells,
        "local_cross_section_fallback_cells": local_fallback_cells,
        "river_cells_evaluated": int(river.sum()), "stages_solved": int((status == 1).sum()),
        "capacity_capped": int((status == 2).sum()), "floodplain_cells": int(floodplain.sum()),
        "river_cells_with_left_overbank": int(left_overbank.sum()),
        "river_cells_with_right_overbank": int(right_overbank.sum()),
        "river_cells_with_two_sided_overbank": int((left_overbank & right_overbank).sum()),
        "river_cells_with_one_sided_overbank": int(one_sided_overbank.sum()),
        "river_cells_reaching_left_search_cap": int(left_search_cap.sum()),
        "river_cells_reaching_right_search_cap": int(right_search_cap.sum()),
        "river_cells_reaching_either_search_cap": int((left_search_cap | right_search_cap).sum()),
        "disconnected_floodplain_cells_removed": disconnected_cells,
        "disconnected_floodplain_area_km2_removed": disconnected_cells * abs(profile["transform"].a * profile["transform"].e) / 1e6,
        "overbank_half_width_m_quantiles": [float(np.nanquantile(half_width[status == 1], q)) for q in (0, .5, .95, 1)] if (status == 1).any() else [],
        "conditioned_minus_raw_bank_m_quantiles": [float(np.nanquantile(bank_mismatch, q)) for q in (0, .05, .5, .95, 1)] if bank_mismatch else [],
        "outputs": paths, "scope": "Mesh-refinement corridor only; not an inundation prediction.",
    }
    report_path = config.out_dir / "design_corridor_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "report": str(report_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a terrain-connected design-flow corridor for adaptive mesh refinement.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_design_corridor(DesignCorridorConfig.from_mapping(load_config_file(args.config))), indent=2))


if __name__ == "__main__":
    main()
