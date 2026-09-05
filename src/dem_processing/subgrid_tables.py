"""HEC-RAS style sub-grid hydraulic property tables.

Two tables, both built once from terrain finer than the mesh:

* per CELL, an elevation-volume curve. The wetted plan area A_s(zeta) is
  piecewise CONSTANT in the sorted patch elevations, so its integral V(zeta) is
  piecewise LINEAR and convex. Storing (zeta, V, A) with the exact recurrence
  V_{j+1} = V_j + A_j*(zeta_{j+1}-zeta_j) makes A the exact derivative of the
  stored V, and that identity is what lets stage be recovered from volume by one
  binary search and one division -- no Newton, no tolerance, no iteration count.

* per FACE, elevation vs flow area A(z), wetted perimeter P(z) and conveyance
  K(z) = sum_i (A_i/n_i) * (A_i/P_i)^(2/3) over roughness-homogeneous panels.
  K is what the momentum equation needs: the hflow^(7/3)/n^2 friction term is
  exactly K^2/A.

Two details that are not optional, both established by measurement:

1. Tables are stored in a CELL-LOCAL coordinate zeta = z - z_min. Keeping
   absolute elevations degrades a 3 cm depth on a 700 m bed to 1.6e-12 relative
   through catastrophic cancellation; local storage holds 4e-16.
2. ``minimum_area_fraction`` is a SAFETY GUARD, not a nicety, and is applied to
   A_s BEFORE integrating. On the Pune mesh 139 cells have a lowest patch under
   1e-6 m2 (smallest 9.2e-10 m2). Without the floor, one nanolitre of residual
   volume puts the free surface a metre above the invert, and stage drives both
   the water-surface slope and the face area. Flooring after integrating instead
   breaks the derivative identity and loses mass.

Reference: HEC-RAS Hydraulic Reference Manual v6.6 pp.89-92 ("Subgrid
Bathymetry", after Casulli 2009, doi 10.1002/fld.1896); tolerance defaults from
the 2D User's Manual v6.6 p.55. Note that HEC-RAS through v6.x uses a single
Manning's n per face; the roughness-varying K(z) here is a deliberate extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FT = 0.3048
CELL_ELEV_VOL_FILTER_M = 0.01 * FT      # HEC-RAS default 0.01 ft
FACE_AREA_ELEV_FILTER_M = 0.01 * FT     # HEC-RAS default 0.01 ft
FACE_CONVEYANCE_TOL = 0.02              # HEC-RAS default 2%
MINIMUM_AREA_FRACTION = 1.0e-2          # HEC-RAS default (2D Manual: "Cell Minimum
                                        # Area Fraction", default 0.01 = 1%). An
                                        # earlier 1e-3 here was a deliberate deviation
                                        # and it was wrong: measured on a 60 m tilted-
                                        # plane cell holding 0.9 mm of rain it raised
                                        # the level-pool stage 1.3x and the face
                                        # conveyance 2.9x.  Do not lower it again
                                        # without re-running grid convergence.


@dataclass(frozen=True)
class CellVolumeTable:
    """Ragged elevation-volume curves, one row per cell, in local coordinates."""

    datum_m: np.ndarray          # (n_cells,) z_ref per cell
    zeta_m: np.ndarray           # (n_cells, max_pts)
    volume_m3: np.ndarray        # (n_cells, max_pts)
    area_m2: np.ndarray          # (n_cells, max_pts) dV/dzeta on each segment
    count: np.ndarray            # (n_cells,) valid points
    plan_area_m2: np.ndarray     # (n_cells,) full cell plan area

    def stage(self, volume_m3: np.ndarray) -> np.ndarray:
        """Water surface elevation from stored volume. Exact, non-iterative."""
        volume = np.maximum(np.asarray(volume_m3, dtype=np.float64), 0.0)
        rows = np.arange(len(volume))
        last = self.count - 1
        # locate the segment: rightmost j with V_j <= volume
        j = np.array([
            int(np.searchsorted(self.volume_m3[i, : self.count[i]], volume[i], side="right")) - 1
            for i in rows
        ])
        j = np.clip(j, 0, last)
        base_v = self.volume_m3[rows, j]
        base_z = self.zeta_m[rows, j]
        slope = self.area_m2[rows, j]
        # above the top breakpoint the curve continues at the full plan area
        top = j >= last
        slope = np.where(top | (slope <= 0.0), self.plan_area_m2, slope)
        return self.datum_m + base_z + (volume - base_v) / np.maximum(slope, 1e-30)

    def volume(self, stage_m: np.ndarray) -> np.ndarray:
        """Volume from water surface elevation. The exact inverse of ``stage``."""
        zeta = np.asarray(stage_m, dtype=np.float64) - self.datum_m
        rows = np.arange(len(zeta))
        last = self.count - 1
        j = np.array([
            int(np.searchsorted(self.zeta_m[i, : self.count[i]], zeta[i], side="right")) - 1
            for i in rows
        ])
        j = np.clip(j, 0, last)
        slope = self.area_m2[rows, j]
        top = j >= last
        slope = np.where(top | (slope <= 0.0), self.plan_area_m2, slope)
        return np.maximum(
            self.volume_m3[rows, j] + (zeta - self.zeta_m[rows, j]) * slope, 0.0
        )


def _fit_bed_planes(
    patch_cell: np.ndarray,
    patch_area_m2: np.ndarray,
    patch_elevation_m: np.ndarray,
    patch_x: np.ndarray,
    patch_y: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Area-weighted least-squares plane through each cell's terrain.

    Returned per patch, so the caller can take ``z - plane`` directly. Centring on
    the area-weighted centroid decouples the constant term from the slopes, which
    both conditions the 2x2 solve and makes the plane's value at the centroid
    exactly the area-weighted mean elevation -- the natural datum for the cell.
    """
    w = np.asarray(patch_area_m2, dtype=np.float64)
    c = np.asarray(patch_cell, dtype=np.int64)
    x = np.asarray(patch_x, dtype=np.float64)
    y = np.asarray(patch_y, dtype=np.float64)
    z = np.asarray(patch_elevation_m, dtype=np.float64)
    good = np.isfinite(z) & (w > 0)
    w = np.where(good, w, 0.0)
    z = np.where(good, z, 0.0)

    def S(v: np.ndarray) -> np.ndarray:
        return np.bincount(c, weights=w * v, minlength=n_cells)

    sw = np.maximum(np.bincount(c, weights=w, minlength=n_cells), 1e-30)
    xc, yc, zc = S(x) / sw, S(y) / sw, S(z) / sw
    dx, dy, dz = x - xc[c], y - yc[c], z - zc[c]
    m11, m12, m22 = S(dx * dx), S(dx * dy), S(dy * dy)
    r1, r2 = S(dx * dz), S(dy * dz)
    det = m11 * m22 - m12 * m12
    # a degenerate footprint (one patch, or a line of patches) has no unique plane;
    # fall back to a horizontal plane at the mean elevation rather than dividing by ~0
    flat = np.abs(det) < 1e-12 * np.maximum(m11 * m22, 1e-30)
    det = np.where(flat, 1.0, det)
    slope_x = np.where(flat, 0.0, (r1 * m22 - r2 * m12) / det)
    slope_y = np.where(flat, 0.0, (r2 * m11 - r1 * m12) / det)
    return zc[c] + slope_x[c] * dx + slope_y[c] * dy


def build_cell_volume_table(
    patch_cell: np.ndarray,
    patch_area_m2: np.ndarray,
    patch_elevation_m: np.ndarray,
    n_cells: int,
    plan_area_m2: np.ndarray,
    filter_m: float = CELL_ELEV_VOL_FILTER_M,
    minimum_area_fraction: float = MINIMUM_AREA_FRACTION,
    datum_mode: str = "horizontal",
    patch_x: np.ndarray | None = None,
    patch_y: np.ndarray | None = None,
    level_pool_mask: np.ndarray | None = None,
    cell_bed_mean_m: np.ndarray | None = None,
) -> CellVolumeTable:
    """Build one elevation-volume curve per cell from exact overlap patches.

    ``patch_*`` are the conservative overlap pairs: which cell, how much area,
    and the terrain elevation of that piece.

    ``datum_mode`` selects what the water surface inside a cell is assumed to be:

    ``"horizontal"``
        A level pool, as HEC-RAS assumes. Correct for floodplains and reservoirs,
        where the cell is flat compared with the water depth.

    ``"bed_plane"`` -- EXPERIMENTAL, NOT WIRED TO THE SOLVER
        Depth above an area-weighted least-squares plane fitted to the cell's own
        terrain, for steep ground where level pool fails. Measured on the
        V-catchment benchmark, a 60 m cell spans 3.85 m of terrain while the water
        is ~0.2 m deep, so a horizontal surface holding the right volume pools in
        the downstream corner and reads 0.52 m where the truth is 0.21 m -- a 148%
        over-prediction. In isolation this mode fixes that: on a synthetic sloping
        cell it recovers the wet area exactly (1800 m2 against a truth of 1800 m2,
        where horizontal mode gives 1100 m2).

        It is NOT usable yet, and the reason is measured, not suspected. The face
        conveyance table is indexed by ABSOLUTE water surface elevation and its
        datum is the lowest terrain point on the face. This mode's datum is instead
        the cell's mean bed, which on a 5% slope sits well above that, so the face
        lookup reads a large flow area with no water present. The V-catchment run
        showed it: the timestep collapsed from 2.51 s to 0.19 s, 13x, with the
        outflow unchanged. Making it work needs the fitted plane evaluated at each
        face midpoint so the cell and face tables share one vertical reference --
        that is an extension past HEC-RAS, not parity with it.

        Requires ``patch_x``/``patch_y``.

    In both modes ``datum_m`` is the elevation that ``stage()`` adds its result to,
    so the returned stage is always a true water surface elevation at the cell
    centroid, and nothing downstream needs to know which mode was used.
    """
    """Gating note -- ``level_pool_mask``.

    A level-pool table is only valid where the water surface really is near
    horizontal inside the cell. Where it is not, the table is actively harmful,
    and the size of the harm is measured: for 0.9 mm of rain on a 60 m cell of the
    V-catchment, the level-pool stage is a 64.8 mm puddle over 1.39% of the cell
    against a true 0.9 mm sheet over 100% -- a 72x depth ratio, hence 17x the
    Manning velocity. On the published Di Giammarco benchmark that makes the
    coarse hydrograph reach equilibrium in 883 s against the reference 1743 s.

    Pass ``level_pool_mask=False`` for such cells (with ``cell_bed_mean_m``) and
    they get a DEGENERATE one-point table: datum at the area-mean bed, slope the
    plan area. ``stage()`` then returns exactly ``bed_mean + volume/plan_area`` --
    the flat-prism answer, bit for bit. So the gate lives entirely in the table and
    the solver needs no branch: one file decides, per cell, which closure applies.
    """
    if datum_mode not in {"horizontal", "bed_plane"}:
        raise ValueError(f"Unknown datum_mode {datum_mode!r}.")
    if datum_mode == "bed_plane" and (patch_x is None or patch_y is None):
        raise ValueError("datum_mode='bed_plane' needs patch_x and patch_y.")
    reference = np.zeros_like(patch_elevation_m)
    if datum_mode == "bed_plane":
        reference = _fit_bed_planes(
            patch_cell, patch_area_m2, patch_elevation_m, patch_x, patch_y, n_cells
        )
    residual_all = patch_elevation_m - reference
    # sort by the residual, not by raw elevation: in bed_plane mode the curve is
    # built in residual space and the two orders differ on sloping ground
    order = np.lexsort((residual_all, patch_cell))
    cell = patch_cell[order]
    area = patch_area_m2[order]
    elev = residual_all[order]
    ref_sorted = reference[order]
    starts = np.r_[0, np.flatnonzero(cell[1:] != cell[:-1]) + 1, len(cell)]

    rows: list[np.ndarray] = []
    for k in range(len(starts) - 1):
        lo, hi = starts[k], starts[k + 1]
        rows.append(np.arange(lo, hi))
    owner = cell[starts[:-1]]

    curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    datum = np.zeros(n_cells)
    counts = np.ones(n_cells, dtype=np.int64)
    blank = (np.zeros(2), np.zeros(2), np.zeros(2))
    table: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [blank] * n_cells

    for idx, m in zip(rows, owner, strict=True):
        z = elev[idx]
        a = area[idx]
        finite = np.isfinite(z) & (a > 0)
        if not finite.any():
            continue
        a_all = area[idx]
        z, a = z[finite], a[finite]
        if level_pool_mask is not None and not bool(level_pool_mask[m]):
            # flat-prism closure for this cell, expressed as a table so the solver
            # stays branch-free.  One point: V=0 at the mean bed, slope = plan area.
            if cell_bed_mean_m is None:
                raise ValueError("level_pool_mask requires cell_bed_mean_m.")
            datum[m] = float(cell_bed_mean_m[m])
            counts[m] = 1
            table[m] = (np.zeros(1), np.zeros(1), np.array([float(plan_area_m2[m])]))
            continue
        if datum_mode == "bed_plane":
            # datum is the fitted plane at the cell's area-weighted centroid, so the
            # residual is already measured from it and zeta may be negative (the
            # channel sits below the plane).  Do NOT re-zero on the minimum here:
            # that would throw away the fact that the low ground is below the datum.
            z_ref = float(np.average(ref_sorted[idx][finite], weights=a))
            zeta = z
        else:
            z_ref = float(z[0])
            zeta = z - z_ref
        # merge patches at (numerically) the same elevation
        keep = np.r_[True, np.diff(zeta) > 1e-12]
        group = np.cumsum(keep) - 1
        zeta_u = zeta[keep]
        area_u = np.bincount(group, weights=a)
        # cumulative wetted plan area: piecewise CONSTANT
        wet = np.cumsum(area_u)
        # SAFETY FLOOR, applied to the AREA before integrating
        floor = minimum_area_fraction * float(plan_area_m2[m])
        wet = np.maximum(wet, floor)
        # exact integral -> piecewise LINEAR volume
        vol = np.zeros(len(zeta_u))
        if len(zeta_u) > 1:
            vol[1:] = np.cumsum(wet[:-1] * np.diff(zeta_u))
        zeta_u, vol, wet = _thin_volume_curve(zeta_u, vol, wet, filter_m)
        datum[m] = z_ref
        counts[m] = len(zeta_u)
        table[m] = (zeta_u, vol, wet)

    width = max(2, int(counts.max()))
    Z = np.zeros((n_cells, width)); V = np.zeros((n_cells, width)); A = np.zeros((n_cells, width))
    for m in range(n_cells):
        z, v, a = table[m]
        k = len(z)
        Z[m, :k], V[m, :k], A[m, :k] = z, v, a
        if k < width:
            Z[m, k:] = z[-1] if k else 0.0
            V[m, k:] = v[-1] if k else 0.0
            A[m, k:] = a[-1] if k else plan_area_m2[m]
    return CellVolumeTable(datum, Z, V, A, counts, np.asarray(plan_area_m2, dtype=np.float64))


def _thin_volume_curve(
    zeta: np.ndarray, volume: np.ndarray, area: np.ndarray, filter_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop breakpoints whose removal moves the curve less than ``filter_m``.

    Thinning happens in ELEVATION, and the retained points keep the exact
    recurrence, so the derivative identity survives and mass is still conserved.
    """
    if filter_m <= 0 or len(zeta) < 3:
        return zeta, volume, area
    keep = [0]
    for j in range(1, len(zeta) - 1):
        if zeta[j] - zeta[keep[-1]] >= filter_m:
            keep.append(j)
    keep.append(len(zeta) - 1)
    keep = np.asarray(keep)
    z = zeta[keep]
    a = area[keep]
    v = np.zeros(len(z))
    if len(z) > 1:
        v[1:] = np.cumsum(a[:-1] * np.diff(z))
    return z, v, a


@dataclass(frozen=True)
class FaceConveyanceTable:
    """Elevation vs flow area, wetted perimeter and conveyance, one row per face."""

    datum_m: np.ndarray          # (n_faces,) lowest terrain point on the face
    zeta_m: np.ndarray           # (n_faces, max_pts)
    area_m2: np.ndarray          # (n_faces, max_pts) flow area through the face
    perimeter_m: np.ndarray      # (n_faces, max_pts) wetted perimeter
    conveyance: np.ndarray       # (n_faces, max_pts) K = sum (A_i/n_i) R_i^(2/3)
    count: np.ndarray            # (n_faces,)
    length_m: np.ndarray         # (n_faces,) full face length

    def _interp(self, table: np.ndarray, stage_m: np.ndarray) -> np.ndarray:
        zeta = np.asarray(stage_m, dtype=np.float64) - self.datum_m
        out = np.zeros(len(zeta))
        for i in range(len(zeta)):
            k = self.count[i]
            out[i] = np.interp(zeta[i], self.zeta_m[i, :k], table[i, :k],
                               left=0.0, right=table[i, k - 1])
        return out

    def flow_area(self, stage_m: np.ndarray) -> np.ndarray:
        return self._interp(self.area_m2, stage_m)

    def wetted_perimeter(self, stage_m: np.ndarray) -> np.ndarray:
        """Piecewise CONSTANT -- a panel is wet or dry, so P steps, it does not ramp."""
        zeta = np.asarray(stage_m, dtype=np.float64) - self.datum_m
        out = np.zeros(len(zeta))
        for i in range(len(zeta)):
            k = self.count[i]
            j = int(np.searchsorted(self.zeta_m[i, :k], zeta[i], side="right")) - 1
            out[i] = self.perimeter_m[i, max(j, 0)] if zeta[i] > 0 else 0.0
        return out

    def conveyance_at(self, stage_m: np.ndarray) -> np.ndarray:
        return self._interp(self.conveyance, stage_m)


def build_face_conveyance_table(
    profiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    length_m: np.ndarray,
    filter_m: float = FACE_AREA_ELEV_FILTER_M,
    freeboard_m: float = 10.0,
    conveyance_tol: float = FACE_CONVEYANCE_TOL,
    max_points: int = 160,
    bottom_decades: int = 4,
    control_elevation_m: np.ndarray | None = None,
) -> FaceConveyanceTable:
    """Build one conveyance curve per face from its sampled terrain profile.

    ``profiles`` is per face: (station along the face, bed elevation, Manning n).
    Panels are the intervals between samples.

    Structure follows HEC-RAS: wetted WIDTH and PERIMETER are piecewise CONSTANT
    in elevation (a panel is either wet or dry), and flow AREA is their exact
    piecewise-linear integral. Interpolating perimeter linearly instead is wrong
    and measurably so -- it returned P = 12 m for a 20 m wide flat channel,
    because P jumps to its full value the moment any water is present.

    Conveyance K = sum_i (A_i/n_i)(A_i/P_i)^(2/3) over roughness-homogeneous
    panels grows as A^(5/3) within a segment, so it is NOT linear either. Rather
    than tabulate it densely everywhere, levels are bisected until linear
    interpolation of K is within ``conveyance_tol`` -- HEC-RAS's published Face
    Conveyance Tol Ratio, default 2%.

    HEC-RAS v6.x uses ONE Manning's n for a whole face; letting n vary along the
    face is a deliberate extension, so this is not HEC-RAS parity.

    ``control_elevation_m`` is REQUIRED for correctness, not optional tuning. The
    face profile is raised to at least this elevation before the curves are built,
    which makes the face a sill at the controlling invert. Pass
    ``max(cell_datum[owner], cell_datum[neighbour])``.

    Without it the tables are catastrophically wrong on sloping ground, and the
    failure is silent in every mass-balance check. A cell's water surface is
    referenced to the cell MINIMUM, so a dry cell reports a stage at its own low
    point -- which is well above the low point of the face it shares with its
    DOWNSLOPE neighbour. The face lookup therefore reads a wet face on dry ground:
    measured on the V-catchment, 439 of 445 internal faces reported flow depth with
    zero water anywhere, median 0.10 m, conveyance up to 62. The volume limiter
    still conserves mass to 1e-11 m3, so nothing looks wrong -- but every cell
    dumps its whole content downslope each step, and the outlet hydrograph reached
    full equilibrium discharge at t = 600 s on a catchment whose time of
    concentration is 5400 s.

    Clipping to the control elevation fixes it by construction: at zero volume the
    face stage equals ``max`` of the two cell datums, which equals the control
    elevation, so the flow area is exactly zero. On flat ground the clip is a no-op
    and the curves reduce to the flat-prism form, as they must.
    """
    n_faces = len(profiles)
    datum = np.zeros(n_faces)
    counts = np.ones(n_faces, dtype=np.int64)
    built: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    if control_elevation_m is not None:
        raise ValueError(
            "control_elevation_m is retired and must not be used. Clipping the face "
            "profile up to the controlling invert flattens the face into a weir: the "
            "whole cross-section becomes a sill and A = L * depth. Measured on a 60 m "
            "face across a 5% plane at a 0.115 m stage, it inflated flow area 34x and "
            "conveyance 57x over the true face geometry, which destroyed the routing. "
            "The dry-face problem it was meant to solve belongs in the solver, as a "
            "water-availability gate, not in the table."
        )
    for face_index, (station, bed, rough) in enumerate(profiles):
        if len(station) < 2 or not np.isfinite(bed).any():
            datum[len(built)] = float(np.nanmin(bed)) if np.isfinite(bed).any() else 0.0
            built.append((np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)))
            continue
        z_ref = float(np.nanmin(bed))
        seg_len = np.diff(station)
        seg_bed = 0.5 * (bed[:-1] + bed[1:]) - z_ref
        seg_wet = np.hypot(seg_len, np.diff(bed))       # wetted length along the bed
        seg_n = np.maximum(0.5 * (rough[:-1] + rough[1:]), 1e-6)

        def panels(zeta: float):
            depth = np.maximum(zeta - seg_bed, 0.0)
            return depth > 0.0, depth

        def area_of(zeta: float) -> float:
            wet, depth = panels(zeta)
            return float(np.sum(depth[wet] * seg_len[wet])) if wet.any() else 0.0

        def perim_of(zeta: float) -> float:
            wet, _ = panels(zeta)
            return float(np.sum(seg_wet[wet])) if wet.any() else 0.0

        def width_of(zeta: float) -> float:
            wet, _ = panels(zeta)
            return float(np.sum(seg_len[wet])) if wet.any() else 0.0

        def conv_of(zeta: float) -> float:
            wet, depth = panels(zeta)
            if not wet.any():
                return 0.0
            a = depth[wet] * seg_len[wet]
            p = np.maximum(seg_wet[wet], 1e-12)
            return float(np.sum(a / seg_n[wet] * (a / p) ** (2.0 / 3.0)))

        # breakpoints: every distinct panel bed, thinned, plus freeboard above
        breaks = np.unique(np.round(seg_bed, 9))
        if filter_m > 0 and len(breaks) > 2:
            keep = [breaks[0]]
            for v in breaks[1:-1]:
                if v - keep[-1] >= filter_m:
                    keep.append(v)
            keep.append(breaks[-1])
            breaks = np.asarray(keep)
        top = float(breaks[-1]) + max(freeboard_m, 10.0 * max(filter_m, 0.001))
        # Seed GEOMETRIC levels near the bottom before the tolerance pass.
        # Conveyance grows as h^(5/3), so linear interpolation across a first
        # interval of finite width over-estimates it badly and the error is worst
        # where the true curve is flattest.  Measured on a flat 20 m face with a
        # 0.039 m first interval: K was 12.4x too high at h = 0.9 mm, 3.9x at 5 mm,
        # 1.6x at 20 mm.  That is exactly the depth a rain-driven sheet lives at,
        # and it made the sub-grid arm over-convey at EVERY mesh resolution.
        # Uniform bisection cannot fix it -- the cap is reached first -- so the
        # bottom decades are seeded directly and the 2% pass refines from there.
        # A geometric ladder spanning the WHOLE range, not just the bottom: with a
        # single wide top interval, uniform bisection reaches only top/2^rounds and
        # left a 1.49x conveyance error near h = 0.02 m even after the bottom was
        # fixed. Log spacing gives roughly uniform RELATIVE accuracy, which is what
        # the 2% tolerance is stated in.
        span = max(top, 1e-6)
        ladder = span * np.logspace(-bottom_decades, 0.0, 6 * bottom_decades + 1)
        levels = sorted(set(np.r_[0.0, ladder, breaks, top].tolist()))

        # HEC-RAS Face Conveyance Tol Ratio: bisect where linear K interpolation errs
        for _ in range(14):
            if len(levels) >= max_points:
                break
            added: list[float] = []
            for lo, hi in zip(levels[:-1], levels[1:], strict=True):
                mid = 0.5 * (lo + hi)
                if hi - lo < 1e-9:
                    continue
                exact = conv_of(mid)
                linear = 0.5 * (conv_of(lo) + conv_of(hi))
                if exact > 0 and abs(linear - exact) > conveyance_tol * exact:
                    added.append(mid)
            if not added:
                break
            levels = sorted(set(levels + added))
        levels = np.asarray(levels[:max_points], dtype=np.float64)

        # width and perimeter are the values just ABOVE each level (right-continuous)
        eps = 1e-9
        W = np.array([width_of(z + eps) for z in levels])
        P = np.array([perim_of(z + eps) for z in levels])
        K = np.array([conv_of(z) for z in levels])
        # area as the EXACT integral of the piecewise-constant width
        A = np.zeros(len(levels))
        if len(levels) > 1:
            A[1:] = np.cumsum(W[:-1] * np.diff(levels))
        datum[len(built)] = z_ref
        counts[len(built)] = len(levels)
        built.append((levels, A, P, K))

    width = max(2, int(counts.max()))
    Z = np.zeros((n_faces, width)); A = np.zeros_like(Z)
    P = np.zeros_like(Z); K = np.zeros_like(Z)
    for f, (z, a, p, k) in enumerate(built):
        m = len(z)
        Z[f, :m], A[f, :m], P[f, :m], K[f, :m] = z, a, p, k
        if m < width:
            Z[f, m:] = z[-1]; A[f, m:] = a[-1]; P[f, m:] = p[-1]; K[f, m:] = k[-1]
    return FaceConveyanceTable(datum, Z, A, P, K, counts, np.asarray(length_m, dtype=np.float64))


SUBGRID_POLICIES = ("all", "river", "river_floodplain", "waterbody", "none")


def subgrid_cell_policy(
    feature_classes,
    policy: str = "river",
    within_cell_relief_m=None,
    relief_threshold_m: float = 0.0,
) -> np.ndarray:
    """Which cells get a level-pool elevation-volume curve.

    ``feature_classes`` is the mesh's per-cell class ("river", "floodplain",
    "waterbody", "urban", "rural", ...). Returns a boolean mask to pass as
    ``level_pool_mask``, and to write to the table file as ``cell_is_subgrid`` so
    the solver can switch the FACE closure with it. A face may only use the
    sub-grid tables if BOTH its cells do -- switching the cell closure alone
    leaves the cell referenced to its area-mean bed while the face table measures
    from the face minimum, about 1.9 m lower on a 60 m mesh, and a nearly dry cell
    then presents its faces metres of phantom head.

    Policies
    --------
    ``"all"``
        Every cell, which is what HEC-RAS does. Correct when the terrain is much
        finer than the mesh AND the water surface is genuinely near-flat inside a
        cell -- e.g. 1 m lidar under 10 m cells. Do NOT use it for coarse cells on
        rain-driven hillslopes: measured on the Di Giammarco V-catchment at 60 m,
        0.9 mm of rain becomes a 64.8 mm pool over 1.39% of the cell, a 72x depth
        ratio, and the hydrograph reaches equilibrium in 883 s against a published
        5400 s (NSE 0.267 against 0.974 for no sub-grid at all).

    ``"river"`` (default)
        Channel cells only. This is the case with the clearest measured gain. On a
        steady uniform-flow test against Manning's normal depth, with a 20 m
        channel inside a 100 m cell:

            no sub-grid   0.2209 m   (-75.4%)
            sub-grid      0.8920 m   ( -0.8%)     analytic 0.8989 m

        and on the same geometry taken overbank (analytic 0.4417 m):

            no sub-grid          0.1528 m   (-65.4%)
            river only           0.4073 m   ( -7.8%)
            river + floodplain   0.3785 m   (-14.3%)

        Discharge is exact in every arm; it is the DEPTH -- the flood map -- that
        a flat prism gets 65-75% wrong.

    ``"river_floodplain"``
        Adds floodplain cells. Measured WORSE than ``"river"`` on the overbank
        test above (-14.3% against -7.8%), because that floodplain is smooth: it
        has no within-cell relief for the tables to capture, so it only inherits
        the level-pool error on a thin downstream-directed sheet. Expect this to
        reverse where a floodplain has real micro-topography (levees, roads,
        ditches, borrow pits) and the water is ponded rather than running --
        that is HEC-RAS's core use case -- but we have no analytic reference for
        it, so treat the choice as case-specific and test it.

    ``"waterbody"``
        Reservoirs and lakes only. Level pool is exactly right for a pond.

    ``"none"``
        Flat prism everywhere. Reproduces the non-sub-grid solver bit for bit.

    ``within_cell_relief_m`` optionally ANDs in a relief floor, so cells with
    nothing to resolve fall back to the flat prism and cost nothing.
    """
    if policy not in SUBGRID_POLICIES:
        raise ValueError(f"policy must be one of {SUBGRID_POLICIES}, got {policy!r}.")
    classes = np.asarray([str(c) for c in feature_classes])
    if policy == "all":
        mask = np.ones(classes.size, dtype=bool)
    elif policy == "none":
        mask = np.zeros(classes.size, dtype=bool)
    elif policy == "river":
        mask = np.isin(classes, ("river", "channel"))
    elif policy == "waterbody":
        mask = np.isin(classes, ("waterbody", "reservoir", "lake"))
    else:
        mask = np.isin(classes, ("river", "channel", "floodplain", "waterbody",
                                 "reservoir", "lake"))
    if within_cell_relief_m is not None and relief_threshold_m > 0.0:
        mask &= np.asarray(within_cell_relief_m, dtype=np.float64) >= relief_threshold_m
    return mask
