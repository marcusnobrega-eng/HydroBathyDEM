"""The face-table builder as it was before the 2026-10 speed-up: the exact reference.

tests/test_subgrid_tables_speedup.py checks that the faster builder gives the same arrays.
"""

import numpy as np

from dem_processing.subgrid_tables import (
    FACE_AREA_ELEV_FILTER_M,
    FACE_CONVEYANCE_TOL,
    FaceConveyanceTable,
)


def reference_build_face_conveyance_table(
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
        seg_wet = np.hypot(seg_len, np.diff(bed))  # wetted length along the bed
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
    Z = np.zeros((n_faces, width))
    A = np.zeros_like(Z)
    P = np.zeros_like(Z)
    K = np.zeros_like(Z)
    for f, (z, a, p, k) in enumerate(built):
        m = len(z)
        Z[f, :m], A[f, :m], P[f, :m], K[f, :m] = z, a, p, k
        if m < width:
            Z[f, m:] = z[-1]
            A[f, m:] = a[-1]
            P[f, m:] = p[-1]
            K[f, m:] = k[-1]
    return FaceConveyanceTable(
        datum, Z, A, P, K, counts, np.asarray(length_m, dtype=np.float64)
    )


