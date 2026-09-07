"""HEC-RAS style computation-point generation for an unstructured mesh.

HEC-RAS builds its 2-D mesh the same way we do -- Delaunay triangulation of a set
of computation points, then the Voronoi dual, cells capped at eight sides.  The
difference is entirely in *where the points go*, and that difference is why it
needs no repair stage:

* A breakline carries ``near_spacing`` (cell size at the line), ``near_repeats``
  (how many rows are held at that size) and ``far_spacing`` (what it grades out
  to).  Points are placed in rows that straddle the line, so the Voronoi bisector
  between a left row and a right row falls *on* the line.  The breakline becomes a
  cell face by construction rather than by cutting finished cells.
* A refinement region carries ``cell_spacing_x/y`` for its interior and a separate
  ``perimeter_spacing`` for its boundary, again with ``near_repeats`` and
  ``far_spacing`` to grade outward.
* Everything a higher-priority feature has already claimed is excluded before the
  next feature seeds, so nothing is placed twice and no two spacings interleave.

Reference: HEC-RAS 2D User's Manual, "Development of the 2D Computational Mesh".

What this module does NOT do is the other half of the HEC-RAS method: the
sub-grid hydraulic property tables (elevation-volume per cell, elevation versus
area / wetted perimeter / roughness per face).  Those are what let a coarse cell
carry a narrow channel, and HydroPol2D cannot read them yet -- it takes a single
bed elevation per cell and a single length per face.  Point placement alone will
not make a 7 m channel work inside a 90 m cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely import contains_xy, prepare, unary_union


@dataclass(frozen=True)
class Breakline:
    """A line the mesh must align to: bank, shoreline, levee, embankment.

    ``enforce`` mirrors HEC-RAS's "Enforce Breakline": points already claimed
    within the graded band are dropped and re-seeded from the line outwards, so
    faces land on the line instead of being cut to it afterwards.
    """

    geometry: LineString
    near_spacing_m: float
    near_repeats: int = 2
    far_spacing_m: float | None = None
    enforce: bool = True
    straddle: bool = True
    # A bank with a channel refinement region on one side must grade AWAY from it
    # only.  Grading both ways is right for a free-standing levee, and wrong for a
    # 30 m channel: each bank then reaches 340 m to meet a 200 m background, so the
    # two bands overlap completely and the second bank loses 277 of its 406 points.
    grade_sign: float = 0.0
    # Set when this line bounds an area, so the offset can be taken from a real
    # buffer of that area rather than from per-station normals.
    closed_area: Any = None
    name: str = "breakline"


@dataclass(frozen=True)
class RefinementRegion:
    """An area meshed at its own resolution: urban, rural, floodplain, lake."""

    geometry: Polygon
    cell_spacing_x_m: float
    cell_spacing_y_m: float | None = None
    perimeter_spacing_m: float | None = None
    near_repeats: int = 1
    far_spacing_m: float | None = None
    # Straddle the boundary rather than sit on it. A point ON the ring puts the
    # ring THROUGH a cell; a pair either side puts the Voronoi bisector -- and so a
    # cell face -- on the ring. This is what makes a reservoir shoreline a hard
    # constraint instead of something 247 cells straddle and a repair pass has to
    # chase. A rectangular shoreline seeded with points on the ring sat 3.93 m from
    # the nearest face (p95 32 m); straddled it lands on it.
    perimeter_straddle: bool = True
    # A region boundary IS a breakline. Treating it as a by-product of the region
    # left only 10.7% of an urban boundary on a cell face (52% for a reservoir
    # shoreline), because the region's own interior lattice sat closer to the
    # boundary than the straddle pair did, and a finer feature crossing it erased
    # the boundary points altogether. Promoting it to a first-class breakline,
    # placed before any lattice and with its own claimed band, fixes all of that.
    boundary_is_breakline: bool = True
    boundary_near_repeats: int = 1
    name: str = "region"

    @property
    def spacing_y(self) -> float:
        return self.cell_spacing_y_m if self.cell_spacing_y_m is not None else self.cell_spacing_x_m


@dataclass(frozen=True)
class ChannelCorridor:
    """A river corridor: lanes inside the channel, then graded rows outside each bank.

    This exists because generating the two banks as independent breaklines does not
    work. Each bank then resamples its OWN curve, so the row that should pair with
    a row from the other bank is staggered along flow, and the Voronoi bisector
    zigzags instead of following the bank. Measured on a sinuous 30 m channel: the
    left bank landed 0.01 m from a cell face and the right bank 0.79 m -- an 80x
    asymmetry between two lines that differ only in sign. Worse, each bank graded
    out 340 m to meet a 200 m background, so the first bank's band swallowed the
    second and it lost 277 of its 406 points.

    A corridor resamples the centreline ONCE and derives every row from those same
    stations and normals, so opposing rows are aligned by construction and both
    banks are equally well resolved.
    """

    centerline: LineString
    width_m: float
    along_spacing_m: float
    cross_spacing_m: float
    far_spacing_m: float | None = None
    near_repeats: int = 2
    name: str = "channel"

    @property
    def lanes(self) -> int:
        return max(1, int(round(self.width_m / self.cross_spacing_m)))

    @property
    def lane_width_m(self) -> float:
        return self.width_m / self.lanes


def _fold_limited(stations: np.ndarray, normals: np.ndarray, half: np.ndarray,
                  safety: float = 0.5) -> np.ndarray:
    """Cap the offset where two consecutive cross-sections would cross.

    An offset row is single valued only while the offset stays inside the point at
    which neighbouring cross-sections meet; past it the row folds back through the
    centreline and the points end up on the wrong side.
    """
    limit = np.full(len(stations), np.inf)
    if len(stations) < 2:
        return np.minimum(half, limit)
    delta = stations[1:] - stations[:-1]
    lead, follow = normals[:-1], normals[1:]
    turn = lead[:, 0] * follow[:, 1] - lead[:, 1] * follow[:, 0]
    parallel = np.abs(turn) < 1e-12
    safe = np.where(parallel, 1.0, turn)
    lead_off = (delta[:, 0] * follow[:, 1] - delta[:, 1] * follow[:, 0]) / safe
    follow_off = (delta[:, 0] * lead[:, 1] - delta[:, 1] * lead[:, 0]) / safe
    limit[:-1] = np.minimum(limit[:-1], np.where(parallel, np.inf, safety * np.abs(lead_off)))
    limit[1:] = np.minimum(limit[1:], np.where(parallel, np.inf, safety * np.abs(follow_off)))
    return np.minimum(half, limit)


def offset_ring_points(
    ring: LineString, offset_m: float, spacing_m: float, closed_area,
    sides: tuple[float, ...] = (1.0, -1.0),
) -> list[np.ndarray]:
    """Points on both offset curves of a ring, at ``spacing_m`` along each.

    Uses a real offset curve rather than per-station normals. A normal offset
    collapses at a sharp corner -- the fold limiter caps it, the straddle pair
    converges onto the ring, and the bisector drifts. On a rectangular reservoir
    that lost every corner and half the shoreline. ``buffer`` resolves corners
    properly, so the pair stays a full ``offset_m`` either side all the way round.
    """
    out: list[np.ndarray] = []
    for side in sides:
        signed = side * offset_m
        try:
            shifted = closed_area.buffer(signed, join_style="mitre", mitre_limit=2.0)
        except Exception:
            continue
        if shifted.is_empty:
            continue
        for part in _rings(shifted):
            stations, _ = _resample(part, spacing_m)
            if len(stations):
                out.append(stations)
    return out


def region_boundary_breaklines(
    region: RefinementRegion, outside_spacing_m: float,
) -> list[Breakline]:
    """The region outline as a breakline, spaced for the cells that will meet it.

    Spacing comes from the FINER of the two sides, because that is the size of the
    cells that will actually touch the boundary. Taking it from the region's own
    interior gave an urban boundary a 45 m straddle against 90 m cells, and the
    interior lattice then sat closer to the line than the pair did.
    """
    if not region.boundary_is_breakline:
        return []
    interior = min(region.cell_spacing_x_m, region.spacing_y)
    spacing = region.perimeter_spacing_m or min(interior, float(outside_spacing_m))
    return [
        Breakline(
            geometry=ring, near_spacing_m=spacing, near_repeats=region.boundary_near_repeats,
            far_spacing_m=region.far_spacing_m, enforce=True, straddle=True,
            closed_area=region.geometry, name=f"{region.name}_boundary",
        )
        for ring in _rings(region.geometry)
    ]


def _inside(geometry, points: np.ndarray) -> np.ndarray:
    """Vectorised point-in-polygon.

    One Python-level shapely call per point dominates everything else at Pune
    scale, where the feature set produces of order a million points.
    ``contains_xy`` on a prepared geometry does the same work in one call.
    """
    if not len(points):
        return np.zeros(0, dtype=bool)
    prepare(geometry)
    return contains_xy(geometry, points[:, 0], points[:, 1])


def _clip_groups(
    groups: list[tuple[np.ndarray, float]], domain: Polygon,
) -> list[tuple[np.ndarray, float]]:
    """Keep only points inside the domain, preserving each group's own spacing.

    Spacing travels with the points because the cull threshold must be local. A
    reservoir region has a 200 m interior and a 60 m perimeter; culling the whole
    region at 200 m deleted the perimeter rows, and the shoreline drifted 6.7 m
    from the nearest cell face instead of landing on it.
    """
    out: list[tuple[np.ndarray, float]] = []
    for points, spacing in groups:
        if not len(points):
            continue
        keep = _inside(domain, points)
        if keep.any():
            out.append((points[keep], float(spacing)))
    return out


def corridor_points(
    corridor: ChannelCorridor, domain: Polygon, default_far_m: float,
) -> tuple[np.ndarray, float]:
    """Lanes inside the channel plus graded rows outside, all from one resampling.

    The bank at +/- width/2 becomes a cell face because the innermost lane sits at
    width/2 - lane/2 and the first outer row at width/2 + lane/2: the bisector
    between them is the bank itself.
    """
    line = corridor.centerline
    if line.length <= 0 or corridor.width_m <= 0:
        return np.empty((0, 2)), 0.0
    stations, normals = _resample(line, corridor.along_spacing_m)
    if len(stations) < 2:
        return np.empty((0, 2)), 0.0
    lanes, lane = corridor.lanes, corridor.lane_width_m
    half = _fold_limited(stations, normals, np.full(len(stations), 0.5 * corridor.width_m))
    collected: list[tuple[np.ndarray, float]] = []

    # lanes inside the channel, at lane centres
    for index in range(lanes):
        fraction = -1.0 + (2.0 * index + 1.0) / lanes
        collected.append((stations + (fraction * half)[:, None] * normals, lane))

    # rows outside each bank, graded out to the background size
    ladder = grading_ladder(
        lane, corridor.far_spacing_m if corridor.far_spacing_m is not None else default_far_m,
        corridor.near_repeats,
    )
    offset = half.copy()
    previous = 0.0
    reach = float(np.max(half)) if len(half) else 0.0
    for width in ladder:
        offset = offset + 0.5 * (previous + width)
        stride = max(1, int(round(width / max(corridor.along_spacing_m, 1e-9))))
        index = np.arange(0, len(stations), stride)
        for sign in (-1.0, 1.0):
            collected.append(
                (stations[index] + (sign * offset[index])[:, None] * normals[index], width)
            )
        previous = width
        reach = max(reach, float(np.max(offset)))
    return _clip_groups(collected, domain), reach


@dataclass
class PointBudget:
    """Where each computation point came from, for QA and for the pre-build report."""

    counts: dict[str, int] = field(default_factory=dict)
    claimed_m2: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, points: np.ndarray) -> None:
        self.counts[name] = self.counts.get(name, 0) + len(points)


def grading_ladder(near_m: float, far_m: float, repeats: int, ratio: float = 2.0) -> list[float]:
    """Row spacings from ``near_m`` out to ``far_m``, HEC-RAS Near/Far style.

    The first ``repeats`` rows stay at ``near_m`` -- that is what ``near_repeats``
    buys: a band of uniform fine cells before any transition starts.  After that
    each row grows by at most ``ratio``, which is what makes the adjacent-size
    rule hold without anyone checking it afterwards.
    """
    if not np.isfinite(near_m) or near_m <= 0:
        return []
    rows = [float(near_m)] * max(1, int(repeats))
    if far_m is None or not np.isfinite(far_m) or far_m <= near_m:
        return rows
    width = float(near_m)
    while width < far_m * (1.0 - 1e-9) and len(rows) < 24:
        width = min(width * max(ratio, 1.05), float(far_m))
        rows.append(width)
    return rows


def _resample(line: LineString, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Stations at (at most) ``spacing_m`` apart, with unit normals."""
    if line.length <= 0 or spacing_m <= 0:
        return np.empty((0, 2)), np.empty((0, 2))
    n = max(1, int(np.ceil(line.length / spacing_m)))
    d = np.linspace(0.0, line.length, n + 1)
    pts = np.asarray([[p.x, p.y] for p in (line.interpolate(item) for item in d)], dtype=np.float64)
    if len(pts) < 2:
        return pts, np.zeros_like(pts)
    tangent = np.gradient(pts, axis=0)
    scale = np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-12)
    tangent /= scale[:, None]
    return pts, np.column_stack((-tangent[:, 1], tangent[:, 0]))


def breakline_points(
    breakline: Breakline, domain: Polygon, default_far_m: float,
) -> tuple[np.ndarray, float]:
    """Rows of points either side of a breakline, graded outward.

    Returns the points and how far the graded band reaches, so the caller can
    exclude that band before seeding anything coarser.
    """
    rows = grading_ladder(
        breakline.near_spacing_m,
        breakline.far_spacing_m if breakline.far_spacing_m is not None else default_far_m,
        breakline.near_repeats,
    )
    if not rows:
        return [], 0.0
    collected: list[tuple[np.ndarray, float]] = []
    offset = 0.0
    previous = 0.0
    for width in rows:
        # Row centre sits half a cell beyond the previous row's outer edge, so the
        # bisector between opposing rows lands on the line itself.
        offset += 0.5 * (previous + width) if previous else 0.5 * width
        stations, normals = _resample(breakline.geometry, width)
        if not len(stations):
            break
        if breakline.closed_area is not None:
            # The first row straddles, so the boundary is a face. Every later row
            # goes OUTWARD only: grading inward as well made the band reach 370 m
            # from a region outline and swallow the region's own interior, which
            # then received 5 points where it needed about 120.
            sides = (1.0, -1.0) if offset <= 0.5 * rows[0] + 1e-9 else (1.0,)
            for group in offset_ring_points(
                breakline.geometry, offset, width, breakline.closed_area, sides,
            ):
                collected.append((group, width))
            previous = width
            continue
        if breakline.grade_sign:
            # first row still straddles, so the line stays a face; later rows go one way
            signs = (-1.0, 1.0) if offset <= 0.5 * rows[0] + 1e-9 else (breakline.grade_sign,)
        else:
            signs = (-1.0, 1.0) if breakline.straddle else (1.0,)
        for sign in signs:
            collected.append((stations + sign * offset * normals, width))
        previous = width
    if not collected:
        return [], 0.0
    return _clip_groups(collected, domain), offset


def region_points(region: RefinementRegion, domain: Polygon) -> np.ndarray:
    """An anisotropic interior lattice plus a separately spaced perimeter."""
    geometry = region.geometry.intersection(domain)
    if geometry.is_empty:
        return []
    xmin, ymin, xmax, ymax = geometry.bounds
    gx = np.arange(xmin + 0.5 * region.cell_spacing_x_m, xmax, region.cell_spacing_x_m)
    gy = np.arange(ymin + 0.5 * region.spacing_y, ymax, region.spacing_y)
    collected: list[tuple[np.ndarray, float]] = []
    interior = min(region.cell_spacing_x_m, region.spacing_y)
    if len(gx) and len(gy):
        mesh = np.column_stack([item.ravel() for item in np.meshgrid(gx, gy, indexing="ij")])
        collected.append((mesh[_inside(geometry, mesh)], interior))
    # The interior lattice only. The boundary is emitted separately, as a
    # breakline, by ``region_boundary_breaklines``.
    if not region.boundary_is_breakline:
        spacing = region.perimeter_spacing_m
        if spacing and spacing > 0:
            for ring in _rings(geometry):
                stations, _ = _resample(ring, spacing)
                if len(stations):
                    collected.append((stations, spacing))
    return _clip_groups(collected, domain)


def _rings(geometry) -> list[LineString]:
    out: list[LineString] = []
    parts = [geometry] if geometry.geom_type == "Polygon" else list(getattr(geometry, "geoms", []))
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        out.append(LineString(part.exterior.coords))
        out.extend(LineString(item.coords) for item in part.interiors)
    return out


class _PlacedIndex:
    """A spatial index over the points placed so far, rebuilt only when it pays.

    Culling used to call ``_drop_claimed`` per point group, and each call did a
    fresh ``np.vstack`` of every prior array plus a fresh ``cKDTree``. Pune's 854
    river corridors emit 7,836 groups over 332,132 points, so that is 7,836 tree
    builds on a growing set -- about a billion distance evaluations, and 26 minutes
    of a build whose point GENERATION takes 0.3 seconds.

    Here the committed points live in one tree that is rebuilt only when the
    pending backlog grows past ``rebuild_at``, and pending points are checked
    against a small tree of their own. Cost becomes near linear.
    """

    def __init__(self, rebuild_at: int = 20_000) -> None:
        self._committed: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._tree: cKDTree | None = None
        self._pending: list[np.ndarray] = []
        self._pending_n = 0
        self._rebuild_at = int(rebuild_at)

    def add(self, points: np.ndarray) -> None:
        if not len(points):
            return
        self._pending.append(np.asarray(points, dtype=np.float64))
        self._pending_n += len(points)
        if self._pending_n >= self._rebuild_at:
            self._consolidate()

    def _consolidate(self) -> None:
        if not self._pending:
            return
        self._committed = np.vstack([self._committed, *self._pending])
        self._pending, self._pending_n = [], 0
        self._tree = cKDTree(self._committed) if len(self._committed) else None

    def keep_farther_than(self, points: np.ndarray, radius: float) -> np.ndarray:
        """Points at least ``radius`` from anything already placed."""
        if not len(points) or radius <= 0:
            return points
        keep = np.ones(len(points), dtype=bool)
        if self._tree is not None:
            keep &= self._tree.query(points)[0] >= radius
        if self._pending:
            near = np.vstack(self._pending)
            if len(near):
                keep &= cKDTree(near).query(points)[0] >= radius
        return points[keep]

    def all_points(self) -> np.ndarray:
        self._consolidate()
        return self._committed


def rows_first(breakline: Breakline) -> float:
    """The finest row width a breakline will emit, for sizing its inner keep-out."""
    return float(breakline.near_spacing_m)


def computation_points(
    domain: Polygon,
    nominal_spacing_x_m: float,
    nominal_spacing_y_m: float | None = None,
    breaklines: Iterable[Breakline] = (),
    regions: Iterable[RefinementRegion] = (),
    corridors: Iterable[ChannelCorridor] = (),
) -> tuple[np.ndarray, PointBudget]:
    """Place every computation point, highest-priority feature first.

    Order is breaklines, then refinement regions, then the nominal background.
    Each stage excludes what earlier stages claimed, which is what stops two
    spacings interleaving and producing the short faces our old pipeline then had
    to repair.
    """
    spacing_y = nominal_spacing_y_m if nominal_spacing_y_m is not None else nominal_spacing_x_m
    nominal = min(float(nominal_spacing_x_m), float(spacing_y))
    budget = PointBudget()
    index = _PlacedIndex()
    claimed: list[tuple[Any, float]] = []
    regions = list(regions)
    corridors = list(corridors)

    # PHASE 1 -- every hard feature, finest first.
    #
    # Hard features are corridors, explicit breaklines, and every region boundary
    # (see ``region_boundary_breaklines``). They all go down before any lattice,
    # and a hard feature is never culled by another hard feature's claimed band:
    # doing so let a 30 m river corridor erase the 90 m urban boundary it crossed,
    # and the boundary simply vanished into the corridor's grid. Only lattices are
    # culled by bands.
    hard: list[tuple[float, str, Any]] = []
    hard += [(float(item.lane_width_m), "corridor", item) for item in corridors]
    hard += [(float(item.near_spacing_m), "breakline", item) for item in breaklines]
    for region in regions:
        hard += [
            (float(item.near_spacing_m), "breakline", item)
            for item in region_boundary_breaklines(region, nominal)
        ]
    hard.sort(key=lambda entry: entry[0])

    for spacing, kind, item in hard:
        if kind == "corridor":
            groups, reach = corridor_points(item, domain, nominal_spacing_x_m)
        else:
            groups, reach = breakline_points(item, domain, nominal_spacing_x_m)
        kept = 0
        for points, group_spacing in sorted(groups, key=lambda entry: entry[1]):
            # Only a tight duplicate check between hard features, so two features
            # that meet keep both their rows.
            points = index.keep_farther_than(points, 0.25 * group_spacing)
            if not len(points):
                continue
            index.add(points)
            kept += len(points)
        if kept:
            budget.counts[item.name] = budget.counts.get(item.name, 0) + kept
        if reach > 0:
            if kind == "corridor":
                band = item.centerline.buffer(reach)
            elif item.closed_area is not None:
                # Only the OUTSIDE of a region outline is claimed; the inside
                # belongs to that region's own lattice.
                band = item.closed_area.buffer(reach).difference(
                    item.closed_area.buffer(-0.5 * rows_first(item))
                )
            else:
                band = item.geometry.buffer(reach)
            claimed.append((band, spacing))
            budget.claimed_m2[item.name] = budget.claimed_m2.get(item.name, 0.0) + band.area

    # PHASE 2 -- lattices, finest first, kept clear of every finer claimed band.
    lattices: list[tuple[float, RefinementRegion]] = [
        (min(item.cell_spacing_x_m, item.spacing_y), item) for item in regions
    ]
    covered = [item.geometry for item in regions]
    covered += [item.centerline.buffer(0.5 * item.width_m) for item in corridors]
    covered += [band for band, _ in claimed]
    background_area = domain if not covered else domain.difference(unary_union(covered))
    lattices.append((
        nominal,
        RefinementRegion(
            geometry=background_area, cell_spacing_x_m=nominal_spacing_x_m,
            cell_spacing_y_m=spacing_y, boundary_is_breakline=False, name="background",
        ),
    ))
    lattices.sort(key=lambda entry: entry[0])

    for spacing, region in lattices:
        blocking = [band for band, band_spacing in claimed
                    if band_spacing < spacing * (1.0 - 1e-9)]
        blocker = unary_union(blocking) if blocking else None
        kept = 0
        for points, group_spacing in sorted(region_points(region, domain), key=lambda e: e[1]):
            if blocker is not None and len(points):
                points = points[~_inside(blocker, points)]
            points = index.keep_farther_than(points, 0.75 * group_spacing)
            if not len(points):
                continue
            index.add(points)
            kept += len(points)
        if kept:
            budget.counts[region.name] = budget.counts.get(region.name, 0) + kept

    points = index.all_points()
    if not len(points):
        return np.empty((0, 2)), budget
    return np.unique(points, axis=0), budget


def _drop_claimed(
    points: np.ndarray, placed: list[np.ndarray], spacing_m: float, factor: float = 0.75,
) -> np.ndarray:
    """Remove points that fall too close to something already placed.

    ``factor`` is 0.75 for a lattice, which must yield to anything finer, and 0.25
    between hard features, where the only thing worth removing is a near duplicate.
    """
    if not len(points) or not placed:
        return points
    existing = np.vstack([item for item in placed if len(item)])
    if not len(existing):
        return points
    distance, _ = cKDTree(existing).query(points)
    return points[distance >= factor * spacing_m]
