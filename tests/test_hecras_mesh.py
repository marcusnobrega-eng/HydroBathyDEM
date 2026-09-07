"""What HEC-RAS style point placement must guarantee, pinned as tests.

Each test states one property the old pipeline could only check after the fact and
then repair.  The point of the rewrite is that these hold by construction, so if
one of them fails the design is wrong, not the mesh.
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely import MultiPoint, intersection, unary_union, voronoi_polygons
from shapely.geometry import LineString, Point, box

from dem_processing.hecras_mesh import (
    Breakline,
    ChannelCorridor,
    RefinementRegion,
    computation_points,
    grading_ladder,
)

DOMAIN = box(-200.0, -900.0, 2200.0, 900.0)


def _cells(points: np.ndarray, domain=DOMAIN) -> list:
    diagram = voronoi_polygons(MultiPoint(points), extend_to=domain)
    cells = [intersection(item, domain) for item in diagram.geoms]
    return [item for item in cells if item.geom_type == "Polygon" and item.area > 1e-9]


def _distance_to_nearest_face(line: LineString, cells: list, samples: int = 400) -> np.ndarray:
    skeleton = unary_union([item.boundary for item in cells])
    return np.asarray([
        skeleton.distance(Point(line.interpolate(item)))
        for item in np.linspace(0.0, line.length, samples)
    ])


def _sinuous(amplitude: float = 120.0, wavelength: float = 400.0) -> LineString:
    x = np.arange(0.0, 2000.0, 25.0)
    return LineString(np.column_stack((x, amplitude * np.sin(x / wavelength))))


def test_grading_ladder_holds_near_rows_then_doubles() -> None:
    assert grading_ladder(30.0, 200.0, repeats=2) == [30.0, 30.0, 60.0, 120.0, 200.0]
    # a single repeat still starts at the fine size
    assert grading_ladder(30.0, 200.0, repeats=1)[0] == 30.0
    # no far target means no transition
    assert grading_ladder(30.0, None, repeats=3) == [30.0, 30.0, 30.0]
    # every step is within the ratio, which is what makes the size rule hold
    ladder = grading_ladder(30.0, 500.0, repeats=1)
    assert all(b / a <= 2.0 + 1e-9 for a, b in zip(ladder, ladder[1:]))


def test_channel_corridor_puts_both_banks_on_a_cell_face() -> None:
    """The defect this rewrite exists to fix.

    Generating each bank as its own breakline gave rows sampled from two different
    curves, staggered along flow, so the bisector zigzagged: the left bank landed
    0.01 m from a face and the right bank 0.79 m, an 80x asymmetry between two
    lines that differ only in sign.  A corridor resamples the centreline ONCE and
    derives every row from it, so both banks are equally well resolved.
    """
    centre = _sinuous()
    corridor = ChannelCorridor(
        centerline=centre, width_m=30.0, along_spacing_m=30.0,
        cross_spacing_m=30.0, far_spacing_m=200.0, near_repeats=2,
    )
    points, _ = computation_points(DOMAIN, 200.0, corridors=[corridor])
    cells = _cells(points)
    left = centre.parallel_offset(15.0, "left")
    right = centre.parallel_offset(15.0, "right")
    dl = _distance_to_nearest_face(left, cells)
    dr = _distance_to_nearest_face(right, cells)
    assert np.median(dl) < 1.0, f"left bank {np.median(dl):.2f} m from a face"
    assert np.median(dr) < 1.0, f"right bank {np.median(dr):.2f} m from a face"
    ratio = max(np.median(dl), np.median(dr)) / max(min(np.median(dl), np.median(dr)), 1e-6)
    assert ratio < 10.0, f"banks resolved {ratio:.0f}x differently"


def test_channel_corridor_seeds_the_channel_interior() -> None:
    """A 30 m channel asked for 30 m cells must actually contain points."""
    centre = _sinuous()
    corridor = ChannelCorridor(
        centerline=centre, width_m=90.0, along_spacing_m=30.0,
        cross_spacing_m=30.0, far_spacing_m=200.0,
    )
    points, budget = computation_points(DOMAIN, 200.0, corridors=[corridor])
    inside = centre.buffer(45.0)
    n = sum(1 for item in points if inside.covers(Point(item)))
    assert n > 0, "no points inside the channel"
    # 90 m wide at 30 m target is three lanes across
    stations = int(centre.length // 30.0)
    assert n >= 2 * stations, f"only {n} points for ~{3 * stations} lane slots"


def test_no_feature_annihilates_a_neighbour_of_equal_fineness() -> None:
    """Two 30 m features 30 m apart must both keep their points.

    Processing in input order let the first feature's graded band swallow the
    second: bank_L kept 406 points and bank_R only 129, of 406.
    """
    centre = _sinuous()
    a = ChannelCorridor(centerline=centre, width_m=30.0, along_spacing_m=30.0,
                        cross_spacing_m=30.0, far_spacing_m=200.0, name="channel_a")
    shifted = LineString(np.asarray(centre.coords) + np.asarray([0.0, 600.0]))
    b = ChannelCorridor(centerline=shifted, width_m=30.0, along_spacing_m=30.0,
                        cross_spacing_m=30.0, far_spacing_m=200.0, name="channel_b")
    _, budget = computation_points(DOMAIN, 200.0, corridors=[a, b])
    na, nb = budget.counts["channel_a"], budget.counts["channel_b"]
    assert min(na, nb) > 0.5 * max(na, nb), f"lopsided: {na} vs {nb}"


def test_cells_respect_the_requested_sizes_per_region() -> None:
    urban = box(200.0, -600.0, 1000.0, 600.0)
    points, _ = computation_points(
        DOMAIN, 200.0,
        regions=[RefinementRegion(urban, 90.0, perimeter_spacing_m=90.0, far_spacing_m=200.0,
                                  name="urban")],
    )
    cells = _cells(points)
    inner = urban.buffer(-120.0)
    urban_side = [np.sqrt(item.area) for item in cells if inner.covers(item.representative_point())]
    assert urban_side, "no cells inside the urban region"
    assert 60.0 < float(np.median(urban_side)) < 130.0, f"urban median {np.median(urban_side):.0f} m"


def test_standalone_breakline_lands_on_a_face() -> None:
    """A levee or road has no region either side, so it straddles both ways."""
    levee = LineString([(0.0, 0.0), (1800.0, 0.0)])
    points, _ = computation_points(
        DOMAIN, 200.0,
        breaklines=[Breakline(levee, 40.0, near_repeats=2, far_spacing_m=200.0, name="levee")],
    )
    cells = _cells(points)
    d = _distance_to_nearest_face(levee, cells)
    assert np.median(d) < 0.5, f"levee {np.median(d):.2f} m from a face"


def test_corridor_holds_the_quality_floor() -> None:
    """A/P is the CFL length for a local-inertial scheme on an unstructured mesh.

    The production mesh reaches A/P = 7.50 m with 30 m river cells, which is what a
    30 m square gives. Point placement must not do worse than that.
    """
    centre = _sinuous()
    corridor = ChannelCorridor(centerline=centre, width_m=30.0, along_spacing_m=30.0,
                               cross_spacing_m=30.0, far_spacing_m=200.0)
    points, _ = computation_points(DOMAIN, 200.0, corridors=[corridor])
    cells = _cells(points)
    ap = np.asarray([item.area / item.length for item in cells])
    assert float(np.min(ap)) > 5.0, f"A/P floor {np.min(ap):.2f} m"
    assert float(np.median(ap)) > 7.0, f"A/P median {np.median(ap):.2f} m"


def test_cells_stay_within_the_eight_side_cap() -> None:
    """HEC-RAS caps a computational cell at eight sides."""
    centre = _sinuous()
    corridor = ChannelCorridor(centerline=centre, width_m=60.0, along_spacing_m=30.0,
                               cross_spacing_m=30.0, far_spacing_m=200.0)
    urban = box(200.0, -600.0, 1000.0, 600.0)
    points, _ = computation_points(
        DOMAIN, 200.0, corridors=[corridor],
        regions=[RefinementRegion(urban, 90.0, perimeter_spacing_m=90.0, name="urban")],
    )
    cells = _cells(points)
    sides = np.asarray([len(item.exterior.coords) - 1 for item in cells])
    over = int((sides > 8).sum())
    assert over <= 0.01 * len(cells), f"{over} of {len(cells)} cells exceed 8 sides"
