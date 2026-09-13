from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from dem_processing.hybrid_mesh import _ring_edges, _snapped_vertices


def test_ring_edges_reports_each_exterior_segment() -> None:
    polygons = [
        Polygon(((0, 0), (2, 0), (2, 2), (0, 2))),
        Polygon(((2, 0), (4, 0), (4, 2), (2, 2))),
    ]

    owner, position, start, end = _ring_edges(polygons)

    assert owner.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert position.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    np.testing.assert_allclose(start[0], (0, 0))
    np.testing.assert_allclose(end[-1], (2, 0))


def test_snapped_vertices_only_rebuilds_affected_cells() -> None:
    affected = Polygon(((0, 0), (2, 0), (2, 2), (0, 2)))
    untouched = Polygon(((4, 0), (6, 0), (6, 2), (4, 2)))

    result = _snapped_vertices([affected, untouched], {(2.0, 2.0): (2.0, 1.5)})

    assert result[0].equals(Polygon(((0, 0), (2, 0), (2, 1.5), (0, 2))))
    assert result[1] is untouched


def test_snapped_vertices_is_a_noop_without_matching_vertices() -> None:
    polygon = Polygon(((0, 0), (2, 0), (2, 2), (0, 2)))

    result = _snapped_vertices([polygon], {(9.0, 9.0): (10.0, 10.0)})

    assert result[0] is polygon
