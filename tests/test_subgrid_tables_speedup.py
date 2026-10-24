"""The faster face-table builder gives exactly the arrays of the one it replaced."""

import numpy as np
import pytest
from face_table_reference import reference_build_face_conveyance_table

from dem_processing.subgrid_tables import build_face_conveyance_table


def _profiles(seed: int):
    rng = np.random.default_rng(seed)
    profiles = []
    for index in range(400):
        kind = index % 5
        points = int(rng.integers(2, 40))
        length = float(rng.choice([30.0, 60.0, 90.0, 120.0, 240.0]))
        station = np.linspace(0.0, length, points)
        if kind == 0:  # flat, many repeated (the off-river boundary faces)
            bed = np.full(points, float(rng.normal(10.0, 1.0)))
        elif kind == 1:  # a V channel
            bed = 10.0 + 3.0 * np.abs(np.linspace(-1.0, 1.0, points))
        elif kind == 2:  # rough natural bed
            bed = 10.0 + np.cumsum(rng.normal(0.0, 0.3, points))
        elif kind == 3:  # steps, with equal beds that merge
            bed = np.round(10.0 + rng.normal(0.0, 1.0, points), 1)
        else:  # nearly flat: tiny differences below the 0.003 m filter
            bed = 10.0 + rng.normal(0.0, 1e-4, points)
        rough = np.where(rng.random(points) < 0.3, 0.03, 0.06)
        profiles.append((station, bed, rough))
    profiles.append((np.asarray([0.0]), np.asarray([5.0]), np.asarray([0.05])))  # degenerate
    profiles.append((np.asarray([0.0, 1.0]), np.full(2, np.nan), np.full(2, 0.05)))
    return profiles


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_face_tables_are_identical_to_the_reference(seed: int) -> None:
    profiles = _profiles(seed)
    length = np.asarray([float(station[-1]) if len(station) else 0.0 for station, *_ in profiles])
    fast = build_face_conveyance_table(profiles, length)
    reference = reference_build_face_conveyance_table(profiles, length)
    for name in ("datum_m", "zeta_m", "area_m2", "perimeter_m", "conveyance", "count", "length_m"):
        np.testing.assert_array_equal(getattr(fast, name), getattr(reference, name), err_msg=name)
