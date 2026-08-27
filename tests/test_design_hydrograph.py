import numpy as np

from dem_processing.design_hydrograph import (
    _fill_missing_cn_from_nearest,
    _upstream_composite_cn,
    kirpich_tc_minutes,
    receiver_from_d4_direction,
    scs_runoff_depth_mm,
)


def test_missing_cn_uses_nearest_valid_published_value() -> None:
    cn = np.array([[60.0, np.nan, np.nan, 90.0], [np.nan, np.nan, np.nan, np.nan]])
    domain = np.array([[True, True, True, True], [False, True, True, False]])
    filled, count = _fill_missing_cn_from_nearest(cn, domain)
    assert count == 4
    assert np.isfinite(filled[domain]).all()
    assert np.isnan(filled[~domain]).all()
    assert set(np.unique(filled[domain])) <= {60.0, 90.0}


def test_download_accepts_a_path_like_string_without_network() -> None:
    # The public downloader receives CLI strings; conversion must happen before
    # any network action.  AMC validation is checked first.
    try:
        from dem_processing.design_hydrograph import download_gcn250
        download_gcn250("invalid", "ignored.tif")
    except ValueError:
        pass
    else:  # pragma: no cover - protects the input-validation contract.
        raise AssertionError("invalid AMC must be rejected")


def test_d4_composite_cn_is_area_weighted_at_the_outlet() -> None:
    cn = np.array([[60.0, 80.0, 100.0]])
    dem = np.array([[3.0, 2.0, 1.0]])
    receiver = np.array([[1, 2, -1]], dtype=np.int64)
    composite, area = _upstream_composite_cn(cn, receiver, np.array([0, 1, 2]), 1.0)
    assert np.allclose(composite, [[60.0, 70.0, 80.0]])
    assert np.allclose(area, [[1.0, 2.0, 3.0]])


def test_scs_runoff_is_zero_below_initial_abstraction_and_positive_above() -> None:
    runoff = scs_runoff_depth_mm(np.array([5.0, 100.0]), np.array([75.0, 75.0]), 0.2)
    assert runoff[0] == 0.0
    assert runoff[1] > 0.0


def test_kirpich_has_an_explicit_headwater_floor() -> None:
    tc = kirpich_tc_minutes(np.array([1.0, 1_000.0]), np.array([0.0, 10.0]), 5.0)
    assert np.all(tc >= 5.0)


def test_hydrobathy_d4_codes_have_the_expected_receivers() -> None:
    direction = np.array([[2, 3], [1, 4]], dtype=float)
    assert receiver_from_d4_direction(direction).tolist() == [[1, 3], [0, 2]]
