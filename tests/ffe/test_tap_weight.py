import numpy as np
import pytest

from serdeskit.ffe import TapWeightFfe


def test_cursor_weight_hand_computed() -> None:
    """(93A-21 normalization): cursor_weight = 1 - sum(|tap_weights|)."""
    tap_weights = np.array([0.1, -0.2, 0.05])
    ffe = TapWeightFfe(tap_weights=tap_weights, n_post=1, tap_delay=1e-11)

    expected = 1.0 - (0.1 + 0.2 + 0.05)
    assert ffe.cursor_weight == pytest.approx(expected)


def test_cursor_weight_is_one_when_all_taps_zero() -> None:
    tap_weights = np.array([0.0, 0.0, 0.0])
    ffe = TapWeightFfe(tap_weights=tap_weights, n_post=1, tap_delay=1e-11)

    assert ffe.cursor_weight == pytest.approx(1.0)


def test_cursor_weight_with_no_pre_post_taps() -> None:
    """An FFE with zero pre/post-cursor taps (empty array) is just a
    straight-through cursor tap of weight 1.
    """
    tap_weights = np.array([])
    ffe = TapWeightFfe(tap_weights=tap_weights, n_post=0, tap_delay=1e-11)

    assert ffe.cursor_weight == pytest.approx(1.0)
