import numpy as np
import pytest
from pychopmarg.utility.filter import calc_Hffe

from serdeskit.ffe import TapWeightFfe


@pytest.mark.usefixtures("exact_pi")
def test_matches_pychopmarg_golden_reference() -> None:
    """(93A-21). n_post=2 (a normal case, at least one post-cursor tap) is
    where PyChOpMarg's calc_Hffe places the cursor tap correctly (see the
    n_post=0 test below for why that case is deliberately *not* compared
    against it).

    calc_Hffe references delay 0 at the *first* (most-precursor) tap;
    TapWeightFfe references delay 0 at the *cursor* tap instead (matching
    MATLAB COM3.70's own FFE — see tests/ffe/test_tap_weight_vs_matlab.py).
    That's a pure linear phase, exp(-j*2*pi*f*T*n_pre), compensated for
    below before comparing.
    """
    tap_weights = np.array([0.05, -0.1, 0.03])  # 1 pre-cursor, 2 post-cursor
    n_post = 2
    n_pre = len(tap_weights) - n_post
    tap_delay = 1e-11
    freqs = np.linspace(0, 50e9, 501)

    ffe = TapWeightFfe(tap_weights=tap_weights, n_post=n_post, tap_delay=tap_delay)

    expected = calc_Hffe(freqs, td=tap_delay, tap_weights=tap_weights, n_post=n_post, hasCurs=False)
    actual = ffe.transfer_function(freqs)
    delay_factor = np.exp(-1j * 2 * np.pi * freqs * n_pre * tap_delay)

    # `exact_pi` gives calc_Hffe full-precision TWOPI, so this can assert
    # exact agreement rather than tolerating the ~1e-6 phase drift
    # PyChOpMarg's truncated PI constant would otherwise introduce.
    np.testing.assert_allclose(actual * delay_factor, expected, atol=1e-12)


def test_n_post_zero_places_cursor_last_hand_verified() -> None:
    """PyChOpMarg's calc_Hffe inserts the cursor tap via
    `np.insert(bs, -n_post, b0)` — when n_post=0, `-n_post` becomes `-0`,
    which Python treats as `0`, inserting the cursor at the *front* instead
    of the (physically correct) end. Deliberately not compared against
    that golden reference here; hand-verified against the tap ordering
    that should actually apply: all given taps are pre-cursor, so cursor
    goes last, at delay 0 (TapWeightFfe references delay 0 at the cursor
    tap — see tests/ffe/test_tap_weight_vs_matlab.py), with the pre-cursor
    taps at delays -3T, -2T, -1T.
    """
    tap_weights = np.array([0.1, 0.2, 0.3])
    n_post = 0
    tap_delay = 1e-11
    freqs = np.array([1e9])

    ffe = TapWeightFfe(tap_weights=tap_weights, n_post=n_post, tap_delay=tap_delay)
    cursor_weight = 1.0 - (0.1 + 0.2 + 0.3)

    bs = np.array([0.1, 0.2, 0.3, cursor_weight])  # cursor last, at index 3
    ns = np.arange(len(bs)) - 3  # cursor-referenced: cursor (index 3) at delay 0
    expected = bs @ np.exp(np.outer(ns, -1j * 2 * np.pi * tap_delay * freqs))

    actual = ffe.transfer_function(freqs)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_cursor_only_is_flat_response() -> None:
    """No pre/post-cursor taps at all: a single term (n=0), so H(f) is a
    frequency-independent constant equal to cursor_weight (=1.0, since
    there are no other taps to subtract from it).
    """
    tap_weights = np.array([])
    ffe = TapWeightFfe(tap_weights=tap_weights, n_post=0, tap_delay=1e-11)

    freqs = np.array([0.0, 1e9, 25e9, 50e9])
    actual = ffe.transfer_function(freqs)

    np.testing.assert_allclose(actual, np.ones_like(freqs, dtype=complex), atol=1e-12)
