import numpy as np
from pychopmarg.utility.general import mk_combs

from serdeskit.optimize import tx_tap_combinations


def test_matches_pychopmarg_golden_reference() -> None:
    """A small, fast-to-enumerate (min, max, step) config — the real
    IEEE_8023dj one has ~900k raw combinations before filtering, fine for
    an actual search but not for a test that should run in milliseconds.
    """
    bounds = [(-0.02, 0.0, 0.01), (0.0, 0.02, 0.01), (-0.05, 0.0, 0.025)]

    actual = tx_tap_combinations(bounds, min_cursor=0.9)

    expected = [v for v in mk_combs(bounds) if (1 - np.abs(v).sum()) >= 0.9]
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e)


def test_hand_verified_small_case() -> None:
    """Two taps, one with a single allowed value (min=max=0, step
    irrelevant), the other ranging over three values — worked out by
    hand which combinations leave at least 0.5 for the cursor.

    tap0 in {0.0} (fixed), tap1 in {-0.5, -0.25, 0.0} (step 0.25, chosen
    as an exact power-of-two fraction so np.arange's own floating-point
    endpoint handling can't add or drop a value — this is a "hand
    verified" test, so the values themselves shouldn't need arange's own
    behavior double-checked to trust it).
    sum(|taps|) for each: 0.5, 0.25, 0.0 -> cursor = 0.5, 0.75, 1.0.
    min_cursor=0.5 keeps all three; min_cursor=0.6 drops the first.
    """
    bounds = [(0.0, 0.0, 0.0), (-0.5, 0.0, 0.25)]

    all_kept = tx_tap_combinations(bounds, min_cursor=0.5)
    assert len(all_kept) == 3
    np.testing.assert_allclose(sorted(v[1] for v in all_kept), [-0.5, -0.25, 0.0])

    filtered = tx_tap_combinations(bounds, min_cursor=0.6)
    assert len(filtered) == 2
    np.testing.assert_allclose(sorted(v[1] for v in filtered), [-0.25, 0.0])


def test_zero_step_means_a_locked_zero_value() -> None:
    """A tap with step=0 contributes exactly one value: 0.0 — per
    mk_combs's own convention, *both* min and max are ignored when step
    is 0 (this is how an unused/nonexistent tap position is represented,
    not a single-point range at its min).
    """
    bounds = [(0.0, 0.0, 0.0), (0.03, 0.07, 0.0)]

    combos = tx_tap_combinations(bounds, min_cursor=0.5)

    assert len(combos) == 1
    np.testing.assert_allclose(combos[0], [0.0, 0.0])


def test_min_cursor_can_exclude_every_combination() -> None:
    bounds = [(-0.1, 0.1, 0.05)]

    combos = tx_tap_combinations(bounds, min_cursor=1.5)

    assert combos == []
