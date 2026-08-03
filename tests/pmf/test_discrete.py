import numpy as np
import numpy.typing as npt
import pytest
from pychopmarg.utility.probability import delta_pmf as pychopmarg_delta_pmf

from serdeskit.pmf import delta_pmf


def _grid(max_y: float, npts: int) -> npt.NDArray[np.float64]:
    return np.linspace(-max_y, max_y, npts)


@pytest.mark.parametrize("levels", [2, 4])
def test_matches_pychopmarg_golden_reference(levels: int) -> None:
    """PyChOpMarg's own `delta_pmf`, given an explicit `y`, skips its
    internal auto-filtering (see its source: the `y is None` branch is the
    only one that calls `filt_pr_samps`) — so passing the same pre-decided
    grid to both sides is an apples-to-apples comparison, not something
    that happens to line up by luck.
    """
    h_samples = np.array([0.05, -0.02, 0.01, 0.008, -0.003])
    y = _grid(max_y=0.15, npts=2001)

    _, expected = pychopmarg_delta_pmf(h_samples, L=levels, y=y)

    actual = delta_pmf(h_samples, levels, y)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_output_is_normalized() -> None:
    h_samples = np.array([0.05, -0.02, 0.01])
    y = _grid(max_y=0.1, npts=1001)

    pmf = delta_pmf(h_samples, levels=2, y=y)

    assert pmf.sum() == pytest.approx(1.0)


def test_single_ui_nrz_is_two_equal_spikes() -> None:
    """Simplest case, hand-verifiable against (93A-39) directly, independent
    of the golden reference: one UI, NRZ (L=2) — the transmitted symbol at
    that UI is +1 or -1 with equal probability, so the voltage it
    contributes is +h or -h with equal probability. No convolution (93A-40)
    needed since there's only one sample.
    """
    h_samples = np.array([0.05])
    y = _grid(max_y=0.1, npts=2001)

    pmf = delta_pmf(h_samples, levels=2, y=y)

    ix_pos = int(np.argmin(np.abs(y - 0.05)))
    ix_neg = int(np.argmin(np.abs(y + 0.05)))
    assert pmf[ix_pos] == pytest.approx(0.5, abs=1e-9)
    assert pmf[ix_neg] == pytest.approx(0.5, abs=1e-9)
    assert pmf.sum() == pytest.approx(1.0)


def test_grid_too_narrow_raises() -> None:
    """`delta_pmf` builds up the result via `np.roll`, which wraps rather
    than drops values that shift past an array's edge. If `y` isn't wide
    enough to hold the worst-case total displacement (every sample's
    contribution landing at its most extreme value, same sign), that
    wraparound silently aliases high-voltage mass onto the low-voltage side
    (or vice versa) instead of erroring — this must be rejected up front
    instead of returning a quietly-wrong PMF.
    """
    h_samples = np.array([0.05, 0.05, 0.05])  # worst case: 0.15V, same sign
    y = _grid(max_y=0.1, npts=21)  # only wide enough for 0.1V

    with pytest.raises(ValueError):
        delta_pmf(h_samples, levels=2, y=y)


def test_grid_exactly_wide_enough_does_not_raise() -> None:
    h_samples = np.array([0.05, 0.05])  # worst case: exactly 0.1V
    y = _grid(max_y=0.1, npts=21)

    delta_pmf(h_samples, levels=2, y=y)  # must not raise


def test_non_uniform_grid_raises() -> None:
    """The shift-per-sample math (`level_values * h / ystep`) assumes a
    single `ystep` applies across the whole array — if `y`'s spacing isn't
    uniform, an index shift computed from `ystep` doesn't correspond to the
    same voltage step everywhere else in the array. Center point (index 3)
    is deliberately kept at exactly 0 here, so this test isolates the
    non-uniform-spacing case from the separate "center isn't 0" case below.
    """
    y = np.array([-0.1, -0.05, -0.02, 0.0, 0.03, 0.07, 0.12])
    h_samples = np.array([0.01])

    with pytest.raises(ValueError):
        delta_pmf(h_samples, levels=2, y=y)


def test_grid_center_not_zero_raises() -> None:
    """The initial delta is placed at `pmf[len(y) // 2]`, assuming that
    index represents y=0 (the running total before any sample is folded
    in). Uniformly spaced but off-center from 0 must still be rejected.
    """
    y = np.linspace(-0.05, 0.2, 21)  # uniform spacing, but y[10] = 0.075, not 0
    h_samples = np.array([0.01])

    with pytest.raises(ValueError):
        delta_pmf(h_samples, levels=2, y=y)
