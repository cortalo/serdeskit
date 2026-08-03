import numpy as np
import numpy.typing as npt
import pytest

from serdeskit.pmf import combine_pmfs, delta_pmf


def _grid(max_y: float, npts: int) -> npt.NDArray[np.float64]:
    return np.linspace(-max_y, max_y, npts)


def test_matches_delta_pmf_folding_h_samples_together() -> None:
    """Convolution is associative: folding h1 then h2 into one `delta_pmf`
    call (93A-40) must equal computing `delta_pmf([h1])` and
    `delta_pmf([h2])` separately and then combining them — the same
    underlying math, reached two different ways. Cross-validates
    `combine_pmfs` against `delta_pmf` (already golden-tested against
    PyChOpMarg) without needing a separate external reference — PyChOpMarg
    doesn't expose this convolution step as a standalone function either
    (inline in `calc_noise`, like `gaussian_pmf`'s formula was).
    """
    levels = 2
    y = _grid(max_y=0.15, npts=2001)
    h1, h2 = 0.05, -0.02

    expected = delta_pmf(np.array([h1, h2]), levels, y)

    pmf1 = delta_pmf(np.array([h1]), levels, y)
    pmf2 = delta_pmf(np.array([h2]), levels, y)
    actual = combine_pmfs(pmf1, pmf2)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_output_is_normalized() -> None:
    levels = 2
    y = _grid(max_y=0.15, npts=2001)
    pmf1 = delta_pmf(np.array([0.05]), levels, y)
    pmf2 = delta_pmf(np.array([-0.02]), levels, y)

    combined = combine_pmfs(pmf1, pmf2)

    assert combined.sum() == pytest.approx(1.0)


def test_hand_verifiable_combination() -> None:
    """Combine two hand-picked, independent distributions on a tiny 7-point
    grid (center index 3) and check every resulting value against a
    manually worked-out convolution — independent of `delta_pmf`/
    `np.convolve`, so it can't share a mistake with either.

    a: +/-2 grid steps, 50/50 (indices 1 and 5).
    b: -1/0/+1 grid steps, weights .25/.5/.25 (indices 2, 3, 4).
    a+b by hand: -3(.125), -2(.25), -1(.125), +1(.125), +2(.25), +3(.125).
    """
    npts = 7
    a = np.zeros(npts)
    a[1] = 0.5
    a[5] = 0.5

    b = np.zeros(npts)
    b[2] = 0.25
    b[3] = 0.5
    b[4] = 0.25

    combined = combine_pmfs(a, b)

    expected = np.array([0.125, 0.25, 0.125, 0.0, 0.125, 0.25, 0.125])
    np.testing.assert_allclose(combined, expected, atol=1e-12)
