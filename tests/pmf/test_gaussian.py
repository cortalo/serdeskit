import numpy as np
import numpy.typing as npt
import pytest
import scipy.stats

from serdeskit.pmf import gaussian_pmf


def _grid(max_y: float, npts: int) -> npt.NDArray[np.float64]:
    return np.linspace(-max_y, max_y, npts)


def test_matches_scipy_normal_pdf() -> None:
    """(93A-42) is exactly a Gaussian PDF, discretized to a PMF by
    multiplying by the grid step. PyChOpMarg doesn't expose this as a
    standalone function (it's inline in `calc_noise`), so `scipy.stats.norm`
    — an independently authored reference for the same density formula,
    already a project dependency — serves as the golden reference instead.
    """
    variance = 0.0009  # sigma = 0.03V
    y = _grid(max_y=0.3, npts=6001)  # +/- 10 sigma
    ystep = y[1] - y[0]

    expected = scipy.stats.norm.pdf(y, loc=0.0, scale=np.sqrt(variance)) * ystep

    actual = gaussian_pmf(variance, y)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_output_is_approximately_normalized_for_wide_grid() -> None:
    """A Gaussian's tails are infinite — any finite grid truncates some
    mass. +/- 10 sigma should capture all but a negligible amount, unlike
    `delta_pmf`'s genuinely-bounded-support distributions, which must sum to
    exactly 1 regardless of grid width.
    """
    variance = 0.0009
    y = _grid(max_y=0.3, npts=6001)  # +/- 10 sigma

    pmf = gaussian_pmf(variance, y)

    assert pmf.sum() == pytest.approx(1.0, abs=1e-9)


def test_peak_is_at_grid_center() -> None:
    variance = 0.0009
    y = _grid(max_y=0.3, npts=6001)

    pmf = gaussian_pmf(variance, y)

    assert int(np.argmax(pmf)) == len(y) // 2


@pytest.mark.parametrize("variance", [0.0, -0.001])
def test_non_positive_variance_raises(variance: float) -> None:
    """variance=0 divides by zero; variance<0 takes sqrt of a negative
    number — both silently produce a NaN-filled array (only a
    RuntimeWarning, no exception) rather than failing at the actual source
    of the bad input.
    """
    y = _grid(max_y=0.3, npts=6001)

    with pytest.raises(ValueError):
        gaussian_pmf(variance, y)


def test_non_uniform_grid_raises() -> None:
    """Only `y[1] - y[0]` is used as the grid step to convert density to
    mass everywhere — a non-uniform `y` silently applies the wrong bin
    width wherever the spacing differs, corrupting the total mass without
    raising (same failure mode already guarded against in `delta_pmf`).
    """
    y = np.concatenate([np.arange(-0.1, 0.0, 0.01), np.arange(0.0, 0.1001, 0.02)])
    variance = 0.0009

    with pytest.raises(ValueError):
        gaussian_pmf(variance, y)
