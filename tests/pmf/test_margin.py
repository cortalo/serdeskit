import numpy as np
import numpy.typing as npt
import pytest
import scipy.stats

from serdeskit.pmf import gaussian_pmf, noise_margin


def _grid(max_y: float, npts: int) -> npt.NDArray[np.float64]:
    return np.linspace(-max_y, max_y, npts)


def test_hand_verifiable_discrete_pmf() -> None:
    """A tiny 5-point PMF with an unambiguous crossing point, worked out by
    hand: cumsum = [.1, .3, .7, .9, 1.0]. At der0=0.25, the cumulative
    probability first reaches >= 0.25 at index 1 (cumsum=.3), y=-0.1 —
    so Ani = -(-0.1) = 0.1.
    """
    pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    y = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

    ani = noise_margin(pmf, y, der0=0.25)

    assert ani == pytest.approx(0.1)


def test_matches_scipy_normal_quantile() -> None:
    """For a Gaussian PMF, the point where the left-tail CDF reaches der0
    is, in the continuous limit, exactly `sigma * norm.ppf(der0)` (negative,
    since der0 < 0.5) — so Ani = -sigma*norm.ppf(der0) = sigma*norm.ppf(1-der0)
    by symmetry. scipy.stats.norm.ppf is an independent reference for this,
    not derived from PyChOpMarg or from our own gaussian_pmf/noise_margin
    code — composes two already-tested pieces (gaussian_pmf + noise_margin)
    and checks the result against real statistics theory.
    """
    sigma = 0.03
    variance = sigma**2
    der0 = 1e-4
    y = _grid(max_y=6 * sigma, npts=200_001)  # +/- 6 sigma, fine grid

    pmf = gaussian_pmf(variance, y)
    ani = noise_margin(pmf, y, der0)

    expected = sigma * scipy.stats.norm.ppf(1 - der0)
    assert ani == pytest.approx(expected, rel=1e-2)


def test_margin_is_positive_for_small_der0() -> None:
    sigma = 0.03
    y = _grid(max_y=6 * sigma, npts=200_001)
    pmf = gaussian_pmf(sigma**2, y)

    ani = noise_margin(pmf, y, der0=1e-4)

    assert ani > 0
