import numpy as np
import pytest
from pychopmarg.utility.probability import filt_pr_samps

from serdeskit.pmf import delta_pmf, filter_samples, gaussian_pmf, voltage_grid


def test_voltage_grid_matches_calc_noise_formula() -> None:
    """Note 1 of 93A.1.7.1. PyChOpMarg builds this inline in `calc_noise`
    rather than exposing it, so the formula is replicated here as the
    golden comparison (same situation as gaussian_pmf's density formula
    and SystemGrid.build's axes).

    Note the 1_000 cap: PyChOpMarg's `delta_pmf` has its own copy of this
    formula using 10_000 instead. That copy only runs when `y` is left
    unspecified, which this project never does, so `calc_noise`'s version
    is the one to match — but the two disagreeing is worth knowing about.
    """
    signal_amplitude = 0.0075

    grid = voltage_grid(signal_amplitude)

    ymax = 1.1 * signal_amplitude
    npts = 2 * min(int(ymax / 0.00001), 1_000) + 1
    expected = np.linspace(-ymax, ymax, npts)

    np.testing.assert_allclose(grid, expected)


def test_voltage_grid_satisfies_what_the_pmf_functions_require() -> None:
    """delta_pmf and gaussian_pmf each validate their `y` argument
    (uniformly spaced, exactly 0 V at the center index). Those guards
    exist for hand-constructed grids; this pins down that the grid this
    package generates never trips them.
    """
    grid = voltage_grid(0.0075)

    assert len(grid) % 2 == 1  # odd, so a point lands exactly on 0 V
    assert grid[len(grid) // 2] == pytest.approx(0.0, abs=1e-18)
    np.testing.assert_allclose(np.diff(grid), grid[1] - grid[0])

    # Neither call raises, which is the actual assertion here.
    delta_pmf(np.array([1e-4, -5e-5]), levels=4, y=grid)
    gaussian_pmf(variance=1e-8, y=grid)


def test_voltage_grid_hits_the_point_cap_for_large_amplitudes() -> None:
    """Below the cap the grid is 10 uV per step; above it, the span keeps
    growing with As while the point count stays put, so the step coarsens
    instead. Both sides of that transition are checked, since the cap is
    the part a naive reading of "10 uV steps" would miss.
    """
    small = voltage_grid(0.001)  # ymax = 1.1 mV -> 110 < 1000, uncapped
    assert len(small) == 2 * 110 + 1
    assert small[1] - small[0] == pytest.approx(1e-5)

    large = voltage_grid(0.05)  # ymax = 55 mV -> 5500 > 1000, capped
    assert len(large) == 2 * 1_000 + 1
    assert large[1] - large[0] > 1e-5  # step coarsened past 10 uV


def test_filter_samples_matches_pychopmarg_golden_reference() -> None:
    """Note 2 of 93A.1.7.1."""
    signal_amplitude = 0.0075
    samples = np.array([1e-2, 5e-6, -3e-3, 7.5e-6, 0.0, -1e-5, 2e-4])

    actual = filter_samples(samples, signal_amplitude)

    expected = filt_pr_samps(samples, As=signal_amplitude)
    np.testing.assert_allclose(actual, expected)


def test_filter_samples_keeps_order_and_drops_by_magnitude() -> None:
    """Hand-checked, independent of PyChOpMarg: threshold is 0.1% of
    As = 1.0, i.e. 1e-3, and sign is irrelevant — only magnitude.
    """
    samples = np.array([5e-3, -5e-4, -2e-3, 0.0, 1e-2])

    actual = filter_samples(samples, signal_amplitude=1.0)

    np.testing.assert_allclose(actual, [5e-3, -2e-3, 1e-2])
