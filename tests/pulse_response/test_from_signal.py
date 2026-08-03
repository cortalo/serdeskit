import numpy as np
import pytest
from pychopmarg.utility.probability import loc_curs

from serdeskit.common.types import Signal
from serdeskit.pulse_response import PulseResponse


def test_matches_pychopmarg_golden_reference() -> None:
    """Cross-checks against PyChOpMarg's loc_curs (93A-25/93A-26) on a
    smooth synthetic pulse. Nonzero t0 exercises the time<->index
    conversion, not just an index-0-aligned special case.
    """
    fs = 100e9
    nspui = 10
    ui = nspui / fs
    n = 300
    peak_ix = 150
    t0 = 3e-9

    samples = np.exp(-(((np.arange(n) - peak_ix) / 12.0) ** 2))
    dfe1_max, dfe1_min = 0.5, -0.5

    expected_ix = loc_curs(samples, nspui, np.array([dfe1_max]), np.array([dfe1_min]))

    signal = Signal(samples=samples, fs=fs, t0=t0)
    pr = PulseResponse.from_signal(signal, ui, dfe1_max, dfe1_min)

    assert pr.cursor_index == expected_ix
    assert pr.cursor_time == pytest.approx(t0 + expected_ix / fs)
    assert pr.ui == ui
    np.testing.assert_array_equal(pr.samples, samples)
    assert pr.fs == fs
    assert pr.t0 == t0


def test_hand_verifiable_symmetric_triangle() -> None:
    """A perfectly symmetric triangular pulse, with the first DFE tap
    clipped to exactly 0 (dfe1_max=dfe1_min=0): the Muller-Mueller residual
    (93A-25) becomes |p[ix-nspui] - p[ix+nspui]|, which is exactly 0 only
    at the peak (unique, since the triangle is strictly monotonic on each
    side) — an unambiguous, hand-verifiable exact solution, independent of
    PyChOpMarg.
    """
    fs = 1.0
    nspui = 10
    ui = nspui / fs
    n = 201
    peak_ix = 100

    samples = np.maximum(0.0, 1.0 - np.abs(np.arange(n) - peak_ix) / 20.0)

    signal = Signal(samples=samples, fs=fs, t0=0.0)
    pr = PulseResponse.from_signal(signal, ui, dfe1_max=0.0, dfe1_min=0.0)

    assert pr.cursor_index == peak_ix
