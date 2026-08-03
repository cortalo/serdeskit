import numpy as np
import pytest

from serdeskit.pulse_response import PulseResponse


def test_nrz_amplitude_equals_cursor_value() -> None:
    """NRZ: levels=2, rlm=1.0 -> As = RLM * cursor_value / (2-1) = cursor_value."""
    fs = 1.0
    samples = np.array([0.0, 0.3, 0.8, 0.3, 0.0])
    cursor_ix = 2

    pr = PulseResponse(samples=samples, fs=fs, cursor_time=cursor_ix / fs, ui=1.0)

    assert pr.signal_amplitude(rlm=1.0, levels=2) == pytest.approx(0.8)


def test_pam4_amplitude_formula() -> None:
    """As = RLM * cursor_value / (levels - 1) (93A.1.6.c), hand-computed."""
    fs = 1.0
    samples = np.array([0.0, 0.3, 0.9, 0.3, 0.0])
    cursor_ix = 2

    pr = PulseResponse(samples=samples, fs=fs, cursor_time=cursor_ix / fs, ui=1.0)

    rlm, levels = 0.95, 4
    expected = rlm * 0.9 / (levels - 1)
    assert pr.signal_amplitude(rlm=rlm, levels=levels) == pytest.approx(expected)
