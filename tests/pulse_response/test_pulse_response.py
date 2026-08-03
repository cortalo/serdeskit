import numpy as np
from pychopmarg.utility.probability import calc_hJ

from serdeskit.pulse_response import PulseResponse


def test_matches_pychopmarg_golden_reference() -> None:
    """Cross-checks against PyChOpMarg's calc_hJ (93A-28) on a synthetic,
    decaying pulse response — the decay is steep enough that some
    post-cursor UI samples fall below the 0.1% threshold and get filtered
    out before reaching the end of the array, exercising that filtering
    behavior as part of the same comparison.
    """
    fs = 100e9  # 100 GS/s
    nspui = 10
    ui = nspui / fs
    n = 300
    cursor_ix = 100
    cursor_time = cursor_ix / fs

    samples = np.exp(-np.abs(np.arange(n) - cursor_ix) / 8.0)
    signal_amplitude = float(samples[cursor_ix])

    pr = PulseResponse(samples=samples, fs=fs, cursor_time=cursor_time, ui=ui)

    expected = calc_hJ(samples, As=signal_amplitude, cursor_ix=cursor_ix, nspui=nspui)
    actual = pr.local_slopes(signal_amplitude)

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert len(actual) < (n - cursor_ix) // nspui  # confirms filtering actually excluded some


def test_hand_verifiable_triangular_pulse() -> None:
    """A simple triangular pulse, one sample per UI, hand-worked by
    central difference (93A-28) — independent of PyChOpMarg.

    samples = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], cursor at index 5 (peak).
    Valid indices (cursor onward, step 1, up to len-1): 5, 6, 7, 8, 9.
    slope[k] = (samples[k+1] - samples[k-1]) / (2 / nspui), nspui=1:
      k=5: (4-4)/2 = 0
      k=6: (3-5)/2 = -1
      k=7: (2-4)/2 = -1
      k=8: (1-3)/2 = -1
      k=9: (0-2)/2 = -1
    """
    fs = 1.0
    nspui = 1
    ui = nspui / fs
    cursor_ix = 5
    cursor_time = cursor_ix / fs

    samples = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=np.float64)
    signal_amplitude = float(samples[cursor_ix])

    pr = PulseResponse(samples=samples, fs=fs, cursor_time=cursor_time, ui=ui)

    actual = pr.local_slopes(signal_amplitude)

    expected = np.array([0.0, -1.0, -1.0, -1.0, -1.0])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
