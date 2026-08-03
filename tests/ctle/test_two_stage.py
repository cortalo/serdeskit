import numpy as np
from pychopmarg.utility.filter import calc_Hctle

from serdeskit.ctle import TwoStageCtle


def test_matches_pychopmarg_golden_reference() -> None:
    """(93A-22)."""
    zero_freq = 4e9
    pole1_freq = 8e9
    pole2_freq = 20e9
    shelf_freq = 2e9
    dc_gain_db = -6.0
    shelf_gain_db = -2.0
    freqs = np.linspace(0, 50e9, 501)

    ctle = TwoStageCtle(
        zero_freq=zero_freq,
        pole1_freq=pole1_freq,
        pole2_freq=pole2_freq,
        shelf_freq=shelf_freq,
        dc_gain_db=dc_gain_db,
        shelf_gain_db=shelf_gain_db,
    )

    expected = calc_Hctle(
        freqs, fz=zero_freq, fp1=pole1_freq, fp2=pole2_freq,
        fLF=shelf_freq, gDC=dc_gain_db, gDC2=shelf_gain_db,
    )
    actual = ctle.transfer_function(freqs)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_dc_gain_hand_verified() -> None:
    """At f=0, every j*f/f_x term vanishes: H(0) = g1*g2 exactly, where
    g1/g2 are the *linear* (not dB) per-stage d.c. gains — independent of
    calc_Hctle, straight from the formula's own definition.
    """
    dc_gain_db = -6.0
    shelf_gain_db = -3.0
    ctle = TwoStageCtle(
        zero_freq=4e9,
        pole1_freq=8e9,
        pole2_freq=20e9,
        shelf_freq=2e9,
        dc_gain_db=dc_gain_db,
        shelf_gain_db=shelf_gain_db,
    )

    g1 = 10 ** (dc_gain_db / 20)
    g2 = 10 ** (shelf_gain_db / 20)

    actual = ctle.transfer_function(np.array([0.0]))

    np.testing.assert_allclose(actual, [g1 * g2], atol=1e-12)
