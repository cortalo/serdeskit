import numpy as np

from serdeskit.tx_filter import TxRisetimeFilter


def test_matches_formula() -> None:
    """(93A-46). PyChOpMarg computes this inline in COM.__init__ rather
    than exposing it as a standalone function (same situation as
    gaussian_pmf's density formula) — replicated directly from that
    source instead, in SI units (no GHz/ns conversion factors needed:
    f*risetime is dimensionless regardless of unit choice).
    """
    risetime = 10e-12  # 10 ps
    freqs = np.linspace(0, 50e9, 501)

    filt = TxRisetimeFilter(risetime=risetime)
    actual = filt.transfer_function(freqs)

    expected = np.exp(-2 * (np.pi * freqs * risetime / 1.6832) ** 2)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_dc_gain_is_unity() -> None:
    """At f=0, exp(0) = 1 exactly — independent hand check."""
    filt = TxRisetimeFilter(risetime=10e-12)

    actual = filt.transfer_function(np.array([0.0]))

    np.testing.assert_allclose(actual, [1.0], atol=1e-12)
