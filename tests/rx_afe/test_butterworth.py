import numpy as np

from serdeskit.rx_afe import RxAfeButterworth


def test_matches_formula() -> None:
    """(93A-20). PyChOpMarg computes this inline in COM.__init__ rather
    than exposing it as a standalone function — replicated directly from
    that source instead (same situation as TxRisetimeFilter/93A-46).
    """
    cutoff_freq = 20e9
    freqs = np.linspace(0, 50e9, 501)

    afe = RxAfeButterworth(cutoff_freq=cutoff_freq)
    actual = afe.transfer_function(freqs)

    f_n = freqs / cutoff_freq
    expected = 1 / (1 - 3.414214 * f_n**2 + f_n**4 + 2.613126j * (f_n - f_n**3))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_dc_gain_is_unity() -> None:
    """At f=0 (f_n=0): 1/(1-0+0+0) = 1 exactly — independent hand check."""
    afe = RxAfeButterworth(cutoff_freq=20e9)

    actual = afe.transfer_function(np.array([0.0]))

    np.testing.assert_allclose(actual, [1.0], atol=1e-12)
