import numpy as np

from serdeskit.adapt import RxDfe


def test_with_taps_returns_a_new_dfe() -> None:
    dfe = RxDfe(taps=np.zeros(2))

    new = dfe.with_taps(np.array([0.1, 0.02]))

    np.testing.assert_array_equal(new.taps, [0.1, 0.02])
    np.testing.assert_array_equal(dfe.taps, [0.0, 0.0])
