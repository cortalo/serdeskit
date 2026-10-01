import numpy as np
import pytest

from serdeskit.adapt import TxFfe


def test_with_weights_keeps_main() -> None:
    ffe = TxFfe(weights=np.array([0.0, 1.0, 0.0]), main=1)

    new = ffe.with_weights(np.array([-0.1, 0.6, -0.3]))

    assert new.main == 1
    np.testing.assert_array_equal(new.weights, [-0.1, 0.6, -0.3])
    np.testing.assert_array_equal(ffe.weights, [0.0, 1.0, 0.0])


def test_main_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        TxFfe(weights=np.zeros(2), main=2)
