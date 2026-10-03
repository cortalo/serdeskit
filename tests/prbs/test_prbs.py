import numpy as np
import pytest

from serdeskit.prbs import TAPS, Prbs


@pytest.mark.parametrize("order", [7, 9, 11, 15])
def test_maximal_length_period_and_balance(order: int) -> None:
    """A full period visits every nonzero state once: 2^(order-1) ones."""
    prbs = Prbs(order)
    bits = prbs.bits(2 * prbs.period)

    np.testing.assert_array_equal(bits[: prbs.period], bits[prbs.period :])
    assert bits[: prbs.period].sum() == 2 ** (order - 1)
    states = {tuple(bits[i : i + order]) for i in range(prbs.period)}
    assert len(states) == prbs.period


@pytest.mark.parametrize("order", sorted(TAPS))
def test_any_order_consecutive_bits_determine_the_rest(order: int) -> None:
    prbs = Prbs(order)
    bits = prbs.bits(500)

    np.testing.assert_array_equal(prbs.continue_from(bits[100 : 100 + order], 200), bits[100 + order : 300 + order])


def test_symbols_map_1_to_plus_1() -> None:
    prbs = Prbs(7)

    np.testing.assert_array_equal(prbs.symbols(50), 2.0 * prbs.bits(50) - 1.0)


def test_unknown_order_raises() -> None:
    with pytest.raises(ValueError):
        Prbs(8)
