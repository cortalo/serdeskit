import numpy as np
import pytest
import skrf
from pychopmarg.utility.sparams import sdd_21

from serdeskit.package import die_model, differential_pair

R0 = 50.0


@pytest.mark.usefixtures("exact_pi_sparams")
def test_matches_pychopmarg_golden_reference() -> None:
    """Two identical, uncoupled copies of a conductor model (die_model's
    output, reused here as an arbitrary already-golden-tested two-port —
    what it represents physically doesn't matter for this test) combined
    into the differential pair they'd form — matches PyChOpMarg's own
    concat_ports([SE, SE], port_order='first') + sdd_21 pipeline.
    """
    freqs = np.linspace(0, 50e9, 501)
    conductor = die_model(
        freqs, R0, capacitances=[4e-5 / 1e9, 9e-5 / 1e9], inductances=[0.13 / 1e9, 0.15 / 1e9],
        bump_capacitance=3e-5 / 1e9,
    )

    actual = differential_pair(conductor)

    four_port = skrf.network.concat_ports([conductor, conductor], port_order="first")
    expected = sdd_21(four_port)
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-12)


def test_matches_the_single_conductor_when_uncoupled() -> None:
    """Two identical, uncoupled legs: the differential-mode response is
    exactly what either leg does on its own (no coupling means the
    differential and common paths don't interact, and both legs being
    identical means "the pair's response" and "one leg's response" are
    the same signal, seen twice) — an independently-derivable fact this
    doesn't share machinery with concat_ports/sdd_21 to check.
    """
    freqs = np.linspace(0, 50e9, 51)
    conductor = die_model(
        freqs, R0, capacitances=[4e-5 / 1e9], inductances=[0.13 / 1e9], bump_capacitance=3e-5 / 1e9,
    )

    pair = differential_pair(conductor)

    np.testing.assert_allclose(pair.s, conductor.s, atol=1e-10)
