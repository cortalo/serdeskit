import numpy as np
import pytest
from pychopmarg.utility.sparams import sCshunt, sLseries

from serdeskit.package import series_inductor, shunt_capacitor


@pytest.mark.usefixtures("exact_pi_sparams")
def test_shunt_capacitor_matches_pychopmarg_golden_reference() -> None:
    """(93A-8)."""
    freqs = np.linspace(0, 50e9, 501)
    capacitance = 3e-14  # 30 fF

    actual = shunt_capacitor(freqs, capacitance, r0=50.0)

    expected = sCshunt(freqs, capacitance, r0=50.0)
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-14)


def test_shunt_capacitor_default_r0_is_50() -> None:
    freqs = np.linspace(0, 50e9, 11)
    capacitance = 3e-14

    default = shunt_capacitor(freqs, capacitance)
    explicit = shunt_capacitor(freqs, capacitance, r0=50.0)

    np.testing.assert_allclose(default.s, explicit.s)


def test_shunt_capacitor_hand_verified_at_dc() -> None:
    """At f=0, a capacitor is an open circuit: no series impedance seen
    through it, so S21=1 (pass-through), S11=0 (no reflection) —
    independent of the capacitance value.
    """
    freqs = np.array([0.0])
    capacitance = 3e-14

    result = shunt_capacitor(freqs, capacitance, r0=50.0)

    np.testing.assert_allclose(result.s[0, 0, 0], 0.0, atol=1e-15)  # S11
    np.testing.assert_allclose(result.s[0, 1, 0], 1.0, atol=1e-15)  # S21


@pytest.mark.usefixtures("exact_pi_sparams")
def test_series_inductor_matches_pychopmarg_golden_reference() -> None:
    freqs = np.linspace(0, 50e9, 501)
    inductance = 1e-10  # 100 pH

    actual = series_inductor(freqs, inductance, r0=50.0)

    expected = sLseries(freqs, inductance, r0=50.0)
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-14)


def test_series_inductor_default_r0_is_50() -> None:
    freqs = np.linspace(0, 50e9, 11)
    inductance = 1e-10

    default = series_inductor(freqs, inductance)
    explicit = series_inductor(freqs, inductance, r0=50.0)

    np.testing.assert_allclose(default.s, explicit.s)


def test_series_inductor_hand_verified_at_dc() -> None:
    """At f=0, an inductor is a short circuit: no impedance in series, so
    S21=1 (pass-through), S11=0 (no reflection) — independent of the
    inductance value.
    """
    freqs = np.array([0.0])
    inductance = 1e-10

    result = series_inductor(freqs, inductance, r0=50.0)

    np.testing.assert_allclose(result.s[0, 0, 0], 0.0, atol=1e-15)  # S11
    np.testing.assert_allclose(result.s[0, 1, 0], 1.0, atol=1e-15)  # S21


def test_shunt_capacitor_and_series_inductor_are_reciprocal_and_symmetric() -> None:
    """Both are passive, reciprocal, symmetric two-ports: S12=S21 and
    S11=S22 at every frequency, independent of the golden reference.
    """
    freqs = np.linspace(0, 50e9, 51)

    cap = shunt_capacitor(freqs, 3e-14, r0=50.0)
    ind = series_inductor(freqs, 1e-10, r0=50.0)

    for network in (cap, ind):
        np.testing.assert_allclose(network.s[:, 0, 1], network.s[:, 1, 0])
        np.testing.assert_allclose(network.s[:, 0, 0], network.s[:, 1, 1])
