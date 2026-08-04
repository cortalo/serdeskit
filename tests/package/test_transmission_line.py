import numpy as np
import pytest
from pychopmarg.utility.sparams import sPkgTline

from serdeskit.package import package_transmission_line

R0 = 50.0
A1 = 0.00089
A2 = 0.0002
TAU = 0.006141
GAMMA0 = 0.0005
SEGMENTS = [(87.5, 33.0), (92.5, 1.8)]  # (characteristic impedance Ohms, length mm)


@pytest.mark.usefixtures("exact_pi_sparams")
def test_matches_pychopmarg_golden_reference() -> None:
    """(93A-9:14): a lossy package transmission line, its segments
    cascaded together. Excludes f=0 — sPkgTline's own gamma(f) special-
    cases it (avoiding log(0) in gamma2), covered separately below.
    """
    freqs = np.linspace(1e8, 50e9, 500)

    actual = package_transmission_line(freqs, R0, A1, A2, TAU, GAMMA0, SEGMENTS)

    expected = sPkgTline(freqs, R0, A1, A2, TAU, GAMMA0, SEGMENTS)
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-12)


@pytest.mark.usefixtures("exact_pi_sparams")
def test_matches_pychopmarg_golden_reference_including_dc() -> None:
    freqs = np.linspace(0, 50e9, 501)

    actual = package_transmission_line(freqs, R0, A1, A2, TAU, GAMMA0, SEGMENTS)

    expected = sPkgTline(freqs, R0, A1, A2, TAU, GAMMA0, SEGMENTS)
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-12)


def test_single_segment_is_reciprocal_and_symmetric() -> None:
    """A single uniform lossy segment sees the same impedance mismatch at
    both ends, so unlike die_ladder_segment's asymmetric cascade of two
    *different* elements, this one is genuinely symmetric — S11=S22 —
    checked independently of the golden reference.
    """
    freqs = np.linspace(0, 50e9, 51)

    line = package_transmission_line(freqs, R0, A1, A2, TAU, GAMMA0, [(87.5, 33.0)])

    np.testing.assert_allclose(line.s[:, 0, 1], line.s[:, 1, 0])
    np.testing.assert_allclose(line.s[:, 0, 0], line.s[:, 1, 1])


def test_zero_length_segment_is_transparent() -> None:
    """A segment of length 0 has nothing to reflect off or attenuate
    through: S21=1, S11=0, independent of its characteristic impedance —
    hand-verified via the formula's own e=exp(-gamma*2*0)=1 collapse
    (s11 = rho*(1-1)/(...) = 0, s21 = (1-rho^2)*1/(1-rho^2*1) = 1).
    """
    freqs = np.linspace(1e8, 50e9, 51)

    line = package_transmission_line(freqs, R0, A1, A2, TAU, GAMMA0, [(87.5, 0.0)])

    np.testing.assert_allclose(line.s[:, 0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(line.s[:, 1, 0], 1.0, atol=1e-12)


def test_matched_impedance_has_no_reflection() -> None:
    """zc = 2*r0 makes rho=0 (the segment's own reference impedance
    exactly matches the line it's inserted into) — no reflection at
    either end, though there's still propagation loss/delay through it.
    """
    freqs = np.linspace(1e8, 50e9, 51)

    line = package_transmission_line(freqs, R0, A1, A2, TAU, GAMMA0, [(2 * R0, 10.0)])

    np.testing.assert_allclose(line.s[:, 0, 0], 0.0, atol=1e-12)
