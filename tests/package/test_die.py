import numpy as np
import pytest
import skrf
from pychopmarg.utility.sparams import sCshunt, sDieLadderSegment

from serdeskit.package import die_model

R0 = 50.0
CAPACITANCES = [4.0e-5 / 1e9, 9.0e-5 / 1e9, 1.1e-4 / 1e9]  # C_d, nF -> F
INDUCTANCES = [0.13 / 1e9, 0.15 / 1e9, 0.14 / 1e9]  # L_s, nH -> H
BUMP_CAPACITANCE = 3.0e-5 / 1e9  # C_b, nF -> F


def _expected(
    freqs: np.ndarray, r0: float, capacitances: list[float], inductances: list[float], bump: float, flip: bool
) -> skrf.Network:
    """PyChOpMarg's own COM.sDie, reproduced directly from its rung-by-rung
    formula rather than via a full COM instance — sDie itself only needs
    freqs/r0/C_d/L_s/C_b, none of which require constructing a whole COM
    (channels, package search grids, ...) just to test this one method.
    """
    r0s = [r0] * len(capacitances)
    result = skrf.network.cascade_list(
        [sDieLadderSegment(freqs, trip) for trip in zip(r0s, capacitances, inductances)]
    )
    result = result ** sCshunt(freqs, bump, r0=r0)
    if flip:
        result = result.copy()
        result.flip()
    return result


@pytest.mark.usefixtures("exact_pi_sparams")
def test_matches_pychopmarg_golden_reference_tx_side() -> None:
    """isRx=False: on-die parasitic ladder (one rung per C_d[i]/L_s[i]
    pair) plus a final bump capacitance C_b, no port flip.
    """
    freqs = np.linspace(0, 50e9, 501)

    actual = die_model(freqs, R0, CAPACITANCES, INDUCTANCES, BUMP_CAPACITANCE, flip=False)

    expected = _expected(freqs, R0, CAPACITANCES, INDUCTANCES, BUMP_CAPACITANCE, flip=False)
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-12)


@pytest.mark.usefixtures("exact_pi_sparams")
def test_matches_pychopmarg_golden_reference_rx_side() -> None:
    """isRx=True: same network, but with its ports swapped (flip=True)."""
    freqs = np.linspace(0, 50e9, 501)

    actual = die_model(freqs, R0, CAPACITANCES, INDUCTANCES, BUMP_CAPACITANCE, flip=True)

    expected = _expected(freqs, R0, CAPACITANCES, INDUCTANCES, BUMP_CAPACITANCE, flip=True)
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-12)


def test_flip_swaps_s11_and_s22() -> None:
    """Independent of the golden reference: flipping a two-port swaps
    which port is "1" and which is "2", so S11<->S22 and S12<->S21.
    """
    freqs = np.linspace(0, 50e9, 51)

    unflipped = die_model(freqs, R0, CAPACITANCES, INDUCTANCES, BUMP_CAPACITANCE, flip=False)
    flipped = die_model(freqs, R0, CAPACITANCES, INDUCTANCES, BUMP_CAPACITANCE, flip=True)

    np.testing.assert_allclose(flipped.s[:, 0, 0], unflipped.s[:, 1, 1])
    np.testing.assert_allclose(flipped.s[:, 1, 1], unflipped.s[:, 0, 0])
    np.testing.assert_allclose(flipped.s[:, 0, 1], unflipped.s[:, 1, 0])
    np.testing.assert_allclose(flipped.s[:, 1, 0], unflipped.s[:, 0, 1])


def test_does_not_mutate_its_inputs() -> None:
    """flip=True must not be implemented via Network.flip() (in-place) on
    a network some other caller might still hold a reference to —
    checked by building the same unflipped network again afterward and
    confirming it's unaffected.
    """
    freqs = np.linspace(0, 50e9, 51)

    before = die_model(freqs, R0, CAPACITANCES, INDUCTANCES, BUMP_CAPACITANCE, flip=False)
    die_model(freqs, R0, CAPACITANCES, INDUCTANCES, BUMP_CAPACITANCE, flip=True)
    after = die_model(freqs, R0, CAPACITANCES, INDUCTANCES, BUMP_CAPACITANCE, flip=False)

    np.testing.assert_allclose(before.s, after.s)
