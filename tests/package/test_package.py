import numpy as np
import pytest
import skrf
from pychopmarg.utility.sparams import sCshunt, sdd_21, sDieLadderSegment, sPkgTline

from serdeskit.package import Package

R0 = 50.0
DIE_CAPACITANCES = [4.0e-5 / 1e9, 9.0e-5 / 1e9, 1.1e-4 / 1e9]
DIE_INDUCTANCES = [0.13 / 1e9, 0.15 / 1e9, 0.14 / 1e9]
BUMP_CAPACITANCE = 3.0e-5 / 1e9
TLINE_A1 = 0.00089
TLINE_A2 = 0.0002
TLINE_TAU = 0.006141
TLINE_GAMMA0 = 0.0005
TLINE_SEGMENTS = [(87.5, 33.0), (92.5, 1.8)]
PAD_CAPACITANCE_TX = 4.0e-5 / 1e9
PAD_CAPACITANCE_RX = 4.0e-5 / 1e9


def _package(is_rx: bool, pad_capacitance: float) -> Package:
    return Package(
        r0=R0,
        die_capacitances=DIE_CAPACITANCES,
        die_inductances=DIE_INDUCTANCES,
        bump_capacitance=BUMP_CAPACITANCE,
        tline_a1=TLINE_A1,
        tline_a2=TLINE_A2,
        tline_tau=TLINE_TAU,
        tline_gamma0=TLINE_GAMMA0,
        tline_segments=TLINE_SEGMENTS,
        pad_capacitance=pad_capacitance,
        is_rx=is_rx,
    )


def _expected_die(freqs: np.ndarray, flip: bool) -> skrf.Network:
    """Same direct-formula replication test_die.py's own golden test
    uses — no COM instance needed, sDie only depends on freqs/r0/C_d/L_s/C_b.
    """
    r0s = [R0] * len(DIE_CAPACITANCES)
    result = skrf.network.cascade_list(
        [sDieLadderSegment(freqs, trip) for trip in zip(r0s, DIE_CAPACITANCES, DIE_INDUCTANCES)]
    )
    result = result ** sCshunt(freqs, BUMP_CAPACITANCE, r0=R0)
    if flip:
        result = result.copy()
        result.flip()
    return result


def _expected(freqs: np.ndarray, is_rx: bool, pad_capacitance: float) -> skrf.Network:
    """PyChOpMarg's own sPkgTx/sPkgRx assembly (com.py __init__), replicated
    directly rather than via a full COM instance — same reasoning as
    test_die.py: none of this depends on a channel, package search grid,
    or anything else a full COM would otherwise force us to set up.
    """
    die = _expected_die(freqs, flip=is_rx)
    tline = sPkgTline(freqs, R0, TLINE_A1, TLINE_A2, TLINE_TAU, TLINE_GAMMA0, TLINE_SEGMENTS)
    pad = sCshunt(freqs, pad_capacitance, r0=R0)
    conductor = skrf.network.cascade_list([pad, tline, die] if is_rx else [die, tline, pad])
    four_port = skrf.network.concat_ports([conductor, conductor], port_order="first")
    return sdd_21(four_port)


@pytest.mark.usefixtures("exact_pi_sparams")
def test_tx_matches_pychopmarg_golden_reference() -> None:
    freqs = np.linspace(0, 50e9, 501)

    actual = _package(is_rx=False, pad_capacitance=PAD_CAPACITANCE_TX).network(freqs)

    expected = _expected(freqs, is_rx=False, pad_capacitance=PAD_CAPACITANCE_TX)
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-12)


@pytest.mark.usefixtures("exact_pi_sparams")
def test_rx_matches_pychopmarg_golden_reference() -> None:
    freqs = np.linspace(0, 50e9, 501)

    actual = _package(is_rx=True, pad_capacitance=PAD_CAPACITANCE_RX).network(freqs)

    expected = _expected(freqs, is_rx=True, pad_capacitance=PAD_CAPACITANCE_RX)
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-12)


def test_tx_and_rx_differ() -> None:
    """Not a golden-reference check: a sanity check that is_rx actually
    changes the cascade order (die-first vs pad-first) rather than
    accidentally producing the same network either way.
    """
    freqs = np.linspace(1e8, 50e9, 51)

    tx = _package(is_rx=False, pad_capacitance=PAD_CAPACITANCE_TX).network(freqs)
    rx = _package(is_rx=True, pad_capacitance=PAD_CAPACITANCE_RX).network(freqs)

    assert not np.allclose(tx.s, rx.s)
