"""Golden test: figure_of_merit (93A-36) against a real COM instance's
own calc_fom(), in PRZF mode with no Rx FFE (nRxTaps=0) — the scope this
project's optimization work starts with, matching every other golden
test's own default configuration.

calc_fom is opt_eq's search objective: a fast, closed-form approximation
to COM, used instead of the full PMF-based calc_noise because running
the exact calculation for every candidate in a large Tx-tap x CTLE-gain
search grid would be far too slow. It differs from calc_noise in three
deliberate ways, not just being "the same formula without a PMF": jitter
combines A_DD and sigma_Rj into one variance term (93A-32) rather than
splitting deterministic jitter into its own delta-PMF; crosstalk is a
direct worst-case-phase energy sum (93A-33/34) rather than routing
through delta_pmf; and every variance is summed directly under one
log10, rather than combined via PMF convolution.
"""
import copy
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pychopmarg.com
import pychopmarg.utility.filter
import pytest
import skrf
from pychopmarg.com import COM
from pychopmarg.common import OptMode
from pychopmarg.config.ieee_8023dj import IEEE_8023dj
from pychopmarg.utility.filter import calc_H21
from pychopmarg.utility.sparams import sdd_21

from serdeskit.channel import SParameterChannel
from serdeskit.com import ComParams
from serdeskit.common.types import Signal
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Link, SystemGrid
from serdeskit.optimize import figure_of_merit
from serdeskit.pulse_response import PulseResponse
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe

G_DC = -6.0
G_DC2 = -2.0
TX_COMB_IX = 5
N_TX_POST_TAPS = 3


@dataclass(frozen=True)
class Setup:
    com: COM
    thru_path: Path
    next_paths: list[Path]
    fext_path: Path


def _synthetic_s4p(
    path: Path, freq: npt.NDArray[np.float64], loss_exponent: float, delay: float, refl_mag: float
) -> None:
    """Same helper tests/com/test_com.py's own uses — see there for why."""
    loss = np.exp(-np.sqrt(freq / 1e9) * loss_exponent) * np.exp(-1j * 2 * np.pi * freq * delay)
    refl = refl_mag * np.exp(-1j * 2 * np.pi * freq * 1e-10)
    s = np.zeros((len(freq), 4, 4), dtype=complex)
    for a, b in [(0, 1), (2, 3)]:
        s[:, a, a] = refl
        s[:, b, b] = refl
        s[:, b, a] = loss
        s[:, a, b] = loss
    skrf.Network(f=freq, s=s, z0=50, f_unit="Hz").write_touchstone(str(path))


def _no_pkg_chnl(com: COM, path: Path, ntype: str) -> tuple[tuple[skrf.Network, str], npt.NDArray[np.complex128]]:
    """Same helper tests/com/test_com.py's own uses — package-free, so
    this test isolates figure_of_merit itself from package modeling.
    """
    ntwk = sdd_21(skrf.Network(str(path)))
    h21 = calc_H21(com.freqs, ntwk, com.gamma1_Tx, com.gamma2_Rx)
    return (ntwk, ntype), h21


@pytest.fixture(scope="module")
def setup() -> Iterator[Setup]:
    """Patches PI/TWOPI for the whole test's duration, not just this
    fixture's own setup: `test_matches_pychopmargs_own_calc_fom` calls
    `com.calc_fom()` itself (fresh, to get `expected`), and that call
    needs the patch active too — a `return` here (instead of `yield`)
    would exit the `with` block, and revert the patch, before the test
    function ever runs.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pychopmarg.utility.filter, "PI", np.pi)
        mp.setattr(pychopmarg.utility.filter, "TWOPI", 2 * np.pi)
        mp.setattr(pychopmarg.com, "PI", np.pi)
        mp.setattr(pychopmarg.com, "TWOPI", 2 * np.pi)
        # See tests/conftest.py's own no_raised_cosine_taper fixture.
        mp.setattr(pychopmarg.utility.filter, "raised_cosine", lambda x: x)

        freq = np.linspace(1e8, 40e9, 400)
        tmp = Path(tempfile.mkdtemp())
        thru_path = tmp / "synthetic_thru.s4p"
        _synthetic_s4p(thru_path, freq, loss_exponent=0.08, delay=3e-10, refl_mag=0.03)
        next1_path = tmp / "synthetic_next1.s4p"
        _synthetic_s4p(next1_path, freq, loss_exponent=0.03, delay=1e-10, refl_mag=0.02)
        next2_path = tmp / "synthetic_next2.s4p"
        _synthetic_s4p(next2_path, freq, loss_exponent=0.04, delay=1.5e-10, refl_mag=0.02)
        fext_path = tmp / "synthetic_fext.s4p"
        _synthetic_s4p(fext_path, freq, loss_exponent=0.09, delay=3.2e-10, refl_mag=0.025)

        cfg = copy.deepcopy(IEEE_8023dj)
        cfg.fstep = 0.1
        cfg.R_d = np.array([50.0, 50.0])
        cfg.C_d = [np.array([4e-05, 9e-05, 0.00011])] * 2  # type: ignore[list-item]
        cfg.L_s = [np.array([0.13, 0.15, 0.14])] * 2  # type: ignore[list-item]
        cfg.C_p = [4e-05, 4e-05]
        cfg.C_b = [3e-05, 3e-05]

        com = COM(
            cfg,
            {"THRU": [thru_path], "FEXT": [fext_path], "NEXT": [next1_path, next2_path]},
            debug=True,
        )
        com.gDC, com.gDC2 = G_DC, G_DC2
        com.tx_ix = TX_COMB_IX
        com.nRxTaps = 0
        com.rx_taps = np.array([])
        # calc_fom (unlike calc_noise) passes self.null_rx_ffe explicitly
        # rather than falling back to self.rx_taps — that placeholder is
        # still the 16-element one __init__ built from cfg's own nRxTaps
        # before this override, so it needs resetting too, or H()'s own
        # len(rx_taps)-based nRxTaps check disagrees with self.nRxTaps.
        com.null_rx_ffe = np.array([])
        com.dfe_taps = np.array([])
        com.opt_mode = OptMode.PRZF
        com.chnls = [
            _no_pkg_chnl(com, thru_path, "THRU"),
            _no_pkg_chnl(com, fext_path, "FEXT"),
            _no_pkg_chnl(com, next1_path, "NEXT"),
            _no_pkg_chnl(com, next2_path, "NEXT"),
        ]

        yield Setup(com=com, thru_path=thru_path, next_paths=[next1_path, next2_path], fext_path=fext_path)


def _link(setup: Setup, path: Path, flat_tx: bool = False) -> Link:
    """flat_tx=True builds the flat, unequalized Tx FFE gen_pulse_resps
    forces for NEXT aggressors (tx_ix=0 there, unconditionally — the
    same rule com.Com._next_links() already implements); FEXT and THRU
    both use the victim's own optimized tap combination.

    tx_taps_min all-zeros (PyChOpMarg's own _tx_combs[0], matching
    tx_ix=0) is *not* the same as an empty tap array: both give unity
    magnitude, but the all-zeros version still has n_post=3 taps either
    side of the cursor (all weight 0), placing the cursor at delay index
    3 rather than 0 — a pure linear-phase (delay) difference from the
    empty-array version, invisible in magnitude but real in the time-
    domain pulse response. Matched exactly here since this test compares
    against calc_fom at rel=1e-9; com.Com._next_links()'s own empty-array
    flat_ffe never needed this distinction, since its own golden tests
    only check crosstalk-derived quantities at the much looser tolerance
    PMF-convolution noise already requires.
    """
    cfg = setup.com.com_params
    tx_taps = np.zeros(len(cfg.tx_taps_min)) if flat_tx else np.array(setup.com._tx_combs[TX_COMB_IX])
    n_post = N_TX_POST_TAPS
    baud_rate = float(cfg.fb) * 1e9
    return Link(
        channel=SParameterChannel.from_touchstone(str(path)),
        ctle=TwoStageCtle(
            zero_freq=float(cfg.f_z) * 1e9,
            pole1_freq=float(cfg.f_p1) * 1e9,
            pole2_freq=float(cfg.f_p2) * 1e9,
            shelf_freq=float(cfg.f_LF) * 1e9,
            dc_gain_db=G_DC,
            shelf_gain_db=G_DC2,
        ),
        ffe=TapWeightFfe(tap_weights=tx_taps, n_post=n_post, tap_delay=1.0 / baud_rate),
        rx_afe=RxAfeButterworth(cutoff_freq=float(cfg.f_r) * baud_rate),
        rx_ffe=TapWeightRxFfe(tap_weights=np.array([1.0]), tap_delay=1.0 / baud_rate),
    )


def _params(setup: Setup) -> ComParams:
    cfg = setup.com.com_params
    return ComParams(
        baud_rate=float(cfg.fb) * 1e9,
        freq_step=float(cfg.fstep) * 1e9,
        samples_per_ui=int(cfg.M),
        levels=int(cfg.L),
        rlm=float(cfg.RLM),
        victim_amplitude=float(cfg.A_v),
        a_ne=float(cfg.A_ne),
        a_fe=float(cfg.A_fe),
        snr_tx=float(cfg.SNR_TX),
        sigma_rj=float(cfg.sigma_Rj),
        eta_0=float(cfg.eta_0),
        a_dd=float(cfg.A_DD),
        der_0=float(cfg.DER_0),
        dfe_min=np.asarray(cfg.dfe_min, dtype=np.float64),
        dfe_max=np.asarray(cfg.dfe_max, dtype=np.float64),
    )


def _delay_convention_shift(setup: Setup, p: ComParams) -> int:
    """TapWeightFfe references delay 0 at the *cursor* tap (matching MATLAB
    COM3.70's own FFE — see tests/ffe/test_tap_weight_vs_matlab.py), while
    PyChOpMarg's calc_fom (via its own Hffe convention) references delay 0
    at the *first* (most-precursor) tap. That's a pure linear-phase
    difference in the frequency domain, i.e. a pure circular shift of
    n_pre UI in the IFFT'd pulse response -- roll it out of every pulse
    response fed into figure_of_merit() before comparing against PyChOpMarg,
    same fix as tests/link/test_pulse_response_vs_pychopmarg.py.
    """
    tx_taps = np.array(setup.com._tx_combs[TX_COMB_IX])
    n_pre = len(tx_taps) - N_TX_POST_TAPS
    return n_pre * p.samples_per_ui


def _figure_of_merit(setup: Setup) -> float:
    p = _params(setup)
    grid = SystemGrid.build(p.baud_rate, p.freq_step, p.samples_per_ui)
    shift = _delay_convention_shift(setup, p)

    victim_link = _link(setup, setup.thru_path)
    signal = victim_link.ffe_channel_ctle_pulse_response(grid).scale(p.victim_amplitude)
    signal = Signal(samples=np.roll(signal.samples, shift), fs=signal.fs, t0=signal.t0)
    pulse_response = PulseResponse.from_signal(
        signal, ui=1.0 / p.baud_rate, dfe1_max=float(p.dfe_max[0]), dfe1_min=float(p.dfe_min[0]),
    )

    aggressor_paths_amplitudes_and_flatness = [(setup.fext_path, p.a_fe, False)] + [
        (path, p.a_ne, True) for path in setup.next_paths
    ]
    aggressor_pulse_responses = [
        np.roll(
            _link(setup, path, flat_tx=flat_tx).ffe_channel_ctle_pulse_response(grid).scale(amplitude).samples,
            shift,
        )
        for path, amplitude, flat_tx in aggressor_paths_amplitudes_and_flatness
    ]

    rx_response = victim_link.ctle.transfer_function(grid.f) * victim_link.rx_afe.transfer_function(  # type: ignore[union-attr]
        grid.f
    )

    return figure_of_merit(pulse_response, aggressor_pulse_responses, rx_response, p)


def test_matches_pychopmargs_own_calc_fom(setup: Setup) -> None:
    actual = _figure_of_merit(setup)

    expected = setup.com.calc_fom(TX_COMB_IX, setup.com.calc_Hctf(G_DC, G_DC2), opt_mode=OptMode.PRZF)
    assert actual == pytest.approx(expected, rel=1e-9)
