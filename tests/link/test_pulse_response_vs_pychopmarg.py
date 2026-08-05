"""End-to-end golden test: our whole (93A-19) transfer-function chain plus
(93A-24) pulse response, against PyChOpMarg's real COM class driving the
same synthetic channel — not the per-formula comparisons the individual
stage tests already do, but the composition of all of them at once.
"""
import copy
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pychopmarg.com
import pytest
import skrf
from pychopmarg.com import COM
from pychopmarg.config.ieee_8023dj import IEEE_8023dj
from pychopmarg.config.template import COMParams

from serdeskit.channel import SParameterChannel
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Link, SystemGrid
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe
from serdeskit.tx_filter import TxRisetimeFilter

G_DC = -6.0
G_DC2 = -2.0
TX_COMB_IX = 5  # arbitrary, just not the all-zeros combination at index 0
N_TX_POST_TAPS = 3  # what COM.Htx() passes to calc_Hffe


@pytest.fixture(scope="module")
def synthetic_s4p(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A lossy, mildly-reflective 4-port thru, written to a Touchstone file
    because COM only accepts channels as file paths. Synthetic rather than
    a real measured channel so this test needs nothing from `reference/`,
    which is gitignored and absent in CI.

    Port order is (TX+, RX+, TX-, RX-) — the ECEN 720 `peters_*`
    convention SParameterChannel documents, which is also the
    "1->2/3->4 thru path" convention PyChOpMarg's own `sdd_21` assumes,
    so both sides read the file the same way.
    """
    freq = np.linspace(1e8, 40e9, 400)
    loss = np.exp(-np.sqrt(freq / 1e9) * 0.08) * np.exp(-1j * 2 * np.pi * freq * 3e-10)
    refl = 0.03 * np.exp(-1j * 2 * np.pi * freq * 1e-10)

    s = np.zeros((len(freq), 4, 4), dtype=complex)
    for a, b in [(0, 1), (2, 3)]:  # two identical, uncoupled traces
        s[:, a, a] = refl
        s[:, b, b] = refl
        s[:, b, a] = loss
        s[:, a, b] = loss

    path = tmp_path_factory.mktemp("chnl") / "synthetic_thru.s4p"
    skrf.Network(f=freq, s=s, z0=50, f_unit="Hz").write_touchstone(str(path))
    return path


def _com_params() -> COMParams:
    """IEEE_8023dj with two independent fixups.

    First, five package fields are restructured. As shipped, `IEEE_8023dj`
    can't construct a COM instance at all: COM.sDie()/COM.__init__ index
    these per-end (`C_d[0]` for Tx, `C_d[1]` for Rx, then `len()` the
    result), but the shipped values are flat — `C_d[0]` is a float, so
    `len()` raises TypeError. Reshaping to one entry per end is what the
    consuming code actually expects. Values themselves are unchanged and
    don't matter here: only `chnls_noPkg` is used below, which excludes
    the package these fields describe.

    Second, `fstep` is coarsened 0.01 -> 0.1 GHz, shrinking the frequency
    vector from 170001 points to 17001 and this test from ~4s to ~1s. Not
    a fidelity concern: both sides use the same grid, and PyChOpMarg's own
    source carries a commented-out `fstep = 50e6  # Dramatically improves
    performance.` for the same reason.
    """
    cfg = copy.deepcopy(IEEE_8023dj)
    cfg.R_d = np.array([50.0, 50.0])
    # COMParams types these as flat list[float], which is exactly the
    # mismatch being corrected here — see this function's docstring.
    cfg.C_d = [np.array([4e-05, 9e-05, 0.00011])] * 2  # type: ignore[list-item]
    cfg.L_s = [np.array([0.13, 0.15, 0.14])] * 2  # type: ignore[list-item]
    cfg.C_p = [4e-05, 4e-05]
    cfg.C_b = [3e-05, 3e-05]
    cfg.fstep = 0.1
    return cfg


def _expected_pulse_response(com: COM) -> npt.NDArray[np.float64]:
    """PyChOpMarg's own (93A-19) composition + (93A-24) pulse response.

    `chnls_noPkg` (debug-mode only) is the terminated channel *without*
    the package model, which this project doesn't implement yet — the
    one deliberate scope difference between the two sides.

    Empty `rx_taps`/`dfe_taps` omit the Rx FFE and DFE, per COM.H()'s own
    documented way of excluding them: this project has no Rx FFE yet, and
    the DFE isn't part of the pulse response at all (it's applied later,
    during ISI extraction).
    """
    _, h21_nopkg = com.chnls_noPkg[0]
    h = com.H(
        h21_nopkg,
        TX_COMB_IX,
        Hctf=com.calc_Hctf(G_DC, G_DC2),
        rx_taps=np.array([]),
        dfe_taps=np.array([]),
    )
    return np.asarray(com.pulse_resp(h), dtype=np.float64)


@pytest.mark.usefixtures("exact_pi", "no_raised_cosine_taper")
def test_matches_pychopmarg_end_to_end(synthetic_s4p: Path) -> None:
    cfg = _com_params()
    com = COM(cfg, {"THRU": [synthetic_s4p], "FEXT": [], "NEXT": []}, debug=True)
    tx_taps = np.array(com._tx_combs[TX_COMB_IX])

    expected = _expected_pulse_response(com)

    baud_rate = cfg.fb * 1e9
    link = Link(
        # gamma1/gamma2 default to 0: this config's R_d equals its R_0, so
        # COM's own reflection coefficients are 0 too.
        channel=SParameterChannel.from_touchstone(str(synthetic_s4p)),
        ctle=TwoStageCtle(
            zero_freq=cfg.f_z * 1e9,
            pole1_freq=cfg.f_p1 * 1e9,
            pole2_freq=cfg.f_p2 * 1e9,
            shelf_freq=cfg.f_LF * 1e9,
            dc_gain_db=G_DC,
            shelf_gain_db=G_DC2,
        ),
        ffe=TapWeightFfe(
            tap_weights=tx_taps, n_post=N_TX_POST_TAPS, tap_delay=1.0 / baud_rate
        ),
        # risetime=0.0 is the identity (H(f)=1 exactly, no floating-point
        # effect on the atol=1e-15 comparison below) -- matches PyChOpMarg's
        # own H() (Htx * H21 * Hr * Hctf * ...), which has no Tx
        # risetime/transition-time factor at all.
        tx_filter=TxRisetimeFilter(risetime=0.0),
        rx_afe=RxAfeButterworth(cutoff_freq=cfg.f_r * baud_rate),
        # A single unity tap is the identity — matches this test's own
        # rx_taps=np.array([]) on the PyChOpMarg side (both mean "no Rx FFE").
        rx_ffe=TapWeightRxFfe(tap_weights=np.array([1.0]), tap_delay=1.0 / baud_rate),
    )

    grid = SystemGrid.build(
        baud_rate=baud_rate, freq_step=cfg.fstep * 1e9, samples_per_ui=cfg.M
    )
    actual = link.ffe_channel_ctle_pulse_response(grid).samples

    assert actual.shape == expected.shape
    # TapWeightFfe references delay 0 at the *cursor* tap (matching MATLAB
    # COM3.70's own FFE — see tests/ffe/test_tap_weight_vs_matlab.py), while
    # PyChOpMarg's calc_Hffe (which com.H() calls) references delay 0 at the
    # *first* (most-precursor) tap. That's a pure linear-phase difference in
    # the frequency domain, i.e. a pure circular shift of n_pre UI in the
    # IFFT'd pulse response -- roll it out before comparing so this still
    # asserts genuine equivalence of the whole (93A-19)+(93A-24) chain (to
    # floating-point noise), not just proximity.
    n_pre = len(tx_taps) - N_TX_POST_TAPS
    aligned = np.roll(actual, n_pre * cfg.M)
    np.testing.assert_allclose(aligned, expected, rtol=0, atol=1e-15)


@pytest.mark.usefixtures("exact_pi", "no_raised_cosine_taper")
def test_matches_pychopmarg_end_to_end_with_rx_ffe(
    synthetic_s4p: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same comparison as test_matches_pychopmarg_end_to_end, but with a
    non-empty rx_taps — the case that test deliberately excludes (Rx FFE
    didn't exist in this project yet when it was written).

    `Hffe_Rx`'s own phase matrix is built inside pychopmarg.com using a
    TWOPI bound at import time from pychopmarg.common — a different
    binding from the one `exact_pi` patches (pychopmarg.utility.filter's)
    — so it needs its own patch here, on top of `exact_pi`, for the same
    exact-agreement reason `exact_pi` exists at all.
    """
    monkeypatch.setattr(pychopmarg.com, "PI", np.pi)
    monkeypatch.setattr(pychopmarg.com, "TWOPI", 2 * np.pi)

    cfg = _com_params()
    com = COM(cfg, {"THRU": [synthetic_s4p], "FEXT": [], "NEXT": []}, debug=True)
    tx_taps = np.array(com._tx_combs[TX_COMB_IX])
    rx_taps = np.array([0.05, 1.0, -0.1])  # cursor at index 1, arbitrary otherwise
    # Hffe_Rx asserts len(taps) == self.nRxTaps and multiplies against
    # rx_ffe_phase_matrix — both still reflect cfg's own (16-tap) Rx FFE
    # search-grid configuration from __init__, so both need rebuilding
    # for this test's 3-tap array.
    com.nRxTaps = len(rx_taps)
    ns = np.arange(len(rx_taps))
    com.rx_ffe_phase_matrix = np.exp(np.outer(ns, -1j * 2 * np.pi * com.ui * com.freqs))

    _, h21_nopkg = com.chnls_noPkg[0]
    h = com.H(
        h21_nopkg,
        TX_COMB_IX,
        Hctf=com.calc_Hctf(G_DC, G_DC2),
        rx_taps=rx_taps,
        dfe_taps=np.array([]),
    )
    expected = np.asarray(com.pulse_resp(h), dtype=np.float64)

    baud_rate = cfg.fb * 1e9
    link = Link(
        channel=SParameterChannel.from_touchstone(str(synthetic_s4p)),
        ctle=TwoStageCtle(
            zero_freq=cfg.f_z * 1e9,
            pole1_freq=cfg.f_p1 * 1e9,
            pole2_freq=cfg.f_p2 * 1e9,
            shelf_freq=cfg.f_LF * 1e9,
            dc_gain_db=G_DC,
            shelf_gain_db=G_DC2,
        ),
        ffe=TapWeightFfe(
            tap_weights=tx_taps, n_post=N_TX_POST_TAPS, tap_delay=1.0 / baud_rate
        ),
        # See test_matches_pychopmarg_end_to_end's own comment: identity,
        # matching PyChOpMarg's H() having no Tx risetime factor.
        tx_filter=TxRisetimeFilter(risetime=0.0),
        rx_afe=RxAfeButterworth(cutoff_freq=cfg.f_r * baud_rate),
        rx_ffe=TapWeightRxFfe(tap_weights=rx_taps, tap_delay=1.0 / baud_rate),
    )

    grid = SystemGrid.build(
        baud_rate=baud_rate, freq_step=cfg.fstep * 1e9, samples_per_ui=cfg.M
    )
    actual = link.ffe_channel_ctle_pulse_response(grid).samples

    assert actual.shape == expected.shape
    # See the FFE delay-convention note in test_matches_pychopmarg_end_to_end.
    n_pre = len(tx_taps) - N_TX_POST_TAPS
    aligned = np.roll(actual, n_pre * cfg.M)
    np.testing.assert_allclose(aligned, expected, rtol=0, atol=1e-15)
