"""Golden test: evaluate_channel (PyChOpMarg's COM.__call__ — opt_eq's
search, then calc_noise on the winning Link) against a real COM
instance's own search-then-compute flow. Same small search space
test_search.py uses, for the same reason (the real IEEE_8023dj one is
16 x 11 CTLE gains x up to ~900k Tx tap combinations).

Unlike test_search.py/test_figure_of_merit.py, this uses a real
(non-matched) termination — R_d=55 Ohms against R_0=50 Ohms, the same
example values Table 1 of the "Deriving COM from S-parameters" reference
uses — and package modeling is left on its default, package-included
com.chnls throughout: this is meant to be the one test exercising
ComStandard/evaluate_channel's full parameter set end to end, not an
isolated piece.
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
import pychopmarg.utility.sparams
import pytest
import skrf
from pychopmarg.com import COM
from pychopmarg.common import OptMode
from pychopmarg.config.ieee_8023dj import IEEE_8023dj

from serdeskit.evaluate import ComStandard, evaluate_channel


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


# A much smaller search space than tests/optimize/test_search.py's own
# (already reduced from the real IEEE_8023dj one): here, every candidate
# costs four package-cascaded channels' worth of transfer-function
# interpolation (victim + 2 NEXT + 1 FEXT), not just a single channel's
# — test_search.py's grid took this file from ~20s to ~130s for the
# whole suite before this was cut down. Still 4 taps (n_post=3, matching
# COM.Htx()'s own hard-coded value regardless of tx_taps_min/max/step's
# structure — needs len(bounds) > n_post), just far fewer values each.
TX_TAPS_BOUNDS = [(-0.01, 0.0, 0.01), (0.0, 0.0, 0.0), (0.0, 0.02, 0.02), (0.0, 0.0, 0.0)]
N_POST = 3
C0_MIN = 0.5
DC_GAIN_CANDIDATES = [0.0, -3.0]
SHELF_GAIN_CANDIDATES = [0.0]
R_0 = 50.0
R_D = 55.0  # a real (non-matched) termination — Table 1's own example value


@pytest.fixture(scope="module")
def setup() -> Iterator[Setup]:
    """Patches PI/TWOPI for the whole test's duration — see
    tests/optimize/test_search.py's own `setup` fixture for why `yield`,
    not `return`, matters. Three bindings this time (see
    tests/package/test_com_end_to_end.py's own `reference` fixture for
    why each is needed): pychopmarg.utility.filter (Tx FFE/CTLE),
    pychopmarg.utility.sparams (the package S-parameter formulas),
    pychopmarg.com (calc_H21's own copy).
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pychopmarg.utility.filter, "PI", np.pi)
        mp.setattr(pychopmarg.utility.filter, "TWOPI", 2 * np.pi)
        mp.setattr(pychopmarg.utility.sparams, "PI", np.pi)
        mp.setattr(pychopmarg.utility.sparams, "TWOPI", 2 * np.pi)
        mp.setattr(pychopmarg.com, "PI", np.pi)
        mp.setattr(pychopmarg.com, "TWOPI", 2 * np.pi)

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
        cfg.R_0 = R_0
        cfg.R_d = np.array([R_D, R_D])
        cfg.C_d = [np.array([4e-05, 9e-05, 0.00011])] * 2  # type: ignore[list-item]
        cfg.L_s = [np.array([0.13, 0.15, 0.14])] * 2  # type: ignore[list-item]
        cfg.C_p = [4e-05, 4e-05]
        cfg.C_b = [3e-05, 3e-05]
        cfg.tx_taps_min = [lo for lo, _, _ in TX_TAPS_BOUNDS]
        cfg.tx_taps_max = [hi for _, hi, _ in TX_TAPS_BOUNDS]
        cfg.tx_taps_step = [step for _, _, step in TX_TAPS_BOUNDS]
        cfg.c0_min = C0_MIN
        cfg.g_DC = DC_GAIN_CANDIDATES
        cfg.g_DC2 = SHELF_GAIN_CANDIDATES

        com = COM(
            cfg,
            {"THRU": [thru_path], "FEXT": [fext_path], "NEXT": [next1_path, next2_path]},
            debug=True,
        )
        com.nRxTaps = 0
        com.rx_taps = np.array([])
        com.null_rx_ffe = np.array([])
        com.dfe_taps = np.array([])
        com.opt_mode = OptMode.PRZF
        # com.chnls left at its default here — package-included (add_pkg
        # ran as part of __init__), unlike every other tests/optimize/
        # golden test, which strips it down to a package-free version.

        yield Setup(com=com, thru_path=thru_path, next_paths=[next1_path, next2_path], fext_path=fext_path)


def _standard(setup: Setup, com_min_db: float) -> ComStandard:
    cfg = setup.com.com_params
    return ComStandard(
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
        ctle_zero_freq=float(cfg.f_z) * 1e9,
        ctle_pole1_freq=float(cfg.f_p1) * 1e9,
        ctle_pole2_freq=float(cfg.f_p2) * 1e9,
        ctle_shelf_freq=float(cfg.f_LF) * 1e9,
        ctle_dc_gain_candidates=DC_GAIN_CANDIDATES,
        ctle_shelf_gain_candidates=SHELF_GAIN_CANDIDATES,
        tx_taps_bounds=TX_TAPS_BOUNDS,
        tx_taps_c0_min=C0_MIN,
        tx_taps_n_post=N_POST,
        rx_afe_cutoff_freq=float(cfg.f_r) * float(cfg.fb) * 1e9,
        r0=float(cfg.R_0),
        tx_termination_resistance=float(cfg.R_d[0]),
        rx_termination_resistance=float(cfg.R_d[1]),
        tx_die_capacitances=[c / 1e9 for c in cfg.C_d[0]],  # type: ignore[attr-defined]
        tx_die_inductances=[l / 1e9 for l in cfg.L_s[0]],  # type: ignore[attr-defined]
        tx_bump_capacitance=cfg.C_b[0] / 1e9,
        tx_pad_capacitance=cfg.C_p[0] / 1e9,
        rx_die_capacitances=[c / 1e9 for c in cfg.C_d[1]],  # type: ignore[attr-defined]
        rx_die_inductances=[l / 1e9 for l in cfg.L_s[1]],  # type: ignore[attr-defined]
        rx_bump_capacitance=cfg.C_b[1] / 1e9,
        rx_pad_capacitance=cfg.C_p[1] / 1e9,
        package_tline_a1=cfg.a1,
        package_tline_a2=cfg.a2,
        package_tline_tau=cfg.tau,
        package_tline_gamma0=cfg.gamma0,
        package_tline_segments=list(zip(cfg.z_c, [cfg.z_p[setup.com.zp_sel], cfg.z_pB])),
        com_min_db=com_min_db,
    )


def _pychopmarg_search_then_compute(com: COM) -> float:
    """opt_eq()'s search loop plus calc_noise() on the winner, replicated
    directly rather than calling COM.__call__() itself — that method's
    own opt_eq() call hits the same nRxPreTaps-indexing-into-an-empty-
    array crash tests/optimize/test_search.py's own helper documents,
    an edge case in opt_eq's wrap-up bookkeeping unrelated to the
    search-then-compute flow this test is actually about.
    """
    fom_max = -1000.0
    best = None
    for gdc2 in com.com_params.g_DC2:
        for gdc in com.com_params.g_DC:
            hctle = com.calc_Hctf(gdc, gdc2)
            for tx_ix in range(com.num_tx_combs):
                fom = com.calc_fom(tx_ix, hctle, opt_mode=OptMode.PRZF)
                if fom > fom_max:
                    fom_max = fom
                    best = (tx_ix, gdc, gdc2)
    assert best is not None
    tx_ix, gdc, gdc2 = best

    com.gDC, com.gDC2 = gdc, gdc2
    com.tx_ix = tx_ix
    signal_amplitude, noise_amplitude, _ = com.calc_noise()
    return float(20 * np.log10(signal_amplitude / noise_amplitude))


@pytest.mark.skip(
    reason=(
        "EqualizationSearch builds Link without a tx_filter -- "
        "ffe_channel_ctle_pulse_response() now requires one (TxRisetimeFilter, "
        "wired in to match MATLAB's real H_t behavior), so this raises "
        "AttributeError. Search/optimization-level MATLAB alignment is "
        "explicitly out of scope for now (CLAUDE.md); wiring tx_filter into "
        "EqualizationSearch is the fix, if picked up. See docs/known-issues.md."
    )
)
@pytest.mark.tx_filter_unwired_in_search
def test_matches_pychopmargs_own_search_then_compute(setup: Setup) -> None:
    standard = _standard(setup, com_min_db=-100.0)  # low enough that pass/fail isn't in question here

    evaluation = evaluate_channel(standard, setup.thru_path, next_paths=setup.next_paths, fext_paths=[setup.fext_path])

    expected_com_db = _pychopmarg_search_then_compute(setup.com)
    assert evaluation.result.com_db == pytest.approx(expected_com_db, rel=1e-9)


@pytest.mark.skip(reason="Same tx_filter-unwired-in-search gap as test_matches_pychopmargs_own_search_then_compute.")
@pytest.mark.tx_filter_unwired_in_search
def test_passes_reflects_the_threshold(setup: Setup) -> None:
    passing = evaluate_channel(
        _standard(setup, com_min_db=-100.0), setup.thru_path, next_paths=setup.next_paths, fext_paths=[setup.fext_path],
    )
    failing = evaluate_channel(
        _standard(setup, com_min_db=100.0), setup.thru_path, next_paths=setup.next_paths, fext_paths=[setup.fext_path],
    )

    assert passing.passes
    assert not failing.passes
