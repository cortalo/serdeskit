"""End-to-end golden test: Com.compute() driven by a package-augmented
channel (Package + cascade_channel + SParameterChannel), against a real
COM instance's own calc_noise() — using its actual, package-included
com.chnls (not the debug-only chnls_noPkg every other tests/com/ golden
test uses to sidestep package modeling, since this project didn't have
any until now).

This is the piece that confirms package modeling doesn't just produce
individually-correct S-parameter networks (test_cascade.py,
test_package_vs_real_com.py, ...) but actually integrates into a real
COM computation end to end, matching PyChOpMarg bit-for-bit.
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

from serdeskit.channel import SParameterChannel
from serdeskit.com import Com, ComParams
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Link
from serdeskit.package import Package, cascade_channel
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe

G_DC = -6.0
G_DC2 = -2.0
TX_COMB_IX = 5
N_TX_POST_TAPS = 3


@dataclass(frozen=True)
class Reference:
    com_db: float
    signal_amplitude: float
    noise_amplitude: float
    sigma_tx: float
    sigma_jitter: float
    sigma_noise: float
    tx_taps: npt.NDArray[np.float64]
    channel_path: Path
    baud_rate: float
    freq_step: float
    samples_per_ui: int
    levels: int
    rlm: float
    victim_amplitude: float
    a_ne: float
    a_fe: float
    snr_tx: float
    sigma_rj: float
    eta_0: float
    a_dd: float
    der_0: float
    dfe_min: npt.NDArray[np.float64]
    dfe_max: npt.NDArray[np.float64]
    afe_cutoff: float
    ctle_freqs: tuple[float, float, float, float]
    tx_package: Package
    rx_package: Package
    freqs: npt.NDArray[np.float64]


@pytest.fixture(scope="module")
def reference() -> Iterator[Reference]:
    """Runs PyChOpMarg's own calc_noise() with its default, package-
    included com.chnls (no override to chnls_noPkg) — three separate
    PI/TWOPI bindings need patching for exact agreement, all of them
    exercised by this pipeline: pychopmarg.utility.filter (Tx FFE/CTLE),
    pychopmarg.utility.sparams (the package S-parameter formulas),
    pychopmarg.com (calc_H21's own copy). calc_noise() itself has to run
    inside the same patched context as __init__, since it's what
    triggers the CTLE/Tx-FFE calculations — not just package assembly,
    which __init__ alone already completes.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pychopmarg.utility.filter, "PI", np.pi)
        mp.setattr(pychopmarg.utility.filter, "TWOPI", 2 * np.pi)
        mp.setattr(pychopmarg.utility.sparams, "PI", np.pi)
        mp.setattr(pychopmarg.utility.sparams, "TWOPI", 2 * np.pi)
        mp.setattr(pychopmarg.com, "PI", np.pi)
        mp.setattr(pychopmarg.com, "TWOPI", 2 * np.pi)
        yield _build_reference()


def _build_reference() -> Reference:
    freq = np.linspace(1e8, 40e9, 400)
    loss = np.exp(-np.sqrt(freq / 1e9) * 0.08) * np.exp(-1j * 2 * np.pi * freq * 3e-10)
    refl = 0.03 * np.exp(-1j * 2 * np.pi * freq * 1e-10)
    s = np.zeros((len(freq), 4, 4), dtype=complex)
    for a, b in [(0, 1), (2, 3)]:
        s[:, a, a] = refl
        s[:, b, b] = refl
        s[:, b, a] = loss
        s[:, a, b] = loss
    path = Path(tempfile.mkdtemp()) / "synthetic_thru.s4p"
    skrf.Network(f=freq, s=s, z0=50, f_unit="Hz").write_touchstone(str(path))

    cfg = copy.deepcopy(IEEE_8023dj)
    cfg.fstep = 0.1
    cfg.R_d = np.array([50.0, 50.0])
    cfg.C_d = [np.array([4e-05, 9e-05, 0.00011])] * 2  # type: ignore[list-item]
    cfg.L_s = [np.array([0.13, 0.15, 0.14])] * 2  # type: ignore[list-item]
    cfg.C_p = [4e-05, 4e-05]
    cfg.C_b = [3e-05, 3e-05]

    com = COM(cfg, {"THRU": [path], "FEXT": [], "NEXT": []}, debug=True)
    com.gDC, com.gDC2 = G_DC, G_DC2
    com.tx_ix = TX_COMB_IX
    com.nRxTaps = 0
    com.rx_taps = np.array([])
    com.dfe_taps = np.array([])
    com.opt_mode = OptMode.PRZF
    # Deliberately *not* overriding com.chnls to chnls_noPkg here, unlike
    # every other tests/com/ golden test — the whole point of this test
    # is exercising the package-included path.
    signal_amplitude, noise_amplitude, _ = com.calc_noise()

    r = com.com_rslts

    def side_package(is_rx: bool) -> Package:
        ix = 1 if is_rx else 0
        return Package(
            r0=cfg.R_0,
            die_capacitances=[c / 1e9 for c in cfg.C_d[ix]],  # type: ignore[attr-defined]
            die_inductances=[l / 1e9 for l in cfg.L_s[ix]],  # type: ignore[attr-defined]
            bump_capacitance=cfg.C_b[ix] / 1e9,
            tline_a1=cfg.a1,
            tline_a2=cfg.a2,
            tline_tau=cfg.tau,
            tline_gamma0=cfg.gamma0,
            tline_segments=list(zip(cfg.z_c, [cfg.z_p[com.zp_sel], cfg.z_pB])),
            pad_capacitance=cfg.C_p[ix] / 1e9,
            is_rx=is_rx,
        )

    return Reference(
        com_db=float(20 * np.log10(signal_amplitude / noise_amplitude)),
        signal_amplitude=float(signal_amplitude),
        noise_amplitude=float(noise_amplitude),
        sigma_tx=float(r["sigma_Tx"]),
        sigma_jitter=float(r["sigma_J"]),
        sigma_noise=float(r["sigma_N"]),
        tx_taps=np.array(com._tx_combs[TX_COMB_IX]),
        channel_path=path,
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
        afe_cutoff=float(cfg.f_r) * float(cfg.fb) * 1e9,
        ctle_freqs=(
            float(cfg.f_z) * 1e9,
            float(cfg.f_p1) * 1e9,
            float(cfg.f_p2) * 1e9,
            float(cfg.f_LF) * 1e9,
        ),
        tx_package=side_package(is_rx=False),
        rx_package=side_package(is_rx=True),
        freqs=np.asarray(com.freqs, dtype=np.float64),
    )


def _raw_channel_differential(path: Path) -> skrf.Network:
    """Same recipe test_cascade.py's own helper uses — see its docstring."""
    network = skrf.Network(str(path)).copy()
    network.renumber([0, 1, 2, 3], [0, 2, 1, 3])
    network.se2gmm(p=2)
    return network.subnetwork([0, 1])


def _build_com(ref: Reference) -> Com:
    zero, pole1, pole2, shelf = ref.ctle_freqs
    channel_network = cascade_channel(
        _raw_channel_differential(ref.channel_path), ref.tx_package, ref.rx_package, ref.freqs,
    )
    link = Link(
        channel=SParameterChannel(channel_network),  # R_d == R_0 here, so gamma1/gamma2 default to 0
        ctle=TwoStageCtle(
            zero_freq=zero,
            pole1_freq=pole1,
            pole2_freq=pole2,
            shelf_freq=shelf,
            dc_gain_db=G_DC,
            shelf_gain_db=G_DC2,
        ),
        ffe=TapWeightFfe(
            tap_weights=ref.tx_taps, n_post=N_TX_POST_TAPS, tap_delay=1.0 / ref.baud_rate,
        ),
        rx_afe=RxAfeButterworth(cutoff_freq=ref.afe_cutoff),
        rx_ffe=TapWeightRxFfe(tap_weights=np.array([1.0]), tap_delay=1.0 / ref.baud_rate),
    )
    params = ComParams(
        baud_rate=ref.baud_rate,
        freq_step=ref.freq_step,
        samples_per_ui=ref.samples_per_ui,
        levels=ref.levels,
        rlm=ref.rlm,
        victim_amplitude=ref.victim_amplitude,
        a_ne=ref.a_ne,
        a_fe=ref.a_fe,
        snr_tx=ref.snr_tx,
        sigma_rj=ref.sigma_rj,
        eta_0=ref.eta_0,
        a_dd=ref.a_dd,
        der_0=ref.der_0,
        dfe_min=ref.dfe_min,
        dfe_max=ref.dfe_max,
    )
    return Com(link=link, params=params)


def test_com_matches_pychopmarg_with_package_modeling(reference: Reference) -> None:
    result = _build_com(reference).compute()

    assert result.com_db == pytest.approx(reference.com_db, rel=1e-9)
    assert result.signal_amplitude == pytest.approx(reference.signal_amplitude, rel=1e-9)
    assert result.noise_amplitude == pytest.approx(reference.noise_amplitude, rel=1e-9)
    assert result.sigma_tx == pytest.approx(reference.sigma_tx, rel=1e-9)
    assert result.sigma_jitter == pytest.approx(reference.sigma_jitter, rel=1e-9)
    assert result.sigma_noise == pytest.approx(reference.sigma_noise, rel=1e-9)
