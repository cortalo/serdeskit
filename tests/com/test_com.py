"""End-to-end golden test of the COM value itself, against PyChOpMarg's
`calc_noise` driving the same synthetic channel.

Distinct from tests/link/test_pulse_response_vs_pychopmarg.py, which stops
at the pulse response: this carries on through residual ISI, the noise
variances, PMF convolution, and the CDF read-off to a COM number in dB.
"""
import copy
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pychopmarg.utility.filter
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
from serdeskit.rx_afe import RxAfeButterworth

G_DC = -6.0
G_DC2 = -2.0
TX_COMB_IX = 5
N_TX_POST_TAPS = 3


@dataclass(frozen=True)
class Reference:
    """PyChOpMarg's answer, plus what's needed to pose it the same question."""

    com_db: float
    signal_amplitude: float
    noise_amplitude: float
    sigma_tx: float
    sigma_jitter: float
    sigma_noise: float
    sigma_gaussian: float
    tx_taps: npt.NDArray[np.float64]
    channel_path: Path
    baud_rate: float
    freq_step: float
    samples_per_ui: int
    levels: int
    rlm: float
    victim_amplitude: float
    snr_tx: float
    sigma_rj: float
    eta_0: float
    a_dd: float
    der_0: float
    dfe_min: npt.NDArray[np.float64]
    dfe_max: npt.NDArray[np.float64]
    afe_cutoff: float
    ctle_freqs: tuple[float, float, float, float]


@pytest.fixture(scope="module")
def reference() -> Iterator[Reference]:
    """Run PyChOpMarg's own calc_noise on a synthetic channel.

    Setup matches tests/link/test_pulse_response_vs_pychopmarg.py — see
    there for why IEEE_8023dj's package fields need reshaping and why
    fstep is coarsened. `nRxTaps = 0` and PRZF mode keep calc_noise on the
    paths this project has: no Rx FFE, and the closed-form (93A-30) Tx
    noise term rather than MMSE's NoiseCalc.

    Applies the full-precision PI patch itself rather than requesting the
    `exact_pi` fixture: that one is function-scoped, and pytest sets
    higher-scoped fixtures up first, so this would otherwise be computed
    with PyChOpMarg's truncated PI while the code under test used np.pi —
    a ~2e-6 relative discrepancy with no bearing on correctness, sitting
    exactly where a real one would show up.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pychopmarg.utility.filter, "PI", np.pi)
        mp.setattr(pychopmarg.utility.filter, "TWOPI", 2 * np.pi)
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
    # calc_noise() would otherwise use `chnls`, which includes the package
    # model this project doesn't implement; `chnls_noPkg` is the same
    # terminated channel without it, matching what our Link composes.
    com.chnls = com.chnls_noPkg
    signal_amplitude, noise_amplitude, _ = com.calc_noise()

    r = com.com_rslts
    return Reference(
        com_db=float(20 * np.log10(signal_amplitude / noise_amplitude)),
        signal_amplitude=float(signal_amplitude),
        noise_amplitude=float(noise_amplitude),
        sigma_tx=float(r["sigma_Tx"]),
        sigma_jitter=float(r["sigma_J"]),
        sigma_noise=float(r["sigma_N"]),
        sigma_gaussian=float(r["sigma_G"]),
        tx_taps=np.array(com._tx_combs[TX_COMB_IX]),
        channel_path=path,
        baud_rate=float(cfg.fb) * 1e9,
        freq_step=float(cfg.fstep) * 1e9,
        samples_per_ui=int(cfg.M),
        levels=int(cfg.L),
        rlm=float(cfg.RLM),
        victim_amplitude=float(cfg.A_v),
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
    )


def _build_com(ref: Reference) -> Com:
    zero, pole1, pole2, shelf = ref.ctle_freqs
    link = Link(
        channel=SParameterChannel.from_touchstone(str(ref.channel_path)),
        ctle=TwoStageCtle(
            zero_freq=zero,
            pole1_freq=pole1,
            pole2_freq=pole2,
            shelf_freq=shelf,
            dc_gain_db=G_DC,
            shelf_gain_db=G_DC2,
        ),
        ffe=TapWeightFfe(
            tap_weights=ref.tx_taps,
            n_post=N_TX_POST_TAPS,
            tap_delay=1.0 / ref.baud_rate,
        ),
        rx_afe=RxAfeButterworth(cutoff_freq=ref.afe_cutoff),
    )
    params = ComParams(
        baud_rate=ref.baud_rate,
        freq_step=ref.freq_step,
        samples_per_ui=ref.samples_per_ui,
        levels=ref.levels,
        rlm=ref.rlm,
        victim_amplitude=ref.victim_amplitude,
        snr_tx=ref.snr_tx,
        sigma_rj=ref.sigma_rj,
        eta_0=ref.eta_0,
        a_dd=ref.a_dd,
        der_0=ref.der_0,
        dfe_min=ref.dfe_min,
        dfe_max=ref.dfe_max,
    )
    return Com(link=link, params=params)


@pytest.mark.usefixtures("exact_pi")
def test_com_value_matches_pychopmarg(reference: Reference) -> None:
    """Weak on its own, and deliberately kept anyway.

    In this configuration the interference distribution is far wider than
    the +/-1.1*As grid it's quantized onto, so the CDF already exceeds
    der_0 at the very first grid point and Ani saturates at the grid edge
    on *both* sides. COM then collapses to 20*log10(1/1.1) = -0.83 dB for
    either implementation regardless of what they computed on the way
    there — so agreement here is close to free.

    Constructing a case that avoids this proved impractical: PyChOpMarg's
    calc_H21 tapers S21 to zero across the measured band, so a channel
    file must reach ~7x Nyquist for the signal band to survive, and every
    combination tried (a real measured backplane, several synthetic
    channels, both IEEE_8023by and IEEE_8023dj, 1 to 14 DFE taps) landed
    in the same saturated regime.

    test_intermediate_quantities_match_pychopmarg below is what actually
    carries the weight — every term feeding this number is checked
    individually, where saturation can't hide a discrepancy.
    """
    result = _build_com(reference).compute()

    assert result.com_db == pytest.approx(reference.com_db, abs=1e-6)


@pytest.mark.usefixtures("exact_pi")
def test_intermediate_quantities_match_pychopmarg(reference: Reference) -> None:
    """The headline number could match while a term inside is wrong, since
    COM is a ratio and the noise terms combine — so each is checked.
    """
    result = _build_com(reference).compute()

    assert result.signal_amplitude == pytest.approx(reference.signal_amplitude, rel=1e-9)
    assert result.noise_amplitude == pytest.approx(reference.noise_amplitude, rel=1e-9)
    assert result.sigma_tx == pytest.approx(reference.sigma_tx, rel=1e-9)
    assert result.sigma_jitter == pytest.approx(reference.sigma_jitter, rel=1e-9)
    assert result.sigma_noise == pytest.approx(reference.sigma_noise, rel=1e-9)
    assert result.sigma_gaussian == pytest.approx(reference.sigma_gaussian, rel=1e-9)


@pytest.mark.usefixtures("exact_pi")
def test_result_carries_a_usable_distribution(reference: Reference) -> None:
    """The PMF and its axis are carried for diagnosis and plotting, so
    they should be a genuine distribution on a genuine axis, not leftovers.
    """
    result = _build_com(reference).compute()

    assert result.noise_pmf.shape == result.voltage_grid.shape
    assert result.noise_pmf.sum() == pytest.approx(1.0)
    assert (result.noise_pmf >= 0).all()
