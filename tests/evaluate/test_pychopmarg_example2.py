"""Runs serdeskit's own Com/Link/Package pipeline against PyChOpMarg's
own bundled "Example 2" (notebook/PyChOpMarg.ipynb): 6 real, measured s4p
files (VITA 68.2, tests/data/pychopmarg_example2/, copied verbatim from
PyChOpMarg's own chnl_data/ under its BSD-3 license), through an
IEEE-802.3by-style config — asserted against hardcoded expected values
rather than a live PyChOpMarg run, so this test has no dependency on
PyChOpMarg at all.

The expected values were captured from this exact pipeline's own output
(not independently re-derived) — this is a regression/coverage test for
two things this project hadn't exercised before, not a golden test
against an external reference:

  - Real, measured Touchstone data (4001 points, DC-40GHz) instead of a
    hand-built synthetic S-parameter network.
  - A single-segment package transmission line (z_c has one entry here,
    vs. every IEEE_8023dj-based test's two) — z_p[0]/z_pB pairing still
    produces a 1-element tline_segments list once z_c is length 1, same
    mechanism as the two-segment case, just never previously exercised
    down that branch.

No search: CTLE gain/Tx taps are fixed at one arbitrary, valid setting
(same role as test_com_end_to_end.py's own G_DC/TX_COMB_IX) rather than
run through EqualizationSearch — this project's own real-data package-
cascaded search grid is expensive (test_evaluate.py already documents
why its own, much smaller synthetic-channel grid needed shrinking); a
fixed setting is enough to exercise the pipeline this test is about.
"""
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.com import Com, ComParams
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Link, SystemGrid
from serdeskit.package import Package, cascade_channel
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe
from serdeskit.tx_filter import TxRisetimeFilter

DATA = Path(__file__).parent.parent / "data" / "pychopmarg_example2"

BAUD_RATE = 25.78125e9
FREQ_STEP = 10e6
SAMPLES_PER_UI = 32
R0 = 50.0
GAMMA = (55.0 - R0) / (55.0 + R0)  # this example's own (non-matched) R_d=55 die termination

TX_PACKAGE = Package(
    r0=R0,
    die_capacitances=[0.250e-12],
    die_inductances=[0.0],
    bump_capacitance=0.001e-12,
    tline_a1=8.9e-4,
    tline_a2=2.0e-4,
    tline_tau=6.141e-3,
    tline_gamma0=5.0e-4,
    tline_segments=[(78.2, 12.0)],  # single segment: z_c=[78.2], z_p[zp_sel=0]=12
    pad_capacitance=0.180e-12,
    is_rx=False,
)
RX_PACKAGE = Package(
    r0=R0,
    die_capacitances=[0.250e-12],
    die_inductances=[0.0],
    bump_capacitance=0.001e-12,
    tline_a1=8.9e-4,
    tline_a2=2.0e-4,
    tline_tau=6.141e-3,
    tline_gamma0=5.0e-4,
    tline_segments=[(78.2, 12.0)],
    pad_capacitance=0.180e-12,
    is_rx=True,
)


def _load_channel(path: Path, freqs: npt.NDArray[np.float64]) -> SParameterChannel:
    raw = differential_network(skrf.Network(str(path)))
    cascaded = cascade_channel(raw, TX_PACKAGE, RX_PACKAGE, freqs)
    return SParameterChannel(cascaded, gamma1=GAMMA, gamma2=GAMMA)


def test_com_on_real_example2_data() -> None:
    grid = SystemGrid.build(BAUD_RATE, FREQ_STEP, SAMPLES_PER_UI)
    tap_delay = 1.0 / BAUD_RATE

    channel = _load_channel(DATA / "example2_THRU.s4p", grid.f)
    ctle = TwoStageCtle(
        zero_freq=6.4453125e9,
        pole1_freq=6.4453125e9,
        pole2_freq=25.78125e9,
        shelf_freq=0.1e9,
        dc_gain_db=-6.0,
        shelf_gain_db=0.0,
    )
    # tx_taps: (pre2, pre1, cursor-adjacent pre, post1, post2, post3) per
    # this config's own tx_taps_min/max/step ([0,0,-0.18,-0.38,0,0] /
    # [0,0,0,0,0,0] / [0,0,0.02,0.02,0,0]); -0.12 = -0.38 + 13*0.02 — an
    # arbitrary, valid combination (see module docstring: no search here).
    ffe = TapWeightFfe(tap_weights=np.array([0.0, 0.0, -0.18, -0.12, 0.0, 0.0]), n_post=3, tap_delay=tap_delay)
    # Identity -- this config has no verified real Tr, and the hardcoded
    # expected values below predate tx_filter existing at all (module
    # docstring: they're this pipeline's own prior output, not an
    # independent reference), so risetime=0.0 keeps them valid unchanged.
    tx_filter = TxRisetimeFilter(risetime=0.0)
    rx_afe = RxAfeButterworth(cutoff_freq=0.75 * BAUD_RATE)
    rx_ffe = TapWeightRxFfe(tap_weights=np.array([1.0]), tap_delay=tap_delay)

    link = Link(channel=channel, ctle=ctle, ffe=ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe)
    params = ComParams(
        baud_rate=BAUD_RATE,
        freq_step=FREQ_STEP,
        samples_per_ui=SAMPLES_PER_UI,
        levels=2,
        rlm=1.0,
        victim_amplitude=0.4,
        a_ne=0.6,
        a_fe=0.4,
        snr_tx=27.0,
        sigma_rj=0.01,
        eta_0=5.20e-8,
        a_dd=0.05,
        der_0=1e-5,
        dfe_min=np.array([-1.0] * 14),
        dfe_max=np.array([1.0] * 14),
    )
    next_channels = [
        _load_channel(DATA / "example2_NEXT1.s4p", grid.f),
        _load_channel(DATA / "example2_NEXT2.s4p", grid.f),
        _load_channel(DATA / "example2_NEXT3.s4p", grid.f),
    ]
    fext_channels = [
        _load_channel(DATA / "example2_FEXT1.s4p", grid.f),
        _load_channel(DATA / "example2_FEXT2.s4p", grid.f),
    ]

    result = Com(link=link, params=params, next_channels=next_channels, fext_channels=fext_channels).compute()

    assert result.com_db == pytest.approx(3.580216471679006, rel=1e-9)
    assert result.signal_amplitude == pytest.approx(0.030576869355415422, rel=1e-9)
    assert result.noise_amplitude == pytest.approx(0.020248002887156096, rel=1e-9)
    assert result.sigma_tx == pytest.approx(0.0013658185840407663, rel=1e-9)
    assert result.sigma_jitter == pytest.approx(0.00025119611998381306, rel=1e-9)
    assert result.sigma_noise == pytest.approx(0.0007599076517400535, rel=1e-9)
