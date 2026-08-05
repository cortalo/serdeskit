"""Runs serdeskit's own compute()/Link/Package pipeline against PyChOpMarg's
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
run through search() -- this project's own real-data package-cascaded
search grid is expensive; a fixed setting is enough to exercise the
pipeline this test is about.
"""
from pathlib import Path

import numpy as np
import pytest

from serdeskit.com import LinkComParams, compute

DATA = Path(__file__).parent.parent / "data" / "pychopmarg_example2"

BAUD_RATE = 25.78125e9
FREQ_STEP = 10e6
SAMPLES_PER_UI = 32
R0 = 50.0
R_D = 55.0  # this example's own (non-matched) die termination
GAMMA = (R_D - R0) / (R_D + R0)


def test_com_on_real_example2_data() -> None:
    params = LinkComParams(
        channel_path=str(DATA / "example2_THRU.s4p"),
        next_channel_paths=[
            str(DATA / "example2_NEXT1.s4p"),
            str(DATA / "example2_NEXT2.s4p"),
            str(DATA / "example2_NEXT3.s4p"),
        ],
        fext_channel_paths=[
            str(DATA / "example2_FEXT1.s4p"),
            str(DATA / "example2_FEXT2.s4p"),
        ],
        port_order=(0, 2, 1, 3),
        gamma1=GAMMA,
        gamma2=GAMMA,
        tx_r0=R0,
        tx_die_capacitances=[0.250e-12],
        tx_die_inductances=[0.0],
        tx_bump_capacitance=0.001e-12,
        tx_tline_a1=8.9e-4,
        tx_tline_a2=2.0e-4,
        tx_tline_tau=6.141e-3,
        tx_tline_gamma0=5.0e-4,
        tx_tline_segments=[(78.2, 12.0)],  # single segment: z_c=[78.2], z_p[zp_sel=0]=12
        tx_pad_capacitance=0.180e-12,
        rx_r0=R0,
        rx_die_capacitances=[0.250e-12],
        rx_die_inductances=[0.0],
        rx_bump_capacitance=0.001e-12,
        rx_tline_a1=8.9e-4,
        rx_tline_a2=2.0e-4,
        rx_tline_tau=6.141e-3,
        rx_tline_gamma0=5.0e-4,
        rx_tline_segments=[(78.2, 12.0)],
        rx_pad_capacitance=0.180e-12,
        ctle_zero_freq=6.4453125e9,
        ctle_pole1_freq=6.4453125e9,
        ctle_pole2_freq=25.78125e9,
        ctle_shelf_freq=0.1e9,
        ctle_dc_gain_db=-6.0,
        ctle_shelf_gain_db=0.0,
        # (pre2, pre1, cursor-adjacent pre, post1, post2, post3) per this
        # config's own tx_taps_min/max/step ([0,0,-0.18,-0.38,0,0] /
        # [0,0,0,0,0,0] / [0,0,0.02,0.02,0,0]); -0.12 = -0.38 + 13*0.02 —
        # an arbitrary, valid combination (see module docstring: no
        # search here).
        ffe_tap_weights=np.array([0.0, 0.0, -0.18, -0.12, 0.0, 0.0]),
        ffe_n_post=3,
        # Identity -- this config has no verified real Tr, and the
        # hardcoded expected values below predate tx_filter existing at
        # all (module docstring: they're this pipeline's own prior
        # output, not an independent reference), so risetime=0.0 keeps
        # them valid unchanged.
        tx_risetime=0.0,
        rx_afe_cutoff_freq=0.75 * BAUD_RATE,
        rx_ffe_tap_weights=np.array([1.0]),
        rx_ffe_n_pre=0,
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

    result = compute(params)

    assert result.com_db == pytest.approx(3.6526596742481168, rel=1e-9)
    assert result.signal_amplitude == pytest.approx(0.03041776030593954, rel=1e-9)
    assert result.noise_amplitude == pytest.approx(0.019975343192910496, rel=1e-9)
    assert result.sigma_tx == pytest.approx(0.001358711443864405, rel=1e-9)
    assert result.sigma_jitter == pytest.approx(0.0002599216840465506, rel=1e-9)
    assert result.sigma_noise == pytest.approx(0.0007599076517400535, rel=1e-9)
