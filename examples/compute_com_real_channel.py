"""End-to-end example: compute COM (IEEE 802.3-2022 Annex 93A) for a real,
measured channel — PyChOpMarg's own bundled "Example 2" data (VITA 68.2,
6 s4p files: 1 THRU + 2 FEXT + 3 NEXT aggressors), through an
IEEE-802.3by-style config with a single-segment package transmission
line. Same computation tests/evaluate/test_pychopmarg_example2.py checks
(COM ~= 3.58 dB), here as a runnable script rather than a test: fixed
CTLE gain and Tx FFE taps, no equalization search (see that test's
docstring for why — a full search over real, 4001-point channel data is
expensive; optimize.search() is where the actual search lives).

Requires tests/data/pychopmarg_example2/*.s4p — checked into the repo
(unlike reference/'s gitignored files): see that directory's README.md
for provenance.

Run: python examples/compute_com_real_channel.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from serdeskit.com import LinkComParams, compute

DATA = Path("../tests/data/pychopmarg_example2")

BAUD_RATE = 25.78125e9
FREQ_STEP = 10e6
SAMPLES_PER_UI = 32
R0 = 50.0
R_D = 55.0  # this example's own (non-matched) die termination
GAMMA = (R_D - R0) / (R_D + R0)


def main() -> None:
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
        port_order=(0, 2, 1, 3),  # differential_network's own default (ECEN720/PyBERT interleaved convention)
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
        # an arbitrary, valid combination (no search — see module docstring).
        ffe_tap_weights=np.array([0.0, 0.0, -0.18, -0.12, 0.0, 0.0]),
        ffe_n_post=3,
        # Identity -- this config has no verified real Tr, and this
        # example's own docstring says it matches test_pychopmarg_example2.py's
        # PyChOpMarg comparison, whose own H() has no Tx risetime factor.
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

    print(f"COM:               {result.com_db:.3f} dB")
    print(f"Signal amplitude:  {result.signal_amplitude * 1e3:.3f} mV")
    print(f"Noise amplitude:   {result.noise_amplitude * 1e3:.3f} mV")
    print(f"sigma_Tx:          {result.sigma_tx * 1e3:.3f} mV")
    print(f"sigma_Jitter:      {result.sigma_jitter * 1e3:.3f} mV")
    print(f"sigma_Noise:       {result.sigma_noise * 1e3:.3f} mV")


if __name__ == "__main__":
    main()
