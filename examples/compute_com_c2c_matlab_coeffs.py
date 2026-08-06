"""Diagnostic example: compute COM for the IEEE 802.3ck "C2C" config using
MATLAB COM3.70's own found-optimal equalization coefficients directly,
instead of serdeskit's own search.

examples/compute_com_c2c.py (search-then-compute, the normal path) gets
COM=2.295 dB against MATLAB's own COM3.70 run of 5.299 dB PASS for the
same channel/config -- a real ~3 dB gap, unlike the KR example's gap
(which turned out to be a noise_margin saturation artifact, see
docs/known-issues.md; C2C's config has floating taps off, so that bug
doesn't apply here). This script isolates one candidate explanation:
does serdeskit's own search simply fail to find as good a point as
MATLAB's exact one, on the *same* coarsened grid compute_com_c2c.py
uses? Plugging MATLAB's coefficients straight in (no search at all)
answers that directly.

Two fixes landed here, each closing part of the gap:

1. `Link.sbr_pulse_response()` (MATLAB's own real time-domain chain,
   what `compute()` always uses), not the old frequency-domain
   composition this project used to have (~1-2% residual against
   MATLAB -- see docs/known-issues.md's "full composed pulse response"
   entry). On its own this barely moved COM (that residual was always
   too small to explain a multi-dB gap) -- but it's the actually-correct
   pulse response, verified independently.
2. RX_PACKAGE's own length: was 13mm (a uniform-package simplification),
   should be 11mm -- this config's own real z_p(RX) for package case 1
   (see compute_com_c2c.py's own docstring: z_p(TX)=13, z_p(NEXT)=11,
   z_p(FEXT)=13, z_p(RX)=11mm). This was the dominant error: fixing it
   alone moves COM from 2.821 dB to 5.565 dB.

Current result, both fixes applied: COM=5.565 dB (search-free numbers
below are updated to match) vs. MATLAB's 5.299 dB -- ~0.27 dB
remains, plausibly from the same NEXT/FEXT package-length simplification
compute_com_c2c.py's own docstring already documents as accepted
(evaluate_channel/this script apply one TX_PACKAGE/RX_PACKAGE pair to
every channel uniformly, victim and aggressors alike, rather than
per-aggressor-type lengths) -- not yet confirmed, next thing to check if
picked up.

MATLAB's case-1 result (package case 1, 13mm TX / 11mm RX):
  CTLE DC gain:     -3 dB
  CTLE shelf gain:  -2 dB
  TXLE_taps:        [-0.02, 0.06, -0.2, 0.68, -0.04]  (c(-3..1), cursor c(0)=0.68)
  COM:              5.299 dB, PASS

Requires the same two things compute_com_c2c.py does (both gitignored):
reference/ck_channels/c2c_pcb/ (see that script's docstring for the
download link) and this config's own parameter values (from the MATLAB
COM tool's C2C config sheet, not redistributed here).

Run: python examples/compute_com_c2c_matlab_coeffs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.com import LinkComParams, compute
from serdeskit.util import plot_pulse_response_cursors, plot_signal

DATA = Path("../reference/ck_channels/c2c_pcb")

BAUD_RATE = 53.125e9
FREQ_STEP = 10e6
SAMPLES_PER_UI = 32
R0 = 50.0  # R_d = [50, 50] in this config too -- matched termination, gamma1 = gamma2 = 0


def main() -> None:
    params = LinkComParams(
        channel_path=str(DATA / "C2C_PCB_SYSVIA_12dB_thru.s4p"),
        next_channel_paths=[
            str(DATA / f"C2C_PCB_SYSVIA_12dB_next{n}.s4p") for n in [1, 2, 3, 4]
        ],
        fext_channel_paths=[
            str(DATA / f"C2C_PCB_SYSVIA_12dB_fext{n}.s4p") for n in [1, 2, 3, 4, 5, 6]
        ],
        # (TX+, RX+, TX-, RX-) matches this file's own S21 (~0.97 near
        # unity at low frequency, between ports 1-2 and 3-4) -- same as
        # compute_com_c2c.py's own evaluate_channel call.
        port_order=(0, 2, 1, 3),
        gamma1=0.0,  # R_d = [50, 50] -- matched termination, no reflection
        gamma2=0.0,
        # package_Z_c = [87.5 87.5; 92.5 92.5] Ohm, z_p (package case 1) =
        # 13/1.8mm for TX, 11/1.8mm for RX (case 1's real z_p(RX) -- see
        # module docstring). One TX/RX package pair applied to every
        # channel uniformly (victim and aggressors alike) -- same
        # simplification compute_com_c2c.py's own docstring documents
        # (real z_p(NEXT)=11mm, z_p(FEXT)=13mm differ from this uniform
        # TX/FEXT=13mm treatment; not separately modeled here either).
        tx_r0=R0,
        tx_die_capacitances=[1.2e-4 * 1e-9],  # C_d = 1.2e-4 nF
        tx_die_inductances=[0.12e-9],  # L_s = 0.12 nH
        tx_bump_capacitance=0.3e-4 * 1e-9,  # C_b = 0.3e-4 nF
        tx_tline_a1=0.0009909,
        tx_tline_a2=0.0002772,
        tx_tline_tau=0.006141,
        tx_tline_gamma0=0.0,
        tx_tline_segments=[(87.5, 13.0), (92.5, 1.8)],
        tx_pad_capacitance=0.87e-4 * 1e-9,  # C_p = 0.87e-4 nF
        rx_r0=R0,
        rx_die_capacitances=[1.2e-4 * 1e-9],
        rx_die_inductances=[0.12e-9],
        rx_bump_capacitance=0.3e-4 * 1e-9,
        rx_tline_a1=0.0009909,
        rx_tline_a2=0.0002772,
        rx_tline_tau=0.006141,
        rx_tline_gamma0=0.0,
        rx_tline_segments=[(87.5, 11.0), (92.5, 1.8)],  # 11mm, package case 1's real z_p(RX)
        rx_pad_capacitance=0.87e-4 * 1e-9,
        # MATLAB's own found-optimal point for this channel, package case
        # 1: "TXFFE coefficients: [-0.02 0.06 -0.2 0.68 -0.04]" = [c(-3),
        # c(-2), c(-1), c(0), c(1)] -- the cursor (0.68) is computed
        # implicitly here (TapWeightFfe's own convention), so only the
        # other four are passed, ffe_n_post=1 (only c(1) is post-cursor).
        ctle_zero_freq=21.25e9,
        ctle_pole1_freq=21.25e9,
        ctle_pole2_freq=53.125e9,
        ctle_shelf_freq=0.6640625e9,
        ctle_dc_gain_db=-3.0,
        ctle_shelf_gain_db=-2.0,
        ffe_tap_weights=np.array([-0.02, 0.06, -0.2, -0.04]),
        ffe_n_post=1,
        tx_risetime=0.0075e-9,  # this config's own T_r -- see matlab_golden/generate/gen_tx_h_t.m
        rx_afe_cutoff_freq=0.75 * BAUD_RATE,
        rx_ffe_tap_weights=np.array([1.0]),
        rx_ffe_n_pre=0,
        baud_rate=BAUD_RATE,
        freq_step=FREQ_STEP,
        samples_per_ui=SAMPLES_PER_UI,
        levels=4,  # PAM4
        rlm=0.95,
        victim_amplitude=0.413,
        a_ne=0.608,
        a_fe=0.413,
        snr_tx=33.0,
        sigma_rj=0.01,
        eta_0=2e-8,
        a_dd=0.02,
        der_0=1e-5,  # unlike KR's 1e-4 -- C2C's own config value
        # b_min/b_max(1..6) from the config sheet -- per-tap DFE bounds,
        # not the uniform +-1 every other example in this project uses.
        dfe_min=np.array([0.3, 0.05, -0.04, -0.04, -0.04, -0.04]),
        dfe_max=np.array([0.65, 0.15, 0.1, 0.1, 0.1, 0.1]),
    )

    result = compute(params)

    print("Using MATLAB's own found-optimal coefficients directly (no search):")
    print("  CTLE DC gain:      -3.0 dB")
    print("  CTLE shelf gain:   -2.0 dB")
    print("  Tx FFE taps:       [-0.02, 0.06, -0.2, -0.04]")
    print()
    verdict = "PASS" if result.com_db >= 3.0 else "FAIL"
    print(f"COM:               {result.com_db:.3f} dB ({verdict} @ 3.0 dB)  -- MATLAB got 5.299 dB PASS")
    print(f"Signal amplitude:  {result.signal_amplitude * 1e3:.3f} mV")
    print(f"Noise amplitude:   {result.noise_amplitude * 1e3:.3f} mV")
    print(f"sigma_Tx:          {result.sigma_tx * 1e3:.3f} mV")
    print(f"sigma_Jitter:      {result.sigma_jitter * 1e3:.3f} mV")
    print(f"sigma_Noise:       {result.sigma_noise * 1e3:.3f} mV")
    print(f"sigma_ISI:         {result.sigma_isi * 1e3:.3f} mV")
    print(f"sigma_Crosstalk:   {result.sigma_crosstalk * 1e3:.3f} mV")

    _, ax = plt.subplots()
    plot_signal(
        result.half_signal_unequalized_pulse_response, ax, label="Half Symbol Unequalized end-to-end PR"
    )
    plot_signal(
        result.half_signal_equalized_pulse_response, ax, label="Half Symbol Equalized end-to-end PR"
    )
    plot_pulse_response_cursors(
        result.half_signal_sampled_pulse_response, ax, dfe_min=params.dfe_min, dfe_max=params.dfe_max
    )
    ax.set_xlabel("seconds")
    ax.set_ylabel("volts")
    ax.set_title("C2C thru, MATLAB coeffs")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
