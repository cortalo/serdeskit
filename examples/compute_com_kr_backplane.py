"""End-to-end example: compute COM (IEEE 802.3-2022 Annex 93A) for the
IEEE 802.3ck "KR" backplane config (53.125 GBd PAM4) against a real,
measured backplane channel — the same pairing this project's own MATLAB
COM3.70 run used (see reference/matlab/COM3.70/): KR_eval got COM=3.608
dB, PASS, for this exact channel/config.

Unlike an earlier version of this script, this calls serdeskit's own
top-level entry point, evaluate.evaluate_channel, directly -- it already
runs the equalization search then the PMF-based compute, the same
search-then-compute flow tests/evaluate/test_evaluate.py exercises. This
script's own job is only to build the ComStandard the KR config
describes and point it at real Touchstone files, not to reimplement
evaluate_channel's own body.

The real KR search space is far too large to run as-is, though: g_DC
alone is 21 candidates ([-20:1:0] dB), g_DC_HP another 7, and the Tx
tap grid (c(-3)/c(-2)/c(-1)/c(1), each its own [min:step:max] range)
multiplies out to ~31k combinations -- ~650k grid points total. Every
point costs a package-cascaded channel-transfer-function interpolation
per victim/FEXT channel (test_evaluate.py's own docstring covers why
that's the expensive part). DC_GAIN_CANDIDATES/SHELF_GAIN_CANDIDATES/
TX_TAPS_BOUNDS below use much coarser steps over the *same* min/max
ranges the config specifies, the same kind of grid-shrink
test_evaluate.py's own fixture applies to a synthetic channel, here
applied to a real one instead.

Don't expect this script's own COM to land near MATLAB's 3.608 dB: this
config's own "Floating Tap Control" section (N_bg=3 groups x N_bf=3
taps, spanning up to N_f=40 UI beyond the 12 fixed DFE taps below)
models a sparse long-reach DFE serdeskit doesn't implement -- com.Com
only ever cancels the N taps dfe_min/dfe_max's length gives it
((93A-26)/(93A-27), pulse_response.residual_isi). Less cancellation
capability than the real KR receiver model means more residual ISI
here, hence a lower COM than MATLAB's own -- a scope gap, not a bug:
this is the first PAM4 config this project has run at all, and it
still executes cleanly end to end, search included.

Requires two things NOT checked into this repo (both gitignored, unlike
tests/data/pychopmarg_example2/):

  - The channel data: 6 s4p files (1 THRU + 5 FEXT + 3 NEXT), a Std_BP_
    12inch_Meg7 traditional backplane model, TE Connectivity 2019.
    Download: https://www.ieee802.org/3/ck/public/tools/backplane/tracy_3ck_03_0119_tradBP.zip
    Unzip into reference/ck_channels/tradBP_15dB/.

  - The KR config's own parameter values below were read out of the
    official MATLAB COM tool's KR config sheet (not redistributed here):
    https://www.ieee802.org/3/ck/public/tools/tools/mellitz_3ck_adhoc_01_032322_COM3p70.zip
    -> config_sheets_3p1/config_com_ieee8023_93a=3ck_d3p1_KR_11_30_21.xlsx,
    'COM_Settings' sheet.

Run: python examples/compute_com_kr_backplane.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from serdeskit.evaluate import ComStandard, evaluate_channel

DATA = Path("../reference/ck_channels/tradBP_15dB")

# Config's own g_DC=[-20:1:0] (21 candidates), coarsened to every 5 dB.
DC_GAIN_CANDIDATES = [-20.0, -15.0, -10.0, -5.0, 0.0]
# Config's own g_DC_HP=[-6:1:0] (7 candidates), coarsened to every 3 dB.
SHELF_GAIN_CANDIDATES = [-6.0, -3.0, 0.0]
# Config's own c(-3)/c(-2)/c(-1)/c(1) ranges, each [min:0.02:max] ->
# coarsened to 2-3 values per tap here (same min/max, bigger step).
TX_TAPS_BOUNDS = [
    (-0.06, 0.0, 0.06),  # c(-3): [-0.06:0.02:0]
    (0.0, 0.12, 0.06),  # c(-2): [0:0.02:0.12]
    (-0.34, 0.0, 0.17),  # c(-1): [-0.34:0.02:0]
    (-0.2, 0.0, 0.2),  # c(1): [-0.2:0.02:0]
]

STANDARD = ComStandard(
    baud_rate=53.125e9,
    freq_step=10e6,
    samples_per_ui=32,
    levels=4,  # PAM4
    rlm=0.95,
    victim_amplitude=0.413,
    a_ne=0.608,
    a_fe=0.413,
    snr_tx=33.0,
    sigma_rj=0.01,
    eta_0=8.2e-9,
    a_dd=0.02,
    der_0=1e-4,
    # b_min/b_max(1..12) from the config sheet -- per-tap DFE bounds,
    # not the uniform +-1 every other example in this project uses.
    dfe_min=np.array([0.3, 0.05, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03]),
    dfe_max=np.array([0.85, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ctle_zero_freq=21.25e9,
    ctle_pole1_freq=21.25e9,
    ctle_pole2_freq=53.125e9,
    ctle_shelf_freq=0.6640625e9,
    ctle_dc_gain_candidates=DC_GAIN_CANDIDATES,
    ctle_shelf_gain_candidates=SHELF_GAIN_CANDIDATES,
    tx_taps_bounds=TX_TAPS_BOUNDS,
    tx_taps_c0_min=0.54,
    tx_taps_n_post=1,  # only c(1) is post-cursor
    rx_afe_cutoff_freq=0.75 * 53.125e9,
    r0=50.0,
    tx_termination_resistance=50.0,  # R_d = [50, 50] in this config too -- matched, gamma1=gamma2=0
    rx_termination_resistance=50.0,
    # package_Z_c = [87.5 87.5; 92.5 92.5] Ohm, z_p/z_pB (package case 1, "z_p select"=1)
    # = 12/1.8 mm identically for TX/RX/NEXT/FEXT -- a two-segment package line,
    # same shape as every IEEE_8023dj-based test in this project (unlike
    # examples/compute_com_real_channel.py's single-segment 802.3by case).
    tx_die_capacitances=[1.2e-4 * 1e-9],  # C_d = 1.2e-4 nF
    tx_die_inductances=[0.12e-9],  # L_s = 0.12 nH
    tx_bump_capacitance=0.3e-4 * 1e-9,  # C_b = 0.3e-4 nF
    tx_pad_capacitance=0.87e-4 * 1e-9,  # C_p = 0.87e-4 nF
    rx_die_capacitances=[1.2e-4 * 1e-9],
    rx_die_inductances=[0.12e-9],
    rx_bump_capacitance=0.3e-4 * 1e-9,
    rx_pad_capacitance=0.87e-4 * 1e-9,
    package_tline_a1=0.0009909,
    package_tline_a2=0.0002772,
    package_tline_tau=0.006141,
    package_tline_gamma0=0.0,
    package_tline_segments=[(87.5, 12.0), (92.5, 1.8)],
    com_min_db=3.0,  # "COM Pass threshold" in the config sheet
)


def main() -> None:
    evaluation = evaluate_channel(
        STANDARD,
        thru_path=DATA / "Std_BP_12inch_Meg7_Thru_B56.s4p",
        next_paths=[
            DATA / "Std_BP_12inch_Meg7_NEXT_A23.s4p",
            DATA / "Std_BP_12inch_Meg7_NEXT_A56.s4p",
            DATA / "Std_BP_12inch_Meg7_NEXT_A89.s4p",
        ],
        fext_paths=[
            DATA / "Std_BP_12inch_Meg7_FEXT_B23.s4p",
            DATA / "Std_BP_12inch_Meg7_FEXT_B89.s4p",
            DATA / "Std_BP_12inch_Meg7_FEXT_C23.s4p",
            DATA / "Std_BP_12inch_Meg7_FEXT_C56.s4p",
            DATA / "Std_BP_12inch_Meg7_FEXT_C89.s4p",
        ],
        # Std_BP_12inch_Meg7's own port order is (TX+, TX-, RX+, RX-) --
        # already adjacent-paired (see Index_S4P-2019-3628.txt: "Port 1/2
        # -> TX side, Port 3/4 -> RX side") -- unlike the ECEN720
        # peters_*/Case4_* files' interleaved (TX+, RX+, TX-, RX-) order
        # differential_network assumes by default.
        port_order=(0, 1, 2, 3),
    )
    result = evaluation.result

    print(f"CTLE DC gain:      {evaluation.dc_gain_db:.1f} dB")
    print(f"CTLE shelf gain:   {evaluation.shelf_gain_db:.1f} dB")
    print(f"Tx FFE taps:       {evaluation.tx_taps}")
    print()
    verdict = "PASS" if evaluation.passes else "FAIL"
    print(f"COM:               {result.com_db:.3f} dB ({verdict} @ {evaluation.com_min_db} dB)")
    print(f"Signal amplitude:  {result.signal_amplitude * 1e3:.3f} mV")
    print(f"Noise amplitude:   {result.noise_amplitude * 1e3:.3f} mV")
    print(f"sigma_Tx:          {result.sigma_tx * 1e3:.3f} mV")
    print(f"sigma_Jitter:      {result.sigma_jitter * 1e3:.3f} mV")
    print(f"sigma_Noise:       {result.sigma_noise * 1e3:.3f} mV")


if __name__ == "__main__":
    main()
