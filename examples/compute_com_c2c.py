"""End-to-end example: compute COM (IEEE 802.3-2022 Annex 93A) for the
IEEE 802.3ck "C2C" (chip-to-chip AUI) config (53.125 GBd PAM4) against a
real on-board channel -- the same pairing this project's own MATLAB
COM3.70 run used (see reference/matlab/COM3.70/): the tool got
COM=5.299 dB, PASS, for this exact channel/config (worst of its two
package test cases).

Unlike examples/compute_com_kr_backplane.py's own KR config, C2C's
"Floating Tap Control" section has N_bg=0 -- the floating-tap DFE
extension serdeskit doesn't implement (see docs/known-issues.md) never
activates here, so this is a fair, apples-to-apples comparison: if
serdeskit's own search-then-compute is correct, it should land in the
same ballpark as MATLAB's 5.299 dB, not collapse to the noise_margin
saturation artifact documented there.

Same grid-shrink rationale as the KR example: C2C's real ranges are
smaller (g_DC still 21 candidates, but g_DC_HP only 5, and each Tx tap's
own range is narrower), but the full cross product is still large enough
to be impractical for a routine script, so DC_GAIN_CANDIDATES/
TX_TAPS_BOUNDS below coarsen it (SHELF_GAIN_CANDIDATES needs no
coarsening -- the real range is only 5 candidates already).

Package note: this config's own z_p(TX)=[13, 31], z_p(NEXT)=[11, 29],
z_p(FEXT)=[13, 31], z_p(RX)=[11, 29] mm (package case 1 values: 13/11/13/11)
-- unlike the KR config, where case 1 happened to be uniform (12mm
everywhere), C2C's NEXT/RX case-1 length (11mm) genuinely differs from
TX/FEXT's (13mm). search() applies one TX/RX package pair to every
channel uniformly (victim and aggressors alike); this script follows
that same simplification rather than extending the architecture to carry
a NEXT-specific package length, so expect a small extra deviation from
MATLAB's own per-aggressor-type package modeling on top of whatever the
search grid's coarseness already contributes.

Requires two things NOT checked into this repo (both gitignored, unlike
tests/data/pychopmarg_example2/):

  - The channel data: 11 s4p files (1 thru + 4 NEXT + 6 FEXT) at the
    12 dB corner of Gore's "Channel Models for 100/200/400 Gb/s C2C AUI"
    (PCB variant -- on-board, no cable).
    Download: https://www.ieee802.org/3/ck/public/tools/c2c/gore_3ck_02_0519_PCB.zip
    Unzip into reference/ck_channels/c2c_pcb/.

  - The C2C config's own parameter values below were read out of the
    official MATLAB COM tool's C2C config sheet (not redistributed here):
    https://www.ieee802.org/3/ck/public/tools/tools/mellitz_3ck_adhoc_01_032322_COM3p70.zip
    -> config_sheets_3p1/config_com_ieee8023_93a=3ck_d3p1_120F_C2C_11_30_21.xlsx,
    'COM_Settings' sheet.

Run: python examples/compute_com_c2c.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from serdeskit.com import compute
from serdeskit.optimize import ComStandard, search

DATA = Path("../reference/ck_channels/c2c_pcb")

# Grid shrunk to a single point: MATLAB's own found-optimal case-1
# coefficients (same numbers compute_com_c2c_matlab_coeffs.py plugs in
# directly, no search) -- CTLE DC gain -3dB, shelf gain -2dB, TXFFE taps
# [-0.02, 0.06, -0.2, 0.68, -0.04] (c(-3..1), cursor c(0)=0.68 derived).
# search()'s own candidate list always prepends the all-zero (flat,
# unequalized) combination unconditionally on top of whatever
# TX_TAPS_BOUNDS produces (matching PyChOpMarg's own com._tx_combs[0]),
# so this is 1 DC gain x 1 shelf gain x 2 Tx-tap candidates (all-zero +
# this one) = 2 grid points total, not exactly 1.
DC_GAIN_CANDIDATES = [-3.0]
SHELF_GAIN_CANDIDATES = [-2.0]
# step=1.0, not 0.02: with lo==hi, the step's own magnitude is
# irrelevant to which values land in range -- but np.arange(lo, lo+step,
# step) is float-rounding-sensitive right at its own endpoint (e.g.
# np.arange(0.06, 0.08, 0.02) spuriously includes 0.08 too, since
# 0.06+0.02 isn't exactly representable), and a step this much larger
# than the values involved stays well clear of that.
TX_TAPS_BOUNDS = [
    (-0.02, -0.02, 1.0),  # c(-3)
    (0.06, 0.06, 1.0),  # c(-2)
    (-0.2, -0.2, 1.0),  # c(-1)
    (-0.04, -0.04, 1.0),  # c(1)
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
    eta_0=2e-8,
    a_dd=0.02,
    der_0=1e-5,  # unlike KR's 1e-4 -- C2C's own config value
    # b_min/b_max(1..6) from the config sheet -- per-tap DFE bounds,
    # not the uniform +-1 every other example in this project uses.
    dfe_min=np.array([0.3, 0.05, -0.04, -0.04, -0.04, -0.04]),
    dfe_max=np.array([0.65, 0.15, 0.1, 0.1, 0.1, 0.1]),
    ctle_zero_freq=21.25e9,
    ctle_pole1_freq=21.25e9,
    ctle_pole2_freq=53.125e9,
    ctle_shelf_freq=0.6640625e9,
    ctle_dc_gain_candidates=DC_GAIN_CANDIDATES,
    ctle_shelf_gain_candidates=SHELF_GAIN_CANDIDATES,
    tx_taps_bounds=TX_TAPS_BOUNDS,
    tx_taps_c0_min=0.54,
    tx_taps_n_post=1,  # only c(1) is post-cursor
    tx_risetime=0.0075e-9,  # this config's own T_r -- see matlab_golden/generate/gen_tx_h_t.m
    rx_afe_cutoff_freq=0.75 * 53.125e9,
    r0=50.0,
    tx_termination_resistance=50.0,  # R_d = [50, 50] in this config too -- matched, gamma1=gamma2=0
    rx_termination_resistance=50.0,
    # package_Z_c = [87.5 87.5; 92.5 92.5] Ohm, z_p (package case 1)
    # = 13/1.8 mm for TX (see module docstring re: NEXT/RX's own,
    # different case-1 length this script doesn't separately model).
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
    package_tline_segments=[(87.5, 13.0), (92.5, 1.8)],
    com_min_db=3.0,  # "COM Pass threshold" in the config sheet
)


def main() -> None:
    params = search(
        STANDARD,
        thru_path=str(DATA / "C2C_PCB_SYSVIA_12dB_thru.s4p"),
        next_paths=[
            str(DATA / "C2C_PCB_SYSVIA_12dB_next1.s4p"),
            str(DATA / "C2C_PCB_SYSVIA_12dB_next2.s4p"),
            str(DATA / "C2C_PCB_SYSVIA_12dB_next3.s4p"),
            str(DATA / "C2C_PCB_SYSVIA_12dB_next4.s4p"),
        ],
        fext_paths=[
            str(DATA / "C2C_PCB_SYSVIA_12dB_fext1.s4p"),
            str(DATA / "C2C_PCB_SYSVIA_12dB_fext2.s4p"),
            str(DATA / "C2C_PCB_SYSVIA_12dB_fext3.s4p"),
            str(DATA / "C2C_PCB_SYSVIA_12dB_fext4.s4p"),
            str(DATA / "C2C_PCB_SYSVIA_12dB_fext5.s4p"),
            str(DATA / "C2C_PCB_SYSVIA_12dB_fext6.s4p"),
        ],
        # Unlike Std_BP_12inch_Meg7 (the KR example's channel), this
        # file's own S21 (~0.97 near unity at low frequency, between
        # ports 1-2 and 3-4) matches differential_network's *default*
        # port_order -- ports are already (TX+, RX+, TX-, RX-), the
        # ECEN720/PyBERT interleaved convention.
        port_order=(0, 2, 1, 3),
        show_progress=True,
    )
    result = compute(params)

    print(f"CTLE DC gain:      {params.ctle_dc_gain_db:.1f} dB")
    print(f"CTLE shelf gain:   {params.ctle_shelf_gain_db:.1f} dB")
    print(f"Tx FFE taps:       {params.ffe_tap_weights}")
    print()
    verdict = "PASS" if result.com_db >= STANDARD.com_min_db else "FAIL"
    print(f"COM:               {result.com_db:.3f} dB ({verdict} @ {STANDARD.com_min_db} dB)")
    print(f"Signal amplitude:  {result.signal_amplitude * 1e3:.3f} mV")
    print(f"Noise amplitude:   {result.noise_amplitude * 1e3:.3f} mV")
    print(f"sigma_Tx:          {result.sigma_tx * 1e3:.3f} mV")
    print(f"sigma_Jitter:      {result.sigma_jitter * 1e3:.3f} mV")
    print(f"sigma_Noise:       {result.sigma_noise * 1e3:.3f} mV")


if __name__ == "__main__":
    main()
