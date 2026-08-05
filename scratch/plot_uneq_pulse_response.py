"""Compare matlab_golden/data/uneq_pulse_response_c2c_thru.csv (MATLAB's own
"Half Symbol Unequalized end-to-end PR") against serdeskit's own
half_symbol_unequalized_pulse_response for the same C2C thru channel/config
(see examples/compute_com_c2c_matlab_coeffs.py, which this duplicates the
LinkComParams from).
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from serdeskit.com import LinkComParams, compute

REPO_ROOT = Path(__file__).parents[1]
MATLAB_CSV = REPO_ROOT / "matlab_golden" / "data" / "uneq_pulse_response_c2c_thru.csv"
DATA = REPO_ROOT / "reference" / "ck_channels" / "c2c_pcb"

BAUD_RATE = 53.125e9
FREQ_STEP = 10e6
SAMPLES_PER_UI = 32
R0 = 50.0


def load_matlab_csv(path: Path) -> tuple[list[float], list[float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = [(float(row["t"]), float(row["pulse"])) for row in reader]
    t, pulse = zip(*rows)
    return list(t), list(pulse)


def compute_serdeskit_pr() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    params = LinkComParams(
        channel_path=str(DATA / "C2C_PCB_SYSVIA_12dB_thru.s4p"),
        next_channel_paths=[str(DATA / f"C2C_PCB_SYSVIA_12dB_next{n}.s4p") for n in [1, 2, 3, 4]],
        fext_channel_paths=[str(DATA / f"C2C_PCB_SYSVIA_12dB_fext{n}.s4p") for n in [1, 2, 3, 4, 5, 6]],
        port_order=(0, 2, 1, 3),
        gamma1=0.0,
        gamma2=0.0,
        tx_r0=R0,
        tx_die_capacitances=[1.2e-4 * 1e-9],
        tx_die_inductances=[0.12e-9],
        tx_bump_capacitance=0.3e-4 * 1e-9,
        tx_tline_a1=0.0009909,
        tx_tline_a2=0.0002772,
        tx_tline_tau=0.006141,
        tx_tline_gamma0=0.0,
        tx_tline_segments=[(87.5, 13.0), (92.5, 1.8)],
        tx_pad_capacitance=0.87e-4 * 1e-9,
        rx_r0=R0,
        rx_die_capacitances=[1.2e-4 * 1e-9],
        rx_die_inductances=[0.12e-9],
        rx_bump_capacitance=0.3e-4 * 1e-9,
        rx_tline_a1=0.0009909,
        rx_tline_a2=0.0002772,
        rx_tline_tau=0.006141,
        rx_tline_gamma0=0.0,
        rx_tline_segments=[(87.5, 11.0), (92.5, 1.8)],
        rx_pad_capacitance=0.87e-4 * 1e-9,
        ctle_zero_freq=21.25e9,
        ctle_pole1_freq=21.25e9,
        ctle_pole2_freq=53.125e9,
        ctle_shelf_freq=0.6640625e9,
        ctle_dc_gain_db=-3.0,
        ctle_shelf_gain_db=-2.0,
        ffe_tap_weights=np.array([-0.02, 0.06, -0.2, -0.04]),
        ffe_n_post=1,
        tx_risetime=0.0075e-9,
        rx_afe_cutoff_freq=0.75 * BAUD_RATE,
        rx_ffe_tap_weights=np.array([1.0]),
        rx_ffe_n_pre=0,
        baud_rate=BAUD_RATE,
        freq_step=FREQ_STEP,
        samples_per_ui=SAMPLES_PER_UI,
        levels=4,
        rlm=0.95,
        victim_amplitude=0.413,
        a_ne=0.608,
        a_fe=0.413,
        snr_tx=33.0,
        sigma_rj=0.01,
        eta_0=2e-8,
        a_dd=0.02,
        der_0=1e-5,
        dfe_min=np.array([0.3, 0.05, -0.04, -0.04, -0.04, -0.04]),
        dfe_max=np.array([0.65, 0.15, 0.1, 0.1, 0.1, 0.1]),
    )
    result = compute(params)
    signal = result.half_signal_unequalized_pulse_response
    t = signal.t0 + np.arange(len(signal.samples)) / signal.fs
    return t, signal.samples


def main() -> None:
    matlab_t, matlab_pulse = load_matlab_csv(MATLAB_CSV)
    serdeskit_t, serdeskit_pulse = compute_serdeskit_pr()

    _, ax = plt.subplots()
    ax.plot(matlab_t, np.array(matlab_pulse) / (4 - 1), label="MATLAB (uneq_pulse_response_c2c_thru.csv)")
    ax.plot(serdeskit_t, serdeskit_pulse, label="serdeskit (half_symbol_unequalized_pulse_response)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("pulse (V)")
    ax.set_title("Half Symbol Unequalized end-to-end PR (C2C thru)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
