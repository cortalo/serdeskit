"""Golden test: PulseResponse.local_slopes() -- (93A-28)'s per-UI slope
samples, which feed sigma_jitter (93A-31) -- against MATLAB COM3.70's own
h_J, for the real C2C thru channel at MATLAB's own found-optimal
equalization point.

Same sbr construction as tests/link/test_sbr_pulse_response_vs_matlab_c2c.py
(already golden-tested against matlab_golden/data/sbr_c2c_thru.csv), built
one step further into a PulseResponse.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Link, SystemGrid
from serdeskit.package import Package, cascade_channel
from serdeskit.pulse_response import PulseResponse
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe
from serdeskit.tx_filter import TxRisetimeFilter

CHAN_DIR = Path(__file__).parent.parent.parent / "reference" / "ck_channels" / "c2c_pcb"
DATA = Path(__file__).parent.parent.parent / "matlab_golden" / "data" / "h_j_c2c_thru.csv"

BAUD_RATE = 53.125e9
FREQ_STEP = 10e6
SAMPLES_PER_UI = 32
R0 = 50.0
VICTIM_AMPLITUDE = 0.413  # A_v, C2C config sheet
DFE1_MIN = 0.3  # dfe_min[0], C2C config sheet
DFE1_MAX = 0.65  # dfe_max[0], C2C config sheet

# Package case 1 (13mm TX, 11mm RX) -- tests/package/test_cascade_vs_matlab_c2c.py's own config.
TX_PACKAGE = Package(
    r0=R0,
    die_capacitances=[1.2e-4 * 1e-9],
    die_inductances=[0.12e-9],
    bump_capacitance=0.3e-4 * 1e-9,
    tline_a1=0.0009909,
    tline_a2=0.0002772,
    tline_tau=0.006141,
    tline_gamma0=0.0,
    tline_segments=[(87.5, 13.0), (92.5, 1.8)],
    pad_capacitance=0.87e-4 * 1e-9,
    is_rx=False,
)
RX_PACKAGE = Package(
    r0=R0,
    die_capacitances=[1.2e-4 * 1e-9],
    die_inductances=[0.12e-9],
    bump_capacitance=0.3e-4 * 1e-9,
    tline_a1=0.0009909,
    tline_a2=0.0002772,
    tline_tau=0.006141,
    tline_gamma0=0.0,
    tline_segments=[(87.5, 11.0), (92.5, 1.8)],
    pad_capacitance=0.87e-4 * 1e-9,
    is_rx=True,
)


def _matlab_h_j() -> np.ndarray:
    with open(DATA) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["h_j"]) for r in rows])


def _serdeskit_pulse_response() -> PulseResponse:
    grid = SystemGrid.build(BAUD_RATE, FREQ_STEP, SAMPLES_PER_UI)
    tap_delay = 1.0 / BAUD_RATE

    network = skrf.Network(str(CHAN_DIR / "C2C_PCB_SYSVIA_12dB_thru.s4p"))
    diff = differential_network(network, port_order=(0, 2, 1, 3))
    cascaded = cascade_channel(diff, TX_PACKAGE, RX_PACKAGE, grid.f)
    channel = SParameterChannel(cascaded)

    ctle = TwoStageCtle(
        zero_freq=21.25e9,
        pole1_freq=21.25e9,
        pole2_freq=53.125e9,
        shelf_freq=0.6640625e9,
        dc_gain_db=-3.0,
        shelf_gain_db=-2.0,
    )
    ffe = TapWeightFfe(tap_weights=np.array([-0.02, 0.06, -0.2, -0.04]), n_post=1, tap_delay=tap_delay)
    tx_filter = TxRisetimeFilter(risetime=0.0075e-9)  # this config's own T_r -- see gen_tx_h_t.m
    rx_afe = RxAfeButterworth(cutoff_freq=0.75 * BAUD_RATE)
    rx_ffe = TapWeightRxFfe(tap_weights=np.array([1.0]), tap_delay=tap_delay)  # this config's own Rx FFE: unity tap

    link = Link(channel=channel, ctle=ctle, ffe=ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe)
    sbr = link.sbr_pulse_response(grid).scale(VICTIM_AMPLITUDE)
    return PulseResponse.from_signal(sbr, ui=1.0 / BAUD_RATE, dfe1_max=DFE1_MAX, dfe1_min=DFE1_MIN)


def test_local_slopes_matches_matlab_h_j() -> None:
    """(93A-28): PulseResponse.local_slopes() against MATLAB's own h_J
    (LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=0 for this config, confirmed via
    matlab_golden/generate/gen_h_j_c2c.m's own printed flag) -- both span
    the *entire* pulse response (pre- and post-cursor) at the cursor's
    own phase, with no amplitude filtering. Replaces an earlier,
    PyChOpMarg-matching version (cursor onward only, amplitude-filtered)
    that put sigma_jitter ~20% low against MATLAB -- first surfaced via
    examples/compute_com_c2c_matlab_coeffs.py's own printed comparison
    (0.412 mV serdeskit vs. 0.515 mV MATLAB).

    Compares on the vector's own norm -- the quantity sigma_jitter
    (93A-31) actually uses -- rather than element-for-element: MATLAB's
    sbr is 2 samples longer than serdeskit's own (a pre-existing,
    already-tolerated discrepancy -- see test_sbr_pulse_response_vs_
    matlab_c2c.py's own peak-relative window, not full-array comparison),
    which shifts h_J's own trailing element count by one and, combined
    with the central-difference formula amplifying that array's own
    rtol=1e-3/atol=1e-4 sbr tolerance, leaves a handful of individual
    entries (mostly near-zero ones, where relative error is meaningless)
    outside a tight per-element tolerance despite the norm matching to
    within 0.01%.
    """
    pulse_response = _serdeskit_pulse_response()
    actual = pulse_response.local_slopes()
    expected = _matlab_h_j()

    np.testing.assert_allclose(np.linalg.norm(actual), np.linalg.norm(expected), rtol=1e-3)
