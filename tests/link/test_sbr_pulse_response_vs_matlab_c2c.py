"""Golden test: Link.sbr_pulse_response() -- the time-domain pipeline
(truncate, then Ctle.process(), box-car, then Ffe.process()) -- against
MATLAB COM3.70's own composed pulse response ("sbr"), for the real C2C
thru channel at MATLAB's own found-optimal equalization point.

Same ground truth as tests/link/test_pulse_response_vs_matlab_c2c.py
(matlab_golden/data/sbr_c2c_thru.csv) -- that test's own
ffe_channel_ctle_pulse_response() (frequency-domain composition, one
IFFT) still has an unexplained ~1-2% residual against it. This one
doesn't: replicating MATLAB's actual time-domain chain (confirmed via
tests/link/test_uneq_pulse_response_vs_matlab_c2c.py, tests/ctle/
test_two_stage_process_vs_matlab.py, tests/ffe/
test_tap_weight_process_vs_matlab.py, each already matching MATLAB to
floating-point precision on their own) closes the gap essentially
completely end to end too -- see docs/known-issues.md.
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
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe
from serdeskit.tx_filter import TxRisetimeFilter

CHAN_DIR = Path(__file__).parent.parent.parent / "reference" / "ck_channels" / "c2c_pcb"
DATA = Path(__file__).parent.parent.parent / "matlab_golden" / "data" / "sbr_c2c_thru.csv"

BAUD_RATE = 53.125e9
FREQ_STEP = 10e6
SAMPLES_PER_UI = 32
R0 = 50.0
VICTIM_AMPLITUDE = 0.413  # A_v, C2C config sheet

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


def _matlab_sbr() -> np.ndarray:
    with open(DATA) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["sbr"]) for r in rows])


def _serdeskit_sbr() -> np.ndarray:
    grid = SystemGrid.build(BAUD_RATE, FREQ_STEP, SAMPLES_PER_UI)
    tap_delay = 1.0 / BAUD_RATE

    network = skrf.Network(str(CHAN_DIR / "C2C_PCB_SYSVIA_12dB_thru.s4p"))
    diff = differential_network(network)
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
    return link.sbr_pulse_response(grid).scale(VICTIM_AMPLITUDE).samples


def test_peak_value_matches_matlab() -> None:
    actual = _serdeskit_sbr()
    expected = _matlab_sbr()
    actual_peak = actual[np.argmax(np.abs(actual))]
    expected_peak = expected[np.argmax(np.abs(expected))]
    np.testing.assert_allclose(actual_peak, expected_peak, rtol=1e-3)


def test_shape_near_peak_matches_matlab() -> None:
    """+-5 UI around each side's own peak -- same peak-relative approach
    as test_uneq_pulse_response_vs_matlab_c2c.py, for the same reason:
    the two sides don't land at the same absolute array offset.
    """
    actual = _serdeskit_sbr()
    expected = _matlab_sbr()

    nspui = SAMPLES_PER_UI
    window = 5
    a_peak = int(np.argmax(np.abs(actual)))
    e_peak = int(np.argmax(np.abs(expected)))
    actual_window = actual[a_peak - window * nspui : a_peak + window * nspui + 1]
    expected_window = expected[e_peak - window * nspui : e_peak + window * nspui + 1]
    np.testing.assert_allclose(actual_window, expected_window, rtol=1e-3, atol=1e-4)
