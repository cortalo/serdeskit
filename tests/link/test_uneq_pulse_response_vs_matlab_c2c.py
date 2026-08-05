"""Golden test: Link.uneq_pulse_response() -- channel + tx_filter + rx_afe,
IFFT'd -- against MATLAB COM3.70's own chdata(1).uneq_pulse_response, for
the real IEEE 802.3ck "C2C" thru channel, package case 1.

Isolation step for tests/link/test_pulse_response_vs_matlab_c2c.py's
residual ~1-2% divergence (docs/known-issues.md's "full composed pulse
response" entry): tests/link/test_uneq_h_vs_matlab_c2c.py already
confirmed the frequency-domain product (channel*tx_filter*rx_afe)
matches MATLAB's chdata(1).sdd21 to floating-point precision. This test
checks the same quantity's time-domain (IFFT'd) counterpart --
chdata(1).uneq_pulse_response (com_ieee8023_93a_370.m:930, `filter(
ones(1,samples_per_ui),1,uneq_imp_response)`), not uneq_imp_response
itself: that box-car integration is the same (93A-24) pulse-shaping step
SystemGrid.pulse_response() does via its own x_sinc multiply, just in
the time domain instead of the frequency domain -- so it's this field,
not the raw impulse response, that's directly comparable to
Link.uneq_pulse_response()'s own output.

Ground truth generated alongside sbr_c2c_thru.csv/uneq_h_c2c_thru.csv by
matlab_golden/generate/gen_sbr_c2c.m.

Peak values already match closely (matlab 0.14806 V, serdeskit 0.14801 V,
~0.03% off) and the peak indices land within half a UI of each other (vs.
~1.6 UI apart for the full composed sbr, most of that gap coming from
FFE's own explicit per-tap circshift, which doesn't happen this early) --
but the shape doesn't fully match: up to ~1.4% of peak in a +-5 UI
window. This is the same order of magnitude as the full composed pulse
response's own residual, found *before* CTLE or FFE ever run --
contradicts this project's earlier leading hypothesis (that the residual
comes from MATLAB composing CTLE/FFE in the time domain where this
project composes them in the frequency domain). Since the frequency-
domain product matches MATLAB exactly, the new leading candidate is the
IFFT/pulse-shaping step itself: MATLAB's `s21_to_impulse_DC` truncates
the impulse response (its own "Truncation ratio" console output) before
the box-car filter runs, while SystemGrid.pulse_response() does one
continuous IFFT over the whole periodic record with no separate
truncation step -- not yet confirmed, next thing to check if picked up.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.link import Link, SystemGrid
from serdeskit.package import Package, cascade_channel
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.tx_filter import TxRisetimeFilter

THRU_PATH = (
    Path(__file__).parent.parent.parent / "reference" / "ck_channels" / "c2c_pcb" / "C2C_PCB_SYSVIA_12dB_thru.s4p"
)
DATA = Path(__file__).parent.parent.parent / "matlab_golden" / "data" / "uneq_pulse_response_c2c_thru.csv"

BAUD_RATE = 53.125e9
FREQ_STEP = 10e6
SAMPLES_PER_UI = 32
R0 = 50.0
VICTIM_AMPLITUDE = 0.413  # A_v, C2C config sheet -- already baked into MATLAB's own captured data

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


def _matlab_uneq_pulse_response() -> np.ndarray:
    with open(DATA) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["pulse"]) for r in rows])


def _serdeskit_uneq_pulse_response() -> np.ndarray:
    grid = SystemGrid.build(BAUD_RATE, FREQ_STEP, SAMPLES_PER_UI)

    network = skrf.Network(str(THRU_PATH))
    diff = differential_network(network)
    cascaded = cascade_channel(diff, TX_PACKAGE, RX_PACKAGE, grid.f)
    channel = SParameterChannel(cascaded)
    tx_filter = TxRisetimeFilter(risetime=0.0075e-9)  # this config's own T_r -- see gen_tx_h_t.m
    rx_afe = RxAfeButterworth(cutoff_freq=0.75 * BAUD_RATE)

    link = Link(channel=channel, tx_filter=tx_filter, rx_afe=rx_afe)
    return link.uneq_pulse_response(grid).scale(VICTIM_AMPLITUDE).samples


def test_peak_value_matches_matlab() -> None:
    actual = _serdeskit_uneq_pulse_response()
    expected = _matlab_uneq_pulse_response()
    actual_peak = actual[np.argmax(np.abs(actual))]
    expected_peak = expected[np.argmax(np.abs(expected))]
    assert actual_peak == pytest.approx(expected_peak, rel=1e-3)


def test_shape_near_peak_matches_matlab() -> None:
    """+-5 UI around each side's own peak -- no cursor/DFE logic involved
    here (this is pre-CTLE/FFE), so a simple peak-relative window is
    enough, unlike PulseResponse.from_signal's Muller-Mueller search used
    for the full composed pulse response.
    """
    actual = _serdeskit_uneq_pulse_response()
    expected = _matlab_uneq_pulse_response()

    nspui = SAMPLES_PER_UI
    window = 5
    a_peak = int(np.argmax(np.abs(actual)))
    e_peak = int(np.argmax(np.abs(expected)))
    actual_window = actual[a_peak - window * nspui : a_peak + window * nspui + 1]
    expected_window = expected[e_peak - window * nspui : e_peak + window * nspui + 1]
    np.testing.assert_allclose(actual_window, expected_window, rtol=1e-3, atol=1e-4)
