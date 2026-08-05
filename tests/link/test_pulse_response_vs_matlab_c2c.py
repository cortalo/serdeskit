"""Golden test: Link.sbr_pulse_response() (MATLAB's own real time-domain
chain) against MATLAB COM3.70's own composed pulse response ("sbr"), for
the real IEEE 802.3ck "C2C" thru channel at MATLAB's own found-optimal
equalization point -- via PulseResponse.from_signal's Muller-Mueller
cursor detection specifically (not just a raw peak/argmax, the simpler
check tests/link/test_sbr_pulse_response_vs_matlab_c2c.py already does),
confirming cursor *detection* also works correctly on this method's
output, not just the waveform values themselves.

Originally written against Link.ffe_channel_ctle_pulse_response()
(frequency-domain composition, one IFFT) -- CLAUDE.md's "full composed
pulse response" known-issue entry, which found a real, still-unexplained
~1-2% residual there (see docs/known-issues.md). Switched to
sbr_pulse_response() once that method existed and was verified
independently (matches MATLAB to floating-point precision) -- no reason
to keep asserting against a known-divergent path when an accurate one is
available. ffe_channel_ctle_pulse_response() itself is untouched and
still has its own residual; it's simply not what this file checks anymore.

Compares in a cursor-relative window rather than raw sample index: the
two sides' pulse responses aren't guaranteed to land at the same array
offset, so each side's own cursor is located independently via
PulseResponse.from_signal's Muller-Mueller search before comparing.

Package case 1 (13mm TX / 11mm RX), matching every other C2C golden
fixture in this project (see tests/package/test_cascade_vs_matlab_c2c.py)
and MATLAB's own found-optimal EQ point (examples/
compute_com_c2c_matlab_coeffs.py): CTLE DC gain -3dB, shelf gain -2dB,
TXFFE taps [-0.02, 0.06, -0.2, 0.68, -0.04].

Needs the real channel files (gitignored -- see examples/compute_com_c2c.py's
own docstring for the download link; CI's own workflow downloads them,
see .github/workflows/ci.yml) and matlab_golden/data/sbr_c2c_thru.csv
(committed once matlab_golden/generate/gen_sbr_c2c.m has been run against
a real MATLAB install).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.common.types import Signal
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Link, SystemGrid
from serdeskit.package import Package, cascade_channel
from serdeskit.pulse_response import PulseResponse
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
DFE1_MAX = 0.65
DFE1_MIN = 0.3

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


def _matlab_pulse_response() -> PulseResponse:
    with open(DATA) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    sbr = np.array([float(r["sbr"]) for r in rows])
    dt = t[1] - t[0]
    signal = Signal(samples=sbr, fs=1.0 / dt, t0=0.0)
    return PulseResponse.from_signal(signal, ui=1.0 / BAUD_RATE, dfe1_max=DFE1_MAX, dfe1_min=DFE1_MIN)


def _serdeskit_pulse_response() -> PulseResponse:
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
    rx_ffe = TapWeightRxFfe(tap_weights=np.array([1.0]), tap_delay=tap_delay)

    link = Link(channel=channel, ctle=ctle, ffe=ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe)
    signal = link.sbr_pulse_response(grid).scale(VICTIM_AMPLITUDE)
    return PulseResponse.from_signal(signal, ui=1.0 / BAUD_RATE, dfe1_max=DFE1_MAX, dfe1_min=DFE1_MIN)


def test_cursor_value_matches_matlab() -> None:
    actual = _serdeskit_pulse_response()
    expected = _matlab_pulse_response()
    assert actual.cursor_value == pytest.approx(expected.cursor_value, rel=1e-3)


def test_shape_near_cursor_matches_matlab() -> None:
    """+-5 UI around the cursor -- where CLAUDE.md's known-issues entry
    describes a persisting "narrower/sharper" shape divergence even after
    the cursor magnitude itself was brought close.
    """
    actual = _serdeskit_pulse_response()
    expected = _matlab_pulse_response()

    nspui = actual.samples_per_ui
    window = 5
    actual_window = actual.samples[actual.cursor_index - window * nspui : actual.cursor_index + window * nspui + 1]
    expected_window = expected.samples[
        expected.cursor_index - window * nspui : expected.cursor_index + window * nspui + 1
    ]
    np.testing.assert_allclose(actual_window, expected_window, rtol=1e-3, atol=1e-4)
