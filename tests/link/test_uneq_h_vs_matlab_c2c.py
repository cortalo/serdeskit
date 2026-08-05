"""Golden test: channel.transfer_function() * tx_filter.transfer_function()
* rx_afe.transfer_function() -- three of Link.ffe_channel_ctle_pulse_
response()'s `h` factors (everything but ctle/ffe/rx_ffe) -- against
MATLAB COM3.70's own pre-CTLE/FFE frequency response (chdata(1).sdd21),
for the real IEEE 802.3ck "C2C" thru channel, package case 1.

Isolation step for tests/link/test_pulse_response_vs_matlab_c2c.py's
confirmed ~12% divergence at the cursor (docs/known-issues.md's "full
composed pulse response" entry). Originally written with just
channel*rx_afe (no tx_filter -- TxRisetimeFilter didn't exist yet), which
failed by exactly MATLAB's H_t factor -- see docs/known-issues.md and
tests/tx_filter/test_risetime_vs_matlab.py for how that was found and
fixed. Now includes tx_filter, so this is a genuine composition check,
not just re-confirming a single already-known-missing factor.

chdata(1).sdd21 is captured once, in COM_FD_to_TD
(com_ieee8023_93a_370.m:904-910), *before* the search loop -- unlike
chdata(k).sdd21ctf (rebuilt on every CTLE candidate inside the search),
so no risk of reading back a stale, non-winning snapshot the way
matlab_golden/README.md's own "two package test cases" cautionary tale
describes. Ground truth generated alongside sbr_c2c_thru.csv by
matlab_golden/generate/gen_sbr_c2c.m.

Bessel-Thomson confirmed off for this config (not even a row in its own
parameter sheet, so xls_parameter's `false` default applies) -- so
chdata(1).sdd21 = (channel+package S21) * H_t * Butterworth, with no
other factor to account for.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.package import Package, cascade_channel
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.tx_filter import TxRisetimeFilter

THRU_PATH = (
    Path(__file__).parent.parent.parent / "reference" / "ck_channels" / "c2c_pcb" / "C2C_PCB_SYSVIA_12dB_thru.s4p"
)
DATA = Path(__file__).parent.parent.parent / "matlab_golden" / "data" / "uneq_h_c2c_thru.csv"

BAUD_RATE = 53.125e9
R0 = 50.0

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


def test_uneq_h_matches_matlab() -> None:
    with open(DATA) as f:
        rows = list(csv.DictReader(f))
    freqs = np.array([float(r["freq"]) for r in rows])
    expected_h = np.array([complex(float(r["h_r"]), float(r["h_i"])) for r in rows])

    network = skrf.Network(str(THRU_PATH))
    diff = differential_network(network)
    cascaded = cascade_channel(diff, TX_PACKAGE, RX_PACKAGE, freqs)
    channel = SParameterChannel(cascaded)  # gamma1=gamma2=0.0 defaults -- matched termination
    tx_filter = TxRisetimeFilter(risetime=0.0075e-9)  # this config's own T_r -- see gen_tx_h_t.m
    rx_afe = RxAfeButterworth(cutoff_freq=0.75 * BAUD_RATE)

    actual_h = channel.transfer_function(freqs) * tx_filter.transfer_function(freqs) * rx_afe.transfer_function(freqs)

    print()
    diff_mag = np.abs(actual_h - expected_h)
    print(f"max |diff|: {diff_mag.max():.6e}  at f={freqs[np.argmax(diff_mag)] / 1e9:.3f} GHz")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        i = min(int(frac * (len(freqs) - 1)), len(freqs) - 1)
        print(
            f"  f={freqs[i]/1e9:7.3f} GHz  matlab={expected_h[i]:.6f}  "
            f"serdeskit={actual_h[i]:.6f}  |diff|={diff_mag[i]:.3e}"
        )

    np.testing.assert_allclose(actual_h, expected_h, rtol=1e-3, atol=1e-4)
