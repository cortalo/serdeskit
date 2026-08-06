"""Golden test: SParameterChannel.transfer_function() -- the actual
method Link.ffe_channel_ctle_pulse_response() calls everywhere in the
real pipeline -- against MATLAB COM3.70's own channel+package H21, for
the real C2C thru channel (matched termination, gamma1=gamma2=0, so
MATLAB's own (93A-18) H21 formula reduces to plain S21, no extra step).

Reuses tests/package/test_cascade_vs_matlab_c2c.py's already-validated
ground truth (matlab_golden/data/channel_plus_package_c2c_thru.csv) --
that test confirmed cascade_channel's own raw S-parameters (no taper)
match MATLAB exactly. This test checks the next layer up:
transfer_function() itself.

Originally found failing here: transfer_function() applied a
raised-cosine taper across the *entire* queried band (not just any
extrapolated tail) before returning H21 -- traced to PyChOpMarg's own
calc_H21 (pychopmarg/utility/filter.py), a faithful transcription, not a
serdeskit-specific bug. But MATLAB never applies anything like this at
the H21 level -- it handles extrapolation/causality entirely
differently, via time-domain alternating-projections causality
correction (com_ieee8023_93a_370.m's own s21_to_impulse_DC), which is
itself off by default (`OP.ENFORCE_CAUSALITY = 0`, "Not recommended") in
every config this project has exercised -- so MATLAB's real behavior
here is simply "no taper, no correction." Fixed by dropping the taper
entirely (docs/known-issues.md); tests/conftest.py's own
no_raised_cosine_taper fixture keeps PyChOpMarg-golden-reference tests
elsewhere comparable by stripping the same taper from calc_H21 for their
duration.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.package import Package, cascade_channel

THRU_PATH = (
    Path(__file__).parent.parent.parent / "reference" / "ck_channels" / "c2c_pcb" / "C2C_PCB_SYSVIA_12dB_thru.s4p"
)
DATA = Path(__file__).parent.parent.parent / "matlab_golden" / "data" / "channel_plus_package_c2c_thru.csv"

R0 = 50.0
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


def test_transfer_function_matches_matlab() -> None:
    with open(DATA) as f:
        rows = list(csv.DictReader(f))
    freqs = np.array([float(r["freq"]) for r in rows])
    expected_s21 = np.array([complex(float(r["s21_r"]), float(r["s21_i"])) for r in rows])

    network = skrf.Network(str(THRU_PATH))
    diff = differential_network(network, port_order=(0, 2, 1, 3))
    cascaded = cascade_channel(diff, TX_PACKAGE, RX_PACKAGE, freqs)
    channel = SParameterChannel(cascaded)  # gamma1=gamma2=0.0 defaults -- matched termination

    actual_s21 = channel.transfer_function(freqs)

    print()
    diff_mag = np.abs(actual_s21 - expected_s21)
    print(f"max |diff|: {diff_mag.max():.6e}  at f={freqs[np.argmax(diff_mag)]/1e9:.3f} GHz")
    print(f"|diff| at first/last point: {diff_mag[0]:.3e} / {diff_mag[-1]:.3e}")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        i = min(int(frac * (len(freqs) - 1)), len(freqs) - 1)
        print(f"  f={freqs[i]/1e9:7.3f} GHz  matlab={expected_s21[i]:.6f}  serdeskit={actual_s21[i]:.6f}  |diff|={diff_mag[i]:.3e}")

    np.testing.assert_allclose(actual_s21, expected_s21, rtol=1e-3, atol=1e-4)
