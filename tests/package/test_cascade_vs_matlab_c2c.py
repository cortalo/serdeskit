"""Golden test: cascade_channel + SParameterChannel against MATLAB
COM3.70's own channel+package computation, for a real IEEE 802.3ck "C2C"
channel -- exposes a real, not-yet-diagnosed discrepancy (see
docs/known-issues.md). Currently EXPECTED TO FAIL: written first, per
TDD, to pin down exactly what "wrong" looks like before chasing the fix.

tests/package/test_cascade.py already golden-tests this same
cascade_channel/SParameterChannel pairing against PyChOpMarg's own
add_pkg() and passes -- so this isn't necessarily a bug in
cascade_channel itself, so much as *something* that diverges from
MATLAB specifically for this channel/package configuration (channel-only,
pre-package, already independently confirmed to match MATLAB bit-for-bit
-- see the C2C pulse-response investigation this test grew out of).

Expected values hardcoded from MATLAB's own chdata(1).sdd21 (channel +
package case 1, before CTLE), captured via an instrumented copy of
com_ieee8023_93a_370.m for
reference/ck_channels/c2c_pcb/C2C_PCB_SYSVIA_12dB_thru.s4p -- package
case 1 (z_p = 13mm/1.8mm, applied uniformly TX and RX -- see
examples/compute_com_c2c.py's own docstring re: the RX-side length
approximation, 13mm used here vs the config's real 11mm for NEXT/RX).

Needs the real channel file (gitignored -- see
examples/compute_com_c2c.py's own docstring for the download link).
Skipped if it isn't present; how CI gets access to it is a separate,
not-yet-solved problem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.link import SystemGrid
from serdeskit.package import Package, cascade_channel

THRU_PATH = (
    Path(__file__).parent.parent.parent / "reference" / "ck_channels" / "c2c_pcb" / "C2C_PCB_SYSVIA_12dB_thru.s4p"
)

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
    tline_segments=[(87.5, 13.0), (92.5, 1.8)],
    pad_capacitance=0.87e-4 * 1e-9,
    is_rx=True,
)

# (freq Hz, real, imag) -- MATLAB COM3.70's own chdata(1).sdd21 for this
# exact file, package case 1, no CTLE applied yet.
MATLAB_SDD21_WITH_PACKAGE = [
    (1e7, 0.9665516714, -0.1115767587),
    (1.001e10, 0.2095719285, 0.3608535777),
    (2.001e10, -0.1708379075, 0.08889911291),
    (2.651e10, -0.08764968005, -0.01665297814),
    (3.001e10, -0.03900120655, -0.04978473472),
    (4.001e10, 0.006192434739, -0.01671421664),
]


@pytest.mark.skipif(not THRU_PATH.exists(), reason="requires the real (gitignored) C2C channel data")
@pytest.mark.skip(
    reason="Expected values predate both the RX tline_segments fix (631f2f5) and "
    "the matlab_golden/ methodology overhaul -- this test's own uniform-13mm "
    "TX/RX package approximation is also now known-wrong (real RX is 11mm). "
    "Needs regenerating via matlab_golden/ before re-enabling."
)
def test_channel_plus_package_matches_matlab() -> None:
    """cascade_channel's own docstring: `freqs` is meant to be "the system
    frequency grid every stage in the link is computed on" -- its
    DC-extrapolation/cubic-interpolation machinery isn't well-conditioned
    on a handful of arbitrary, widely-spaced query points, so this
    queries the same dense SystemGrid the real pipeline uses and picks
    out the entries nearest each target frequency, rather than cascading
    directly on the sparse comparison-point array.
    """
    grid = SystemGrid.build(baud_rate=53.125e9, freq_step=10e6, samples_per_ui=32)

    network = skrf.Network(str(THRU_PATH))
    diff = differential_network(network)
    cascaded = cascade_channel(diff, TX_PACKAGE, RX_PACKAGE, grid.f)
    channel = SParameterChannel(cascaded)
    full = channel.transfer_function(grid.f)

    target_freqs = np.array([f for f, _, _ in MATLAB_SDD21_WITH_PACKAGE])
    expected = np.array([complex(r, i) for _, r, i in MATLAB_SDD21_WITH_PACKAGE])
    indices = np.searchsorted(grid.f, target_freqs)
    actual = full[indices]

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
