"""End-to-end example: compute COM (IEEE 802.3-2022 Annex 93A) for a real,
measured channel — PyChOpMarg's own bundled "Example 2" data (VITA 68.2,
6 s4p files: 1 THRU + 2 FEXT + 3 NEXT aggressors), through an
IEEE-802.3by-style config with a single-segment package transmission
line. Same computation tests/evaluate/test_pychopmarg_example2.py checks
(COM ~= 3.58 dB), here as a runnable script rather than a test: fixed
CTLE gain and Tx FFE taps, no equalization search (see that test's
docstring for why — a full search over real, 4001-point channel data is
expensive; evaluate.evaluate_channel is where the actual search lives).

Requires tests/data/pychopmarg_example2/*.s4p — checked into the repo
(unlike reference/'s gitignored files): see that directory's README.md
for provenance.

Run: python examples/compute_com_real_channel.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.com import Com, ComParams
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Link, SystemGrid
from serdeskit.package import Package, cascade_channel
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe

DATA = Path("../tests/data/pychopmarg_example2")

BAUD_RATE = 25.78125e9
FREQ_STEP = 10e6
SAMPLES_PER_UI = 32
R0 = 50.0
GAMMA = (55.0 - R0) / (55.0 + R0)  # this example's own (non-matched) R_d=55 die termination

TX_PACKAGE = Package(
    r0=R0,
    die_capacitances=[0.250e-12],
    die_inductances=[0.0],
    bump_capacitance=0.001e-12,
    tline_a1=8.9e-4,
    tline_a2=2.0e-4,
    tline_tau=6.141e-3,
    tline_gamma0=5.0e-4,
    tline_segments=[(78.2, 12.0)],  # single segment: z_c=[78.2], z_p[zp_sel=0]=12
    pad_capacitance=0.180e-12,
    is_rx=False,
)
RX_PACKAGE = Package(
    r0=R0,
    die_capacitances=[0.250e-12],
    die_inductances=[0.0],
    bump_capacitance=0.001e-12,
    tline_a1=8.9e-4,
    tline_a2=2.0e-4,
    tline_tau=6.141e-3,
    tline_gamma0=5.0e-4,
    tline_segments=[(78.2, 12.0)],
    pad_capacitance=0.180e-12,
    is_rx=True,
)


def _load_channel(path: Path, freqs: npt.NDArray[np.float64]) -> SParameterChannel:
    raw = differential_network(skrf.Network(str(path)))
    cascaded = cascade_channel(raw, TX_PACKAGE, RX_PACKAGE, freqs)
    return SParameterChannel(cascaded, gamma1=GAMMA, gamma2=GAMMA)


def main() -> None:
    grid = SystemGrid.build(BAUD_RATE, FREQ_STEP, SAMPLES_PER_UI)
    tap_delay = 1.0 / BAUD_RATE

    channel = _load_channel(DATA / "example2_THRU.s4p", grid.f)
    ctle = TwoStageCtle(
        zero_freq=6.4453125e9,
        pole1_freq=6.4453125e9,
        pole2_freq=25.78125e9,
        shelf_freq=0.1e9,
        dc_gain_db=-6.0,
        shelf_gain_db=0.0,
    )
    # tx_taps: (pre2, pre1, cursor-adjacent pre, post1, post2, post3) per
    # this config's own tx_taps_min/max/step ([0,0,-0.18,-0.38,0,0] /
    # [0,0,0,0,0,0] / [0,0,0.02,0.02,0,0]); -0.12 = -0.38 + 13*0.02 — an
    # arbitrary, valid combination (no search — see module docstring).
    ffe = TapWeightFfe(tap_weights=np.array([0.0, 0.0, -0.18, -0.12, 0.0, 0.0]), n_post=3, tap_delay=tap_delay)
    rx_afe = RxAfeButterworth(cutoff_freq=0.75 * BAUD_RATE)
    rx_ffe = TapWeightRxFfe(tap_weights=np.array([1.0]), tap_delay=tap_delay)

    link = Link(channel=channel, ctle=ctle, ffe=ffe, rx_afe=rx_afe, rx_ffe=rx_ffe)
    params = ComParams(
        baud_rate=BAUD_RATE,
        freq_step=FREQ_STEP,
        samples_per_ui=SAMPLES_PER_UI,
        levels=2,
        rlm=1.0,
        victim_amplitude=0.4,
        a_ne=0.6,
        a_fe=0.4,
        snr_tx=27.0,
        sigma_rj=0.01,
        eta_0=5.20e-8,
        a_dd=0.05,
        der_0=1e-5,
        dfe_min=np.array([-1.0] * 14),
        dfe_max=np.array([1.0] * 14),
    )
    next_channels = [
        _load_channel(DATA / "example2_NEXT1.s4p", grid.f),
        _load_channel(DATA / "example2_NEXT2.s4p", grid.f),
        _load_channel(DATA / "example2_NEXT3.s4p", grid.f),
    ]
    fext_channels = [
        _load_channel(DATA / "example2_FEXT1.s4p", grid.f),
        _load_channel(DATA / "example2_FEXT2.s4p", grid.f),
    ]

    result = Com(link=link, params=params, next_channels=next_channels, fext_channels=fext_channels).compute()

    print(f"COM:               {result.com_db:.3f} dB")
    print(f"Signal amplitude:  {result.signal_amplitude * 1e3:.3f} mV")
    print(f"Noise amplitude:   {result.noise_amplitude * 1e3:.3f} mV")
    print(f"sigma_Tx:          {result.sigma_tx * 1e3:.3f} mV")
    print(f"sigma_Jitter:      {result.sigma_jitter * 1e3:.3f} mV")
    print(f"sigma_Noise:       {result.sigma_noise * 1e3:.3f} mV")


if __name__ == "__main__":
    main()
