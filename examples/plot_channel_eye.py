"""End-to-end example: random NRZ bits -> Link (SParameterChannel, real B12
backplane data) -> eye diagram, no equalization — reproducing ECEN 720
Lab 2 part (c) (10k random bits; MATLAB's channel_data.m is the reference,
and it has no CTLE/FFE/DFE either: its TX FIR taps are `[1]`, an identity
filter).

Requires reference/ecen720/peters_01_0605_B12_thru.s4p — gitignored, see
docs/eye-diagram-design.md for where it comes from.

Run: python examples/plot_channel_eye.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.channel import SParameterChannel
from serdeskit.link import Link
from serdeskit.util import plot_eye

TOUCHSTONE_PATH = "../reference/ecen720/peters_01_0605_B12_thru.s4p"
SYMBOL_RATE = 5e9
N_SYMBOLS = 10_000
SAMPLES_PER_UI = 20


def main() -> None:
    # peters_* is ECEN720/PyBERT interleaved (TX+, RX+, TX-, RX-) -- see
    # differential_network's own docstring.
    channel = SParameterChannel.from_touchstone(TOUCHSTONE_PATH, port_order=(0, 2, 1, 3))
    link = Link(channel=channel)

    rng = np.random.default_rng(seed=0)
    bits = rng.choice(np.array([-1.0, 1.0]), size=N_SYMBOLS)
    fs = SAMPLES_PER_UI * SYMBOL_RATE

    result = link.simulate(bits=bits, fs=fs, symbol_rate=SYMBOL_RATE)

    plot_eye(result.eye, title=f"B12 Backplane Channel - {SYMBOL_RATE / 1e9:.0f} Gb/s")
    plt.show()


if __name__ == "__main__":
    main()
