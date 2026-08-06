"""Plot |S21| for the B12 backplane channel, for visual comparison against
reference/ecen720/read_sparam.m's own plot (same axis: 0-15 GHz, -80-0 dB).

Requires reference/ecen720/peters_01_0605_B12_thru.s4p — gitignored, see
docs/eye-diagram-design.md for where it comes from.

Run: python examples/plot_s21.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.channel import SParameterChannel
from serdeskit.util import plot_s21

TOUCHSTONE_PATH = "../reference/ecen720/peters_01_0605_B12_thru.s4p"


def main() -> None:
    # peters_* is ECEN720/PyBERT interleaved (TX+, RX+, TX-, RX-) -- see
    # differential_network's own docstring.
    channel = SParameterChannel.from_touchstone(TOUCHSTONE_PATH, port_order=(0, 2, 1, 3))
    freq = np.linspace(50e6, 15e9, 1000)

    plot_s21(channel, freq, title="B12 Backplane Channel")
    plt.show()


if __name__ == "__main__":
    main()
