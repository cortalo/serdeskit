"""End-to-end example: random bits -> Link (PassThroughChannel) -> eye diagram.

This is the "outer layer" the domain code deliberately stays out of: Link
and its stages never import matplotlib, so plotting LinkResult uses
serdeskit.util.plot_eye instead.

Run: python examples/pass_through_link.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.channel import PassThroughChannel
from serdeskit.link import Link
from serdeskit.util import plot_eye

FS = 100e9  # simulation sample rate, Hz
SYMBOL_RATE = 10e9  # 10 Gb/s -> 10 samples/UI at FS
N_SYMBOLS = 2000


def main() -> None:
    rng = np.random.default_rng(seed=0)
    bits = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=N_SYMBOLS)

    link = Link(channel=PassThroughChannel())
    result = link.simulate(bits=bits, fs=FS, symbol_rate=SYMBOL_RATE)

    eye = result.eye
    print(f"traces: {eye.traces.shape[1]} traces, {eye.traces.shape[0]} samples/trace")
    print(f"ui: {eye.ui * 1e12:.1f} ps, fs: {eye.fs / 1e9:.1f} GHz")

    plot_eye(eye, title="Eye diagram - PassThroughChannel")
    plt.show()


if __name__ == "__main__":
    main()
