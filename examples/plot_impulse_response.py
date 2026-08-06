"""Plot the impulse response of the B12 backplane channel, for visual
comparison against reference/ecen720/read_sparam.m's own impulse response
plot (main peak around 3-6 ns, ~5.3 mV, for this channel).

dt=1ps matches read_sparam.m's Ts=1ps — peak amplitude only lines up with
MATLAB's ~5.3 mV at that same dt; see SParameterChannel.impulse_response()'s
docstring for why the two aren't comparable at different dt.

Requires reference/ecen720/peters_01_0605_B12_thru.s4p — gitignored, see
docs/eye-diagram-design.md for where it comes from.

Run: python examples/plot_impulse_response.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from serdeskit.channel import SParameterChannel
from serdeskit.util import plot_impulse_response

TOUCHSTONE_PATH = "../reference/ecen720/peters_01_0605_B12_thru.s4p"


def main() -> None:
    # peters_* is ECEN720/PyBERT interleaved (TX+, RX+, TX-, RX-) -- see
    # differential_network's own docstring.
    channel = SParameterChannel.from_touchstone(TOUCHSTONE_PATH, port_order=(0, 2, 1, 3))

    plot_impulse_response(
        channel, title="B12 Backplane Channel", dt=1e-12, xlim=(3e-9, 6e-9)
    )
    plt.show()


if __name__ == "__main__":
    main()
