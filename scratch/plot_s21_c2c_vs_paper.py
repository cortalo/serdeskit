"""Plot |S21| of examples/compute_com_c2c.py's own channel (IEEE 802.3ck
C2C, 12 dB corner, PCB-only) out to 60 GHz, to eyeball against Fig. 9 of
"A 1.41-pJ/b 224-Gb/s PAM4 6-bit ADC-Based SerDes Receiver..." (Khairi et
al., JSSC 2023): that paper's loopback channel measures -31.36 dB @ 56 GHz.

This is NOT the paper's channel -- the paper gives no Touchstone data, only
this one insertion-loss curve and four Nyquist-loss numbers (20.6/27/31.4/
38 dB @ 56 GHz). The point of this plot is just to see how far off C2C's
own on-board channel (built for a 53.125 GBd config, not 112 GBd) is from
the paper's long-reach numbers -- a sanity check before trying to build a
synthetic channel that actually matches Fig. 9.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.channel import SParameterChannel
from serdeskit.util import plot_s21

REPO_ROOT = Path(__file__).parents[1]
CHANNEL_PATH = REPO_ROOT / "reference" / "ck_channels" / "c2c_pcb" / "C2C_PCB_SYSVIA_12dB_thru.s4p"

PAPER_NYQUIST_FREQ = 56e9
PAPER_LOSS_DB = -31.36  # Fig. 9, the 31.4-dB loopback case (this work's 3rd of 4 measured channels)


def main() -> None:
    # C2C_PCB_SYSVIA_* is ECEN720/PyBERT interleaved too (see
    # compute_com_c2c.py's own port_order comment) -- same convention as
    # differential_network's C2C default.
    channel = SParameterChannel.from_touchstone(str(CHANNEL_PATH), port_order=(0, 2, 1, 3))
    freq = np.linspace(50e6, 60e9, 2000)

    ax = plot_s21(channel, freq, title="C2C PCB thru (12 dB corner) vs. paper's loopback channel")
    c2c_loss_at_56ghz = 20 * np.log10(np.abs(channel.s21(np.array([PAPER_NYQUIST_FREQ]))))[0]

    ax.axvline(PAPER_NYQUIST_FREQ / 1e9, color="gray", linestyle="--", linewidth=1)
    ax.plot([PAPER_NYQUIST_FREQ / 1e9], [PAPER_LOSS_DB], "rx", markersize=10,
            label=f"paper Fig. 9: {PAPER_LOSS_DB} dB @ 56 GHz")
    ax.plot([PAPER_NYQUIST_FREQ / 1e9], [c2c_loss_at_56ghz], "bo", markersize=8,
            label=f"C2C thru: {c2c_loss_at_56ghz:.1f} dB @ 56 GHz")
    ax.legend()

    print(f"C2C PCB thru |S21| @ 56 GHz:  {c2c_loss_at_56ghz:.2f} dB")
    print(f"Paper's loopback channel:     {PAPER_LOSS_DB} dB @ 56 GHz (Fig. 9)")
    print(f"Gap:                          {c2c_loss_at_56ghz - PAPER_LOSS_DB:.2f} dB")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
