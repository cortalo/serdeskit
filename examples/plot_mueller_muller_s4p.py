"""Linear Mueller-Muller timing recovery on the ECEN720 T20 backplane at
10 Gb/s, no equalization, known data: where it locks relative to the pulse
peak, and how fast it gets there.

Requires reference/ecen720/peters_01_0605_T20_thru.s4p -- gitignored, see
plot_tx_ffe_adaptation_s4p.py.

Run: python examples/plot_mueller_muller_s4p.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.cdr import MuellerMuller, PulseChannel, SamplingPhaseLink
from serdeskit.channel import SParameterChannel

TOUCHSTONE_PATH = Path(__file__).parents[1] / "reference/ecen720/peters_01_0605_T20_thru.s4p"
SYMBOL_RATE = 10e9
SAMPLES_PER_UI = 64
N_PRE, N_POST = 3, 30
GAIN = 0.5  # UI per volt of block-averaged z
BLOCK = 1000
N_ITERATIONS = 200
STARTS = (-0.5, 0.6)


def main() -> None:
    channel = SParameterChannel.from_touchstone(str(TOUCHSTONE_PATH), port_order=(0, 2, 1, 3))
    pulse = PulseChannel.from_waveform(channel, SYMBOL_RATE, SAMPLES_PER_UI)

    grid = np.linspace(-0.75, 0.75, 301)
    cursors = np.array([pulse.cursors(ph, 1, 1) for ph in grid])  # h_-1, h_0, h_1
    detector = cursors[:, 2] - cursors[:, 0]
    root = grid[np.flatnonzero(np.diff(np.sign(detector)))[0]]

    rng = np.random.default_rng(0)
    runs = {}
    for start in STARTS:
        link = SamplingPhaseLink(pulse, N_PRE, N_POST, phase=start)
        data = rng.choice(np.array([-1.0, 1.0]), size=(N_ITERATIONS, BLOCK))
        runs[start] = MuellerMuller(gain=GAIN).run(link, data)

    for start, phases in runs.items():
        locked = phases[-100:]
        h = pulse.cursors(locked.mean(), 1, 1)
        print(f"start {start:+.2f} UI: locked {locked.mean():+.3f} UI "
              f"(rms {locked.std():.3f}), h_-1 {h[0]:.3f} h_0 {h[1]:.3f} h_1 {h[2]:.3f}")
    print(f"h_1 = h_-1 at {root:+.3f} UI; h_0 there {pulse.cursors(root, 0, 0)[0]:.3f}, "
          f"at the peak {pulse.cursors(0.0, 0, 0)[0]:.3f}")

    fig, (ax_pd, ax_phase, ax_pulse) = plt.subplots(3, 1, figsize=(7, 9))
    ax_pd.plot(grid, detector)
    ax_pd.axhline(0, color="k", lw=0.5)
    ax_pd.axvline(root, color="C3", ls="--", label=f"lock {root:+.3f} UI")
    ax_pd.set_xlabel("sampling phase (UI from pulse peak)")
    ax_pd.set_ylabel("$h_1 - h_{-1}$ (V)")
    ax_pd.legend()
    ax_pd.grid(True)

    for start, phases in runs.items():
        ax_phase.plot(phases, label=f"start {start:+.2f} UI")
    ax_phase.axhline(root, color="C3", ls="--", lw=1)
    ax_phase.set_xlabel(f"iteration ({BLOCK} symbols each)")
    ax_phase.set_ylabel("sampling phase (UI)")
    ax_phase.legend()
    ax_phase.grid(True)

    t = (np.arange(len(pulse.samples)) - pulse.peak) / SAMPLES_PER_UI
    ax_pulse.plot(t, pulse.samples, color="C7")
    for phase, color, name in [(0.0, "C0", "peak"), (root, "C3", "MM lock")]:
        ui = np.arange(-2, 6)
        ax_pulse.plot(phase + ui, pulse.cursors(phase, 2, 5), "o", color=color, label=name)
    ax_pulse.set_xlim(-3, 6)
    ax_pulse.set_xlabel("time (UI from pulse peak)")
    ax_pulse.set_ylabel("pulse response (V)")
    ax_pulse.legend()
    ax_pulse.grid(True)

    fig.suptitle(f"T20 thru, {SYMBOL_RATE / 1e9:.0f} Gb/s: linear Mueller-Muller, known data")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
