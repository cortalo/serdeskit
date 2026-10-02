"""Sign-sign Mueller-Muller timing recovery with its dLev loop, on the ECEN720
T20 backplane at 10 Gb/s, no equalization, known data: both loops start far
off (dLev at 0) and lock together; the phase should land where the linear
detector does, h_1 = h_-1.

Requires reference/ecen720/peters_01_0605_T20_thru.s4p -- gitignored, see
plot_tx_ffe_adaptation_s4p.py.

Run: python examples/plot_sign_sign_mueller_muller_s4p.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from serdeskit.cdr import PulseChannel, SamplingPhaseLink, SignSignMuellerMuller, TimingTrace
from serdeskit.channel import SParameterChannel

TOUCHSTONE_PATH = Path(__file__).parents[1] / "reference/ecen720/peters_01_0605_T20_thru.s4p"
SYMBOL_RATE = 10e9
SAMPLES_PER_UI = 64
N_PRE, N_POST = 3, 30
STEP_PHASE = 1 / 128  # UI
STEP_DLEV = 2e-3  # V
NOISE_RMS = 5e-3  # V
BLOCK = 512
N_ITERATIONS = 600
STARTS = (-0.5, 0.6)


def track(pulse: PulseChannel, start: float, rng: np.random.Generator) -> TimingTrace:
    """Phase and dLev per iteration, phase starting at `start` UI, dLev at 0."""
    link = SamplingPhaseLink(pulse, N_PRE, N_POST, phase=start)
    data = rng.choice(np.array([-1.0, 1.0]), size=(N_ITERATIONS, BLOCK))
    loop = SignSignMuellerMuller(STEP_PHASE, STEP_DLEV, noise_rms=NOISE_RMS, rng=rng)
    return loop.run(link, data)


def detector_curve(
    pulse: PulseChannel,
    grid: npt.NDArray[np.float64],
    dlev: float,
    rng: np.random.Generator,
    n_symbols: int = 100_000,
) -> npt.NDArray[np.float64]:
    """Mean sign-sign phase vote d[n]d[n-1](ERR[n] - ERR[n-1]) per symbol, at
    each phase in `grid`, for a fixed dLev.
    """
    d = rng.choice(np.array([-1.0, 1.0]), size=n_symbols)
    votes = []
    for phase in grid:
        rx = SamplingPhaseLink(pulse, N_PRE, N_POST, phase=phase).respond(d)
        r = rx.r + rng.normal(0.0, NOISE_RMS, n_symbols)
        err = np.where(r * rx.d > dlev, 1.0, -1.0)
        vote = rx.d[1:] * rx.d[:-1] * (err[1:] - err[:-1])
        votes.append(vote[N_POST:].mean())  # skip the idle-line start
    return np.array(votes)


def main() -> None:
    channel = SParameterChannel.from_touchstone(str(TOUCHSTONE_PATH), port_order=(0, 2, 1, 3))
    pulse = PulseChannel.from_waveform(channel, SYMBOL_RATE, SAMPLES_PER_UI)

    rng = np.random.default_rng(0)
    runs = {start: track(pulse, start, rng) for start in STARTS}

    for start, trace in runs.items():
        phase, dlev = trace.phase[-200:], trace.dlev[-200:]
        h = pulse.cursors(phase.mean(), 1, 1)
        print(f"start {start:+.2f} UI: phase {phase.mean():+.3f} UI (rms {phase.std():.3f}), "
              f"dLev {dlev.mean():.3f} V; h_-1 {h[0]:.3f} h_0 {h[1]:.3f} h_1 {h[2]:.3f}")

    fig, (ax_phase, ax_dlev) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for start, trace in runs.items():
        ax_phase.plot(trace.phase, label=f"start {start:+.2f} UI")
        ax_dlev.plot(trace.dlev, label=f"start {start:+.2f} UI")
    ax_phase.set_ylabel("sampling phase (UI from pulse peak)")
    ax_phase.legend()
    ax_phase.grid(True)
    ax_dlev.set_ylabel("dLev (V)")
    ax_dlev.set_xlabel(f"iteration ({BLOCK} symbols each)")
    ax_dlev.grid(True)
    fig.suptitle(f"T20 thru, {SYMBOL_RATE / 1e9:.0f} Gb/s: sign-sign Mueller-Muller + dLev, known data")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
