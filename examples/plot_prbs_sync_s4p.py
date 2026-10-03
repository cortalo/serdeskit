"""The receiver recovering the training data itself, then timing on it. On
the ECEN720 T20 backplane at 10 Gb/s, unequalized, sampling starts at a bad
phase: a PrbsSync locks onto the transmitted PRBS15 from its own
(error-prone) decisions and free-runs to give an error-free d; until then
the sign-sign Mueller-Muller CDR has no data to work with and the phase
holds. Then the CDR + dLev converge on the recovered d.

Requires reference/ecen720/peters_01_0605_T20_thru.s4p -- gitignored, see
plot_tx_ffe_adaptation_s4p.py.

Run: python examples/plot_prbs_sync_s4p.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.cdr import (
    PulseChannel,
    RecoveredDataLink,
    SamplingPhaseLink,
    SignSignMuellerMuller,
    TimingTrace,
)
from serdeskit.channel import SParameterChannel
from serdeskit.prbs import Prbs, PrbsSync

TOUCHSTONE_PATH = Path(__file__).parents[1] / "reference/ecen720/peters_01_0605_T20_thru.s4p"
SYMBOL_RATE = 10e9
SAMPLES_PER_UI = 64
N_PRE, N_POST = 3, 30
START_PHASE = -0.5  # UI from the pulse peak: between two symbols, the worst
ORDER = 15  # PRBS31 can false-lock on a closed eye: see PrbsSync
BLOCK, N_ITERATIONS = 64, 300
STEP_PHASE, STEP_DLEV = 1 / 256, 2e-3


def simulate(pulse: PulseChannel) -> tuple[TimingTrace, PrbsSync]:
    """The CDR's phase/dLev trace, and the sync (locked_at, attempts)."""
    prbs = Prbs(ORDER)
    sync = PrbsSync(prbs)
    link = RecoveredDataLink(SamplingPhaseLink(pulse, N_PRE, N_POST, phase=START_PHASE), sync)
    data = prbs.symbols(N_ITERATIONS * BLOCK, start=12345).reshape(N_ITERATIONS, BLOCK)
    loop = SignSignMuellerMuller(STEP_PHASE, STEP_DLEV, noise_rms=5e-3, rng=np.random.default_rng(0))
    return loop.run(link, data), sync


def main() -> None:
    channel = SParameterChannel.from_touchstone(str(TOUCHSTONE_PATH), port_order=(0, 2, 1, 3))
    pulse = PulseChannel.from_waveform(channel, SYMBOL_RATE, SAMPLES_PER_UI)
    trace, sync = simulate(pulse)

    assert sync.locked_at is not None
    final = trace.phase[-100:].mean()
    print(f"PRBS{ORDER} locked after {sync.locked_at} bits ({sync.attempts} seeds); "
          f"phase settles at {final:+.3f} UI, dLev {trace.dlev[-100:].mean():.3f} V")

    bits = np.arange(len(trace.phase)) * BLOCK
    fig, (ax, ax_dlev) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    ax.plot(bits, trace.phase)
    ax.annotate(f"PRBS{ORDER} locked ({sync.locked_at} bits)", (sync.locked_at, START_PHASE),
                xytext=(8, -14), textcoords="offset points", fontsize=9)
    # With known data the loop locks at +0.106 UI; the sync may label the
    # data one bit off, which moves the same lock point by a whole UI.
    ax.axhline(0.106 - 1, color="gray", ls="--", lw=1)
    ax.annotate("h_1 = h_-1 lock, one UI over (+0.106 - 1)", (bits[-1], 0.106 - 1),
                xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=9, color="gray")
    ax.set_ylabel("sampling phase (UI from pulse peak)")
    ax.set_title(f"T20, 10 Gb/s: hold until the PRBS{ORDER} sync locks, then SS-MM CDR on the recovered d")
    ax_dlev.plot(bits, trace.dlev, color="C1")
    ax_dlev.set_ylabel("dLev (V)")
    ax_dlev.set_xlabel("received bits")
    for a in (ax, ax_dlev):
        a.axvline(sync.locked_at, color="k", ls=":", lw=1)
        a.grid(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
