"""Dual-loop sign-sign LMS TX FFE adaptation (Stojanovic, JSSC 2005) over a
real backplane channel: the ECEN720 T20 server thru channel at 10 Gb/s,
where the unequalized worst-case eye is closed (PDA: a_0 < sum|a_m|).

The channel enters the symbol-rate model as its UI-spaced pulse-response
samples on the main cursor's phase (taken at the pulse peak). The adapted
taps are then checked with the waveform-level serdeskit.link.Link: eye
diagrams before (main tap 1) and after (adapted taps) TX equalization.

Requires reference/ecen720/peters_01_0605_T20_thru.s4p -- gitignored, see
docs/eye-diagram-design.md for where it comes from.

Run: python examples/plot_tx_ffe_adaptation_s4p.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from serdeskit.adapt import SampledPulseChannel, SignSignLms, SymbolRateLink, TxFfe
from serdeskit.channel import SParameterChannel
from serdeskit.common.types import Signal
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import EyeData, Link
from serdeskit.util import plot_eye

TOUCHSTONE_PATH = Path(__file__).parents[1] / "reference/ecen720/peters_01_0605_T20_thru.s4p"
SYMBOL_RATE = 10e9
SAMPLES_PER_UI = 32
N_PRE, N_POST = 3, 40  # cursors kept around the main one
N_EYE_SYMBOLS = 10_000


def sampled_cursors(channel: SParameterChannel) -> npt.NDArray[np.float64]:
    """The pulse response to one isolated 1 V, 1 UI pulse, sampled once per
    UI on the peak's phase: N_PRE pre-cursors, a_0, N_POST post-cursors.
    """
    pulse = np.zeros(400 * SAMPLES_PER_UI)
    pulse[200 * SAMPLES_PER_UI : 201 * SAMPLES_PER_UI] = 1.0
    y = channel.process(Signal(samples=pulse, fs=SAMPLES_PER_UI * SYMBOL_RATE)).samples
    peak = int(np.argmax(y))
    return np.asarray(y[peak + SAMPLES_PER_UI * np.arange(-N_PRE, N_POST + 1)], dtype=np.float64)


def tx_ffe_eye(channel: SParameterChannel, ffe: TxFfe, bits: npt.NDArray[np.float64]) -> EyeData:
    """Waveform-level eye through Link with `ffe` as its TX FFE. TapWeightFfe
    derives its cursor tap as 1 - sum(|others|), the same swing constraint
    the adaptation enforced, so the adapted main tap carries over as is.
    """
    tx = TapWeightFfe(
        tap_weights=np.delete(ffe.weights, ffe.main),
        n_post=len(ffe.weights) - 1 - ffe.main,
        tap_delay=1 / SYMBOL_RATE,
    )
    fs = SAMPLES_PER_UI * SYMBOL_RATE
    return Link(channel=channel, ffe=tx).simulate(bits=bits, fs=fs, symbol_rate=SYMBOL_RATE).eye


def main() -> None:
    # peters_* is ECEN720/PyBERT interleaved (TX+, RX+, TX-, RX-) -- see
    # differential_network's own docstring.
    channel = SParameterChannel.from_touchstone(str(TOUCHSTONE_PATH), port_order=(0, 2, 1, 3))
    a = sampled_cursors(channel)

    ffe = TxFfe(weights=np.array([0.0, 1.0, 0.0, 0.0]), main=1)  # pre1, main, post1, post2
    link = SymbolRateLink(channel=SampledPulseChannel(cursors=a, n_pre=N_PRE), ffe=ffe)
    rng = np.random.default_rng(0)
    trace = SignSignLms(step_tap=1e-3, step_dlev=1e-3, noise_rms=5e-3, rng=rng).run(
        link=link,
        data=rng.choice(np.array([-1.0, 1.0]), size=(3000, 512)),
    )

    # Sign-sign keeps wandering around the solution (slowly, along the
    # weakly-observed main/post2 direction here), so report an average.
    taps = trace.tap_weights[-500:].mean(axis=0)
    p = np.convolve(taps, a)  # equalized pulse, p_0 at index N_PRE + main
    i0 = N_PRE + ffe.main
    ui = np.arange(-N_PRE, N_POST + 1)
    for name, cursors, c in [("unequalized", a, N_PRE), ("equalized", p, i0)]:
        isi = np.abs(np.delete(cursors, c)).sum()
        print(f"{name:12s} main cursor {cursors[c]:.3f}  sum|ISI| {isi:.3f}  "
              f"PDA eye height {2 * (cursors[c] - isi):+.3f}")
    print("taps [pre1, main, post1, post2], last 500 iterations' mean:", taps.round(3))

    fig, (ax_taps, ax_dlev, ax_pulse) = plt.subplots(3, 1, figsize=(7, 9))
    for col, name in enumerate(["pre1", "main", "post1", "post2"]):
        ax_taps.plot(trace.tap_weights[:, col], label=name)
    ax_taps.set_ylabel("tap weight")
    ax_taps.set_xlabel("iteration")
    ax_taps.legend()
    ax_taps.grid(True)

    ax_dlev.plot(trace.dlev)
    ax_dlev.set_ylabel("dLev (V)")
    ax_dlev.set_xlabel("iteration")
    ax_dlev.grid(True)

    ax_pulse.stem(ui, a, linefmt="C7-", markerfmt="C7o", basefmt=" ", label="unequalized")
    ax_pulse.stem(ui, p[i0 - N_PRE : i0 + N_POST + 1], linefmt="C3-", markerfmt="C3o",
                  basefmt=" ", label="TX FFE equalized")
    ax_pulse.axhline(0, color="k", lw=0.5)
    ax_pulse.set_xlim(-N_PRE - 0.5, 12.5)
    ax_pulse.set_xlabel("UI from main cursor")
    ax_pulse.set_ylabel("pulse response (V)")
    ax_pulse.legend()
    ax_pulse.grid(True)

    fig.suptitle(f"T20 thru, {SYMBOL_RATE / 1e9:.0f} Gb/s: TX FFE sign-sign LMS")
    fig.tight_layout()

    bits = rng.choice(np.array([-1.0, 1.0]), size=N_EYE_SYMBOLS)
    fig_eye, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(11, 4))
    plot_eye(tx_ffe_eye(channel, ffe, bits), ax=ax_before, title="Before: main tap 1")
    adapted = ffe.with_weights(taps)
    plot_eye(tx_ffe_eye(channel, adapted, bits), ax=ax_after,
             title=f"After: taps {np.array2string(taps, precision=3)}")
    fig_eye.suptitle(f"T20 thru, {SYMBOL_RATE / 1e9:.0f} Gb/s eye, {N_EYE_SYMBOLS} random bits")
    fig_eye.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
