"""TX FFE + RX DFE + dLev adapted together by sign-sign LMS over the ECEN720
T20 server thru channel at 10 Gb/s, compared with the TX FFE alone.

Configuration follows Stojanovic, JSSC 2005: a 4-tap TX FIR (pre1, main,
post1, post2) and a 1-tap DFE. The DFE takes the first post-cursor, so
the TX post1 tap is held at 0 and its swing goes to the main tap instead.
(The paper finds the DFE tap by locking dLev to two signal levels; here it
is adapted directly by sign-sign LMS.) The adapted settings are then
checked with the waveform-level serdeskit.link.Link: eyes unequalized,
with the TX FFE alone, and with TX FFE + DFE.

Requires reference/ecen720/peters_01_0605_T20_thru.s4p -- gitignored, see
docs/eye-diagram-design.md for where it comes from.

Run: python examples/plot_tx_ffe_dfe_adaptation_s4p.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from serdeskit.adapt import (
    AdaptationTrace,
    RxDfe,
    SampledPulseChannel,
    SignSignLms,
    SymbolRateLink,
    TxFfe,
)
from serdeskit.channel import SParameterChannel
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import EyeData, Link
from serdeskit.util import plot_eye

TOUCHSTONE_PATH = Path(__file__).parents[1] / "reference/ecen720/peters_01_0605_T20_thru.s4p"
SYMBOL_RATE = 10e9
SAMPLES_PER_UI = 32
N_PRE, N_POST = 3, 40  # channel cursors kept around the main one
TX_PRE, TX_POST = 1, 2  # TX FIR taps around the main one
N_DFE = 1
N_BLOCKS, BLOCK = 3000, 512
N_EYE_SYMBOLS = 10_000


def adapt(a: SampledPulseChannel, n_dfe: int) -> AdaptationTrace:
    weights = np.zeros(TX_PRE + 1 + TX_POST)
    weights[TX_PRE] = 1.0
    link = SymbolRateLink(
        channel=a, ffe=TxFfe(weights=weights, main=TX_PRE), dfe=RxDfe(taps=np.zeros(n_dfe))
    )
    rng = np.random.default_rng(0)
    lms = SignSignLms(step_tap=1e-3, step_dlev=1e-3, step_dfe=1e-3, noise_rms=5e-3, rng=rng)
    return lms.run(link=link, data=rng.choice(np.array([-1.0, 1.0]), size=(N_BLOCKS, BLOCK)))


def residual(
    a: npt.NDArray[np.float64], taps: npt.NDArray[np.float64], alpha: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Pulse after the TX FFE, the same after DFE cancellation, and the main
    cursor's index.
    """
    p = np.convolve(taps, a)
    i0 = N_PRE + TX_PRE
    after_dfe = p.copy()
    after_dfe[i0 + 1 : i0 + 1 + len(alpha)] -= alpha
    return p, after_dfe, i0


def waveform_eye(
    s4p: SParameterChannel,
    taps: npt.NDArray[np.float64],
    alpha: npt.NDArray[np.float64],
    bits: npt.NDArray[np.float64],
) -> EyeData:
    """Eye through the waveform-level Link. TapWeightFfe derives its cursor
    tap as 1 - sum(|others|), the swing constraint the adaptation enforced,
    so only the non-main taps are passed.
    """
    ffe = TapWeightFfe(
        tap_weights=np.delete(taps, TX_PRE), n_post=TX_POST, tap_delay=1 / SYMBOL_RATE
    )
    link = Link(channel=s4p, ffe=ffe, dfe=RxDfe(taps=alpha))
    return link.simulate(bits=bits, fs=SAMPLES_PER_UI * SYMBOL_RATE, symbol_rate=SYMBOL_RATE).eye


def main() -> None:
    # peters_* is ECEN720/PyBERT interleaved (TX+, RX+, TX-, RX-) -- see
    # differential_network's own docstring.
    s4p = SParameterChannel.from_touchstone(str(TOUCHSTONE_PATH), port_order=(0, 2, 1, 3))
    channel = SampledPulseChannel.from_waveform(s4p, SYMBOL_RATE, SAMPLES_PER_UI, N_PRE, N_POST)
    a = channel.cursors

    # Sign-sign keeps wandering around the solution, so report averages.
    results = {}
    for n_dfe in (0, N_DFE):
        trace = adapt(channel, n_dfe)
        taps = trace.tap_weights[-500:].mean(axis=0)
        alpha = trace.dfe_taps[-500:].mean(axis=0)
        results[n_dfe] = (trace, taps, alpha)

    print(f"{'':20s} {'TX taps':28s} {'DFE':8s} {'main':>6s} {'sum|ISI|':>9s} {'PDA eye':>8s}")
    print(f"{'unequalized':20s} {'':28s} {'':8s} {a[N_PRE]:6.3f} "
          f"{np.abs(np.delete(a, N_PRE)).sum():9.3f} {2 * (a[N_PRE] - np.abs(np.delete(a, N_PRE)).sum()):+8.3f}")
    for n_dfe, (_, taps, alpha) in results.items():
        _, eq, i0 = residual(a, taps, alpha)
        isi = np.abs(np.delete(eq, i0)).sum()
        name = f"TX FFE + {n_dfe}-tap DFE" if n_dfe else "TX FFE only"
        print(f"{name:20s} {np.array2string(taps, precision=3):28s} "
              f"{np.array2string(alpha, precision=3):8s} {eq[i0]:6.3f} {isi:9.3f} {2 * (eq[i0] - isi):+8.3f}")

    trace, taps, alpha = results[N_DFE]
    p, eq, i0 = residual(a, taps, alpha)

    fig, (ax_taps, ax_rx, ax_pulse) = plt.subplots(3, 1, figsize=(7, 9))
    for col in range(trace.tap_weights.shape[1]):
        ax_taps.plot(trace.tap_weights[:, col], label=f"w[{col - TX_PRE:+d}]")
    ax_taps.set_ylabel("TX tap weight")
    ax_taps.set_xlabel("iteration")
    ax_taps.legend(ncol=4)
    ax_taps.grid(True)

    for k in range(trace.dfe_taps.shape[1]):
        ax_rx.plot(trace.dfe_taps[:, k], label=f"DFE alpha_{k + 1}")
    ax_rx.plot(trace.dlev, label="dLev")
    ax_rx.set_ylabel("V")
    ax_rx.set_xlabel("iteration")
    ax_rx.legend()
    ax_rx.grid(True)

    ui = np.arange(-N_PRE, 13)
    window = slice(i0 - N_PRE, i0 - N_PRE + len(ui))
    ax_pulse.stem(ui - 0.15, a[: len(ui)], linefmt="C7-", markerfmt="C7o", basefmt=" ",
                  label="unequalized")
    ax_pulse.stem(ui, p[window], linefmt="C0-", markerfmt="C0o", basefmt=" ", label="after TX FFE")
    ax_pulse.stem(ui + 0.15, eq[window], linefmt="C3-", markerfmt="C3o", basefmt=" ",
                  label="after TX FFE + DFE")
    ax_pulse.axhline(0, color="k", lw=0.5)
    ax_pulse.set_xlabel("UI from main cursor")
    ax_pulse.set_ylabel("pulse response (V)")
    ax_pulse.legend()
    ax_pulse.grid(True)

    fig.suptitle(f"T20 thru, {SYMBOL_RATE / 1e9:.0f} Gb/s: TX FFE + {N_DFE}-tap DFE, sign-sign LMS")
    fig.tight_layout()

    bits = np.random.default_rng(1).choice(np.array([-1.0, 1.0]), size=N_EYE_SYMBOLS)
    no_eq = np.zeros(TX_PRE + 1 + TX_POST)
    no_eq[TX_PRE] = 1.0
    fig_eye, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (title, eye_taps, eye_alpha) in zip(axes, [
        ("Unequalized", no_eq, np.zeros(0)),
        ("TX FFE only", results[0][1], np.zeros(0)),
        (f"TX FFE + {N_DFE}-tap DFE", taps, alpha),
    ], strict=True):
        plot_eye(waveform_eye(s4p, eye_taps, eye_alpha, bits), ax=ax, title=title)
        ax.set_ylabel("Amplitude (V)")
    fig_eye.suptitle(f"T20 thru, {SYMBOL_RATE / 1e9:.0f} Gb/s, {N_EYE_SYMBOLS} random bits")
    fig_eye.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
