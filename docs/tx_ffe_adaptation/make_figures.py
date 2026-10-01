"""Regenerate the report's figures into figures/ from the same code as
examples/plot_tx_ffe_adaptation_s4p.py.

Run: python docs/tx_ffe_adaptation/make_figures.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.adapt import SampledPulseChannel, SignSignLms, SymbolRateLink, TxFfe
from serdeskit.channel import SParameterChannel
from serdeskit.util import plot_eye

HERE = Path(__file__).parent
REPO = HERE.parents[1]
OUT = HERE / "figures"

spec = importlib.util.spec_from_file_location("example", REPO / "examples/plot_tx_ffe_adaptation_s4p.py")
assert spec and spec.loader
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def main() -> None:
    channel = SParameterChannel.from_touchstone(str(ex.TOUCHSTONE_PATH), port_order=(0, 2, 1, 3))

    # S21
    f = np.linspace(50e6, 15e9, 600)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(f / 1e9, 20 * np.log10(np.abs(channel.s21(f))))
    ax.axvline(ex.SYMBOL_RATE / 2e9, color="k", ls="--", lw=0.8)
    ax.text(ex.SYMBOL_RATE / 2e9 + 0.3, -8, "Nyquist\n5 GHz", fontsize=8)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|SDD21| (dB)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "s21.pdf")

    # Adaptation
    a = ex.sampled_cursors(channel)
    ffe = TxFfe(weights=np.array([0.0, 1.0, 0.0, 0.0]), main=1)
    link = SymbolRateLink(channel=SampledPulseChannel(cursors=a, n_pre=ex.N_PRE), ffe=ffe)
    rng = np.random.default_rng(0)
    trace = SignSignLms(step_tap=1e-3, step_dlev=1e-3, noise_rms=5e-3, rng=rng).run(
        link=link, data=rng.choice(np.array([-1.0, 1.0]), size=(3000, 512))
    )
    taps = trace.tap_weights[-500:].mean(axis=0)

    fig, (ax_w, ax_d) = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
    for col, name in enumerate(["$w_{-1}$", "$w_0$", "$w_1$", "$w_2$"]):
        ax_w.plot(trace.tap_weights[:, col], label=name, lw=0.9)
    ax_w.set_ylabel("TX tap weight")
    ax_w.set_ylim(-0.4, 1.3)
    ax_w.legend(ncol=4, fontsize=7, loc="upper right")
    ax_w.grid(True)
    ax_d.plot(trace.dlev, lw=0.9)
    ax_d.set_ylabel("dLev (V)")
    ax_d.set_xlabel("iteration (512 symbols each)")
    ax_d.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.pdf")

    # Pulse response cursors
    p = np.convolve(taps, a)
    i0 = ex.N_PRE + ffe.main
    ui = np.arange(-ex.N_PRE, 11)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.stem(ui, a[: len(ui)], linefmt="C7-", markerfmt="C7o", basefmt=" ", label="unequalized")
    ax.stem(ui + 0.12, p[i0 - ex.N_PRE : i0 - ex.N_PRE + len(ui)], linefmt="C3-",
            markerfmt="C3o", basefmt=" ", label="TX FFE equalized")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("UI from main cursor")
    ax.set_ylabel("pulse response (V)")
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "pulse.pdf")

    # Eyes (raster: 10k traces)
    bits = rng.choice(np.array([-1.0, 1.0]), size=ex.N_EYE_SYMBOLS)
    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(7, 2.6))
    plot_eye(ex.tx_ffe_eye(channel, ffe, bits), ax=ax_b, title="Unequalized")
    plot_eye(ex.tx_ffe_eye(channel, ffe.with_weights(taps), bits), ax=ax_a, title="Adapted TX FFE")
    for ax in (ax_b, ax_a):
        ax.set_ylabel("Amplitude (V)")
    fig.tight_layout()
    fig.savefig(OUT / "eyes.png", dpi=200)

    for name, cursors, c in [("unequalized", a, ex.N_PRE), ("equalized", p, i0)]:
        isi = np.abs(np.delete(cursors, c)).sum()
        print(f"{name}: main {cursors[c]:.3f}, sum|ISI| {isi:.3f}, PDA eye {2 * (cursors[c] - isi):+.3f}")
    print("taps:", taps.round(3), " dLev:", round(float(trace.dlev[-500:].mean()), 3))


if __name__ == "__main__":
    main()
