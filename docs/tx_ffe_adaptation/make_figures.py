"""Regenerate the report's figures into figures/ from the same code as
examples/plot_tx_ffe_adaptation_s4p.py, plot_tx_ffe_dfe_adaptation_s4p.py,
plot_mueller_muller_s4p.py and plot_sign_sign_mueller_muller_s4p.py.

Run: python docs/tx_ffe_adaptation/make_figures.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from serdeskit.adapt import SampledPulseChannel, SignSignLms, SymbolRateLink, TxFfe
from serdeskit.cdr import PulseChannel
from serdeskit.channel import SParameterChannel
from serdeskit.util import plot_eye

HERE = Path(__file__).parent
REPO = HERE.parents[1]
OUT = HERE / "figures"


def _load_example(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, REPO / "examples" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _load_example("plot_tx_ffe_adaptation_s4p")
ex_dfe = _load_example("plot_tx_ffe_dfe_adaptation_s4p")
ex_mm = _load_example("plot_mueller_muller_s4p")
ex_ssmm = _load_example("plot_sign_sign_mueller_muller_s4p")

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

    dfe_figures(channel)
    mm_figures(channel)
    ssmm_figures(channel)


def dfe_figures(s4p: SParameterChannel) -> None:
    """TX FFE + 1-tap DFE, against the TX FFE alone."""
    channel = SampledPulseChannel.from_waveform(
        s4p, ex_dfe.SYMBOL_RATE, ex_dfe.SAMPLES_PER_UI, ex_dfe.N_PRE, ex_dfe.N_POST
    )
    a = channel.cursors
    runs = {}
    for n_dfe in (0, ex_dfe.N_DFE):
        trace = ex_dfe.adapt(channel, n_dfe)
        runs[n_dfe] = (trace, trace.tap_weights[-500:].mean(axis=0), trace.dfe_taps[-500:].mean(axis=0))
    trace, taps, alpha = runs[ex_dfe.N_DFE]
    p, eq, i0 = ex_dfe.residual(a, taps, alpha)

    fig, (ax_w, ax_rx) = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
    for col, name in enumerate(["$w_{-1}$", "$w_0$", "$w_1$", "$w_2$"]):
        ax_w.plot(trace.tap_weights[:, col], label=name, lw=0.9)
    ax_w.set_ylabel("TX tap weight")
    ax_w.set_ylim(-0.4, 1.3)
    ax_w.legend(ncol=4, fontsize=7, loc="upper right")
    ax_w.grid(True)
    ax_rx.plot(trace.dfe_taps[:, 0], label=r"DFE $\alpha_1$", lw=0.9)
    ax_rx.plot(trace.dlev, label="dLev", lw=0.9)
    ax_rx.set_ylabel("V")
    ax_rx.set_xlabel("iteration (512 symbols each)")
    ax_rx.legend(fontsize=7)
    ax_rx.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "dfe_convergence.pdf")

    ui = np.arange(-ex_dfe.N_PRE, 11)
    window = slice(i0 - ex_dfe.N_PRE, i0 - ex_dfe.N_PRE + len(ui))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.stem(ui - 0.15, a[: len(ui)], linefmt="C7-", markerfmt="C7o", basefmt=" ", label="unequalized")
    ax.stem(ui, p[window], linefmt="C0-", markerfmt="C0o", basefmt=" ", label="after TX FFE")
    ax.stem(ui + 0.15, eq[window], linefmt="C3-", markerfmt="C3o", basefmt=" ", label="after TX FFE + DFE")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("UI from main cursor")
    ax.set_ylabel("pulse response (V)")
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "dfe_pulse.pdf")

    bits = np.random.default_rng(1).choice(np.array([-1.0, 1.0]), size=ex_dfe.N_EYE_SYMBOLS)
    no_eq = np.zeros(len(taps))
    no_eq[ex_dfe.TX_PRE] = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
    for ax, (title, eye_taps, eye_alpha) in zip(axes, [
        ("Unequalized", no_eq, np.zeros(0)),
        ("TX FFE only", runs[0][1], np.zeros(0)),
        ("TX FFE + 1-tap DFE", taps, alpha),
    ], strict=True):
        plot_eye(ex_dfe.waveform_eye(s4p, eye_taps, eye_alpha, bits), ax=ax, title=title)
        ax.set_ylabel("Amplitude (V)")
    fig.tight_layout()
    fig.savefig(OUT / "dfe_eyes.png", dpi=200)

    for n_dfe, (_, w, al) in runs.items():
        _, res, c = ex_dfe.residual(a, w, al)
        isi = np.abs(np.delete(res, c)).sum()
        print(f"DFE={n_dfe}: taps {w.round(3)} alpha {al.round(3)} main {res[c]:.3f} "
              f"sum|ISI| {isi:.3f} PDA eye {2 * (res[c] - isi):+.3f}")


def mm_figures(s4p: SParameterChannel) -> None:
    """Linear Mueller-Muller on the unequalized channel, known data."""
    pulse = PulseChannel.from_waveform(s4p, ex_mm.SYMBOL_RATE, ex_mm.SAMPLES_PER_UI)
    grid = np.linspace(-0.75, 0.75, 301)
    detector, root = ex_mm.detector_curve(pulse, grid)
    rng = np.random.default_rng(0)
    runs = {start: ex_mm.track(pulse, start, rng) for start in ex_mm.STARTS}

    fig, (ax_pd, ax_ph) = plt.subplots(2, 1, figsize=(3.4, 3.6))
    ax_pd.plot(grid, detector, lw=0.9)
    ax_pd.axhline(0, color="k", lw=0.5)
    ax_pd.axvline(root, color="C3", ls="--", lw=0.9)
    ax_pd.set_xlabel("sampling phase (UI from pulse peak)")
    ax_pd.set_ylabel("$h_1 - h_{-1}$ (V)")
    ax_pd.grid(True)
    for start, phases in runs.items():
        ax_ph.plot(phases, lw=0.9, label=f"start {start:+.1f} UI")
    ax_ph.axhline(root, color="C3", ls="--", lw=0.9)
    ax_ph.set_xlabel(f"iteration ({ex_mm.BLOCK} symbols each)")
    ax_ph.set_ylabel("phase (UI)")
    ax_ph.set_xlim(0, 60)
    ax_ph.legend(fontsize=7)
    ax_ph.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "mm_loop.pdf")

    t = (np.arange(len(pulse.samples)) - pulse.peak) / ex_mm.SAMPLES_PER_UI
    ui = np.arange(-2, 6)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(t, pulse.samples, color="C7", lw=0.9)
    ax.plot(ui, pulse.cursors(0.0, 2, 5), "o", ms=4, color="C0", label="pulse peak")
    ax.plot(root + ui, pulse.cursors(root, 2, 5), "o", ms=4, color="C3", label="MM lock")
    ax.set_xlim(-3, 6)
    ax.set_xlabel("time (UI from pulse peak)")
    ax.set_ylabel("pulse response (V)")
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "mm_pulse.pdf")

    for start, phases in runs.items():
        locked = phases[-100:]
        print(f"MM start {start:+.2f}: locked {locked.mean():+.3f} UI, rms {locked.std():.4f}")
    print(f"MM root {root:+.3f} UI, cursors there {pulse.cursors(root, 1, 1).round(3)}, "
          f"at the peak {pulse.cursors(0.0, 1, 1).round(3)}")


def ssmm_figures(s4p: SParameterChannel) -> None:
    """Sign-sign Mueller-Muller with its dLev loop, known data."""
    pulse = PulseChannel.from_waveform(s4p, ex_ssmm.SYMBOL_RATE, ex_ssmm.SAMPLES_PER_UI)
    rng = np.random.default_rng(0)
    runs = {start: ex_ssmm.track(pulse, start, rng) for start in ex_ssmm.STARTS}

    fig, (ax_ph, ax_d) = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
    for start, trace in runs.items():
        ax_ph.plot(trace.phase, lw=0.9, label=f"start {start:+.1f} UI")
        ax_d.plot(trace.dlev, lw=0.9)
    ax_ph.set_ylabel("phase (UI)")
    ax_ph.legend(fontsize=7)
    ax_ph.grid(True)
    ax_d.set_ylabel("dLev (V)")
    ax_d.set_xlabel(f"iteration ({ex_ssmm.BLOCK} symbols each)")
    ax_d.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "ssmm_loop.pdf")

    grid = np.linspace(-0.5, 0.6, 45)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for dlev in (0.1, 0.2, 0.316, 0.45):
        ax.plot(grid, ex_ssmm.detector_curve(pulse, grid, dlev, rng), lw=0.9,
                label=f"dLev {dlev:.2f} V")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("sampling phase (UI from pulse peak)")
    ax.set_ylabel("mean phase vote")
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / "ssmm_detector.pdf")

    for start, trace in runs.items():
        print(f"SS-MM start {start:+.2f}: phase {trace.phase[-200:].mean():+.3f} UI "
              f"(rms {trace.phase[-200:].std():.3f}), dLev {trace.dlev[-200:].mean():.3f} V")


if __name__ == "__main__":
    main()
