"""Plotting helpers: the "outer layer" link/link.py's docstring refers to —
consumes LinkResult/EyeData, never the reverse. Unlike every domain package
under serdeskit (link/, channel/, common/), this one is allowed to import
matplotlib.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from serdeskit.channel import SParameterChannel
from serdeskit.common.types import Signal
from serdeskit.link import EyeData
from serdeskit.pulse_response import PulseResponse


def plot_eye(eye: EyeData, ax: Axes | None = None, title: str = "Eye Diagram") -> Axes:
    """Overlay every trace in `eye` on `ax` (a new Axes if none given)."""
    if ax is None:
        _, ax = plt.subplots()

    t = (np.arange(eye.traces.shape[0]) / eye.fs) * 1e12  # ps
    ax.plot(t, eye.traces, color="steelblue", linewidth=0.3, alpha=0.5)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)
    return ax


def plot_s21(
    channel: SParameterChannel,
    freq: npt.NDArray[np.float64],
    ax: Axes | None = None,
    title: str = "S21",
) -> Axes:
    """Plot |S21| (dB) vs. frequency (GHz) on `ax` (a new Axes if none given)."""
    if ax is None:
        _, ax = plt.subplots()

    s21 = channel.s21(freq)
    mag_db = 20 * np.log10(np.abs(s21))
    ax.plot(freq / 1e9, mag_db, color="steelblue")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 (dB)")
    ax.set_title(title)
    ax.grid(True)
    return ax


def plot_signal(
    signal: Signal, ax: Axes | None = None, label: str | None = None, title: str = "Signal"
) -> Axes:
    """Plot `signal.samples` against its absolute time axis (`signal.t0` +
    sample index / `signal.fs`) on `ax` (a new Axes if none given). `label`
    feeds `ax.legend()`, for overlaying several signals on one Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    t = signal.t0 + np.arange(len(signal.samples)) / signal.fs
    ax.plot(t, signal.samples, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title(title)
    ax.grid(True)
    return ax


def plot_pulse_response_cursors(
    pulse_response: PulseResponse,
    ax: Axes | None = None,
    dfe_min: npt.NDArray[np.float64] | None = None,
    dfe_max: npt.NDArray[np.float64] | None = None,
) -> Axes:
    """Scatter the cursor, pre-cursor, and post-cursor UI-spaced samples of
    `pulse_response` on `ax` (a new Axes if none given) -- same one-sample-
    per-UI convention MATLAB COM3.70's own plot uses (`sampled_best_sbr_
    precursors`/`_postcursors`, com_ieee8023_93a_370.m), starting at the
    cursor and stepping by `samples_per_ui` in each direction.

    If `dfe_min`/`dfe_max` are given, also stems each DFE-cancelled tap
    (`PulseResponse.dfe_cancelled_cursors`) from 0 up to the cancelled
    amount (pre-cancellation minus post-cancellation), matching MATLAB's
    own "DFE-canceled cursors" magenta stem plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    nspui = pulse_response.samples_per_ui
    cursor_ix = pulse_response.cursor_index
    n = len(pulse_response.samples)
    t = pulse_response.t0 + np.arange(n) / pulse_response.fs

    pre_ixs = np.arange(cursor_ix - nspui, -1, -nspui)[::-1]
    post_ixs = np.arange(cursor_ix + nspui, n, nspui)

    ax.plot(t[pre_ixs], pulse_response.samples[pre_ixs], "kx", label="Pre cursors")
    ax.plot(
        t[post_ixs], pulse_response.samples[post_ixs], "o", markeredgecolor="black",
        markerfacecolor="none", label="Post cursors",
    )
    ax.vlines(t[cursor_ix], 0, pulse_response.cursor_value, colors="green")
    ax.plot(
        t[cursor_ix], pulse_response.cursor_value, "o", markeredgecolor="green",
        markerfacecolor="none", label="Cursor (sample point)",
    )

    if dfe_min is not None and dfe_max is not None:
        cursors = pulse_response.dfe_cancelled_cursors(dfe_min, dfe_max)
        for i, (dfe_t, pre, post) in enumerate(cursors):
            ax.plot(
                [dfe_t, dfe_t], [0, pre - post], color="magenta", linewidth=2,
                label="DFE-canceled cursors" if i == 0 else None,
            )
        dfe_ts, dfe_pres, dfe_posts = zip(*cursors) if cursors else ((), (), ())
        cancelled = [pre - post for pre, post in zip(dfe_pres, dfe_posts)]
        ax.plot(dfe_ts, cancelled, "o", markeredgecolor="magenta", markerfacecolor="none")
        ax.plot(dfe_ts, dfe_posts, "o", markeredgecolor="black", markerfacecolor="none")

    return ax


def plot_impulse_response(
    channel: SParameterChannel,
    ax: Axes | None = None,
    title: str = "Impulse Response",
    dt: float | None = None,
    xlim: tuple[float, float] | None = None,
) -> Axes:
    """Plot the channel's impulse response (time in ns, amplitude in mV) on
    `ax` (a new Axes if none given). `dt` is passed straight through to
    SParameterChannel.impulse_response() — see its docstring: peak
    amplitude isn't comparable across different `dt`. `xlim` is (start,
    stop) in seconds (SI, matching every other time value in this codebase),
    e.g. to zoom in on the main peak.
    """
    if ax is None:
        _, ax = plt.subplots()

    t, h = channel.impulse_response(dt=dt)
    ax.plot(t * 1e9, h * 1e3, color="steelblue")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Impulse response (mV)")
    ax.set_title(title)
    ax.grid(True)
    if xlim is not None:
        ax.set_xlim(xlim[0] * 1e9, xlim[1] * 1e9)
    return ax
