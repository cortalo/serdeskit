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
