"""Plotting helpers: the "outer layer" link/link.py's docstring refers to —
consumes LinkResult/EyeData, never the reverse. Unlike every domain package
under serdeskit (link/, channel/, common/), this one is allowed to import
matplotlib.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

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
    return ax
