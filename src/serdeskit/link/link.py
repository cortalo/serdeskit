"""Link: the top-level object. Currently holds a single Channel and runs it
end to end, producing a LinkResult — plain data, not a plot. Rendering (eye
diagrams as matplotlib figures) is deliberately a separate, outer layer:
Link never imports matplotlib, so its output can be asserted on directly in
tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from serdeskit.common.types import Signal


class Channel(Protocol):
    """What Link needs from a channel: turn the signal launched into it into
    the signal that arrives at the receiver. Satisfied implicitly — a
    concrete channel class needs no relation to this Protocol beyond having
    a matching `process` method.
    """

    def process(self, sig: Signal) -> Signal: ...


@dataclass(frozen=True, slots=True)
class EyeData:
    """One column per unit-interval-pair trace, ready to overlay-plot."""

    traces: npt.NDArray[np.float64]  # shape: (samples_per_2ui, n_traces)
    ui: float  # unit interval, seconds
    fs: float  # sample rate the traces were sliced at, Hz — dt between rows is 1/fs


@dataclass(frozen=True, slots=True)
class LinkResult:
    eye: EyeData


@dataclass
class Link:
    """channel is the (currently only) stage the launched signal passes
    through.
    """

    channel: Channel

    def simulate(self, bits: npt.NDArray[np.float64], fs: float, symbol_rate: float) -> LinkResult:
        """bits is one value per symbol; symbol_rate is what turns it into a
        proper analog-time waveform at fs before it ever reaches the channel.
        """
        samples = _upsample_bits(bits, fs, symbol_rate)
        sig = Signal(samples=samples, fs=fs, t0=0.0)
        sig = self.channel.process(sig)

        eye = _extract_eye(sig, symbol_rate)
        return LinkResult(eye=eye)


def _upsample_bits(bits: npt.NDArray[np.float64], fs: float, symbol_rate: float) -> npt.NDArray[np.float64]:
    """Zero-order-hold: each symbol value held for its full UI, turning a
    per-symbol sequence into a per-sample waveform at fs (mirrors the
    `repmat` upsampling step in reference/ecen720/channel_data.m).
    """
    samples_per_ui = round(fs / symbol_rate)
    return np.repeat(bits, samples_per_ui)


def _extract_eye(sig: Signal, symbol_rate: float) -> EyeData:
    """Sliding window: 1 UI step, 2 UI width (see docs/eye-diagram-design.md
    for why — matches MATLAB/PyBERT, not serdespy's non-overlapping split).
    No zero-crossing alignment yet: PassThroughChannel has no delay, so
    windows starting at sample 0 are already UI-aligned.
    """
    samples_per_ui = round(sig.fs / symbol_rate)
    window = 2 * samples_per_ui
    n_traces = (len(sig.samples) - window) // samples_per_ui + 1

    traces = np.empty((window, n_traces), dtype=np.float64)
    for i in range(n_traces):
        start = i * samples_per_ui
        traces[:, i] = sig.samples[start : start + window]

    return EyeData(traces=traces, ui=1.0 / symbol_rate, fs=sig.fs)
