"""PulseChannel: a channel known by its whole response to one isolated pulse,
finely sampled -- so the UI-spaced cursors can be read at any sampling
phase, which is what a CDR loop moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from serdeskit.common.types import Signal


class WaveformChannel(Protocol):
    """A channel run on a sampled waveform, e.g. SParameterChannel."""

    def process(self, sig: Signal) -> Signal: ...


@dataclass(frozen=True)
class PulseChannel:
    """`samples[peak]` is the pulse's peak, sampling phase 0."""

    samples: npt.NDArray[np.float64]
    samples_per_ui: int
    peak: int

    def __post_init__(self) -> None:
        if not 0 <= self.peak < len(self.samples):
            raise ValueError(
                f"peak must index into samples (length {len(self.samples)}), got {self.peak}."
            )
        self.samples.setflags(write=False)

    def cursors(self, phase: float, n_pre: int, n_post: int) -> npt.NDArray[np.float64]:
        """The pulse at (phase + k) UI from its peak, k = -n_pre..n_post,
        linearly interpolated. `phase` in UI, positive = sampling later.
        """
        t = self.peak + (phase + np.arange(-n_pre, n_post + 1)) * self.samples_per_ui
        if t[0] < 0 or t[-1] > len(self.samples) - 1:
            raise ValueError("phase/n_pre/n_post reach past the stored pulse response.")
        y = np.interp(t, np.arange(len(self.samples)), self.samples)
        return np.asarray(y, dtype=np.float64)

    @classmethod
    def from_waveform(
        cls,
        channel: WaveformChannel,
        symbol_rate: float,
        samples_per_ui: int,
        n_ui: int = 400,
    ) -> PulseChannel:
        """`channel`'s response to one isolated 1 V, 1 UI pulse placed in the
        middle of an `n_ui`-long record.
        """
        pulse = np.zeros(n_ui * samples_per_ui)
        start = n_ui // 2 * samples_per_ui
        pulse[start : start + samples_per_ui] = 1.0
        y = channel.process(Signal(samples=pulse, fs=samples_per_ui * symbol_rate)).samples
        samples = np.asarray(y, dtype=np.float64)
        return cls(samples=samples, samples_per_ui=samples_per_ui, peak=int(np.argmax(samples)))
