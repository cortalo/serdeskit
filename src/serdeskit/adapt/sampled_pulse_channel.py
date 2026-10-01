"""SampledPulseChannel: a channel known only by its UI-spaced pulse-response
samples (cursors) on the sampling phase -- the symbol-rate view used by
adaptation loops, where nothing between sampling instants matters.
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
class SampledPulseChannel:
    """`cursors[n_pre]` is the main cursor a_0; entries before it are
    pre-cursors, after it post-cursors.
    """

    cursors: npt.NDArray[np.float64]
    n_pre: int

    def __post_init__(self) -> None:
        if not 0 <= self.n_pre < len(self.cursors):
            raise ValueError(
                f"n_pre must index into cursors (length {len(self.cursors)}), got {self.n_pre}."
            )
        self.cursors.setflags(write=False)

    @classmethod
    def from_waveform(
        cls,
        channel: WaveformChannel,
        symbol_rate: float,
        samples_per_ui: int,
        n_pre: int,
        n_post: int,
    ) -> SampledPulseChannel:
        """`channel`'s response to one isolated 1 V, 1 UI pulse, sampled once
        per UI on the peak's phase: n_pre pre-cursors, a_0, n_post post-cursors.
        """
        pulse = np.zeros((2 * (n_pre + n_post) + 400) * samples_per_ui)
        start = (n_pre + 200) * samples_per_ui
        pulse[start : start + samples_per_ui] = 1.0
        y = channel.process(Signal(samples=pulse, fs=samples_per_ui * symbol_rate)).samples
        idx = int(np.argmax(y)) + samples_per_ui * np.arange(-n_pre, n_post + 1)
        if idx[0] < 0 or idx[-1] >= len(y):
            raise ValueError("n_pre/n_post reach past the simulated pulse response.")
        return cls(cursors=np.asarray(y[idx], dtype=np.float64), n_pre=n_pre)
