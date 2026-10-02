"""SamplingPhaseLink: a channel sampled once per UI at a phase the RX can move
at runtime -- the plant a CDR loop drives. Streams like
serdeskit.adapt.SymbolRateLink: each respond() call picks up where the last
one left off.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from serdeskit.adapt import Received
from serdeskit.cdr.pulse_channel import PulseChannel


class SamplingPhaseLink:
    """Cursors outside -n_pre..n_post are dropped. A phase change only
    affects samples taken afterwards; the waveform already in flight is
    unchanged, it is just read at a different instant.
    """

    def __init__(self, channel: PulseChannel, n_pre: int, n_post: int, phase: float = 0.0) -> None:
        self._channel = channel
        self._n_pre = n_pre
        self._n_post = n_post
        self._cursors = channel.cursors(phase, n_pre, n_post)
        self._phase = phase
        self._history = np.zeros(n_pre + n_post)  # idle line before the stream

    @property
    def sampling_phase(self) -> float:
        """UI from the pulse peak, positive = later."""
        return self._phase

    def set_sampling_phase(self, phase: float) -> None:
        self._cursors = self._channel.cursors(phase, self._n_pre, self._n_post)
        self._phase = phase

    def respond(self, symbols: npt.NDArray[np.float64]) -> Received:
        """Slices at 0; d lags the input by n_pre, the pre-cursors' latency."""
        ext = np.concatenate([self._history, symbols])
        self._history = ext[len(symbols) :]
        r = np.convolve(ext, self._cursors, "valid")
        d = ext[self._n_post : self._n_post + len(symbols)]
        return Received(r=r, decisions=np.where(r > 0, 1.0, -1.0), d=d)
