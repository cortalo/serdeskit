"""SymbolRateLink: TX FFE followed by a channel, simulated one sample per UI
as a continuous stream -- each respond() call picks up where the last one
left off, so callers can feed it any block sizes they like.

Deliberately separate from serdeskit.link.Link: that one is a static
composition for waveform/COM analysis, while this is the plant an
adaptation loop drives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from serdeskit.adapt.sampled_pulse_channel import SampledPulseChannel
from serdeskit.adapt.tx_ffe import TxFfe


@dataclass(frozen=True, slots=True)
class Received:
    """Received samples r[n] and the symbol d[n] each one carries as its
    main cursor. Before the stream's first symbol, d is 0 (idle line).
    """

    r: npt.NDArray[np.float64]
    d: npt.NDArray[np.float64]


class _StreamingFir:
    """y[t] = sum_j h_j x[t-j] over a stream fed in chunks, `taps` ordered by
    offset j. Pre-taps (j < 0) need future input, so output lags input by
    the number of pre-taps.
    """

    def __init__(self, taps: npt.NDArray[np.float64]) -> None:
        self.taps = taps
        self._history = np.zeros(len(taps) - 1)  # idle line before the stream

    def feed(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """One output per input sample."""
        ext = np.concatenate([self._history, x])
        self._history = ext[len(x) :]
        return np.convolve(ext, self.taps, "valid")


class SymbolRateLink:
    """Satisfies serdeskit.adapt.AdaptableLink implicitly. Stateful: it
    remembers what is already in flight, and a tap change only affects what
    the TX sends afterwards.
    """

    def __init__(self, channel: SampledPulseChannel, ffe: TxFfe) -> None:
        self._ffe = ffe
        self._tx = _StreamingFir(ffe.weights)
        self._channel = _StreamingFir(channel.cursors)
        # d[n] must line up with r[n]: delay the symbols by the link's
        # latency (TX + channel pre-cursors) -- a FIR with a single 1.
        latency = ffe.main + channel.n_pre
        self._symbols = _StreamingFir(np.eye(1, latency + 1, latency)[0])

    @property
    def ffe(self) -> TxFfe:
        return self._ffe

    @property
    def tap_weights(self) -> npt.NDArray[np.float64]:
        """All TX taps, main cursor included, ordered by UI offset."""
        return self._ffe.weights

    @property
    def main_tap(self) -> int:
        return self._ffe.main

    def set_tap_weights(self, tap_weights: npt.NDArray[np.float64]) -> None:
        self._ffe = self._ffe.with_weights(tap_weights)
        self._tx.taps = self._ffe.weights

    def respond(self, symbols: npt.NDArray[np.float64]) -> Received:
        r = self._channel.feed(self._tx.feed(symbols))
        return Received(r=r, d=self._symbols.feed(symbols))
