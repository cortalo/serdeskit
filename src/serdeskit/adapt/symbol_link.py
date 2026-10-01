"""SymbolRateLink: TX FFE, channel and RX DFE, simulated one sample per UI
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

from serdeskit.adapt.rx_dfe import RxDfe
from serdeskit.adapt.sampled_pulse_channel import SampledPulseChannel
from serdeskit.adapt.tx_ffe import TxFfe


@dataclass(frozen=True, slots=True)
class Received:
    """Per UI: the DFE summer output r[n] (the received sample once the DFE
    feedback is subtracted), the slicer's decision on it, and the symbol
    d[n] it carries as its main cursor. Before the stream's first symbol,
    d is 0 (idle line).
    """

    r: npt.NDArray[np.float64]
    decisions: npt.NDArray[np.float64]
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

    def __init__(self, channel: SampledPulseChannel, ffe: TxFfe, dfe: RxDfe | None = None) -> None:
        self._ffe = ffe
        self._dfe = dfe if dfe is not None else RxDfe(taps=np.zeros(0))
        self._past = [0.0] * len(self._dfe.taps)  # decisions, most recent first
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

    @property
    def dfe_taps(self) -> npt.NDArray[np.float64]:
        return self._dfe.taps

    def set_dfe_taps(self, taps: npt.NDArray[np.float64]) -> None:
        self._dfe = self._dfe.with_taps(taps)

    def respond(self, symbols: npt.NDArray[np.float64]) -> Received:
        r = self._channel.feed(self._tx.feed(symbols))
        y, decisions = self._decide(r)
        return Received(r=y, decisions=decisions, d=self._symbols.feed(symbols))

    def _decide(
        self, r: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Subtract the DFE feedback, then slice at 0 -- one UI at a time,
        since each decision feeds the next.
        """
        taps = self._dfe.taps
        y = np.empty_like(r)
        decisions = np.empty_like(r)
        for n in range(len(r)):
            y[n] = r[n] - sum(t * past for t, past in zip(taps, self._past))
            decisions[n] = 1.0 if y[n] > 0 else -1.0
            if self._past:
                self._past = [float(decisions[n])] + self._past[:-1]
        return y, decisions
