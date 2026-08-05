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
from serdeskit.link.system_grid import SystemGrid


class Channel(Protocol):
    """What Link needs from a channel: turn the signal launched into it into
    the signal that arrives at the receiver, and/or expose its frequency
    response for COM-style pulse-response generation (SystemGrid.
    pulse_response). Satisfied implicitly — a concrete channel class needs
    no relation to this Protocol beyond having matching methods.
    """

    def process(self, sig: Signal) -> Signal: ...
    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...


class Ctle(Protocol):
    """What Link needs from a CTLE. Same shape as Channel, but kept as its
    own Protocol rather than a shared generic one — see CLAUDE.md's
    architecture note on why.
    """

    def process(self, sig: Signal) -> Signal: ...
    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...


class Ffe(Protocol):
    """What Link needs from an FFE."""

    def process(self, sig: Signal) -> Signal: ...
    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...


class TxFilter(Protocol):
    """What Link needs from a Tx-side filter — frequency-domain only, no
    `process`. Concrete example: `tx_filter.TxRisetimeFilter`, the
    transmitter output driver's finite risetime. Consumed by
    `ffe_channel_ctle_pulse_response()`'s `h` composition, matching
    MATLAB COM3.70's own real behavior (`s21_pkg_tester`'s `H_t`, gated
    by `OP.FORCE_TR`/`T_r_filter_type` — confirmed against MATLAB,
    `tests/tx_filter/test_risetime_vs_matlab.py`, docs/known-issues.md's
    "full composed pulse response" entry) — this project previously
    assumed, incorrectly, that a Tx risetime filter "has no role in the
    pulse response".
    """

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...


class RxAfe(Protocol):
    """What Link needs from an Rx analog front-end — same situation as
    TxFilter, frequency-domain only.
    """

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...


class RxFfe(Protocol):
    """What Link needs from an Rx FFE — same shape as RxAfe, but its own
    Protocol rather than reusing it: RxAfe is a fixed, always-present
    stage, while an Rx FFE may genuinely be absent (see Link.rx_ffe).
    """

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...


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
    """channel is the stage the bit-domain `simulate()` pipeline runs the
    launched signal through. `ctle`/`ffe`/`tx_filter`/`rx_afe`/`rx_ffe`
    are only used by `ffe_channel_ctle_pulse_response()`'s frequency-
    domain composition — optional since `simulate()` doesn't need them;
    left unset and then used there raises a plain AttributeError, which
    is fine (a caller building a pulse response without equalization is
    a caller error, not a case worth a defensive check).

    A link with no real Rx FFE still needs an `rx_ffe` set: a single
    unity tap (`RxFfe` with one tap of weight 1.0) is the identity —
    PyChOpMarg's own H() skips its Hrx factor entirely rather than
    treating "none" as a trivial one-tap case, but the two are
    equivalent, and this way every equalization stage here follows the
    same "set it, even trivially" rule instead of rx_ffe alone needing a
    conditional.
    """

    channel: Channel
    ctle: Ctle | None = None
    ffe: Ffe | None = None
    tx_filter: TxFilter | None = None
    rx_afe: RxAfe | None = None
    rx_ffe: RxFfe | None = None

    def simulate(self, bits: npt.NDArray[np.float64], fs: float, symbol_rate: float) -> LinkResult:
        """bits is one value per symbol; symbol_rate is what turns it into a
        proper analog-time waveform at fs before it ever reaches the channel.
        """
        samples = _upsample_bits(bits, fs, symbol_rate)
        sig = Signal(samples=samples, fs=fs, t0=0.0)
        sig = self.channel.process(sig)

        eye = _extract_eye(sig, symbol_rate)
        return LinkResult(eye=eye)

    def ffe_channel_ctle_pulse_response(self, grid: SystemGrid) -> Signal:
        """(93A-19)/(93A-24) pulse response: composes channel's, tx_filter's,
        ctle's, ffe's, rx_afe's, and rx_ffe's transfer functions on a
        shared SystemGrid and inverse-transforms the result into a
        Signal. Requires `self.ctle`/`self.ffe`/`self.tx_filter`/
        `self.rx_afe`/`self.rx_ffe` to be set (see the class docstring
        for what happens if not).

        `tx_filter` (the Tx driver's finite risetime) is included here
        despite this project previously assuming, per its reading of
        (93A-19), that it "has no role in the pulse response — it only
        shapes the transmitter noise PSD". That assumption was wrong:
        MATLAB COM3.70's own real behavior bakes an equivalent factor
        (`s21_pkg_tester`'s `H_t`) into the same `chdata(i).sdd21` this
        method's `h` is meant to match, whenever `OP.FORCE_TR` is set —
        confirmed against MATLAB, see docs/known-issues.md's "full
        composed pulse response" entry and `tests/tx_filter/
        test_risetime_vs_matlab.py`.

        Not yet cursor-located — pass the result to
        `PulseResponse.from_signal(...)` for that; kept as a separate
        step here rather than folded in, so this package doesn't need to
        depend on `serdeskit.pulse_response`.

        Args:
            grid: The time/frequency axes to compute on. Taken rather
                than derived here so a caller needing the same axes for
                something else — the (93A-35) noise integral, say — uses
                one grid instead of deriving a second that has to agree.

        Returns:
            The link's pulse response, as a Signal.
        """
        h = (
            self.channel.transfer_function(grid.f)
            * self.tx_filter.transfer_function(grid.f)  # type: ignore[union-attr]
            * self.ctle.transfer_function(grid.f)  # type: ignore[union-attr]
            * self.ffe.transfer_function(grid.f)  # type: ignore[union-attr]
            * self.rx_afe.transfer_function(grid.f)  # type: ignore[union-attr]
            * self.rx_ffe.transfer_function(grid.f)  # type: ignore[union-attr]
        )

        return grid.pulse_response(h)


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
