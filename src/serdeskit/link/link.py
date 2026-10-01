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
    response for COM-style pulse-response generation (Link.
    sbr_pulse_response). Satisfied implicitly — a concrete channel class
    needs no relation to this Protocol beyond having matching methods.
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


class Dfe(Protocol):
    """What Link.simulate() needs from an RX DFE: its taps, volts --
    taps[k - 1] times the decision k UI earlier is subtracted.
    """

    @property
    def taps(self) -> npt.NDArray[np.float64]: ...


class TxFilter(Protocol):
    """What Link needs from a Tx-side filter — frequency-domain only, no
    `process`. Concrete example: `tx_filter.TxRisetimeFilter`, the
    transmitter output driver's finite risetime. Consumed by
    `sbr_pulse_response()`'s `h` composition, matching MATLAB COM3.70's
    own real behavior (`s21_pkg_tester`'s `H_t`, gated by `OP.FORCE_TR`/
    `T_r_filter_type` — confirmed against MATLAB, `tests/tx_filter/
    test_risetime_vs_matlab.py`, docs/known-issues.md's "full composed
    pulse response" entry) — this project previously assumed,
    incorrectly, that a Tx risetime filter "has no role in the pulse
    response".
    """

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...


class RxAfe(Protocol):
    """What Link needs from an Rx analog front-end — same situation as
    TxFilter, frequency-domain only.
    """

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...


class RxFfe(Protocol):
    """What Link needs from an Rx FFE. `process()` is consumed by
    `sbr_pulse_response()` (MATLAB's own real Rx FFE application,
    `force()`, confirmed by reading `Apply_EQ`'s call site — see that
    method's own docstring); `transfer_function()` by com.compute()'s own
    receiver-noise integral (93A-35), CTLE * Rx AFE * Rx FFE evaluated
    on the frequency grid.
    """

    def process(self, sig: Signal) -> Signal: ...
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


def uneq_truncated_impulse_response(
    channel: Channel, tx_filter: TxFilter, rx_afe: RxAfe, grid: SystemGrid
) -> Signal:
    h = channel.transfer_function(grid.f) * tx_filter.transfer_function(grid.f) * rx_afe.transfer_function(grid.f)
    return grid.truncated_impulse_response(h)


@dataclass
class Link:
    """channel is the stage the bit-domain `simulate()` pipeline runs the
    launched signal through, after the TX `ffe` if one is set and before
    the RX `dfe` if one is set.
    `ctle`/`tx_filter`/`rx_afe`/`rx_ffe` are only used by
    `sbr_pulse_response()` — optional since `simulate()` doesn't need
    them; left unset and then used there raises a plain
    AttributeError, which is fine (a caller building a pulse response
    without equalization is a caller error, not a case worth a
    defensive check).

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
    dfe: Dfe | None = None

    def simulate(self, bits: npt.NDArray[np.float64], fs: float, symbol_rate: float) -> LinkResult:
        """bits is one value per symbol; symbol_rate is what turns it into a
        proper analog-time waveform at fs before it ever reaches the channel.
        """
        samples = _upsample_bits(bits, fs, symbol_rate)
        sig = Signal(samples=samples, fs=fs, t0=0.0)
        if self.ffe is not None:
            sig = self.ffe.process(sig)
        sig = self.channel.process(sig)
        if self.dfe is not None and len(self.dfe.taps):
            sig = self._apply_dfe(sig, len(bits), symbol_rate)

        eye = _extract_eye(sig, symbol_rate)
        return LinkResult(eye=eye)

    def _apply_dfe(self, sig: Signal, n_symbols: int, symbol_rate: float) -> Signal:
        """Decide each symbol at its main cursor's sampling instant (the
        isolated pulse's peak), subtracting the DFE feedback from past
        decisions; that feedback is held over the 1 UI centered on the
        sampling instant, so the eye around it shows the cancelled ISI.
        """
        taps = self.dfe.taps  # type: ignore[union-attr]
        samples_per_ui = round(sig.fs / symbol_rate)
        delay = self._cursor_delay(sig.fs, symbol_rate)

        out = sig.samples.copy()
        past = [0.0] * len(taps)  # decisions, most recent first
        for n in range(n_symbols):
            center = (n / symbol_rate + delay - sig.t0) * sig.fs
            i = int(np.floor(center + 0.5))
            if not 0 <= i < len(out):
                continue
            feedback = sum(t * d for t, d in zip(taps, past))
            decision = 1.0 if sig.samples[i] - feedback > 0 else -1.0
            start = int(np.ceil(center - samples_per_ui / 2))  # window centered on `center`
            out[max(start, 0) : max(start + samples_per_ui, 0)] -= feedback
            past = [decision] + past[:-1]
        return Signal(samples=out, fs=sig.fs, t0=sig.t0)

    def _cursor_delay(self, fs: float, symbol_rate: float) -> float:
        """Seconds from a symbol's start to its main cursor: the peak of one
        isolated pulse through the TX FFE and channel (the middle of the
        peak, if it is flat).
        """
        samples_per_ui = round(fs / symbol_rate)
        m = 200  # UI of idle line before the pulse, so the channel settles
        pulse = np.zeros(2 * m * samples_per_ui)
        pulse[m * samples_per_ui : (m + 1) * samples_per_ui] = 1.0
        sig = Signal(samples=pulse, fs=fs, t0=0.0)
        if self.ffe is not None:
            sig = self.ffe.process(sig)
        y = self.channel.process(sig)
        peak = np.flatnonzero(y.samples >= y.samples.max() * (1 - 1e-9)).mean()
        return float(y.t0 + peak / fs - m / symbol_rate)

    def sbr_pulse_response(self, grid: SystemGrid) -> Signal:
        impulse = uneq_truncated_impulse_response(
            self.channel, self.tx_filter, self.rx_afe, grid  # type: ignore[arg-type]
        )
        ctle_impulse = self.ctle.process(impulse)  # type: ignore[union-attr]
        pulse = grid.box_car_integrate(ctle_impulse)
        eq_pulse = self.ffe.process(pulse)  # type: ignore[union-attr]
        return self.rx_ffe.process(eq_pulse)  # type: ignore[union-attr]

    def unequalized_impulse_response(self, grid: SystemGrid, victim_amplitude: float) -> Signal:
        impulse = uneq_truncated_impulse_response(
            self.channel, self.tx_filter, self.rx_afe, grid  # type: ignore[arg-type]
        )
        return impulse.scale(victim_amplitude)



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
    Windows are aligned so the eye center lands at 0.5 UI, giving two full,
    symmetric eyes at 0.5 and 1.5 UI whatever the channel's delay; a signal
    that never crosses zero keeps windows starting at sample 0.
    """
    samples_per_ui = round(sig.fs / symbol_rate)
    window = 2 * samples_per_ui
    center = _eye_center_phase(sig.samples, samples_per_ui)
    offset = 0 if center is None else round(center - samples_per_ui / 2) % samples_per_ui
    n_traces = (len(sig.samples) - offset - window) // samples_per_ui + 1

    traces = np.empty((window, n_traces), dtype=np.float64)
    for i in range(n_traces):
        start = offset + i * samples_per_ui
        traces[:, i] = sig.samples[start : start + window]

    return EyeData(traces=traces, ui=1.0 / symbol_rate, fs=sig.fs)


def _eye_center_phase(samples: npt.NDArray[np.float64], samples_per_ui: int) -> float | None:
    """Eye center's position within a UI, in samples: the middle of the
    longest (circular) stretch of UI phases where zero crossings are rarest.
    None if the signal never crosses zero.

    Not "mean crossing + 0.5 UI": strong equalization splits crossings into
    several clusters by data pattern, and their mean can fall between them
    rather than opposite the opening.
    """
    after = np.flatnonzero(np.signbit(samples[1:]) != np.signbit(samples[:-1])) + 1
    if len(after) == 0:
        return None
    hist = np.bincount(after % samples_per_ui, minlength=samples_per_ui).astype(np.float64)
    width = max(1, samples_per_ui // 8)  # smooth over noise in sparse histograms
    smooth = np.convolve(np.tile(hist, 3), np.ones(width) / width, "same")[
        samples_per_ui : 2 * samples_per_ui
    ]
    rare = np.isclose(smooth, smooth.min())

    # Longest circular run of `rare` phases, and its midpoint.
    best_len, best_mid = 0, 0.0
    for begin in range(samples_per_ui):
        if not rare[begin] or rare[begin - 1] and not rare.all():
            continue
        length = 0
        while length < samples_per_ui and rare[(begin + length) % samples_per_ui]:
            length += 1
        if length > best_len:
            best_len, best_mid = length, begin + (length - 1) / 2
    return best_mid % samples_per_ui
