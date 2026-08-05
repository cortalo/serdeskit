"""Link: the top-level object. Currently holds a single Channel and runs it
end to end, producing a LinkResult — plain data, not a plot. Rendering (eye
diagrams as matplotlib figures) is deliberately a separate, outer layer:
Link never imports matplotlib, so its output can be asserted on directly in
tests.
"""
from __future__ import annotations

import functools
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

    `__hash__` is part of the contract (not just inherited from `object`
    implicitly) because `_uneq_truncated_impulse_response` below caches
    on channel identity — a concrete Channel is expected to be frozen/
    immutable (see SParameterChannel) so that identity-based caching can
    never return a stale result.
    """

    def process(self, sig: Signal) -> Signal: ...
    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...
    def __hash__(self) -> int: ...


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
    `sbr_pulse_response()`'s `h` composition, matching MATLAB COM3.70's
    own real behavior (`s21_pkg_tester`'s `H_t`, gated by `OP.FORCE_TR`/
    `T_r_filter_type` — confirmed against MATLAB, `tests/tx_filter/
    test_risetime_vs_matlab.py`, docs/known-issues.md's "full composed
    pulse response" entry) — this project previously assumed,
    incorrectly, that a Tx risetime filter "has no role in the pulse
    response". `__hash__` is part of the contract for the same reason as
    Channel's own — see there.
    """

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...
    def __hash__(self) -> int: ...


class RxAfe(Protocol):
    """What Link needs from an Rx analog front-end — same situation as
    TxFilter, frequency-domain only, `__hash__` included.
    """

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]: ...
    def __hash__(self) -> int: ...


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


@functools.lru_cache(maxsize=128)
def _uneq_truncated_impulse_response(
    channel: Channel, tx_filter: TxFilter, rx_afe: RxAfe, grid: SystemGrid
) -> Signal:
    """`channel + tx_filter + rx_afe`, truncated impulse response --
    `sbr_pulse_response()`'s own first step, factored out and cached: it
    doesn't depend on ctle/ffe/rx_ffe at all, so a grid search sweeping
    those (src/serdeskit/optimize/search.py) would otherwise redo this
    same IFFT+truncate on every one of its (CTLE gain x Tx tap) grid
    points, for every channel (victim and every NEXT/FEXT aggressor)
    that doesn't actually change between them. Safe to cache across Link
    instances: keyed on the stage objects themselves (identity, not
    value), all four of which are frozen/immutable (Channel
    implementations, TxFilter, RxAfe, SystemGrid) so a cache hit can
    never return a stale result.
    """
    h = channel.transfer_function(grid.f) * tx_filter.transfer_function(grid.f) * rx_afe.transfer_function(grid.f)
    return grid.truncated_impulse_response(h)


@dataclass
class Link:
    """channel is the stage the bit-domain `simulate()` pipeline runs the
    launched signal through. `ctle`/`ffe`/`tx_filter`/`rx_afe`/`rx_ffe`
    are only used by `sbr_pulse_response()` — optional since `simulate()`
    doesn't need them; left unset and then used there raises a plain
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

    def simulate(self, bits: npt.NDArray[np.float64], fs: float, symbol_rate: float) -> LinkResult:
        """bits is one value per symbol; symbol_rate is what turns it into a
        proper analog-time waveform at fs before it ever reaches the channel.
        """
        samples = _upsample_bits(bits, fs, symbol_rate)
        sig = Signal(samples=samples, fs=fs, t0=0.0)
        sig = self.channel.process(sig)

        eye = _extract_eye(sig, symbol_rate)
        return LinkResult(eye=eye)

    def sbr_pulse_response(self, grid: SystemGrid) -> Signal:
        """MATLAB COM3.70's real computation, end to end — named after its
        own `sbr`/`eq_pulse_response` (`Apply_EQ`, `com_ieee8023_93a_370.m:
        680-735`, "returns pulse response with CTLE, TXLE, RXFFE"). Two
        separate steps rather than one call, since CTLE's own time-domain
        path (`Ctle.process()`) needs the impulse response, before
        `box_car_integrate()` turns it into a pulse response:
        `channel + tx_filter + rx_afe` (truncated impulse response) ->
        `ctle.process()` (time-domain IIR, twice, per MATLAB's own CL120d
        two-stage structure) -> box-car integration -> `ffe.process()`
        (Tx, time-domain circular-shift-sum) -> `rx_ffe.process()` (same
        shape, Rx-side taps). Requires `self.ctle`/`self.ffe`/`self.
        tx_filter`/`self.rx_afe`/`self.rx_ffe` to be set.
        """
        # No `# type: ignore[union-attr]` needed here (unlike the None-
        # unsafe accesses below): @functools.lru_cache's own typeshed
        # stub types every argument as plain Hashable, not this
        # function's real (Channel, TxFilter, RxAfe, SystemGrid)
        # signature -- so mypy can't see the Optional-ness here to flag
        # it either. Still a genuine caller error at runtime if unset
        # (AttributeError on None), same as the others.
        impulse = _uneq_truncated_impulse_response(self.channel, self.tx_filter, self.rx_afe, grid)
        ctle_impulse = self.ctle.process(impulse)  # type: ignore[union-attr]
        pulse = grid.box_car_integrate(ctle_impulse)
        eq_pulse = self.ffe.process(pulse)  # type: ignore[union-attr]
        return self.rx_ffe.process(eq_pulse)  # type: ignore[union-attr]


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
