from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from serdeskit.adapt import RxDfe
from serdeskit.channel import PassThroughChannel
from serdeskit.common.types import Signal
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Link


def test_simulate_slices_eye_traces_1ui_step_2ui_window() -> None:
    link = Link(channel=PassThroughChannel())
    fs = 100e9
    symbol_rate = 10e9  # 10 samples/UI
    bits = np.array([0.0, 1.0, 2.0, 3.0])  # 4 symbols, one value each

    result = link.simulate(bits=bits, fs=fs, symbol_rate=symbol_rate)

    upsampled = np.repeat(bits, 10)  # zero-order-hold, 10 samples/UI

    assert result.eye.ui == 1.0 / symbol_rate
    assert result.eye.fs == fs
    assert result.eye.traces.shape == (20, 3)  # 2 UI window, 1 UI step -> 3 traces
    np.testing.assert_array_equal(result.eye.traces[:, 0], upsampled[0:20])
    np.testing.assert_array_equal(result.eye.traces[:, 1], upsampled[10:30])
    np.testing.assert_array_equal(result.eye.traces[:, 2], upsampled[20:40])


def test_simulate_applies_tx_ffe_before_the_channel() -> None:
    """A symbol-spaced TX FIR on an NRZ driver: each UI holds
    sum_j w_j d[n-j]. Here one post-tap of +0.25 (cursor 0.75) -- positive
    so the waveform has no zero crossing to align the eye to.
    """
    ffe = TapWeightFfe(tap_weights=np.array([0.25]), n_post=1, tap_delay=1e-10)
    link = Link(channel=PassThroughChannel(), ffe=ffe)
    bits = np.array([1.0, 0.0, 0.0, 0.0])

    result = link.simulate(bits=bits, fs=100e9, symbol_rate=10e9)

    expected = np.repeat([0.75, 0.25, 0.0, 0.0], 10)
    np.testing.assert_allclose(result.eye.traces[:, 0], expected[0:20])
    np.testing.assert_allclose(result.eye.traces[:, 1], expected[10:30])


@dataclass
class _DelayChannel:
    """Delays the signal by `samples`, holding its first value meanwhile."""

    samples: int

    def process(self, sig: Signal) -> Signal:
        pad = np.full(self.samples, sig.samples[0])
        return Signal(samples=np.concatenate([pad, sig.samples]), fs=sig.fs, t0=sig.t0)

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        raise NotImplementedError


def _crossing_rows(traces: npt.NDArray[np.float64]) -> set[int]:
    """Row indices (within the 2 UI window) where any trace changes sign."""
    flips = np.signbit(traces[1:]) != np.signbit(traces[:-1])
    return {int(i) + 1 for i in np.flatnonzero(flips.any(axis=1))}


def test_eye_windows_start_on_the_zero_crossings() -> None:
    """Crossings land at 0, 1 and 2 UI (only the middle one falls between
    two rows of a trace), so two full eyes sit at 0.5 and 1.5 UI --
    whatever the channel's delay.
    """
    rng = np.random.default_rng(0)
    bits = rng.choice(np.array([-1.0, 1.0]), size=200)

    for delay in [0, 3, 7]:
        link = Link(channel=_DelayChannel(samples=delay))
        traces = link.simulate(bits=bits, fs=100e9, symbol_rate=10e9).eye.traces

        assert _crossing_rows(traces) == {10}, f"delay={delay}"


@dataclass
class _EchoChannel:
    """Adds `gain` times the signal one UI (`delay` samples) later: a pure
    first post-cursor."""

    gain: float
    delay: int

    def process(self, sig: Signal) -> Signal:
        echo = np.concatenate([np.zeros(self.delay), sig.samples[: -self.delay]])
        return Signal(samples=sig.samples + self.gain * echo, fs=sig.fs, t0=sig.t0)

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        raise NotImplementedError


def test_dfe_cancels_the_post_cursor_over_each_sampled_ui() -> None:
    """Echo of 0.6 one UI later; a 1-tap DFE of 0.6 removes it everywhere in
    the UI around each sampling instant, leaving a clean +/-1 eye.
    """
    rng = np.random.default_rng(1)
    bits = rng.choice(np.array([-1.0, 1.0]), size=300)
    link = Link(channel=_EchoChannel(gain=0.6, delay=10), dfe=RxDfe(taps=np.array([0.6])))

    traces = link.simulate(bits=bits, fs=100e9, symbol_rate=10e9).eye.traces

    np.testing.assert_allclose(np.abs(traces[:, 1:]), 1.0, atol=1e-12)


def test_without_dfe_the_echo_stays_in_the_eye() -> None:
    rng = np.random.default_rng(1)
    bits = rng.choice(np.array([-1.0, 1.0]), size=300)

    traces = Link(channel=_EchoChannel(gain=0.6, delay=10)).simulate(
        bits=bits, fs=100e9, symbol_rate=10e9
    ).eye.traces

    assert np.isclose(np.abs(traces), 0.4).any()
