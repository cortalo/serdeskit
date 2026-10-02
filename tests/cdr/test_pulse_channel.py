import numpy as np
import pytest

from serdeskit.cdr import PulseChannel
from serdeskit.channel import PassThroughChannel


def _triangle() -> PulseChannel:
    """p(t) = max(0, 1 - |t|), t in UI, peak at index 12."""
    t = np.arange(-12, 13) / 4
    return PulseChannel(samples=np.maximum(0.0, 1.0 - np.abs(t)), samples_per_ui=4, peak=12)


def test_peak_must_index_into_samples() -> None:
    with pytest.raises(ValueError):
        PulseChannel(samples=np.array([0.5, 0.25]), samples_per_ui=1, peak=2)


def test_samples_are_read_only() -> None:
    with pytest.raises(ValueError):
        _triangle().samples[0] = 1.0


def test_cursors_read_the_pulse_at_phase_plus_k_ui() -> None:
    channel = _triangle()

    np.testing.assert_allclose(channel.cursors(0.0, 1, 1), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(channel.cursors(0.25, 1, 1), [0.25, 0.75, 0.0])
    np.testing.assert_allclose(channel.cursors(-0.125, 1, 1), [0.0, 0.875, 0.125])


def test_cursors_past_the_stored_pulse_raise() -> None:
    with pytest.raises(ValueError):
        _triangle().cursors(0.5, 1, 3)


def test_from_waveform_keeps_the_pulse_with_its_peak_at_phase_0() -> None:
    """Through a pass-through channel the pulse stays a 1 UI rectangle."""
    channel = PulseChannel.from_waveform(PassThroughChannel(), symbol_rate=10e9, samples_per_ui=8)

    np.testing.assert_allclose(channel.cursors(0.0, 2, 3), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
