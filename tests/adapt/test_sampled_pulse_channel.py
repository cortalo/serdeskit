import numpy as np
import pytest

from serdeskit.adapt import SampledPulseChannel
from serdeskit.channel import PassThroughChannel


def test_n_pre_must_index_into_cursors() -> None:
    with pytest.raises(ValueError):
        SampledPulseChannel(cursors=np.array([0.5, 0.25]), n_pre=2)


def test_cursors_are_read_only() -> None:
    channel = SampledPulseChannel(cursors=np.array([0.5, 0.25]), n_pre=0)

    with pytest.raises(ValueError):
        channel.cursors[0] = 1.0


def test_from_waveform_samples_the_pulse_once_per_ui_at_its_peak() -> None:
    """Through a pass-through channel the pulse stays a 1 UI rectangle: the
    main cursor is 1 and every other UI-spaced sample is 0.
    """
    channel = SampledPulseChannel.from_waveform(
        PassThroughChannel(), symbol_rate=10e9, samples_per_ui=8, n_pre=2, n_post=3
    )

    assert channel.n_pre == 2
    np.testing.assert_allclose(channel.cursors, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
