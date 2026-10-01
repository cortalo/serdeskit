import numpy as np
import pytest

from serdeskit.adapt import SampledPulseChannel


def test_n_pre_must_index_into_cursors() -> None:
    with pytest.raises(ValueError):
        SampledPulseChannel(cursors=np.array([0.5, 0.25]), n_pre=2)


def test_cursors_are_read_only() -> None:
    channel = SampledPulseChannel(cursors=np.array([0.5, 0.25]), n_pre=0)

    with pytest.raises(ValueError):
        channel.cursors[0] = 1.0
