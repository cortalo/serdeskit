import numpy as np

from serdeskit.channel import PassThroughChannel
from serdeskit.common.types import Signal


def test_process_returns_signal_unchanged() -> None:
    sig = Signal(samples=np.array([1.0, -1.0, 1.0]), fs=100e9, t0=0.0)
    out = PassThroughChannel().process(sig)
    assert out is sig
