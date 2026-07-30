import numpy as np

from serdeskit.channel import PassThroughChannel
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
