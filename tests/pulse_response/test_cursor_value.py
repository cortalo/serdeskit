import numpy as np
import pytest

from serdeskit.pulse_response import PulseResponse


def test_cursor_value_reads_the_sample_at_cursor_index() -> None:
    samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    pr = PulseResponse(samples=samples, fs=1.0, cursor_time=4.0, ui=1.0)

    assert pr.cursor_index == 4
    assert pr.cursor_value == pytest.approx(4.0)


def test_signal_amplitude_and_residual_isi_agree_with_cursor_value() -> None:
    """signal_amplitude() and residual_isi() both derive from the cursor
    sample internally — this pins down that they're reading the same value
    cursor_value exposes, not something computed independently that could
    drift from it.
    """
    samples = np.arange(10, dtype=np.float64)
    pr = PulseResponse(samples=samples, fs=1.0, cursor_time=4.0, ui=1.0)

    assert pr.cursor_value == pytest.approx(4.0)
    assert pr.signal_amplitude(rlm=1.0, levels=2) == pytest.approx(pr.cursor_value)

    residual = pr.residual_isi(dfe_min=np.array([-2.0]), dfe_max=np.array([2.0]))
    assert residual[5] == pytest.approx(0.0)  # fully cancelled: needs 5/4, well inside limits
