import numpy as np
import pytest

from serdeskit.link import SystemGrid


def test_build_matches_pychopmarg_formula() -> None:
    """PyChOpMarg doesn't expose t/f/x_sinc construction as a standalone
    function — it's inline in COM.__init__, which requires a fully
    configured COM instance (real channel files, a complete COMParams) to
    exercise at all. Replicated here directly from that source instead
    (com.py's __init__: fb/fstep -> tmax/ui/tstep -> t -> fmax -> f ->
    Xsinc) as the golden comparison.
    """
    baud_rate = 100e9
    freq_step = 50e6
    samples_per_ui = 32

    grid = SystemGrid.build(baud_rate, freq_step, samples_per_ui)

    ui = 1 / baud_rate
    tstep = ui / samples_per_ui
    tmax = 1 / freq_step
    expected_t = np.arange(0, tmax, tstep)
    fmax = 0.5 / expected_t[1]
    expected_f = np.arange(0, fmax + freq_step, freq_step)
    expected_x_sinc = int(ui / expected_t[1]) * np.sinc(ui * expected_f)

    np.testing.assert_allclose(grid.t, expected_t)
    np.testing.assert_allclose(grid.f, expected_f)
    np.testing.assert_allclose(grid.x_sinc, expected_x_sinc)


def test_x_sinc_dc_equals_samples_per_ui() -> None:
    """At f=0, sinc(0)=1, so x_sinc[0] = int(ui/tstep) — and since
    tstep = ui/samples_per_ui exactly, that's just samples_per_ui.
    Independent sanity check of the Xsinc scaling factor, not derived from
    the same formula replication as the test above.
    """
    grid = SystemGrid.build(baud_rate=100e9, freq_step=50e6, samples_per_ui=32)

    assert grid.f[0] == 0.0
    assert grid.x_sinc[0] == pytest.approx(32.0)
