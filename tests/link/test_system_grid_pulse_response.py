from types import SimpleNamespace
from typing import cast

import numpy as np
from pychopmarg.com import COM

from serdeskit.ffe import TapWeightFfe
from serdeskit.link import SystemGrid


def test_matches_pychopmarg_golden_reference() -> None:
    """(93A-24). PyChOpMarg's `pulse_resp()` only ever touches
    `self.Xsinc`, `self.freqs`, `self.times` — bypass constructing a full
    COM/COMParams instance (which needs real channel files) with a
    lightweight stand-in object carrying just those three attributes, and
    call the unbound method directly.
    """
    grid = SystemGrid.build(baud_rate=100e9, freq_step=50e6, samples_per_ui=32)
    ffe = TapWeightFfe(tap_weights=np.array([0.05, -0.1, 0.03]), n_post=2, tap_delay=1e-11)
    h = ffe.transfer_function(grid.f)

    mock = cast(COM, SimpleNamespace(Xsinc=grid.x_sinc, freqs=grid.f, times=grid.t))
    expected_samples = COM.pulse_resp(mock, h)

    sig = grid.pulse_response(h)

    np.testing.assert_allclose(sig.samples, expected_samples)
    assert sig.fs == 1.0 / (grid.t[1] - grid.t[0])
    assert sig.t0 == 0.0
