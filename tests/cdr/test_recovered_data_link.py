import numpy as np

from serdeskit.cdr import PulseChannel, RecoveredDataLink, SamplingPhaseLink
from serdeskit.prbs import Prbs, PrbsSync


def test_reports_recovered_data_once_locked() -> None:
    """Clean channel: decisions are right, so the sync locks on its first
    seed and the recovered d then equals the transmitted d.
    """
    t = np.arange(-20, 21) / 4
    pulse = PulseChannel(samples=np.maximum(0.0, 1.0 - np.abs(t)), samples_per_ui=4, peak=20)
    inner = SamplingPhaseLink(pulse, n_pre=2, n_post=3)
    reference = SamplingPhaseLink(pulse, n_pre=2, n_post=3)
    link = RecoveredDataLink(inner, PrbsSync(Prbs(7), verify_bits=64))
    symbols = Prbs(7).symbols(400)

    rx = link.respond(symbols)
    truth = reference.respond(symbols)

    np.testing.assert_array_equal(rx.r, truth.r)
    locked = link.sync.locked_at
    assert locked is not None
    np.testing.assert_array_equal(rx.d[: locked + 1], 0.0)
    np.testing.assert_array_equal(rx.d[locked + 1 :], truth.d[locked + 1 :])


def test_phase_passes_through() -> None:
    t = np.arange(-20, 21) / 4
    pulse = PulseChannel(samples=np.maximum(0.0, 1.0 - np.abs(t)), samples_per_ui=4, peak=20)
    link = RecoveredDataLink(SamplingPhaseLink(pulse, 2, 3), PrbsSync(Prbs(7)))

    link.set_sampling_phase(0.3)

    assert link.sampling_phase == 0.3
