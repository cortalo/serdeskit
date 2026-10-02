import numpy as np
import pytest

from serdeskit.cdr import PulseChannel, SamplingPhaseLink, SignSignMuellerMuller

# Rises over 1 UI, decays over 3: h_1 = h_-1 at phase 0.5, where h_0 = 5/6.
_T = np.arange(-24, 49) / 8
_PULSE = np.where(_T < 0, np.maximum(0.0, 1.0 + _T), np.maximum(0.0, 1.0 - _T / 3))
LOCK = 0.5
H0_AT_LOCK = 5 / 6


def _link(phase: float) -> SamplingPhaseLink:
    channel = PulseChannel(samples=_PULSE.copy(), samples_per_ui=8, peak=24)
    return SamplingPhaseLink(channel, n_pre=2, n_post=4, phase=phase)


@pytest.mark.parametrize("start", [0.0, 0.9])
def test_phase_and_dlev_lock_together_from_either_side(start: float) -> None:
    """dLev starts at 0, far from h_0; the phase still ends at h_1 = h_-1.
    The noise is large because this pulse has only three ISI terms: with
    dLev near h_0 their few discrete values otherwise leave the detector
    almost no gain around the lock point.
    """
    rng = np.random.default_rng(0)
    data = rng.choice(np.array([-1.0, 1.0]), size=(600, 300))

    trace = SignSignMuellerMuller(step_phase=0.01, step_dlev=0.005, noise_rms=0.2, rng=rng).run(
        _link(start), data
    )

    assert trace.phase[-200:].mean() == pytest.approx(LOCK, abs=0.02)
    assert trace.dlev[-200:].mean() == pytest.approx(H0_AT_LOCK, abs=0.03)


def test_trace_has_the_start_and_one_row_per_iteration() -> None:
    data = np.ones((5, 10))

    trace = SignSignMuellerMuller(step_phase=0.01, step_dlev=0.01).run(_link(0.3), data)

    assert trace.phase.shape == trace.dlev.shape == (6,)
    assert (trace.phase[0], trace.dlev[0]) == (0.3, 0.0)
