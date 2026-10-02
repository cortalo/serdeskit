import numpy as np
import numpy.typing as npt
import pytest

from serdeskit.cdr import MuellerMuller, PulseChannel, SamplingPhaseLink

# Rises over 1 UI, decays over 3: h_-1 = phase, h_1 = 1 - (1 + phase)/3 for
# 0 <= phase <= 1, so h_1 = h_-1 at phase 0.5.
_T = np.arange(-24, 49) / 8
_PULSE = np.where(_T < 0, np.maximum(0.0, 1.0 + _T), np.maximum(0.0, 1.0 - _T / 3))
LOCK = 0.5


def _link(phase: float) -> SamplingPhaseLink:
    channel = PulseChannel(samples=_PULSE.copy(), samples_per_ui=8, peak=24)
    return SamplingPhaseLink(channel, n_pre=2, n_post=4, phase=phase)


def _random_symbols(shape: tuple[int, int], seed: int) -> npt.NDArray[np.float64]:
    return np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), size=shape)


def test_one_step_moves_the_phase_by_gain_times_h1_minus_h_minus1() -> None:
    gain = 1e-9
    phases = MuellerMuller(gain=gain).run(_link(0.2), _random_symbols((1, 200_000), seed=0))

    h_m1, h_1 = 0.2, 1.0 - 1.2 / 3
    assert (phases[1] - phases[0]) / gain == pytest.approx(h_1 - h_m1, abs=0.02)


@pytest.mark.parametrize("start", [0.0, 0.9])
def test_locks_where_h1_equals_h_minus1_from_either_side(start: float) -> None:
    phases = MuellerMuller(gain=0.2).run(_link(start), _random_symbols((300, 500), seed=1))

    assert phases[-100:].mean() == pytest.approx(LOCK, abs=0.015)
