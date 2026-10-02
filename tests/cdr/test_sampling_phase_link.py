import numpy as np
import numpy.typing as npt

from serdeskit.cdr import PulseChannel, SamplingPhaseLink


def _triangle() -> PulseChannel:
    """p(t) = max(0, 1 - |t|), t in UI: cursors are linear in the phase."""
    t = np.arange(-20, 21) / 4
    return PulseChannel(samples=np.maximum(0.0, 1.0 - np.abs(t)), samples_per_ui=4, peak=20)


def _link(phase: float = 0.0) -> SamplingPhaseLink:
    return SamplingPhaseLink(_triangle(), n_pre=2, n_post=3, phase=phase)


def _random_symbols(n: int, seed: int) -> npt.NDArray[np.float64]:
    return np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), size=n)


def test_isolated_bit_gives_the_cursors_around_its_own_sample() -> None:
    link = _link(phase=0.25)
    symbols = np.zeros(16)
    symbols[8] = 1.0

    rx = link.respond(symbols)

    i0 = int(np.flatnonzero(rx.d == 1.0)[0])
    np.testing.assert_allclose(rx.r[i0 - 2 : i0 + 4], _triangle().cursors(0.25, 2, 3))


def test_d_is_the_input_delayed_by_n_pre() -> None:
    symbols = np.arange(1.0, 11.0)

    rx = _link().respond(symbols)

    np.testing.assert_array_equal(rx.d, np.concatenate([[0.0, 0.0], symbols[:-2]]))


def test_chunked_stream_matches_one_shot() -> None:
    symbols = _random_symbols(200, seed=0)

    whole = _link(phase=0.3).respond(symbols)
    chunked = _link(phase=0.3)
    parts = [chunked.respond(c) for c in np.split(symbols, [37, 38, 128])]

    np.testing.assert_allclose(np.concatenate([p.r for p in parts]), whole.r, atol=1e-15)
    np.testing.assert_array_equal(np.concatenate([p.d for p in parts]), whole.d)


def test_phase_change_only_affects_samples_taken_afterwards() -> None:
    """The waveform in flight is the same; after the change it is just read
    at the new phase.
    """
    symbols = _random_symbols(60, seed=1)

    link = _link(phase=0.0)
    first = link.respond(symbols[:30])
    link.set_sampling_phase(-0.4)
    second = link.respond(symbols[30:])

    assert link.sampling_phase == -0.4
    np.testing.assert_allclose(first.r, _link(phase=0.0).respond(symbols).r[:30], atol=1e-15)
    np.testing.assert_allclose(second.r, _link(phase=-0.4).respond(symbols).r[30:], atol=1e-15)


def test_decisions_slice_at_zero() -> None:
    rx = _link(phase=0.1).respond(_random_symbols(100, seed=2))

    np.testing.assert_array_equal(rx.decisions, np.where(rx.r > 0, 1.0, -1.0))
