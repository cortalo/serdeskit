import numpy as np
import numpy.typing as npt

from serdeskit.adapt import SampledPulseChannel, SymbolRateLink, TxFfe

A = np.array([0.05, 0.5, 0.25, 0.1])  # one pre-cursor
TAPS = np.array([-0.05, 0.7, -0.2, 0.05])  # one pre-tap -> latency 1 + 1


def _link(tap_weights: npt.NDArray[np.float64] = TAPS) -> SymbolRateLink:
    return SymbolRateLink(
        channel=SampledPulseChannel(cursors=A, n_pre=1),
        ffe=TxFfe(weights=tap_weights, main=1),
    )


def test_isolated_bit_gives_equalized_pulse_response_around_its_own_sample() -> None:
    """p = w * a, with p_0 on the sample that the link reports as d = 1."""
    link = _link()
    symbols = np.zeros(16)
    symbols[8] = 1.0

    rx = link.respond(symbols)

    i0 = int(np.flatnonzero(rx.d == 1.0)[0])
    p = np.convolve(link.ffe.weights, A)  # p_0 at index 2
    np.testing.assert_allclose(rx.r[i0 - 2 : i0 - 2 + len(p)], p, atol=1e-15)


def test_d_is_the_input_delayed_by_the_latency() -> None:
    symbols = np.arange(1.0, 11.0)

    rx = _link().respond(symbols)

    np.testing.assert_array_equal(rx.d, np.concatenate([[0.0, 0.0], symbols[:-2]]))


def test_chunked_stream_matches_one_shot() -> None:
    rng = np.random.default_rng(0)
    symbols = rng.choice(np.array([-1.0, 1.0]), size=200)

    whole = _link().respond(symbols)
    chunked = _link()
    parts = [chunked.respond(c) for c in np.split(symbols, [37, 38, 128])]

    np.testing.assert_allclose(np.concatenate([p.r for p in parts]), whole.r, atol=1e-15)
    np.testing.assert_array_equal(np.concatenate([p.d for p in parts]), whole.d)


def test_tap_change_only_affects_what_is_transmitted_afterwards() -> None:
    """ISI already in the channel was launched with the old taps."""
    rng = np.random.default_rng(1)
    symbols = rng.choice(np.array([-1.0, 1.0]), size=60)
    new_taps = np.array([-0.1, 0.58, -0.3, 0.02])

    link = _link()
    first = link.respond(symbols[:30])
    link.set_tap_weights(new_taps)
    second = link.respond(symbols[30:])

    # u[t] = sum_j w_j d[t-j] starts at t = -1 (the first symbol's pre-tap)
    # and needs d[t+1], so the first call sends u[-1..28] with the old taps,
    # the rest with the new.
    padded = np.concatenate([np.zeros(3), symbols, [0.0]])  # idle line before the stream
    u_old = np.convolve(padded, TAPS, "valid")  # u[-1..59]
    u_new = np.convolve(padded, new_taps, "valid")
    u = np.concatenate([u_old[:30], u_new[30:]])

    # r[n] = sum_m a_m u[n-m]; with latency 2 the link reports r[-2..57].
    expected = np.convolve(u, A)[:60]
    np.testing.assert_allclose(np.concatenate([first.r, second.r]), expected, atol=1e-12)


def test_tap_view_and_update() -> None:
    link = _link()

    link.set_tap_weights(np.array([-0.1, 0.58, -0.3, 0.02]))

    np.testing.assert_array_equal(link.tap_weights, [-0.1, 0.58, -0.3, 0.02])
    assert link.main_tap == 1
