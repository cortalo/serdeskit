from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pytest

from serdeskit.adapt import (
    Received,
    RxDfe,
    SampledPulseChannel,
    SignSignLms,
    SymbolRateLink,
    TxFfe,
)

A = np.array([0.05, 0.5, 0.25, 0.1])  # one pre-cursor, two post-cursors
N_PRE_CH = 1
MAIN = 1  # taps: [pre1, main, post1, post2]
IDLE_TX = np.array([0.0, 1.0, 0.0, 0.0])
STEP = 1e-3


def _link(tap_weights: npt.NDArray[np.float64], n_dfe: int = 0) -> SymbolRateLink:
    return SymbolRateLink(
        channel=SampledPulseChannel(cursors=A, n_pre=N_PRE_CH),
        ffe=TxFfe(weights=tap_weights, main=MAIN),
        dfe=RxDfe(taps=np.zeros(n_dfe)),
    )


def _zero_forcing_taps(signs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Solve p_j = 0 over the FFE's span (j = -1, 1, 2) with the cursor tap
    tied to 1 - sum(|w|). With the tap signs fixed, p is affine in the
    taps, so a 3x3 linear solve is exact.
    """
    def p_at_span(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        w = np.array([x[0], 1.0 - signs @ x, x[1], x[2]])  # |x| -> signs * x
        p = np.convolve(w, A)  # p_0 at index 1 + 1
        return p[[1, 3, 4]]

    p0 = p_at_span(np.zeros(3))
    jac = np.column_stack([p_at_span(e) - p0 for e in np.eye(3)])
    return np.asarray(np.linalg.solve(jac, -p0), dtype=np.float64)


def test_converges_to_zero_forcing_with_dlev_tracking_the_cursor() -> None:
    rng = np.random.default_rng(1)
    data = rng.choice(np.array([-1.0, 1.0]), size=(3000, 512))

    lms = SignSignLms(step_tap=STEP, step_dlev=STEP, noise_rms=0.01, rng=np.random.default_rng(2))
    trace = lms.run(
        link=_link(IDLE_TX),
        data=data,
    )

    taps = trace.tap_weights[-500:].mean(axis=0)  # average out sign-sign dither
    dlev = trace.dlev[-500:].mean()
    p = np.convolve(taps, A)  # p_0 at index 2

    others = np.delete(taps, MAIN)
    np.testing.assert_allclose(others, _zero_forcing_taps(np.sign(others)), atol=3 * STEP)
    np.testing.assert_allclose(p[[1, 3, 4]], 0.0, atol=3 * STEP)
    assert dlev == pytest.approx(p[2], abs=3 * STEP)


@dataclass
class _ConstantLink:
    """A stand-in AdaptableLink: the receiver always sees `level`."""

    level: float
    tap_weights: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    main_tap: int = 1
    dfe_taps: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    def respond(self, symbols: npt.NDArray[np.float64]) -> Received:
        r = np.full(len(symbols), self.level)
        return Received(r=r, decisions=np.where(r > 0, 1.0, -1.0), d=symbols)

    def set_tap_weights(self, tap_weights: npt.NDArray[np.float64]) -> None:
        self.tap_weights = tap_weights

    def set_dfe_taps(self, taps: npt.NDArray[np.float64]) -> None:
        self.dfe_taps = taps


def test_main_tap_keeps_the_swing_constraint_every_iteration() -> None:
    rng = np.random.default_rng(4)
    trace = SignSignLms(step_tap=0.01, step_dlev=0.01).run(
        link=_link(IDLE_TX),
        data=rng.choice(np.array([-1.0, 1.0]), size=(50, 256)),
    )

    np.testing.assert_allclose(np.abs(trace.tap_weights).sum(axis=1), 1.0)
    assert not np.allclose(trace.tap_weights[-1], IDLE_TX)  # it did adapt


def test_dlev_ramps_one_step_per_iteration_toward_the_signal() -> None:
    """Signal always above dLev: sign(e) = +1 every time, so dLev climbs
    exactly step_dlev per iteration -- the slew-limited ramp of sign-sign.
    """
    rng = np.random.default_rng(3)
    trace = SignSignLms(step_tap=STEP, step_dlev=0.01).run(
        link=_ConstantLink(level=1.0),
        data=rng.choice(np.array([-1.0, 1.0]), size=(20, 256)),
    )

    np.testing.assert_allclose(trace.dlev, 0.01 * np.arange(21))


def _zero_forcing_taps_with_dfe(signs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """As _zero_forcing_taps, with post1 left to a DFE: TX taps [x0, cursor,
    0, x1], solving p_j = 0 for j = -1, 2.
    """
    def p_at_span(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        w = np.array([x[0], 1.0 - signs @ x, 0.0, x[1]])
        return np.convolve(w, A)[[1, 4]]

    p0 = p_at_span(np.zeros(2))
    jac = np.column_stack([p_at_span(e) - p0 for e in np.eye(2)])
    return np.asarray(np.linalg.solve(jac, -p0), dtype=np.float64)


def test_with_a_1_tap_dfe_the_tx_leaves_post1_to_it() -> None:
    """TX post1 stays 0; the DFE tap converges to the equalized first
    post-cursor and the TX zero-forces the rest of its span (j = -1, 2).
    """
    rng = np.random.default_rng(5)
    lms = SignSignLms(
        step_tap=STEP, step_dlev=STEP, step_dfe=STEP, noise_rms=0.01, rng=np.random.default_rng(6)
    )
    trace = lms.run(
        link=_link(IDLE_TX, n_dfe=1),
        data=rng.choice(np.array([-1.0, 1.0]), size=(3000, 512)),
    )

    taps = trace.tap_weights[-500:].mean(axis=0)
    p = np.convolve(taps, A)  # p_0 at index 2
    free = taps[[0, 3]]

    np.testing.assert_array_equal(trace.tap_weights[:, MAIN + 1], 0.0)
    np.testing.assert_allclose(free, _zero_forcing_taps_with_dfe(np.sign(free)), atol=3 * STEP)
    assert trace.dfe_taps[-500:, 0].mean() == pytest.approx(p[3], abs=3 * STEP)
    assert trace.dlev[-500:].mean() == pytest.approx(p[2], abs=3 * STEP)


def test_dfe_covered_tx_taps_are_forced_to_zero() -> None:
    rng = np.random.default_rng(7)
    trace = SignSignLms(step_tap=STEP, step_dlev=STEP).run(
        link=_link(np.array([0.0, 0.7, 0.2, 0.1]), n_dfe=2),
        data=rng.choice(np.array([-1.0, 1.0]), size=(3, 256)),
    )

    np.testing.assert_array_equal(trace.tap_weights[1:, MAIN + 1 :], 0.0)
