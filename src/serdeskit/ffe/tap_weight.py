"""TapWeightFfe: a feed-forward equalizer specified by its pre/post-cursor
tap weights, satisfying serdeskit.link.Ffe's `process` implicitly.

Physical picture: an FFE's frequency response, per IEEE 802.3-2022 Annex
93A equation 93A-21, is the DTFT of its tap sequence — H(f) = sum_n b_n *
exp(-j*2*pi*n*T*f), T (`tap_delay`) being the spacing between adjacent
taps (one UI, for a baud-spaced FFE). Delay is referenced to the
cursor/main tap (n=0) rather than the first (most-precursor) tap, matching
MATLAB COM3.70's own `FFE` (a time-domain circshift-based tap-delay sum) —
see tests/ffe/test_tap_weight_vs_matlab.py.

The cursor/main tap, b_0, is deliberately *not* a constructor argument:
per COM's convention, it isn't a free parameter — only the pre/post-cursor
taps are (they're what COM's tap-combination search optimizes over). The
cursor tap is derived so the whole tap set satisfies sum(|b_n|) = 1, a
gain-normalization constraint — see `cursor_weight`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from serdeskit.common.types import Signal


@dataclass(frozen=True)
class TapWeightFfe:
    # Pre/post-cursor tap weights — NOT including the cursor/main tap
    # (see `cursor_weight`). Ordered [pre-cursor taps..., post-cursor
    # taps...].
    tap_weights: npt.NDArray[np.float64]
    # How many of `tap_weights`' entries (counting from the end) are
    # post-cursor taps; the rest, at the front, are pre-cursor.
    n_post: int
    tap_delay: float  # T (seconds), the spacing between adjacent taps (93A-21)

    def __post_init__(self) -> None:
        # `frozen=True` blocks reassigning `tap_weights`, not in-place
        # mutation of the array it points to -- locked so identity-based
        # caching keyed on a TapWeightFfe can't be invalidated out from
        # under it without an error (see SystemGrid.build's own
        # setflags(write=False) for the same reasoning).
        self.tap_weights.setflags(write=False)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def cursor_weight(self) -> float:
        """(93A-21's normalization convention): the main/cursor tap isn't
        a free parameter — it's derived so that sum(|all taps|) = 1.
        """
        return float(1.0 - np.abs(self.tap_weights).sum())

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """(93A-21): this FFE's complex voltage transfer function H(f), at
        each frequency in `freqs` (Hz).
        """
        n_pre = len(self.tap_weights) - self.n_post
        taps = np.concatenate(
            [self.tap_weights[:n_pre], [self.cursor_weight], self.tap_weights[n_pre:]]
        )
        delays = np.arange(len(taps)) - n_pre
        return np.asarray(
            taps @ np.exp(np.outer(delays, -1j * 2 * np.pi * self.tap_delay * freqs)),
            dtype=np.complex128,
        )

    def process(self, sig: Signal) -> Signal:
        """Time-domain path: a circular-shift-and-sum tap-delay line —
        matching MATLAB COM3.70's own `FFE` (`matlab_golden/lib/FFE.m`)
        directly, rather than this class's own `transfer_function()`
        multiplied into a shared frequency-domain composition. Verified
        against MATLAB's own `FFE` output directly:
        `tests/ffe/test_tap_weight_process_vs_matlab.py`.
        """
        n_pre = len(self.tap_weights) - self.n_post
        taps = np.concatenate(
            [self.tap_weights[:n_pre], [self.cursor_weight], self.tap_weights[n_pre:]]
        )
        samples_per_ui = round(self.tap_delay * sig.fs)

        out = np.zeros_like(sig.samples)
        for i, c in enumerate(taps):
            out += np.roll(sig.samples, (i - n_pre) * samples_per_ui) * c
        return Signal(samples=out, fs=sig.fs, t0=sig.t0)
