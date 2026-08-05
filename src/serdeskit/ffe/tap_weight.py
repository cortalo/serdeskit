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

import numpy as np
import numpy.typing as npt

from serdeskit.common.types import Signal


class TapWeightFfe:
    def __init__(
        self,
        tap_weights: npt.NDArray[np.float64],
        n_post: int,
        tap_delay: float,
    ) -> None:
        """
        Args:
            tap_weights: Pre/post-cursor tap weights — NOT including the
                cursor/main tap (see `cursor_weight`). Ordered
                [pre-cursor taps..., post-cursor taps...].
            n_post: How many of `tap_weights`' entries (counting from the
                end) are post-cursor taps; the rest, at the front, are
                pre-cursor.
            tap_delay: T (seconds), the spacing between adjacent taps
                (93A-21) — one UI, for a baud-spaced FFE.
        """
        self.tap_weights = tap_weights
        self.n_post = n_post
        self.tap_delay = tap_delay

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
        """Apply this FFE to `sig` via its frequency-domain transfer
        function.
        """
        raise NotImplementedError
