"""TapWeightRxFfe: an Rx feed-forward equalizer specified by its full tap
sequence, satisfying serdeskit.link.RxFfe's `transfer_function` implicitly.

Physical picture: same DTFT-of-tap-sequence shape as the Tx FFE (93A-21),
H(f) = sum_n b_n * exp(-j*2*pi*n*T*f) — but a different convention. Tx FFE
(TapWeightFfe) is given only pre/post-cursor taps and derives the cursor so
sum(|b_n|) = 1 (COM's Tx tap-search convention). Rx FFE (PyChOpMarg's
Hffe_Rx) has no such normalization: every tap, cursor included, is given
directly as `tap_weights[n]`, n=0..N-1. Deliberately its own class rather
than a TapWeightFfe variant — the tap conventions genuinely differ, not
just the constructor arguments.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class TapWeightRxFfe:
    def __init__(self, tap_weights: npt.NDArray[np.float64], tap_delay: float) -> None:
        """
        Args:
            tap_weights: All tap weights, cursor included, ordered by
                increasing delay (tap_weights[0] is the earliest tap).
            tap_delay: T (seconds), the spacing between adjacent taps —
                one UI, for a baud-spaced Rx FFE.
        """
        self.tap_weights = tap_weights
        self.tap_delay = tap_delay

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """This Rx FFE's complex voltage transfer function H(f), at each
        frequency in `freqs` (Hz) — matches PyChOpMarg's Hffe_Rx.
        """
        delays = np.arange(len(self.tap_weights))
        return np.asarray(
            self.tap_weights @ np.exp(np.outer(delays, -1j * 2 * np.pi * self.tap_delay * freqs)),
            dtype=np.complex128,
        )
