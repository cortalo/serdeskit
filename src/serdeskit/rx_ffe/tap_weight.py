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

`process()` doesn't exist yet — MATLAB's real Rx FFE signal-path
application (`force()`, `com_ieee8023_93a_370.m:3693-3850`, called from
`Apply_EQ` with already-chosen taps) isn't just a filter application: it
also runs a "DNH" search that can zero trailing post-cursor taps, out of
scope here the same way `EqualizationSearch` is (see CLAUDE.md). TDD:
`tests/rx_ffe/test_rx_ffe_tap_weight_process_vs_matlab.py` is written first, against this
stub, scoped to n_post_taps=0 so that search provably can't modify
anything (confirmed directly against a real MATLAB run, not assumed).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from serdeskit.common.types import Signal


class TapWeightRxFfe:
    def __init__(self, tap_weights: npt.NDArray[np.float64], tap_delay: float, n_pre: int = 0) -> None:
        """
        Args:
            tap_weights: All tap weights, cursor included, ordered by
                increasing delay (tap_weights[0] is the earliest tap).
            tap_delay: T (seconds), the spacing between adjacent taps —
                one UI, for a baud-spaced Rx FFE.
            n_pre: How many of `tap_weights`' entries, counting from the
                start, are pre-cursor. Default 0 (first tap is the
                cursor) preserves this class's original PyChOpMarg-
                matching convention — irrelevant for a single tap, the
                only case this project has exercised in production so
                far. `process()` needs this to reference delay 0 at the
                cursor tap, matching MATLAB's own `FFE.m` (see
                tests/ffe/test_tap_weight_process_vs_matlab.py's
                equivalent note for Tx FFE); `transfer_function()`
                doesn't use it (unverified against MATLAB, still matches
                PyChOpMarg's own first-tap-referenced Hffe_Rx only).
        """
        self.tap_weights = tap_weights
        self.tap_delay = tap_delay
        self.n_pre = n_pre

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """This Rx FFE's complex voltage transfer function H(f), at each
        frequency in `freqs` (Hz) — matches PyChOpMarg's Hffe_Rx.
        """
        delays = np.arange(len(self.tap_weights))
        return np.asarray(
            self.tap_weights @ np.exp(np.outer(delays, -1j * 2 * np.pi * self.tap_delay * freqs)),
            dtype=np.complex128,
        )

    def process(self, sig: Signal) -> Signal:
        """Circular-shift-and-sum, cursor-referenced via `n_pre` —
        MATLAB's own `force()` reduces to exactly this (`FFE.m`'s own
        algorithm, same as `TapWeightFfe.process()`) whenever its "DNH"
        backoff search can't modify anything, the scope this class's
        `process()` covers (see module docstring). Verified against
        MATLAB's own `force()` output directly: `tests/rx_ffe/
        test_rx_ffe_tap_weight_process_vs_matlab.py`.
        """
        samples_per_ui = round(self.tap_delay * sig.fs)
        out = np.zeros_like(sig.samples)
        for i, c in enumerate(self.tap_weights):
            out += np.roll(sig.samples, (i - self.n_pre) * samples_per_ui) * c
        return Signal(samples=out, fs=sig.fs, t0=sig.t0)
