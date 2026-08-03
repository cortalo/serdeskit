"""TwoStageCtle: a two-stage continuous-time linear equalizer, satisfying
serdeskit.link.Ctle's `process` implicitly.

Physical picture, per IEEE 802.3-2022 Annex 93A equation 93A-22:

- First stage: a classic one-zero/two-pole high-frequency boost — a zero
  at `zero_freq` provides the boost, two poles (`pole1_freq`, `pole2_freq`)
  roll it back off, and `dc_gain_db` sets the (possibly negative, i.e.
  attenuating) gain at DC.
- Second stage: a low-frequency "shelf" — zero and pole at the *same*
  frequency (`shelf_freq`), gain `shelf_gain_db` at DC, gain -> 1 (0 dB)
  as frequency -> infinity. Shapes low-frequency gain independently of the
  first stage's high-frequency boost shape.

H(f) = (g1 + j*f/zero_freq)(g2 + j*f/shelf_freq)
       / [(1 + j*f/pole1_freq)(1 + j*f/pole2_freq)(1 + j*f/shelf_freq)]

where g1, g2 are dc_gain_db/shelf_gain_db converted from dB to linear
voltage gain (g = 10**(dB/20) — voltage, not power, hence /20 not /10).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from serdeskit.common.types import Signal


class TwoStageCtle:
    def __init__(
        self,
        zero_freq: float,
        pole1_freq: float,
        pole2_freq: float,
        shelf_freq: float,
        dc_gain_db: float,
        shelf_gain_db: float,
    ) -> None:
        """
        Args:
            zero_freq: First stage zero frequency (Hz).
            pole1_freq: First stage lower pole frequency (Hz).
            pole2_freq: First stage upper pole frequency (Hz).
            shelf_freq: Second stage zero/pole frequency (Hz).
            dc_gain_db: First stage d.c. gain (dB).
            shelf_gain_db: Second stage d.c. gain (dB).
        """
        self.zero_freq = zero_freq
        self.pole1_freq = pole1_freq
        self.pole2_freq = pole2_freq
        self.shelf_freq = shelf_freq
        self.dc_gain_db = dc_gain_db
        self.shelf_gain_db = shelf_gain_db

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """(93A-22): this CTLE's complex voltage transfer function H(f), at
        each frequency in `freqs` (Hz).
        """
        g1 = 10 ** (self.dc_gain_db / 20)
        g2 = 10 ** (self.shelf_gain_db / 20)

        num = (g1 + 1j * freqs / self.zero_freq) * (g2 + 1j * freqs / self.shelf_freq)
        den = (
            (1 + 1j * freqs / self.pole1_freq)
            * (1 + 1j * freqs / self.pole2_freq)
            * (1 + 1j * freqs / self.shelf_freq)
        )
        return np.asarray(num / den, dtype=np.complex128)

    def process(self, sig: Signal) -> Signal:
        """Not needed by COM's own methodology — COM composes each stage's
        transfer_function() in the frequency domain and does a single
        IFFT, rather than calling a per-stage Signal -> Signal process()
        (same situation as ffe.TapWeightFfe.process()).
        """
        raise NotImplementedError
