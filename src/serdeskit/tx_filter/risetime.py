"""TxRisetimeFilter: the transmitter output driver's finite risetime,
modeled as a low-pass filter (IEEE 802.3-2022 Annex 93A equation 93A-46).

Physical picture: a real Tx driver can't produce an instantaneous
transition — it has a finite 20%-80% risetime. That band-limits the
transmitted signal; 93A-46 models the effect as a Gaussian low-pass:

H(f) = exp(-2 * (pi * f * risetime / 1.6832)^2)

(PyChOpMarg's own formula is written with f in GHz and risetime in ns,
purely a config-sheet convention — f*risetime is dimensionless either
way, so passing both in SI units (Hz, seconds) here needs no conversion
factor.)
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class TxRisetimeFilter:
    def __init__(self, risetime: float) -> None:
        """
        Args:
            risetime: The Tx output driver's 20%-80% risetime (seconds).
        """
        self.risetime = risetime

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """(93A-46): this filter's (real-valued, but returned as complex
        for composability with other stages') transfer function H(f), at
        each frequency in `freqs` (Hz).
        """
        h = np.exp(-2 * (np.pi * freqs * self.risetime / 1.6832) ** 2)
        return np.asarray(h, dtype=np.complex128)
