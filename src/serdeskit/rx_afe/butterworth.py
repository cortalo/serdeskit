"""RxAfeButterworth: the receiver analog front-end's fourth-order
Butterworth low-pass response (IEEE 802.3-2022 Annex 93A equation 93A-20).

Physical picture: the Rx AFE band-limits the incoming signal before it
reaches the CTLE/slicer. Modeled as a normalized fourth-order Butterworth:

H(f) = 1 / (1 - 3.414214*f_n^2 + f_n^4 + 2.613126j*(f_n - f_n^3))

where f_n = f / cutoff_freq.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class RxAfeButterworth:
    def __init__(self, cutoff_freq: float) -> None:
        """
        Args:
            cutoff_freq: The AFE's cutoff frequency (Hz).
        """
        self.cutoff_freq = cutoff_freq

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """(93A-20): this filter's complex voltage transfer function H(f),
        at each frequency in `freqs` (Hz).
        """
        f_n = freqs / self.cutoff_freq
        h = 1 / (1 - 3.414214 * f_n**2 + f_n**4 + 2.613126j * (f_n - f_n**3))
        return np.asarray(h, dtype=np.complex128)
