"""PassThroughChannel: the simplest possible Channel — returns the signal
unchanged. Exists to exercise the Link -> Channel -> LinkResult pipeline
end to end before a real S-parameter channel model is built.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from serdeskit.common.types import Signal


class PassThroughChannel:
    """Satisfies serdeskit.link.Channel implicitly — no inheritance needed."""

    def process(self, sig: Signal) -> Signal:
        return sig

    def transfer_function(self, freqs: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """Unity gain at every frequency — same "no-op" identity as
        `process`.
        """
        return np.ones_like(freqs, dtype=np.complex128)
