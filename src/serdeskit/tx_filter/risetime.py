"""TxRisetimeFilter: the transmitter output driver's finite risetime.

Physical picture: a real Tx driver can't produce an instantaneous
transition — it has a finite 20%-80% risetime. That band-limits the
transmitted signal.

Not the textbook 93A-46 Gaussian low-pass (magnitude only) this class
used to implement: MATLAB COM3.70's own `s21_pkg_tester`
(`com_ieee8023_93a_370.m:9839-9856`) bakes a *different* version into
every `chdata(i).sdd21` whenever `OP.FORCE_TR` is set (which its own
comment says "should be set to 1 in most later config sheets") —
confirmed against MATLAB (`tests/tx_filter/test_risetime_vs_matlab.py`,
`matlab_golden/generate/gen_tx_h_t.m`; empirically cross-checked against
a real live run to 5.3e-10 precision first, see docs/known-issues.md's
"full composed pulse response" entry):

H(f) = exp(-2 * (pi * f * risetime / 1.6832)^2)
     * exp(-1j * 2*pi * f * risetime * 3)

— the same Gaussian magnitude term as 93A-46, but with an extra
`exp(-1j*2*pi*f*risetime*3)` phase (a pure `3*risetime` delay) this
class's old implementation was missing entirely. MATLAB's own formula is
written with f in GHz and risetime in ns, purely a config-sheet
convention — f*risetime is dimensionless either way, so passing both in
SI units (Hz, seconds) here needs no conversion factor.

Only the `T_r_filter_type==1`/`T_r_meas_point==0` branch of
`s21_pkg_tester`'s H_t — the one `OP.FORCE_TR` forces every config this
project has exercised into (`matlab_golden/lib/tx_transition_time_filter.m`
keeps the other branches, for whenever that stops being true).
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
        """This filter's complex voltage transfer function H(f), at each
        frequency in `freqs` (Hz). See the module docstring for the formula.
        """
        h = np.exp(-2 * (np.pi * freqs * self.risetime / 1.6832) ** 2) * np.exp(
            -1j * 2 * np.pi * freqs * self.risetime * 3
        )
        return np.asarray(h, dtype=np.complex128)
