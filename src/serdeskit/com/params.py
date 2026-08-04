"""ComParams: the configuration a COM calculation needs that isn't already
carried by the Link's own stages.

Deliberately does *not* restate anything a stage already owns — the CTLE's
pole/zero/gain settings live on TwoStageCtle, the Tx tap weights on
TapWeightFfe, the Rx AFE's cutoff on RxAfeButterworth, the termination
reflection coefficients on SParameterChannel. Only what has no such home
appears here, which is why this is ~11 fields rather than the ~40 of
PyChOpMarg's flat COMParams.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ComParams:
    baud_rate: float  # fb (Hz)
    freq_step: float  # fstep (Hz) — frequency resolution; see SystemGrid.build
    samples_per_ui: int  # M
    levels: int  # L — 2 for NRZ, 4 for PAM4
    rlm: float  # RLM, relative level mismatch — see PulseResponse.signal_amplitude
    # A_v (V): the victim transmitter's launch amplitude. Scales the pulse
    # response, and must be applied *before* the cursor is located, not
    # after: the Muller-Mueller search's `eps` is an absolute voltage, so
    # an unscaled response can settle on a different cursor sample.
    victim_amplitude: float
    # A_ne/A_fe (V): NEXT/FEXT aggressor launch amplitudes — same role as
    # victim_amplitude, one per crosstalk kind. Applied to an aggressor's
    # pulse response the same way (before its worst-case-phase samples are
    # taken), per PyChOpMarg's gen_pulse_resps.
    a_ne: float
    a_fe: float
    snr_tx: float  # SNR_TX (dB) — transmitter noise, (93A-30)
    sigma_rj: float  # random jitter (UI RMS) — (93A-31)
    eta_0: float  # one-sided noise spectral density (V^2/GHz) — (93A-35)
    a_dd: float  # dual-Dirac deterministic jitter amplitude (UI) — (93A-40)
    der_0: float  # target detector error ratio — where Ani is read off
    dfe_min: npt.NDArray[np.float64]  # per-tap DFE limits; length sets the DFE's span
    dfe_max: npt.NDArray[np.float64]

    @property
    def level_variance(self) -> float:
        """varX (93A-29): the variance of a transmitted symbol, taken over
        `levels` equally likely, evenly spaced values in [-1, 1].
        """
        return (self.levels**2 - 1) / (3 * (self.levels - 1) ** 2)
