"""SystemGrid: the time/frequency axes a pulse response gets computed on
(IEEE 802.3-2022 Annex 93A equation 93A-24's setup), plus x_sinc — the
closed-form frequency-domain spectrum of the UI-wide rectangular pulse a
single transmitted symbol actually is (zero-order-hold, not an idealized
delta impulse — the same physical picture link.py's own `_upsample_bits`
already encodes in the time domain).

Not itself a pulse response: `pulse_response()` is the later step that
multiplies x_sinc by a system's actual transfer function (channel * CTLE
* FFE * ...) and inverse-transforms the result back to the time domain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from serdeskit.common.types import Signal


@dataclass(frozen=True, slots=True)
class SystemGrid:
    t: npt.NDArray[np.float64]
    f: npt.NDArray[np.float64]
    x_sinc: npt.NDArray[np.float64]

    @classmethod
    def build(cls, baud_rate: float, freq_step: float, samples_per_ui: int) -> SystemGrid:
        """Constructs t/f/x_sinc from (baud rate, frequency step, samples
        per UI), per PyChOpMarg's own recipe.

        Args:
            baud_rate: Symbol rate (Hz).
            freq_step: Frequency-domain resolution, Δf (Hz) — determines
                how much time domain the system's time vector spans
                (1/Δf), per the reciprocal relationship between frequency
                resolution and time-domain record length in an IFFT.
            samples_per_ui: Number of time-domain samples per UI.

        Returns:
            A SystemGrid with `t` spanning one full cycle of the
            fundamental (1/freq_step), `f` up to Nyquist, and `x_sinc` —
            the rectangular-pulse spectrum evaluated on `f`.
        """
        ui = 1.0 / baud_rate
        tstep = ui / samples_per_ui
        tmax = 1.0 / freq_step
        t = np.arange(0.0, tmax, tstep)

        fmax = 0.5 / t[1]
        f = np.arange(0.0, fmax + freq_step, freq_step)

        x_sinc = int(ui / t[1]) * np.sinc(ui * f)

        return cls(t=t, f=f, x_sinc=x_sinc)

    def pulse_response(self, h: npt.NDArray[np.complex128]) -> Signal:
        """(93A-24): p(t) = IFFT[x_sinc(f) * H(f)], the pulse response of
        a system whose transfer function is `h` (evaluated on `self.f`),
        as a Signal — not yet a PulseResponse; cursor location is a
        separate step (PulseResponse.from_signal).

        Args:
            h: The system's transfer function (channel * CTLE * FFE *
                ...), evaluated on `self.f`.

        Returns:
            The pulse response, as a Signal.
        """
        p = np.fft.irfft(self.x_sinc * h)[: len(self.t)]
        fs = 1.0 / (self.t[1] - self.t[0])
        return Signal(samples=p, fs=fs, t0=0.0)
