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
    samples_per_ui: int

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

        return cls(t=t, f=f, x_sinc=x_sinc, samples_per_ui=samples_per_ui)

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

    def truncated_impulse_response(self, h: npt.NDArray[np.complex128], threshold: float = 1e-3) -> Signal:
        """MATLAB COM3.70's own `s21_to_impulse_DC`
        (`com_ieee8023_93a_370.m:9895-9964`), not this class's usual
        `pulse_response()`: IFFT `h`, then truncate once the result's
        magnitude decays below `threshold * peak` (MATLAB's own default,
        `OP.impulse_response_truncation_threshold`) — the result is no
        longer the periodic record `pulse_response()` assumes. Not a
        pulse response yet — `box_car_integrate()` is the next step
        MATLAB's own pipeline takes; kept separate here since CTLE's own
        time-domain path (`Ctle.process()`) runs on the impulse response,
        before that box-car integration.

        Args:
            h: The system's transfer function, evaluated on `self.f`.
            threshold: Truncate once |impulse| drops below this fraction
                of its own peak. Default matches MATLAB's own default.

        Returns:
            The (shorter than `self.t`) impulse response, as a Signal.
        """
        impulse = np.fft.irfft(h)[: len(self.t)]
        peak = np.max(np.abs(impulse))
        last = int(np.max(np.nonzero(np.abs(impulse) > peak * threshold)))
        fs = 1.0 / (self.t[1] - self.t[0])
        return Signal(samples=impulse[: last + 1], fs=fs, t0=0.0)

    def box_car_integrate(self, sig: Signal) -> Signal:
        """Integrate `sig` over one UI (`self.samples_per_ui` samples) via
        linear (causal, non-circular) convolution — matching MATLAB's own
        `filter(ones(1,samples_per_ui),1,...)` (`com_ieee8023_93a_370.m:930`),
        the step that turns an impulse response into a pulse response.
        Linear, not `pulse_response()`'s circular treatment, because
        `sig` (from `truncated_impulse_response()`, or `Ctle.process()`'s
        output on it) is no longer a periodic record.
        """
        box = np.ones(self.samples_per_ui)
        p = np.convolve(sig.samples, box, mode="full")[: len(sig.samples)]
        return Signal(samples=p, fs=sig.fs, t0=sig.t0)

    def truncated_pulse_response(self, h: npt.NDArray[np.complex128], threshold: float = 1e-3) -> Signal:
        """`truncated_impulse_response()` then `box_car_integrate()` —
        MATLAB's own two-step recipe end to end, with no CTLE/FFE in
        between. Exists for `Link.uneq_pulse_response()`; see
        `truncated_impulse_response()`'s own docstring for why CTLE needs
        these as two separate steps rather than this one.
        """
        return self.box_car_integrate(self.truncated_impulse_response(h, threshold))
