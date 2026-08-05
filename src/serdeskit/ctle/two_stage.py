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
from scipy.signal import lfilter

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
        """Time-domain path: two bilinear-transform IIR filters applied in
        sequence, one per stage — matching MATLAB COM3.70's own `TD_CTLE`
        (`com_ieee8023_93a_370.m:2228-2243`), called twice
        (`:5925`/`:5930`) rather than this class's own `transfer_function()`
        multiplied into a shared frequency-domain composition. Verified
        against MATLAB's own `TD_CTLE` output directly:
        `tests/ctle/test_two_stage_process_vs_matlab.py`.
        """
        stage1 = _bilinear_iir(sig.samples, sig.fs, self.zero_freq, self.pole1_freq, self.pole2_freq, self.dc_gain_db)
        stage2 = _bilinear_iir(stage1, sig.fs, self.shelf_freq, self.shelf_freq, 100e100, self.shelf_gain_db)
        return Signal(samples=stage2, fs=sig.fs, t0=sig.t0)


def _bilinear_iir(
    ir_in: npt.NDArray[np.float64], fs: float, f_z: float, f_p1: float, f_p2: float, kacdc_db: float
) -> npt.NDArray[np.float64]:
    """One (93A-22) zero/two-pole stage, bilinear-transformed to a
    discrete IIR filter at sample rate `fs` and applied to `ir_in` —
    MATLAB's own `TD_CTLE`, transcribed directly (`bilinear_fs = 2*fb*
    oversampling` there is just `2*fs` here: `fb*oversampling` is the
    sample rate by definition, so there's nothing to recover separately).
    `f_p2=100e100` (as `TwoStageCtle.process()` passes for the shelf
    stage) makes its own bilinear-transformed pole land at `z=-1`,
    canceling the zero this filter's numerator always has there —
    collapsing it to a true single-pole/zero stage.
    """
    p1 = -2 * np.pi * f_p1
    p2 = -2 * np.pi * f_p2
    z = -2 * np.pi * f_z * 10 ** (kacdc_db / 20)
    k = -p2
    bilinear_fs = 2 * fs
    p1d = (1 + p1 / bilinear_fs) / (1 - p1 / bilinear_fs)
    p2d = (1 + p2 / bilinear_fs) / (1 - p2 / bilinear_fs)
    zd = (1 + z / bilinear_fs) / (1 - z / bilinear_fs)
    kd = (bilinear_fs - z) / ((bilinear_fs - p1) * (bilinear_fs - p2)) * f_p1 / f_z

    b = k * kd * np.array([1, 1 - zd, -zd])
    a = np.array([1, -(p1d + p2d), p1d * p2d])
    return np.asarray(lfilter(b, a, ir_in), dtype=np.float64)
