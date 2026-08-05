"""LinkComParams: every raw value a COM calculation needs, flat enough to
come straight off a config sheet — numbers, strings (Touchstone file
paths), and sequences of either. `compute()` is what turns this into the
actual Channel/Ctle/Ffe/... objects; this dataclass deliberately holds no
domain objects itself; see `compute()`'s own docstring for which concrete
class each group of fields feeds.

No field has a default: every value here is something a real config sheet
actually specifies, so a caller building one is forced to make each
choice explicitly rather than silently inheriting a value that happens to
be wrong for their config (e.g. `rx_ffe_n_pre=0`, or an empty tap-weight
array meaning "unequalized").
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class LinkComParams:
    # Victim + crosstalk aggressor channels: Touchstone file paths.
    channel_path: str
    next_channel_paths: Sequence[str]
    fext_channel_paths: Sequence[str]
    port_order: Sequence[int]  # see channel.differential_network
    gamma1: float  # near-end reflection coefficient — see SParameterChannel
    gamma2: float  # far-end reflection coefficient

    # Tx/Rx package (Package's own fields, tx_/rx_ prefixed) — see
    # package.Package for what each one means physically.
    tx_r0: float
    tx_die_capacitances: Sequence[float]
    tx_die_inductances: Sequence[float]
    tx_bump_capacitance: float
    tx_tline_a1: float
    tx_tline_a2: float
    tx_tline_tau: float
    tx_tline_gamma0: float
    tx_tline_segments: Sequence[tuple[float, float]]
    tx_pad_capacitance: float

    rx_r0: float
    rx_die_capacitances: Sequence[float]
    rx_die_inductances: Sequence[float]
    rx_bump_capacitance: float
    rx_tline_a1: float
    rx_tline_a2: float
    rx_tline_tau: float
    rx_tline_gamma0: float
    rx_tline_segments: Sequence[tuple[float, float]]
    rx_pad_capacitance: float

    # CTLE (TwoStageCtle's own fields) — see ctle.TwoStageCtle.
    ctle_zero_freq: float
    ctle_pole1_freq: float
    ctle_pole2_freq: float
    ctle_shelf_freq: float
    ctle_dc_gain_db: float
    ctle_shelf_gain_db: float

    # Tx FFE (TapWeightFfe's own fields, minus tap_delay -- derived from
    # baud_rate) — see ffe.TapWeightFfe.
    ffe_tap_weights: npt.NDArray[np.float64]
    ffe_n_post: int

    tx_risetime: float  # TxRisetimeFilter's own field
    rx_afe_cutoff_freq: float  # RxAfeButterworth's own field

    # Rx FFE (TapWeightRxFfe's own fields, minus tap_delay) — see
    # rx_ffe.TapWeightRxFfe.
    rx_ffe_tap_weights: npt.NDArray[np.float64]
    rx_ffe_n_pre: int

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
