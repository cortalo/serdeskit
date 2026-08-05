"""802.3-2022
IEEE Standard for Ethernet: https://doi.org/10.1109/IEEESTD.2022.9844436
8023ck-2022: https://doi.org/10.1109/IEEESTD.2022.9999414
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ComStandard:
    # Table 93A-1 parameters
    baud_rate: float  # Signaling rate (Baud), 93A.1.1
    # Missing Maximum start frequency (Hz), 93A.1.1
    freq_step: float  # Maximum frequency step (Hz), 93A.1.1
    tx_die_capacitances: Sequence[float]  # Single-ended device capacitance (F), 93A.1.2
    tx_die_inductances: Sequence[float]  # Single-ended device series inductance (H), 93A.1.2
    tx_bump_capacitance: float  # Single-ended bump capacitance (F), 93A.1.2
    tx_tline_segments: Sequence[tuple[float, float]]  # (characteristic impedance Ohms, length mm) pairs, 93A.1.2
    tx_tline_gamma0: float # Transmission line parameter, Table 93A-3
    tx_tline_a1: float # Transmission line model parameter, a1, Table 93A-3
    tx_tline_a2: float # Transmission line model parameter, a2, Table 93A-3
    tx_tline_tau: float # Transmission line model parameter, Table 93A-3
    tx_pad_capacitance: float  # Single-ended package capacitance at package-to-board interface (F), 93A.1.2
    rx_die_capacitances: Sequence[float]  # Single-ended device capacitance (F), 93A.1.2
    rx_die_inductances: Sequence[float]  # Single-ended device series inductance (H), 93A.1.2
    rx_bump_capacitance: float  # Single-ended bump capacitance (F), 93A.1.2
    rx_tline_segments: Sequence[tuple[float, float]] # (characteristic impedance Ohms, length mm) pairs, 93A.1.2
    rx_tline_gamma0: float # Transmission line model parameter, Table 93A-3
    rx_tline_a1: float # Transmission line model parameter, a1, Table 93A-3
    rx_tline_a2: float # Transmission line model parameter, a2, Table 93A-3
    rx_tline_tau: float # Transmission line model parameter, Table 93A-3
    rx_pad_capacitance: float  # Single-ended package capacitance at package-to-board interface (F), 93A.1.2
    r0: float  # Single-ended reference resistance (Ohm), 93.A.1.2
    tx_termination_resistance: float  # Single-ended termination resistance (Ohm), 93A.1.3
    rx_termination_resistance: float  # Single-ended termination resistance (Ohm), 93A.1.3
    rx_afe_cutoff_freq: float # Receiver 3 dB bandwidth (Hz), 93A.1.4.1
    tx_taps_c0_min: float # Transmitter equalizer, minimum cursor coefficient, 93A.1.4.2
    tx_taps_bounds: Sequence[tuple[float, float, float]] # Transmitter equalizer, pre-cursor and post-cursor coefficient (minimum value, maximum value, step size), 93A.1.4.2
    tx_taps_n_post: int
    ctle_dc_gain_candidates: Sequence[float] # Continuous time filter, DC gain candidates (dB), 93A.1.4.3
    ctle_shelf_gain_candidates: Sequence[float] # Continuous time filter, DC gain 2 candidates (dB), 93A.1.4.3
    ctle_zero_freq: float # Continuous time filter, zero frequency for gDC = 0 (Hz), 93A.1.4.3
    ctle_pole1_freq: float # Continuous time filter, pole frequency (Hz), 93.A.1.4.3
    ctle_pole2_freq: float # Continuous time filter, pole frequency (Hz), 93.A.1.4.3
    ctle_shelf_freq: float # Continuous time filter, low frequency pole/zero (Hz), 93A.1.4.3
    victim_amplitude: float  # Transmitter differential peak output voltage, Victim (V), 93A.1.5
    a_fe: float  # Transmitter differential peak output voltage, Far-end aggressor (V), 93A.1.5
    a_ne: float  # Transmitter differential peak output voltage, Near-end aggressor (V), 93A.1.5
    levels: int  # Number of signal levels, 93A.1.6
    rlm: float # Level separation mismatch ratio, 93A.1.6
    snr_tx: float  # Transmitter signal-to-noise ratio (dB), 93A.1.6
    samples_per_ui: int  # Number of samples per unit interval, 93A.1.6
    # Decision feedback equalizer (DFE) length is equal to dfe_min.length, also equal to dfe_max.length, 93A.1.6
    dfe_max: npt.NDArray[np.float64] # Normalized DFE coefficient maximum limit, 93A.1.6
    dfe_min: npt.NDArray[np.float64] # Normalized DFE coefficient minimum limit, 93A.1.6
    sigma_rj: float  # Random jitter, RMS (UI), 93A.1.6
    a_dd: float  # Dual-Dirac jitter, peak (UI), 93A.1.6
    eta_0: float  # One-sided noise spectral density (V^2/GHz), 93A.1.6
    der_0: float  # Target detector error ratio, 93A.1.7

    tx_risetime: float  # T_r (s), 20%-80% -- see tx_filter.TxRisetimeFilter. Fixed across the search.

    com_min_db: float


    @property
    def level_variance(self) -> float:
        """varX (93A-29): the variance of a transmitted symbol, taken over
        `levels` equally likely, evenly spaced values in [-1, 1].
        """
        return (self.levels**2 - 1) / (3 * (self.levels - 1) ** 2)

    @property
    def gamma1(self) -> float:
        """(93A-18): the reflection coefficient looking out of the Tx end."""
        return (self.tx_termination_resistance - self.r0) / (self.tx_termination_resistance + self.r0)

    @property
    def gamma2(self) -> float:
        """(93A-18): the reflection coefficient looking out of the Rx end."""
        return (self.rx_termination_resistance - self.r0) / (self.rx_termination_resistance + self.r0)
