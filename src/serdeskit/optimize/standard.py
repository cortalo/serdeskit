"""ComStandard: every input parameter a COM standard configuration
needs, top to bottom — what com.compute() needs on its own, plus what
running the equalization search first requires: the CTLE's fixed shape
and gain search grid, the Tx FFE's tap search grid, and the Rx AFE's
cutoff. Flat, plain data — no nested objects — mirroring PyChOpMarg's
own COMParams shape (one flat dataclass covering the whole standard
configuration) rather than this project's usual small-composed-objects
convention, since this is meant to be the one place every standard
parameter lives, not a bundle of already-separate stage configs.

search() turns this into a winning LinkComParams (search space
resolved to one point); compute() turns that into a ComResult.

`com_min_db` is the pass/fail threshold ("COM Pass threshold" in the
standard's own configuration tables) — kept here rather than hard-coded
anywhere, since PyChOpMarg itself doesn't hard-code it either: it's
specific to which PMD type/clause is being evaluated, external to the
COM calculation proper.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ComStandard:
    # What com.compute() needs, given a fixed equalization — same
    # fields as com.LinkComParams' own numeric config.
    baud_rate: float  # fb (Hz)
    freq_step: float  # fstep (Hz)
    samples_per_ui: int  # M
    levels: int  # L
    rlm: float
    victim_amplitude: float  # A_v (V)
    a_ne: float  # A_ne (V)
    a_fe: float  # A_fe (V)
    snr_tx: float  # SNR_TX (dB)
    sigma_rj: float  # random jitter (UI RMS)
    eta_0: float  # one-sided noise spectral density (V^2/GHz)
    a_dd: float  # dual-Dirac deterministic jitter amplitude (UI)
    der_0: float  # target detector error ratio
    dfe_min: npt.NDArray[np.float64]
    dfe_max: npt.NDArray[np.float64]

    # CTLE: shape fixed, gains searched.
    ctle_zero_freq: float
    ctle_pole1_freq: float
    ctle_pole2_freq: float
    ctle_shelf_freq: float
    ctle_dc_gain_candidates: Sequence[float]
    ctle_shelf_gain_candidates: Sequence[float]

    # Tx FFE: every (pre-cursor, post-cursor) tap combination this wide,
    # this finely stepped, leaving at least tx_taps_c0_min for the
    # cursor (93A-21's cursor = 1 - sum(|taps|) convention) is searched.
    tx_taps_bounds: Sequence[tuple[float, float, float]]
    tx_taps_c0_min: float
    tx_taps_n_post: int

    tx_risetime: float  # T_r (s), 20%-80% -- see tx_filter.TxRisetimeFilter. Fixed across the search.
    rx_afe_cutoff_freq: float

    # Reference/termination impedances (Ohms) — set gamma1/gamma2 (93A-18),
    # the reflection coefficients looking out of the channel's Tx/Rx ends.
    # Separate Tx/Rx values, not one shared "rd": PyChOpMarg's own R_d is
    # itself per-side (R_d[0] for Tx, R_d[1] for Rx), and the two dies
    # need not be identical.
    r0: float  # R_0
    tx_termination_resistance: float  # R_d[0]
    rx_termination_resistance: float  # R_d[1]

    # Package model — die and package transmission line, both per side.
    # z_p (package trace length, in `*_tline_segments`) genuinely
    # differs Tx vs Rx even under "package case 1" naming (e.g. IEEE
    # 802.3ck C2C: z_p(TX)=13mm, z_p(RX)=11mm) -- kept fully split
    # rather than one shared package_tline_* pair (this class's own
    # earlier design), which silently re-collapsed Tx/Rx to the same
    # length and cost ~2.7dB of COM accuracy versus MATLAB before this
    # was caught (see examples/compute_com_c2c.py's git history).
    tx_die_capacitances: Sequence[float]  # C_d[0] (F), one per on-die ladder rung
    tx_die_inductances: Sequence[float]  # L_s[0] (H)
    tx_bump_capacitance: float  # C_b[0] (F)
    tx_pad_capacitance: float  # C_p[0] (F)
    tx_tline_a1: float
    tx_tline_a2: float
    tx_tline_tau: float
    tx_tline_gamma0: float
    tx_tline_segments: Sequence[tuple[float, float]]  # (characteristic impedance Ohms, length mm) pairs
    rx_die_capacitances: Sequence[float]  # C_d[1]
    rx_die_inductances: Sequence[float]  # L_s[1]
    rx_bump_capacitance: float  # C_b[1]
    rx_pad_capacitance: float  # C_p[1]
    rx_tline_a1: float
    rx_tline_a2: float
    rx_tline_tau: float
    rx_tline_gamma0: float
    rx_tline_segments: Sequence[tuple[float, float]]

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
