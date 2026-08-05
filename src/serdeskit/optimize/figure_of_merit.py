"""figure_of_merit: a fast, closed-form approximation to COM (IEEE
802.3-2022 Annex 93A, equation 93A-36) — PyChOpMarg's calc_fom, the
objective opt_eq's grid search maximizes over Tx tap combinations and
CTLE gain settings.

Deliberately not "calc_noise without the PMF": four of its five variance
terms are computed differently, not just skipping the PMF step
com.Com.compute() ends with —

- Jitter (93A-32) combines deterministic (A_DD) and random (sigma_Rj)
  jitter into one variance term, rather than random jitter joining varG
  while deterministic jitter gets its own delta-PMF.
- Crosstalk (93A-33/34) is a direct worst-case-phase energy sum per
  aggressor, rather than routing each aggressor's samples through
  delta_pmf and combine_pmfs.
- Receiver noise (93A-35) does *not* exclude the DC frequency bin — a
  real formula difference from calc_noise's own varN (which does), not
  an oversight; confirmed by reading calc_fom's own source rather than
  assumed from calc_noise's shape.
- Every variance is summed directly under one log10 (93A-36), rather
  than combined via PMF convolution and read off a CDF at der_0.

Tx noise (93A-30) and ISI (93A-31 applied to residual ISI) are the same
formulas com.compute() already uses — composed here from the same
PulseResponse/ComStandard pieces, not reimplemented.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from serdeskit.optimize.standard import ComStandard
from serdeskit.pmf import filter_samples
from serdeskit.pulse_response import PulseResponse


def figure_of_merit(
    pulse_response: PulseResponse,
    aggressor_pulse_responses: Sequence[npt.NDArray[np.float64]],
    rx_response: npt.NDArray[np.complex128],
    params: ComStandard,
) -> float:
    """(93A-36): 10*log10(As^2 / sum of variances).

    Args:
        pulse_response: The victim's cursor-located pulse response,
            already scaled by `params.victim_amplitude` — built the same
            way com.Com.compute() builds its own.
        aggressor_pulse_responses: Each NEXT/FEXT aggressor's raw pulse
            response samples, already scaled by A_ne/A_fe — unfiltered
            and not yet reduced to a single phase (this function does
            both itself, per aggressor).
        rx_response: CTLE * Rx AFE (* Rx FFE, if any), evaluated on the
            frequency grid `params.freq_step` describes.
        params: The rest of what these formulas need.

    Returns:
        The figure of merit, dB.
    """
    signal_amplitude = pulse_response.signal_amplitude(params.rlm, params.levels)

    var_tx = pulse_response.cursor_value**2 * 10 ** (-params.snr_tx / 10)  # (93A-30)

    residual = pulse_response.residual_isi(params.dfe_min, params.dfe_max)  # (93A-26)/(93A-27)
    var_isi = params.level_variance * float((residual**2).sum())  # (93A-31)

    slopes = pulse_response.local_slopes(signal_amplitude)
    var_jitter = (
        (params.a_dd**2 + params.sigma_rj**2) * params.level_variance * float((slopes**2).sum())
    )  # (93A-32)

    # Unlike com.Com.compute()'s own varN, calc_fom's own (93A-35) does
    # *not* exclude the DC bin (calc_noise's does) — a genuine formula
    # difference between the two, not an oversight here.
    var_noise = params.eta_0 * float((np.abs(rx_response) ** 2).sum()) * (params.freq_step / 1e9)  # (93A-35)

    var_crosstalk = params.level_variance * sum(
        _worst_case_phase_energy(pr, params.samples_per_ui, signal_amplitude)
        for pr in aggressor_pulse_responses
    )  # (93A-33)/(93A-34)

    total_variance = var_tx + var_isi + var_jitter + var_crosstalk + var_noise
    return float(10 * np.log10(signal_amplitude**2 / total_variance))


def _worst_case_phase_energy(
    pulse_response: npt.NDArray[np.float64], samples_per_ui: int, signal_amplitude: float
) -> float:
    """(93A-33): the largest per-UI energy any sub-UI phase of this
    aggressor's pulse response reaches, filtering each phase's samples
    (Note 2 of 93A.1.7.1) before summing — unlike
    `crosstalk.worst_case_phase_samples`, which picks its phase by raw
    (unfiltered) energy and leaves filtering to the caller, since that
    one feeds delta_pmf rather than a variance sum directly.
    """
    return max(
        float((filter_samples(pulse_response[phase::samples_per_ui], signal_amplitude) ** 2).sum())
        for phase in range(samples_per_ui)
    )
