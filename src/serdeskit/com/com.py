"""Com: the Channel Operating Margin calculation (IEEE 802.3-2022 Annex
93A), orchestrating the pieces the other packages provide.

Deliberately a thin orchestrator, in contrast to PyChOpMarg's COM class
(53 methods, a 191-line __init__ that loads channel files and builds
package models before anything can be tested). Three rules keep it that
way:

1. The constructor stores its two arguments and nothing else. No
   precomputation, so constructing a Com can't fail for reasons unrelated
   to what a caller is about to ask it for.
2. It doesn't absorb neighbouring responsibilities — no config-file
   parsing, no plotting (that stays in util/, consuming the result).
3. It composes rather than reimplements: every formula it needs already
   lives in link/, pulse_response/, or pmf/, each golden-tested there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from serdeskit.com.params import ComParams
from serdeskit.link import Link, SystemGrid
from serdeskit.pmf import (
    combine_pmfs,
    delta_pmf,
    filter_samples,
    gaussian_pmf,
    noise_margin,
    voltage_grid,
)
from serdeskit.pulse_response import PulseResponse


@dataclass(frozen=True, slots=True)
class ComResult:
    """A COM calculation's outcome, as plain data — never a figure, so it
    can be asserted on directly (same rationale as LinkResult).

    Carries the intermediate quantities alongside the headline number
    because they're what diagnosis needs: which noise term dominates, and
    where the distribution actually crosses the target error ratio.
    """

    com_db: float  # 20*log10(signal_amplitude / noise_amplitude)
    signal_amplitude: float  # As (V), (93A.1.6.c)
    noise_amplitude: float  # Ani (V), read off the CDF at der_0
    sigma_tx: float  # sqrt(varTx), (93A-30)
    sigma_jitter: float  # sqrt(varJ), (93A-31)
    sigma_noise: float  # sqrt(varN), (93A-35)
    sigma_gaussian: float  # sqrt(varG) = the three above combined, (93A-41)
    voltage_grid: npt.NDArray[np.float64]  # y (V) the PMFs below are on
    noise_pmf: npt.NDArray[np.float64]  # the combined interference+noise PMF, (93A-45)


@dataclass
class Com:
    link: Link
    params: ComParams

    def compute(self) -> ComResult:
        """Run the calculation end to end: pulse response -> cursor ->
        residual ISI and jitter-induced amplitude noise -> their
        distributions, combined -> the noise amplitude at the target
        error ratio -> COM.

        Returns:
            The COM value in dB, with the intermediate quantities that
            produced it.
        """
        p = self.params
        grid = SystemGrid.build(p.baud_rate, p.freq_step, p.samples_per_ui)

        signal = self.link.ffe_channel_ctle_pulse_response(grid).scale(p.victim_amplitude)
        # Scale to the victim's launch amplitude before locating the
        # cursor — see ComParams.victim_amplitude for why the order matters.
        pulse_response = PulseResponse.from_signal(
            signal,
            ui=1.0 / p.baud_rate,
            dfe1_max=float(p.dfe_max[0]),
            dfe1_min=float(p.dfe_min[0]),
        )
        signal_amplitude = pulse_response.signal_amplitude(p.rlm, p.levels)
        y = voltage_grid(signal_amplitude)

        # Deterministic jitter converts to amplitude through the pulse
        # response's local slope; random jitter uses the same slopes but
        # stays Gaussian, so it joins varG rather than getting its own PMF.
        slopes = pulse_response.local_slopes(signal_amplitude)
        p_jitter = delta_pmf(
            filter_samples(p.a_dd * slopes, signal_amplitude), p.levels, y
        )

        rx_response = self.link.ctle.transfer_function(grid.f) * self.link.rx_afe.transfer_function(  # type: ignore[union-attr]
            grid.f
        )
        var_tx, var_jitter, var_noise = _gaussian_variances(
            pulse_response.cursor_value, slopes, rx_response, p
        )
        var_gaussian = var_tx + var_jitter + var_noise  # (93A-41)
        p_gaussian = gaussian_pmf(var_gaussian, y)

        residual = pulse_response.residual_isi(p.dfe_min, p.dfe_max)  # (93A-26)/(93A-27)
        p_isi = delta_pmf(filter_samples(residual, signal_amplitude), p.levels, y)

        # No crosstalk aggressors yet, so this omits (93A-44)'s per-aggressor
        # terms — a call with more arguments once they exist, not a
        # different shape of call. (93A-43)+(93A-45) combined into one step.
        p_total = combine_pmfs(p_gaussian, p_jitter, p_isi)

        noise_amplitude = noise_margin(p_total, y, p.der_0)

        return ComResult(
            com_db=float(20 * np.log10(signal_amplitude / noise_amplitude)),
            signal_amplitude=signal_amplitude,
            noise_amplitude=noise_amplitude,
            sigma_tx=float(np.sqrt(var_tx)),
            sigma_jitter=float(np.sqrt(var_jitter)),
            sigma_noise=float(np.sqrt(var_noise)),
            sigma_gaussian=float(np.sqrt(var_gaussian)),
            voltage_grid=y,
            noise_pmf=p_total,
        )


def _gaussian_variances(
    cursor_value: float,
    slopes: npt.NDArray[np.float64],
    rx_response: npt.NDArray[np.complex128],
    params: ComParams,
) -> tuple[float, float, float]:
    """(93A-30)/(93A-31)/(93A-35): the three variances that sum to varG
    (93A-41) — transmitter noise, random jitter projected to amplitude
    through the pulse response's local slopes, and receiver-referred
    noise integrated through the CTLE and Rx AFE.

    Args:
        cursor_value: The pulse response's cursor sample, volts.
        slopes: Local slopes at each valid UI, V/UI — see
            PulseResponse.local_slopes.
        rx_response: CTLE * Rx AFE, evaluated on the frequency grid
            `params.freq_step` describes.
        params: The rest of what these formulas need.

    Returns:
        (var_tx, var_jitter, var_noise), volts^2.
    """
    var_tx = cursor_value**2 * 10 ** (-params.snr_tx / 10)  # (93A-30)
    var_jitter = params.sigma_rj**2 * params.level_variance * float((slopes**2).sum())  # (93A-31)
    # DC excluded; eta_0 is per GHz while freq_step is in Hz, hence the
    # 1e9 — both following PyChOpMarg's own varN computation.
    var_noise = (
        params.eta_0 * float((np.abs(rx_response[1:]) ** 2).sum()) * (params.freq_step / 1e9)
    )  # (93A-35)
    return var_tx, var_jitter, var_noise
