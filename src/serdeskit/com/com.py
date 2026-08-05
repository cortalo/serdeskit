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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from serdeskit.com.params import ComParams
from serdeskit.crosstalk import worst_case_phase_samples
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Channel, Link, SystemGrid
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
    sigma_isi: float  # sqrt(varISI): (93A-31)'s variance formula applied to residual ISI
    sigma_crosstalk: float  # sqrt(varXT): the combined crosstalk PMF's own variance, sum(y^2 * p)
    voltage_grid: npt.NDArray[np.float64]  # y (V) the PMFs below are on
    noise_pmf: npt.NDArray[np.float64]  # the combined interference+noise PMF, (93A-45)


@dataclass
class Com:
    """next_channels/fext_channels: one Channel per crosstalk aggressor.
    Com builds each aggressor's Link itself, from `self.link`'s own
    ctle/rx_afe — those are receiver-side and shared by every signal
    arriving at the victim's Rx, aggressor or not — plus a Tx FFE that
    depends on the aggressor kind: NEXT gets a flat, unequalized FFE (its
    neighbor's own Tx FFE isn't known), FEXT reuses the victim's own tap
    weights (same link, assumed same equalization). Same rule
    PyChOpMarg's gen_pulse_resps applies. A caller only supplies the
    channel; it doesn't reconstruct this rule itself.
    """

    link: Link
    params: ComParams
    next_channels: Sequence[Channel] = ()
    fext_channels: Sequence[Channel] = ()

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
        #
        # `1.1 * signal_amplitude` here (and below, for ISI and each
        # aggressor) matches PyChOpMarg's calc_noise, which filters all
        # three against `ymax` — voltage_grid's own half-width — rather
        # than signal_amplitude itself. Unlike local_slopes just above,
        # whose validity threshold PyChOpMarg's calc_hJ applies against
        # signal_amplitude directly.
        slopes = pulse_response.local_slopes(signal_amplitude)
        p_jitter = delta_pmf(
            filter_samples(p.a_dd * slopes, 1.1 * signal_amplitude), p.levels, y
        )

        rx_response = (
            self.link.ctle.transfer_function(grid.f)  # type: ignore[union-attr]
            * self.link.rx_afe.transfer_function(grid.f)  # type: ignore[union-attr]
            * self.link.rx_ffe.transfer_function(grid.f)  # type: ignore[union-attr]
        )
        var_tx, var_jitter, var_noise = _gaussian_variances(
            pulse_response.cursor_value, slopes, rx_response, p
        )
        var_gaussian = var_tx + var_jitter + var_noise  # (93A-41)
        p_gaussian = gaussian_pmf(var_gaussian, y)

        residual = pulse_response.residual_isi(p.dfe_min, p.dfe_max)  # (93A-26)/(93A-27)
        p_isi = delta_pmf(filter_samples(residual, 1.1 * signal_amplitude), p.levels, y)
        var_isi = p.level_variance * float((residual**2).sum())  # (93A-31), applied to residual ISI

        # Each aggressor: scale to its launch amplitude (A_ne/A_fe — same
        # rule as the victim's A_v), pick its worst-case sub-UI phase
        # (93A-33), then the same delta_pmf treatment ISI/jitter get.
        aggressors = [(link, p.a_ne) for link in self._next_links()] + [
            (link, p.a_fe) for link in self._fext_links()
        ]
        p_aggressors = [
            delta_pmf(
                filter_samples(
                    worst_case_phase_samples(
                        link.ffe_channel_ctle_pulse_response(grid).scale(amplitude).samples,
                        p.samples_per_ui,
                    ),
                    1.1 * signal_amplitude,
                ),
                p.levels,
                y,
            )
            for link, amplitude in aggressors
        ]

        # Aggressors combined among themselves before joining the rest,
        # not folded in one at a time: np.convolve(..., mode="same")
        # truncates whenever a combined distribution outgrows the shared
        # voltage grid (it does here — p_isi alone is wide enough that
        # folding it and each aggressor together one at a time loses mass
        # differently than combining the aggressors as their own group
        # first, since truncation depends on what's already been folded
        # into the array at each step — matches PyChOpMarg's own grouping
        # exactly. combine_pmfs's renormalization timing doesn't matter on
        # its own (a scalar divide commutes with a subsequent same-mode
        # convolution+crop) — only this grouping does.
        p_total = combine_pmfs(p_gaussian, p_jitter, p_isi)  # (93A-42)/(93A-43)
        var_crosstalk = 0.0
        if p_aggressors:
            p_crosstalk = combine_pmfs(*p_aggressors)  # (93A-44): pXT
            var_crosstalk = float((y**2 * p_crosstalk).sum())
            p_total = combine_pmfs(p_total, p_crosstalk)  # (93A-45)

        noise_amplitude = noise_margin(p_total, y, p.der_0)

        return ComResult(
            com_db=float(20 * np.log10(signal_amplitude / noise_amplitude)),
            signal_amplitude=signal_amplitude,
            noise_amplitude=noise_amplitude,
            sigma_tx=float(np.sqrt(var_tx)),
            sigma_jitter=float(np.sqrt(var_jitter)),
            sigma_noise=float(np.sqrt(var_noise)),
            sigma_gaussian=float(np.sqrt(var_gaussian)),
            sigma_isi=float(np.sqrt(var_isi)),
            sigma_crosstalk=float(np.sqrt(var_crosstalk)),
            voltage_grid=y,
            noise_pmf=p_total,
        )

    def compute_sbr(self) -> ComResult:
        """Same calculation as `compute()`, but building the pulse
        response via `Link.sbr_pulse_response()` (MATLAB's own real
        time-domain chain: truncate, `Ctle.process()`, box-car,
        `Ffe.process()`, `RxFfe.process()`) instead of `Link.
        ffe_channel_ctle_pulse_response()` (frequency-domain composition,
        one IFFT) — the latter has a known, unexplained ~1-2% residual
        against MATLAB; `sbr_pulse_response()` matches MATLAB's real
        `eq_pulse_response` to floating-point precision (see
        docs/known-issues.md's "full composed pulse response" entry).

        A new, separate method rather than a change to `compute()`
        itself: `compute()` is exercised by tests comparing against
        PyChOpMarg (a different reference, whose own methodology doesn't
        do this truncate-then-time-domain-chain composition at all), not
        just this project's own MATLAB-alignment tests — switching it in
        place would conflate the two.

        Everything past pulse-response construction (the noise integral,
        PMF construction, `noise_margin`) is unchanged: those formulas
        are frequency-domain regardless of which pulse response feeds
        the cursor/residual-ISI/local-slope calculations upstream of them.

        Returns:
            The COM value in dB, with the intermediate quantities that
            produced it.
        """
        p = self.params
        grid = SystemGrid.build(p.baud_rate, p.freq_step, p.samples_per_ui)

        signal = self.link.sbr_pulse_response(grid).scale(p.victim_amplitude)
        pulse_response = PulseResponse.from_signal(
            signal,
            ui=1.0 / p.baud_rate,
            dfe1_max=float(p.dfe_max[0]),
            dfe1_min=float(p.dfe_min[0]),
        )
        signal_amplitude = pulse_response.signal_amplitude(p.rlm, p.levels)
        y = voltage_grid(signal_amplitude)

        slopes = pulse_response.local_slopes(signal_amplitude)
        p_jitter = delta_pmf(
            filter_samples(p.a_dd * slopes, 1.1 * signal_amplitude), p.levels, y
        )

        rx_response = (
            self.link.ctle.transfer_function(grid.f)  # type: ignore[union-attr]
            * self.link.rx_afe.transfer_function(grid.f)  # type: ignore[union-attr]
            * self.link.rx_ffe.transfer_function(grid.f)  # type: ignore[union-attr]
        )
        var_tx, var_jitter, var_noise = _gaussian_variances(
            pulse_response.cursor_value, slopes, rx_response, p
        )
        var_gaussian = var_tx + var_jitter + var_noise  # (93A-41)
        p_gaussian = gaussian_pmf(var_gaussian, y)

        residual = pulse_response.residual_isi(p.dfe_min, p.dfe_max)  # (93A-26)/(93A-27)
        p_isi = delta_pmf(filter_samples(residual, 1.1 * signal_amplitude), p.levels, y)
        var_isi = p.level_variance * float((residual**2).sum())  # (93A-31), applied to residual ISI

        aggressors = [(link, p.a_ne) for link in self._next_links()] + [
            (link, p.a_fe) for link in self._fext_links()
        ]
        p_aggressors = [
            delta_pmf(
                filter_samples(
                    worst_case_phase_samples(
                        link.sbr_pulse_response(grid).scale(amplitude).samples,
                        p.samples_per_ui,
                    ),
                    1.1 * signal_amplitude,
                ),
                p.levels,
                y,
            )
            for link, amplitude in aggressors
        ]

        p_total = combine_pmfs(p_gaussian, p_jitter, p_isi)  # (93A-42)/(93A-43)
        var_crosstalk = 0.0
        if p_aggressors:
            p_crosstalk = combine_pmfs(*p_aggressors)  # (93A-44): pXT
            var_crosstalk = float((y**2 * p_crosstalk).sum())
            p_total = combine_pmfs(p_total, p_crosstalk)  # (93A-45)

        noise_amplitude = noise_margin(p_total, y, p.der_0)

        return ComResult(
            com_db=float(20 * np.log10(signal_amplitude / noise_amplitude)),
            signal_amplitude=signal_amplitude,
            noise_amplitude=noise_amplitude,
            sigma_tx=float(np.sqrt(var_tx)),
            sigma_jitter=float(np.sqrt(var_jitter)),
            sigma_noise=float(np.sqrt(var_noise)),
            sigma_gaussian=float(np.sqrt(var_gaussian)),
            sigma_isi=float(np.sqrt(var_isi)),
            sigma_crosstalk=float(np.sqrt(var_crosstalk)),
            voltage_grid=y,
            noise_pmf=p_total,
        )

    def _next_links(self) -> list[Link]:
        flat_ffe = _flat_ffe(1.0 / self.params.baud_rate)
        return [
            Link(
                channel=channel,
                ctle=self.link.ctle,
                ffe=flat_ffe,
                tx_filter=self.link.tx_filter,
                rx_afe=self.link.rx_afe,
                rx_ffe=self.link.rx_ffe,
            )
            for channel in self.next_channels
        ]

    def _fext_links(self) -> list[Link]:
        return [
            Link(
                channel=channel,
                ctle=self.link.ctle,
                ffe=self.link.ffe,
                tx_filter=self.link.tx_filter,
                rx_afe=self.link.rx_afe,
                rx_ffe=self.link.rx_ffe,
            )
            for channel in self.fext_channels
        ]


def _flat_ffe(tap_delay: float) -> TapWeightFfe:
    """A flat, unequalized Tx FFE — the reference NEXT aggressors are
    assumed to use, per PyChOpMarg's gen_pulse_resps (`tx_ix=0`, the
    all-zero tap combination). Empty tap_weights make TapWeightFfe.
    cursor_weight 1.0 with no other taps, so its transfer function is
    unity at every frequency regardless of tap_delay — passed through
    only for constructor completeness.
    """
    return TapWeightFfe(tap_weights=np.array([]), n_post=0, tap_delay=tap_delay)


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
