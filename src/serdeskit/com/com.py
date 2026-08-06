"""compute(): the Channel Operating Margin calculation (IEEE 802.3-2022
Annex 93A), orchestrating the pieces the other packages provide.

Deliberately a thin, stateless function rather than an object: it takes a
flat LinkComParams (config-sheet-shaped: numbers, file paths, sequences of
either — see that class's own docstring) and does the two things a
config sheet can't do for itself:

1. Builds every Channel/Ctle/Ffe/... object LinkComParams' raw fields
   describe (each stage's own class, chosen here — this project doesn't
   yet have more than one implementation per stage kind to choose
   between).
2. Composes them into a Link and runs the actual (93A) calculation.

Every formula it uses already lives in link/, pulse_response/, or pmf/,
each golden-tested there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.com.params import LinkComParams
from serdeskit.common.types import Signal
from serdeskit.crosstalk import worst_case_phase_samples
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Channel, Ctle, Ffe, Link, RxAfe, RxFfe, SystemGrid, TxFilter
from serdeskit.link.com_link_with_cache import com_link_with_cache
from serdeskit.package import Package, cascade_channel
from serdeskit.pmf import (
    combine_pmfs,
    delta_pmf,
    filter_samples,
    gaussian_pmf,
    noise_margin,
    voltage_grid,
)
from serdeskit.pulse_response import PulseResponse
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe
from serdeskit.tx_filter import TxRisetimeFilter


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
    half_signal_unequalized_pulse_response: Signal
    half_signal_equalized_pulse_response: Signal


def compute(params: LinkComParams) -> ComResult:
    """Builds a Link from `params`' raw fields, then runs the (93A)
    calculation on it via `Link.sbr_pulse_response()` (MATLAB COM3.70's
    own real time-domain chain: truncate, `Ctle.process()`, box-car,
    `Ffe.process()`, `RxFfe.process()`) — matches MATLAB's real
    `eq_pulse_response` to floating-point precision (see docs/known-
    issues.md's "full composed pulse response" entry).

    NEXT/FEXT aggressors: each gets the shared ctle/tx_filter/rx_afe/
    rx_ffe (receiver-side, common to every signal arriving at the
    victim's Rx) plus a Tx FFE that depends on aggressor kind — NEXT gets
    a flat, unequalized FFE (its neighbor's own Tx FFE isn't known), FEXT
    reuses the victim's own tap weights (same link, assumed same
    equalization). Same rule PyChOpMarg's gen_pulse_resps applies.

    Returns:
        The COM value in dB, with the intermediate quantities that
        produced it.
    """
    grid = SystemGrid.build(params.baud_rate, params.freq_step, params.samples_per_ui)
    tap_delay = 1.0 / params.baud_rate

    tx_package = Package(
        r0=params.tx_r0,
        die_capacitances=params.tx_die_capacitances,
        die_inductances=params.tx_die_inductances,
        bump_capacitance=params.tx_bump_capacitance,
        tline_a1=params.tx_tline_a1,
        tline_a2=params.tx_tline_a2,
        tline_tau=params.tx_tline_tau,
        tline_gamma0=params.tx_tline_gamma0,
        tline_segments=params.tx_tline_segments,
        pad_capacitance=params.tx_pad_capacitance,
        is_rx=False,
    )
    rx_package = Package(
        r0=params.rx_r0,
        die_capacitances=params.rx_die_capacitances,
        die_inductances=params.rx_die_inductances,
        bump_capacitance=params.rx_bump_capacitance,
        tline_a1=params.rx_tline_a1,
        tline_a2=params.rx_tline_a2,
        tline_tau=params.rx_tline_tau,
        tline_gamma0=params.rx_tline_gamma0,
        tline_segments=params.rx_tline_segments,
        pad_capacitance=params.rx_pad_capacitance,
        is_rx=True,
    )

    def load_channel_with_pkg_model(path: str) -> Channel:
        raw = differential_network(skrf.Network(path), port_order=params.port_order)
        cascaded = cascade_channel(raw, tx_package, rx_package, grid.f)
        return SParameterChannel(cascaded, gamma1=params.gamma1, gamma2=params.gamma2)

    channel = load_channel_with_pkg_model(params.channel_path)
    next_channels = [load_channel_with_pkg_model(path) for path in params.next_channel_paths]
    fext_channels = [load_channel_with_pkg_model(path) for path in params.fext_channel_paths]

    ctle: Ctle = TwoStageCtle(
        zero_freq=params.ctle_zero_freq,
        pole1_freq=params.ctle_pole1_freq,
        pole2_freq=params.ctle_pole2_freq,
        shelf_freq=params.ctle_shelf_freq,
        dc_gain_db=params.ctle_dc_gain_db,
        shelf_gain_db=params.ctle_shelf_gain_db,
    )
    ffe: Ffe = TapWeightFfe(tap_weights=params.ffe_tap_weights, n_post=params.ffe_n_post, tap_delay=tap_delay)
    tx_filter: TxFilter = TxRisetimeFilter(risetime=params.tx_risetime)
    rx_afe: RxAfe = RxAfeButterworth(cutoff_freq=params.rx_afe_cutoff_freq)
    rx_ffe: RxFfe = TapWeightRxFfe(
        tap_weights=params.rx_ffe_tap_weights, tap_delay=tap_delay, n_pre=params.rx_ffe_n_pre
    )

    link = Link(channel=channel, ctle=ctle, ffe=ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe)
    unequalized_impulse_signal = link.unequalized_impulse_response(grid, params.victim_amplitude)
    cached_link = com_link_with_cache(link=link, unequalized_impulse_signal=unequalized_impulse_signal)
    equalized_pulse_signal = cached_link.sbr_pulse_response(grid)
    pulse_response = PulseResponse.from_signal(
        equalized_pulse_signal,
        ui=1.0 / params.baud_rate,
        dfe1_max=float(params.dfe_max[0]),
        dfe1_min=float(params.dfe_min[0]),
    )
    signal_amplitude = pulse_response.signal_amplitude(params.rlm, params.levels)
    y = voltage_grid(signal_amplitude)

    slopes = pulse_response.local_slopes(signal_amplitude)
    p_jitter = delta_pmf(
        filter_samples(params.a_dd * slopes, 1.1 * signal_amplitude), params.levels, y
    )

    rx_response = ctle.transfer_function(grid.f) * rx_afe.transfer_function(grid.f) * rx_ffe.transfer_function(grid.f)
    var_tx, var_jitter, var_noise = _gaussian_variances(pulse_response.cursor_value, slopes, rx_response, params)
    var_gaussian = var_tx + var_jitter + var_noise  # (93A-41)
    p_gaussian = gaussian_pmf(var_gaussian, y)

    residual = pulse_response.residual_isi(params.dfe_min, params.dfe_max)  # (93A-26)/(93A-27)
    p_isi = delta_pmf(filter_samples(residual, 1.1 * signal_amplitude), params.levels, y)
    var_isi = params.level_variance * float((residual**2).sum())  # (93A-31), applied to residual ISI

    flat_ffe = TapWeightFfe(tap_weights=np.array([]), n_post=0, tap_delay=tap_delay)
    aggressor_links = [
        Link(channel=aggressor, ctle=ctle, ffe=flat_ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe)
        for aggressor in next_channels
    ] + [
        Link(channel=aggressor, ctle=ctle, ffe=ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe)
        for aggressor in fext_channels
    ]
    aggressor_amplitudes = [params.a_ne] * len(next_channels) + [params.a_fe] * len(fext_channels)
    p_aggressors = [
        delta_pmf(
            filter_samples(
                worst_case_phase_samples(
                    aggressor_link.sbr_pulse_response(grid).scale(amplitude).samples,
                    params.samples_per_ui,
                ),
                1.1 * signal_amplitude,
            ),
            params.levels,
            y,
        )
        for aggressor_link, amplitude in zip(aggressor_links, aggressor_amplitudes, strict=True)
    ]

    p_total = combine_pmfs(p_gaussian, p_jitter, p_isi)  # (93A-42)/(93A-43)
    var_crosstalk = 0.0
    if p_aggressors:
        p_crosstalk = combine_pmfs(*p_aggressors)  # (93A-44): pXT
        var_crosstalk = float((y**2 * p_crosstalk).sum())
        p_total = combine_pmfs(p_total, p_crosstalk)  # (93A-45)

    noise_amplitude = noise_margin(p_total, y, params.der_0)

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
        half_signal_unequalized_pulse_response=grid.box_car_integrate(unequalized_impulse_signal).scale(1/(params.levels - 1)),
        half_signal_equalized_pulse_response=equalized_pulse_signal.scale(1/(params.levels - 1)),
    )


def _gaussian_variances(
    cursor_value: float,
    slopes: npt.NDArray[np.float64],
    rx_response: npt.NDArray[np.complex128],
    params: LinkComParams,
) -> tuple[float, float, float]:
    """(93A-30)/(93A-31)/(93A-35): the three variances that sum to varG
    (93A-41) — transmitter noise, random jitter projected to amplitude
    through the pulse response's local slopes, and receiver-referred
    noise integrated through the CTLE and Rx AFE.

    Args:
        cursor_value: The pulse response's cursor sample, volts.
        slopes: Local slopes at each valid UI, V/UI — see
            PulseResponse.local_slopes.
        rx_response: CTLE * Rx AFE * Rx FFE, evaluated on the frequency
            grid `params.freq_step` describes.
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
