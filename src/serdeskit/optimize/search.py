"""search(): PyChOpMarg's opt_eq (PRZF mode, no Rx FFE) — a Tx-tap-
combination x CTLE-gain grid search maximizing figure_of_merit, plus the
channel loading PyChOpMarg's own COM.__call__ does first (given raw
Touchstone paths and a ComStandard, not pre-built Channel objects, since
this is the actual top-level search entry point).

Rx FFE is fixed at the identity (single unity tap): this project's search
doesn't optimize it yet (PyChOpMarg's own przf isn't implemented here).

Mirrors com.compute()'s own shape: takes a flat, plain-data config
(ComStandard) and raw file paths, builds every Channel/Ctle/Ffe/... object
itself, and returns a LinkComParams — the search space resolved to one
point, ready to hand straight to com.compute() for the actual (93A) COM
number, or to inspect directly for the winning equalization settings.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import skrf
from tqdm import tqdm

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.com import LinkComParams
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Channel, Link, SystemGrid
from serdeskit.optimize.figure_of_merit import figure_of_merit
from serdeskit.optimize.standard import ComStandard
from serdeskit.optimize.tap_combinations import tx_tap_combinations
from serdeskit.package import Package, cascade_channel
from serdeskit.pulse_response import PulseResponse
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe
from serdeskit.tx_filter import TxRisetimeFilter


def search(
    standard: ComStandard,
    thru_path: str,
    next_paths: Sequence[str],
    fext_paths: Sequence[str],
    port_order: Sequence[int],
    show_progress: bool,
) -> LinkComParams:
    """Runs the full grid over `standard`'s CTLE-gain x Tx-tap search
    space, returning a LinkComParams for the combination with the largest
    figure_of_merit — never empty, since the all-zero Tx tap combination
    (flat, unequalized) is always included, matching PyChOpMarg's own
    com._tx_combs[0].

    Args:
        standard: Every parameter the package models, the search space,
            and the eventual COM computation need.
        thru_path: The victim's raw THRU channel, as a Touchstone path.
        next_paths: Zero or more NEXT aggressor channel files.
        fext_paths: Zero or more FEXT aggressor channel files.
        port_order: Forwarded to `differential_network` for every channel
            loaded (victim and aggressors alike — same file family is
            assumed throughout).
        show_progress: Show a tqdm progress bar (rate, elapsed, ETA) over
            the grid — a slow real search (e.g. a full standard's grid
            against a real channel) has no other visible progress
            otherwise.

    Returns:
        A LinkComParams naming `thru_path`/`next_paths`/`fext_paths`
        directly (not pre-loaded Channels) and carrying the winning CTLE
        gains and Tx FFE taps -- pass it to com.compute() for the COM
        number.
    """
    tx_package = Package(
        r0=standard.r0,
        die_capacitances=standard.tx_die_capacitances,
        die_inductances=standard.tx_die_inductances,
        bump_capacitance=standard.tx_bump_capacitance,
        tline_a1=standard.tx_tline_a1,
        tline_a2=standard.tx_tline_a2,
        tline_tau=standard.tx_tline_tau,
        tline_gamma0=standard.tx_tline_gamma0,
        tline_segments=standard.tx_tline_segments,
        pad_capacitance=standard.tx_pad_capacitance,
        is_rx=False,
    )
    rx_package = Package(
        r0=standard.r0,
        die_capacitances=standard.rx_die_capacitances,
        die_inductances=standard.rx_die_inductances,
        bump_capacitance=standard.rx_bump_capacitance,
        tline_a1=standard.rx_tline_a1,
        tline_a2=standard.rx_tline_a2,
        tline_tau=standard.rx_tline_tau,
        tline_gamma0=standard.rx_tline_gamma0,
        tline_segments=standard.rx_tline_segments,
        pad_capacitance=standard.rx_pad_capacitance,
        is_rx=True,
    )
    grid = SystemGrid.build(standard.baud_rate, standard.freq_step, standard.samples_per_ui)
    tap_delay = 1.0 / standard.baud_rate

    def load_channel(path: str) -> Channel:
        raw = differential_network(skrf.Network(path), port_order=port_order)
        cascaded = cascade_channel(raw, tx_package, rx_package, grid.f)
        return SParameterChannel(cascaded, gamma1=standard.gamma1, gamma2=standard.gamma2)

    channel = load_channel(thru_path)
    next_channels = [load_channel(path) for path in next_paths]
    fext_channels = [load_channel(path) for path in fext_paths]

    tx_filter = TxRisetimeFilter(risetime=standard.tx_risetime)
    rx_afe = RxAfeButterworth(cutoff_freq=standard.rx_afe_cutoff_freq)
    rx_ffe = TapWeightRxFfe(tap_weights=np.array([1.0]), tap_delay=tap_delay)

    all_zero = np.zeros(len(standard.tx_taps_bounds))
    candidates = [all_zero, *tx_tap_combinations(standard.tx_taps_bounds, standard.tx_taps_c0_min)]
    total = len(standard.ctle_shelf_gain_candidates) * len(standard.ctle_dc_gain_candidates) * len(candidates)
    progress = tqdm(total=total, disable=not show_progress)

    # NEXT aggressors always use this same flat, unequalized Tx FFE
    # (never the victim's own candidate taps) — but their pulse
    # responses still depend on the current CTLE gain, so they're
    # rebuilt once per (dc_gain_db, shelf_gain_db) pair below, not per
    # Tx-tap candidate.
    flat_ffe = TapWeightFfe(
        tap_weights=np.zeros(len(standard.tx_taps_bounds)), n_post=standard.tx_taps_n_post, tap_delay=tap_delay
    )

    best: LinkComParams | None = None
    best_fom = -np.inf
    for shelf_gain_db in standard.ctle_shelf_gain_candidates:
        for dc_gain_db in standard.ctle_dc_gain_candidates:
            ctle = TwoStageCtle(
                zero_freq=standard.ctle_zero_freq,
                pole1_freq=standard.ctle_pole1_freq,
                pole2_freq=standard.ctle_pole2_freq,
                shelf_freq=standard.ctle_shelf_freq,
                dc_gain_db=dc_gain_db,
                shelf_gain_db=shelf_gain_db,
            )
            rx_response = (
                ctle.transfer_function(grid.f) * rx_afe.transfer_function(grid.f) * rx_ffe.transfer_function(grid.f)
            )
            next_prs = [
                Link(channel=nc, ctle=ctle, ffe=flat_ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe)
                .sbr_pulse_response(grid)
                .scale(standard.a_ne)
                .samples
                for nc in next_channels
            ]

            for tx_taps in candidates:
                ffe = TapWeightFfe(tap_weights=tx_taps, n_post=standard.tx_taps_n_post, tap_delay=tap_delay)
                link = Link(channel=channel, ctle=ctle, ffe=ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe)
                signal = link.sbr_pulse_response(grid).scale(standard.victim_amplitude)
                pulse_response = PulseResponse.from_signal(
                    signal,
                    ui=1.0 / standard.baud_rate,
                    dfe1_max=float(standard.dfe_max[0]),
                    dfe1_min=float(standard.dfe_min[0]),
                )
                fext_prs = [
                    Link(channel=fc, ctle=ctle, ffe=ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe)
                    .sbr_pulse_response(grid)
                    .scale(standard.a_fe)
                    .samples
                    for fc in fext_channels
                ]

                fom = figure_of_merit(pulse_response, next_prs + fext_prs, rx_response, standard)
                if fom > best_fom:
                    best_fom = fom
                    best = _link_com_params(
                        standard, thru_path, next_paths, fext_paths, port_order, dc_gain_db, shelf_gain_db, tx_taps
                    )

                progress.update(1)

    progress.close()
    assert best is not None  # candidates always has at least the all-zero entry
    return best


def _link_com_params(
    standard: ComStandard,
    thru_path: str,
    next_paths: Sequence[str],
    fext_paths: Sequence[str],
    port_order: Sequence[int],
    dc_gain_db: float,
    shelf_gain_db: float,
    tx_taps: npt.NDArray[np.float64],
) -> LinkComParams:
    return LinkComParams(
        channel_path=thru_path,
        next_channel_paths=next_paths,
        fext_channel_paths=fext_paths,
        port_order=port_order,
        gamma1=standard.gamma1,
        gamma2=standard.gamma2,
        tx_r0=standard.r0,
        tx_die_capacitances=standard.tx_die_capacitances,
        tx_die_inductances=standard.tx_die_inductances,
        tx_bump_capacitance=standard.tx_bump_capacitance,
        tx_tline_a1=standard.tx_tline_a1,
        tx_tline_a2=standard.tx_tline_a2,
        tx_tline_tau=standard.tx_tline_tau,
        tx_tline_gamma0=standard.tx_tline_gamma0,
        tx_tline_segments=standard.tx_tline_segments,
        tx_pad_capacitance=standard.tx_pad_capacitance,
        rx_r0=standard.r0,
        rx_die_capacitances=standard.rx_die_capacitances,
        rx_die_inductances=standard.rx_die_inductances,
        rx_bump_capacitance=standard.rx_bump_capacitance,
        rx_tline_a1=standard.rx_tline_a1,
        rx_tline_a2=standard.rx_tline_a2,
        rx_tline_tau=standard.rx_tline_tau,
        rx_tline_gamma0=standard.rx_tline_gamma0,
        rx_tline_segments=standard.rx_tline_segments,
        rx_pad_capacitance=standard.rx_pad_capacitance,
        ctle_zero_freq=standard.ctle_zero_freq,
        ctle_pole1_freq=standard.ctle_pole1_freq,
        ctle_pole2_freq=standard.ctle_pole2_freq,
        ctle_shelf_freq=standard.ctle_shelf_freq,
        ctle_dc_gain_db=dc_gain_db,
        ctle_shelf_gain_db=shelf_gain_db,
        ffe_tap_weights=tx_taps,
        ffe_n_post=standard.tx_taps_n_post,
        tx_risetime=standard.tx_risetime,
        rx_afe_cutoff_freq=standard.rx_afe_cutoff_freq,
        rx_ffe_tap_weights=np.array([1.0]),
        rx_ffe_n_pre=0,
        baud_rate=standard.baud_rate,
        freq_step=standard.freq_step,
        samples_per_ui=standard.samples_per_ui,
        levels=standard.levels,
        rlm=standard.rlm,
        victim_amplitude=standard.victim_amplitude,
        a_ne=standard.a_ne,
        a_fe=standard.a_fe,
        snr_tx=standard.snr_tx,
        sigma_rj=standard.sigma_rj,
        eta_0=standard.eta_0,
        a_dd=standard.a_dd,
        der_0=standard.der_0,
        dfe_min=standard.dfe_min,
        dfe_max=standard.dfe_max,
    )
