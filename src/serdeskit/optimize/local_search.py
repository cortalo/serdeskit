"""local_search(): a coordinate-descent alternative to search()'s
exhaustive grid search.

MATLAB COM3.70's own search (`param.LOCAL_SEARCH`, its own comment:
"Decreases COM compute time... if 0 search is full grid") skips
evaluating the vast majority of the (CTLE gain x Tx tap) grid by only
exploring within a small step radius of the current best point, moving
that point whenever a neighbor improves on it -- not a faster way to
evaluate the same grid, a fundamentally smaller one. This is an
independent implementation of that same idea (not a line-for-line port
of MATLAB's own single-pass, loop-order-dependent funnel): classic
iterative coordinate descent -- probe every dimension's neighbors
(CTLE dc_gain, CTLE shelf_gain, and each Tx tap) around the current
best, move to the best improving neighbor found in that dimension, and
repeat full passes until no dimension has one left within
`step_radius`.

Cheaper than search()'s own exhaustive grid at the same tradeoff
MATLAB's own docs accept: it can settle on a local, not global, optimum.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import skrf
from tqdm import tqdm

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.com import LinkComParams
from serdeskit.common.types import Signal
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Channel, Link, SystemGrid
from serdeskit.link.com_link_with_cache import ComLinkWithCache
from serdeskit.optimize.figure_of_merit import figure_of_merit
from serdeskit.optimize.search import _link_com_params
from serdeskit.optimize.standard import ComStandard
from serdeskit.package import Package, cascade_channel
from serdeskit.pulse_response import PulseResponse
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe
from serdeskit.tx_filter import TxRisetimeFilter


def local_search(
    standard: ComStandard,
    thru_path: str,
    next_paths: Sequence[str],
    fext_paths: Sequence[str],
    port_order: Sequence[int],
    show_progress: bool,
    step_radius: int = 2,
) -> LinkComParams:
    """Coordinate descent over `standard`'s CTLE-gain x Tx-tap search
    space, returning a LinkComParams for the best point found.

    Args:
        standard: Same as search()'s own.
        thru_path: The victim's raw THRU channel, as a Touchstone path.
        next_paths: Zero or more NEXT aggressor channel files.
        fext_paths: Zero or more FEXT aggressor channel files.
        port_order: Forwarded to `differential_network` for every channel
            loaded.
        show_progress: Show a tqdm progress bar (evaluation count) --
            unlike search()'s own, the total isn't known upfront (it
            depends on how many passes convergence takes).
        step_radius: How many index steps, in either direction, a
            dimension may move in one pass -- MATLAB's own C2C config
            uses 2 (`param.LOCAL_SEARCH`).

    Returns:
        A LinkComParams naming `thru_path`/`next_paths`/`fext_paths`
        directly and carrying the winning CTLE gains and Tx FFE taps.
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

    def unequalized_impulse(ch: Channel, amplitude: float) -> Signal:
        return Link(channel=ch, tx_filter=tx_filter, rx_afe=rx_afe).unequalized_impulse_response(grid, amplitude)

    victim_impulse = unequalized_impulse(channel, standard.victim_amplitude)
    next_impulses = [unequalized_impulse(nc, standard.a_ne) for nc in next_channels]
    fext_impulses = [unequalized_impulse(fc, standard.a_fe) for fc in fext_channels]

    flat_ffe = TapWeightFfe(
        tap_weights=np.zeros(len(standard.tx_taps_bounds)), n_post=standard.tx_taps_n_post, tap_delay=tap_delay
    )

    # Per-tap candidate value arrays -- same construction tx_tap_combinations
    # uses internally (a step of 0 locks that position to 0.0), but kept
    # per-tap here for coordinate-wise stepping instead of a flat cross
    # product.
    tap_value_arrays = [
        np.arange(lo, hi + step, step) if step else np.array([0.0]) for lo, hi, step in standard.tx_taps_bounds
    ]
    dc_gain_values = list(standard.ctle_dc_gain_candidates)
    shelf_gain_values = list(standard.ctle_shelf_gain_candidates)
    dim_sizes = [len(dc_gain_values), len(shelf_gain_values)] + [len(arr) for arr in tap_value_arrays]

    # (dc_gain_index, shelf_gain_index)-keyed cache: everything that only
    # depends on ctle, not the victim's own tx taps (rx_response, NEXT
    # aggressors' pulse responses) -- built on first use, reused for every
    # tap-dimension probe at that same CTLE setting.
    ctle_cache: dict[
        tuple[int, int], tuple[TwoStageCtle, npt.NDArray[np.complex128], list[npt.NDArray[np.float64]]]
    ] = {}

    def ctle_context(
        dc_ix: int, shelf_ix: int
    ) -> tuple[TwoStageCtle, npt.NDArray[np.complex128], list[npt.NDArray[np.float64]]]:
        key = (dc_ix, shelf_ix)
        if key not in ctle_cache:
            ctle = TwoStageCtle(
                zero_freq=standard.ctle_zero_freq,
                pole1_freq=standard.ctle_pole1_freq,
                pole2_freq=standard.ctle_pole2_freq,
                shelf_freq=standard.ctle_shelf_freq,
                dc_gain_db=dc_gain_values[dc_ix],
                shelf_gain_db=shelf_gain_values[shelf_ix],
            )
            rx_response = (
                ctle.transfer_function(grid.f) * rx_afe.transfer_function(grid.f) * rx_ffe.transfer_function(grid.f)
            )
            next_prs = [
                ComLinkWithCache(
                    link=Link(channel=nc, ctle=ctle, ffe=flat_ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe),
                    unequalized_impulse_signal=next_impulse,
                )
                .sbr_pulse_response(grid)
                .samples
                for nc, next_impulse in zip(next_channels, next_impulses, strict=True)
            ]
            ctle_cache[key] = (ctle, rx_response, next_prs)
        return ctle_cache[key]

    evaluated: dict[tuple[int, ...], float] = {}

    def evaluate(state: tuple[int, ...]) -> float:
        if state in evaluated:
            return evaluated[state]

        dc_ix, shelf_ix, *tap_ixs = state
        tx_taps = np.array([tap_value_arrays[i][tap_ixs[i]] for i in range(len(tap_ixs))])
        if (1 - np.abs(tx_taps).sum()) < standard.tx_taps_c0_min:
            evaluated[state] = -np.inf
            return -np.inf

        ctle, rx_response, next_prs = ctle_context(dc_ix, shelf_ix)
        ffe = TapWeightFfe(tap_weights=tx_taps, n_post=standard.tx_taps_n_post, tap_delay=tap_delay)
        cached_link = ComLinkWithCache(
            link=Link(channel=channel, ctle=ctle, ffe=ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe),
            unequalized_impulse_signal=victim_impulse,
        )
        signal = cached_link.sbr_pulse_response(grid)
        pulse_response = PulseResponse.from_signal(
            signal,
            ui=1.0 / standard.baud_rate,
            dfe1_max=float(standard.dfe_max[0]),
            dfe1_min=float(standard.dfe_min[0]),
        )
        fext_prs = [
            ComLinkWithCache(
                link=Link(channel=fc, ctle=ctle, ffe=ffe, tx_filter=tx_filter, rx_afe=rx_afe, rx_ffe=rx_ffe),
                unequalized_impulse_signal=fext_impulse,
            )
            .sbr_pulse_response(grid)
            .samples
            for fc, fext_impulse in zip(fext_channels, fext_impulses, strict=True)
        ]
        fom = figure_of_merit(pulse_response, next_prs + fext_prs, rx_response, standard)
        evaluated[state] = fom
        return fom

    # Start at each CTLE candidate list's own midpoint (a reasonable,
    # arbitrary point to descend from) and all-zero Tx taps -- always a
    # valid combination, since the cursor is 1.0 when every tap is 0.
    zero_tap_ixs = tuple(int(np.argmin(np.abs(arr))) for arr in tap_value_arrays)
    state = (len(dc_gain_values) // 2, len(shelf_gain_values) // 2) + zero_tap_ixs
    best_fom = evaluate(state)

    progress = tqdm(disable=not show_progress)
    improved = True
    while improved:
        improved = False
        for dim in range(len(dim_sizes)):
            best_neighbor: tuple[int, ...] | None = None
            for delta in range(-step_radius, step_radius + 1):
                if delta == 0:
                    continue
                new_val = state[dim] + delta
                if not (0 <= new_val < dim_sizes[dim]):
                    continue
                candidate = state[:dim] + (new_val,) + state[dim + 1 :]
                fom = evaluate(candidate)
                progress.update(1)
                if fom > best_fom:
                    best_fom = fom
                    best_neighbor = candidate
            if best_neighbor is not None:
                state = best_neighbor
                improved = True
    progress.close()

    dc_ix, shelf_ix, *tap_ixs = state
    tx_taps = np.array([tap_value_arrays[i][tap_ixs[i]] for i in range(len(tap_ixs))])
    return _link_com_params(
        standard, thru_path, next_paths, fext_paths, port_order,
        dc_gain_values[dc_ix], shelf_gain_values[shelf_ix], tx_taps,
    )
