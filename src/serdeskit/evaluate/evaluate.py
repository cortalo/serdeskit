"""evaluate_channel: PyChOpMarg's COM.__call__ — opt_eq's search first,
then calc_noise (com.Com.compute()) on the winning Link, compared
against ComStandard.com_min_db. Takes raw Touchstone paths, not
pre-built Channel objects: this is the actual top-level entry point
("given a channel's S-parameters and a standard's parameters, does it
pass"), so loading the file, converting it to a differential network,
and cascading it with the Tx/Rx package models `standard` describes all
happen here.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import skrf

from serdeskit.channel import SParameterChannel, differential_network
from serdeskit.com.com import Com, ComResult
from serdeskit.com.params import ComParams
from serdeskit.evaluate.standard import ComStandard
from serdeskit.link import Channel, SystemGrid
from serdeskit.optimize.search import EqualizationSearch
from serdeskit.package import Package, cascade_channel
from serdeskit.rx_afe import RxAfeButterworth
from serdeskit.rx_ffe import TapWeightRxFfe


@dataclass(frozen=True)
class ComEvaluation:
    passes: bool
    com_min_db: float
    tx_taps: npt.NDArray[np.float64]
    dc_gain_db: float
    shelf_gain_db: float
    result: ComResult


def evaluate_channel(
    standard: ComStandard,
    thru_path: Path,
    next_paths: Sequence[Path] = (),
    fext_paths: Sequence[Path] = (),
) -> ComEvaluation:
    """Loads `thru_path` (and any aggressor paths), cascades each with
    `standard`'s Tx/Rx package models (PyChOpMarg's own add_pkg — every
    channel, victim and aggressor alike, gets the same treatment), runs
    the equalization search over `standard`'s search space, then
    computes the full PMF-based COM on the winning Link — the same
    two-step order PyChOpMarg's COM.__call__ follows.

    Rx FFE is fixed at the identity (single unity tap): this project's
    search doesn't optimize it yet (PyChOpMarg's own przf isn't
    implemented here), matching every other golden test's own default.

    Args:
        standard: Every parameter the package models, the search, and
            the final COM computation need.
        thru_path: The victim's raw THRU channel, as a Touchstone file.
        next_paths: Zero or more NEXT aggressor channel files.
        fext_paths: Zero or more FEXT aggressor channel files.

    Returns:
        The winning equalization settings, the full COM result, and
        whether it clears `standard.com_min_db`.
    """
    params = ComParams(
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

    tx_package = Package(
        r0=standard.r0,
        die_capacitances=standard.tx_die_capacitances,
        die_inductances=standard.tx_die_inductances,
        bump_capacitance=standard.tx_bump_capacitance,
        tline_a1=standard.package_tline_a1,
        tline_a2=standard.package_tline_a2,
        tline_tau=standard.package_tline_tau,
        tline_gamma0=standard.package_tline_gamma0,
        tline_segments=standard.package_tline_segments,
        pad_capacitance=standard.tx_pad_capacitance,
        is_rx=False,
    )
    rx_package = Package(
        r0=standard.r0,
        die_capacitances=standard.rx_die_capacitances,
        die_inductances=standard.rx_die_inductances,
        bump_capacitance=standard.rx_bump_capacitance,
        tline_a1=standard.package_tline_a1,
        tline_a2=standard.package_tline_a2,
        tline_tau=standard.package_tline_tau,
        tline_gamma0=standard.package_tline_gamma0,
        tline_segments=standard.package_tline_segments,
        pad_capacitance=standard.rx_pad_capacitance,
        is_rx=True,
    )
    freqs = SystemGrid.build(standard.baud_rate, standard.freq_step, standard.samples_per_ui).f

    def load_channel(path: Path) -> Channel:
        raw = differential_network(skrf.Network(str(path)))
        cascaded = cascade_channel(raw, tx_package, rx_package, freqs)
        return SParameterChannel(cascaded, gamma1=standard.gamma1, gamma2=standard.gamma2)

    channel = load_channel(thru_path)
    next_channels = [load_channel(path) for path in next_paths]
    fext_channels = [load_channel(path) for path in fext_paths]

    search = EqualizationSearch(
        channel=channel,
        rx_afe=RxAfeButterworth(cutoff_freq=standard.rx_afe_cutoff_freq),
        rx_ffe=TapWeightRxFfe(tap_weights=np.array([1.0]), tap_delay=1.0 / standard.baud_rate),
        zero_freq=standard.ctle_zero_freq,
        pole1_freq=standard.ctle_pole1_freq,
        pole2_freq=standard.ctle_pole2_freq,
        shelf_freq=standard.ctle_shelf_freq,
        dc_gain_candidates=standard.ctle_dc_gain_candidates,
        shelf_gain_candidates=standard.ctle_shelf_gain_candidates,
        tx_taps_bounds=standard.tx_taps_bounds,
        c0_min=standard.tx_taps_c0_min,
        n_post=standard.tx_taps_n_post,
        params=params,
        next_channels=next_channels,
        fext_channels=fext_channels,
    )
    search_result = search.search()

    com = Com(
        link=search_result.link, params=params, next_channels=next_channels, fext_channels=fext_channels,
    )
    result = com.compute()

    return ComEvaluation(
        passes=result.com_db >= standard.com_min_db,
        com_min_db=standard.com_min_db,
        tx_taps=search_result.tx_taps,
        dc_gain_db=search_result.dc_gain_db,
        shelf_gain_db=search_result.shelf_gain_db,
        result=result,
    )
