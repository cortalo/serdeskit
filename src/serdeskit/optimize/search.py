"""EqualizationSearch: PyChOpMarg's opt_eq (PRZF mode, no Rx FFE) — a
Tx-tap-combination x CTLE-gain grid search maximizing figure_of_merit.

Builds a fresh Link per candidate (channel/rx_afe/rx_ffe fixed; CTLE gain
and Tx FFE taps vary) and evaluates figure_of_merit on it, the same way
com.Com evaluates the full PMF-based calc_noise on one already-fixed
Link — this is the layer above that decides which Link to feed it.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from serdeskit.com.params import ComParams
from serdeskit.ctle import TwoStageCtle
from serdeskit.ffe import TapWeightFfe
from serdeskit.link import Channel, Link, RxAfe, RxFfe, SystemGrid
from serdeskit.optimize.figure_of_merit import figure_of_merit
from serdeskit.optimize.tap_combinations import tx_tap_combinations
from serdeskit.pulse_response import PulseResponse


@dataclass(frozen=True)
class SearchResult:
    tx_taps: npt.NDArray[np.float64]
    dc_gain_db: float
    shelf_gain_db: float
    fom: float
    link: Link  # the winning victim Link, ready to hand to com.Com alongside next_channels/fext_channels


@dataclass
class EqualizationSearch:
    """channel/rx_afe/rx_ffe and the CTLE's zero/pole1/pole2/shelf
    frequencies are fixed across the whole search — only the CTLE's two
    gains and the Tx FFE's tap weights vary, one combination per grid
    point. next_channels/fext_channels follow com.Com's own rule: NEXT
    aggressors always get a flat, unequalized Tx FFE (PyChOpMarg forces
    tx_ix=0 for them, unconditionally); FEXT aggressors use whatever Tx
    FFE the current candidate is trying, same as the victim.
    """

    channel: Channel
    rx_afe: RxAfe
    rx_ffe: RxFfe
    zero_freq: float
    pole1_freq: float
    pole2_freq: float
    shelf_freq: float
    dc_gain_candidates: Sequence[float]
    shelf_gain_candidates: Sequence[float]
    tx_taps_bounds: Sequence[tuple[float, float, float]]
    c0_min: float
    n_post: int
    params: ComParams
    next_channels: Sequence[Channel] = ()
    fext_channels: Sequence[Channel] = ()

    def search(self, on_progress: Callable[[int, int], None] | None = None) -> SearchResult:
        """Runs the full grid, returning the combination with the
        largest figure_of_merit.

        Args:
            on_progress: Called as `on_progress(done, total)` after each
                grid point is evaluated, `done` counting from 1 — a slow
                real search (e.g. a full standard's grid against a real
                channel) has no other visible progress otherwise. Kept
                optional, not a logger this class reaches for itself:
                the domain layer stays presentation-free (see this
                package's own module docstring), same reasoning as
                `Link`/`Com` never importing matplotlib — a caller that
                wants a progress bar/print supplies one.

        Returns:
            The winning combination — never empty, since the all-zero
            Tx tap combination (flat, unequalized) is always included,
            matching PyChOpMarg's own com._tx_combs[0].
        """
        p = self.params
        grid = SystemGrid.build(p.baud_rate, p.freq_step, p.samples_per_ui)
        tap_delay = 1.0 / p.baud_rate

        all_zero = np.zeros(len(self.tx_taps_bounds))
        candidates = [all_zero, *tx_tap_combinations(self.tx_taps_bounds, self.c0_min)]
        total = len(self.shelf_gain_candidates) * len(self.dc_gain_candidates) * len(candidates)
        done = 0

        # NEXT aggressors always use this same flat, unequalized Tx FFE
        # (never the victim's own candidate taps) — but their pulse
        # responses still depend on the current CTLE gain, so they're
        # rebuilt once per (dc_gain_db, shelf_gain_db) pair below, not
        # per Tx-tap candidate.
        flat_ffe = TapWeightFfe(tap_weights=np.zeros(len(self.tx_taps_bounds)), n_post=self.n_post, tap_delay=tap_delay)

        best: SearchResult | None = None
        best_fom = -math.inf
        for shelf_gain_db in self.shelf_gain_candidates:
            for dc_gain_db in self.dc_gain_candidates:
                ctle = TwoStageCtle(
                    zero_freq=self.zero_freq,
                    pole1_freq=self.pole1_freq,
                    pole2_freq=self.pole2_freq,
                    shelf_freq=self.shelf_freq,
                    dc_gain_db=dc_gain_db,
                    shelf_gain_db=shelf_gain_db,
                )
                rx_response = (
                    ctle.transfer_function(grid.f)
                    * self.rx_afe.transfer_function(grid.f)
                    * self.rx_ffe.transfer_function(grid.f)
                )
                next_prs = [
                    link.ffe_channel_ctle_pulse_response(grid).scale(p.a_ne).samples
                    for link in [
                        Link(channel=channel, ctle=ctle, ffe=flat_ffe, rx_afe=self.rx_afe, rx_ffe=self.rx_ffe)
                        for channel in self.next_channels
                    ]
                ]

                for tx_taps in candidates:
                    ffe = TapWeightFfe(tap_weights=tx_taps, n_post=self.n_post, tap_delay=tap_delay)
                    link = Link(
                        channel=self.channel, ctle=ctle, ffe=ffe, rx_afe=self.rx_afe, rx_ffe=self.rx_ffe,
                    )
                    signal = link.ffe_channel_ctle_pulse_response(grid).scale(p.victim_amplitude)
                    pulse_response = PulseResponse.from_signal(
                        signal,
                        ui=1.0 / p.baud_rate,
                        dfe1_max=float(p.dfe_max[0]),
                        dfe1_min=float(p.dfe_min[0]),
                    )
                    fext_prs = [
                        Link(channel=channel, ctle=ctle, ffe=ffe, rx_afe=self.rx_afe, rx_ffe=self.rx_ffe)
                        .ffe_channel_ctle_pulse_response(grid)
                        .scale(p.a_fe)
                        .samples
                        for channel in self.fext_channels
                    ]

                    fom = figure_of_merit(pulse_response, next_prs + fext_prs, rx_response, p)
                    if fom > best_fom:
                        best_fom = fom
                        best = SearchResult(
                            tx_taps=tx_taps, dc_gain_db=dc_gain_db, shelf_gain_db=shelf_gain_db, fom=fom, link=link,
                        )

                    done += 1
                    if on_progress is not None:
                        on_progress(done, total)

        assert best is not None  # candidates always has at least the all-zero entry
        return best
