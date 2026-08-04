"""cascade_channel: Tx package ** channel ** Rx package — the step that
turns Package's isolated building blocks into something usable, by
combining a raw channel's own differential two-port network with a Tx
and an Rx Package (PyChOpMarg's own add_pkg).

The result is a plain skrf.Network, same as every other package/
function returns — handing it to channel.SParameterChannel is the
caller's job (package/ stays unaware of channel/, same reasoning as
elsewhere in this package).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import skrf

from serdeskit.package.package import Package


def cascade_channel(
    channel: skrf.Network, tx: Package, rx: Package, freqs: npt.NDArray[np.float64]
) -> skrf.Network:
    """Tx package -> channel -> Rx package, all three on `freqs`.

    Args:
        channel: The raw channel's own differential two-port network —
            not necessarily already defined on `freqs` (a channel's own
            measured band is usually narrower); extended onto `freqs`
            first via `_extend_to_grid`, matching PyChOpMarg's own
            add_pkg (DC-extrapolated, cubic-interpolated in-band, then
            edge-padded beyond — no raised-cosine taper at this stage,
            unlike channel.SParameterChannel.transfer_function's own
            interpolation; that one happens once, later, when *this*
            function's result is eventually turned into a transfer
            function).
        tx: The Tx-side package model.
        rx: The Rx-side package model.
        freqs: Frequencies to evaluate at (Hz) — the system frequency
            grid every stage in the link is computed on.

    Returns:
        The cascaded differential two-port network.
    """
    extended = _extend_to_grid(channel, freqs)
    return tx.network(freqs) ** extended ** rx.network(freqs)


def _extend_to_grid(network: skrf.Network, freqs: npt.NDArray[np.float64]) -> skrf.Network:
    """DC-extrapolate and cubic-interpolate `network` onto the in-band
    portion of `freqs`, then edge-pad (repeat the last in-band value)
    for anything beyond its measured range — matches PyChOpMarg's own
    add_pkg, which does this to a raw channel before cascading it with
    the package models (which are already defined on the full `freqs`
    grid, so cascading needs the channel to match).
    """
    in_band = network.extrapolate_to_dc().interpolate(
        freqs[freqs <= network.f[-1]], kind="cubic", coords="polar", basis="t", assume_sorted=True,
    )
    pad_len = len(freqs) - len(in_band.f)
    s11 = np.pad(in_band.s[:, 0, 0], (0, pad_len), mode="edge")
    s12 = np.pad(in_band.s[:, 0, 1], (0, pad_len), mode="edge")
    s21 = np.pad(in_band.s[:, 1, 0], (0, pad_len), mode="edge")
    s22 = np.pad(in_band.s[:, 1, 1], (0, pad_len), mode="edge")
    s = np.stack(
        [np.stack([s11, s12], axis=-1), np.stack([s21, s22], axis=-1)], axis=-2,
    )
    return skrf.Network(s=s, f=freqs, z0=in_band.z0[0], f_unit="Hz")
