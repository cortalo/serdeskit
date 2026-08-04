"""differential_pair: combines two identical, uncoupled single-ended
two-port models — one per conductor of a differential pair — into the
differential two-port they form together.

Physical picture: PyChOpMarg's own Tx/Rx package models treat the P and N
conductors as independently modeled (same die/package parasitics on each,
no coupling between them) — `concat_ports([conductor, conductor],
port_order='first')` builds the resulting 4-port (P's two ends, then N's
two ends: [P_in, P_out, N_in, N_out]), which happens to be exactly the
port order `channel.SParameterChannel` already assumes for raw 4-port
Touchstone files ((TX+, RX+, TX-, RX-) is the same [P_in, P_out, N_in,
N_out] shape) — so the same renumber-then-mixed-mode-convert recipe
applies here too. Kept as its own small function rather than importing
SParameterChannel to reuse its version: `package/` and `channel/` are
meant to stay mutually unaware of each other (see this package's other
modules' docstrings) and SParameterChannel doesn't expose this step on
its own anyway.
"""
from __future__ import annotations

import skrf


def differential_pair(conductor: skrf.Network) -> skrf.Network:
    """Args:
        conductor: A two-port model of one differential pair's conductor
            (P or N) — the same model stands in for both, since they're
            assumed identical.

    Returns:
        The pair's differential-mode two-port network (Sdd).
    """
    four_port = skrf.network.concat_ports([conductor, conductor], port_order="first")
    four_port = four_port.copy()
    four_port.renumber([0, 1, 2, 3], [0, 2, 1, 3])
    four_port.se2gmm(p=2)
    return four_port.subnetwork([0, 1])
