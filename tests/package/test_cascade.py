"""End-to-end golden test: a package-augmented channel, built entirely
from serdeskit.package + serdeskit.channel pieces, against PyChOpMarg's
own add_pkg()-computed H21 (self.chnls[0][1], always computed in
COM.__init__ — not the debug-only chnls_noPkg this project's other
channel-level golden test uses).

This is the piece that actually closes the loop: Package.network()
produces skrf.Network objects, but nothing before this test cascades one
with a real channel and hands the result to SParameterChannel — the
point where "package modeling exists" becomes "package modeling is
usable."
"""
import copy
import tempfile
from pathlib import Path

import numpy as np
import pychopmarg.com
import pychopmarg.utility.sparams
import pytest
import skrf
from pychopmarg.com import COM
from pychopmarg.config.ieee_8023dj import IEEE_8023dj

from serdeskit.channel import SParameterChannel
from serdeskit.package import Package, cascade_channel


@pytest.fixture(scope="module")
def com_and_channel_path() -> tuple[COM, Path]:
    """Same reasoning as test_package_vs_real_com.py's own `com` fixture:
    a real COM instance, PI/TWOPI patched around construction (both
    bindings this pipeline touches — pychopmarg.utility.sparams for the
    package S-parameter formulas, pychopmarg.com for calc_H21's own
    module-level import of the same constants) since self.chnls is
    computed once in __init__ and cached. Also returns the synthetic
    channel's own Touchstone path, needed on our side to load and
    convert the same raw channel independently.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pychopmarg.utility.sparams, "PI", np.pi)
        mp.setattr(pychopmarg.utility.sparams, "TWOPI", 2 * np.pi)
        mp.setattr(pychopmarg.com, "PI", np.pi)
        mp.setattr(pychopmarg.com, "TWOPI", 2 * np.pi)

        freq = np.linspace(1e8, 40e9, 400)
        loss = np.exp(-np.sqrt(freq / 1e9) * 0.08) * np.exp(-1j * 2 * np.pi * freq * 3e-10)
        refl = 0.03 * np.exp(-1j * 2 * np.pi * freq * 1e-10)
        s = np.zeros((len(freq), 4, 4), dtype=complex)
        for a, b in [(0, 1), (2, 3)]:
            s[:, a, a] = refl
            s[:, b, b] = refl
            s[:, b, a] = loss
            s[:, a, b] = loss
        path = Path(tempfile.mkdtemp()) / "thru.s4p"
        skrf.Network(f=freq, s=s, z0=50, f_unit="Hz").write_touchstone(str(path))

        cfg = copy.deepcopy(IEEE_8023dj)
        cfg.fstep = 0.1
        cfg.R_d = np.array([50.0, 50.0])
        cfg.C_d = [np.array([4e-05, 9e-05, 0.00011])] * 2  # type: ignore[list-item]
        cfg.L_s = [np.array([0.13, 0.15, 0.14])] * 2  # type: ignore[list-item]
        cfg.C_p = [4e-05, 4e-05]
        cfg.C_b = [3e-05, 3e-05]

        com = COM(cfg, {"THRU": [path], "FEXT": [], "NEXT": []}, debug=True)
        return com, path


def _package(com: COM, is_rx: bool) -> Package:
    cfg = com.com_params
    ix = 1 if is_rx else 0
    return Package(
        r0=cfg.R_0,
        die_capacitances=[c / 1e9 for c in cfg.C_d[ix]],  # type: ignore[attr-defined]
        die_inductances=[l / 1e9 for l in cfg.L_s[ix]],  # type: ignore[attr-defined]
        bump_capacitance=cfg.C_b[ix] / 1e9,
        tline_a1=cfg.a1,
        tline_a2=cfg.a2,
        tline_tau=cfg.tau,
        tline_gamma0=cfg.gamma0,
        tline_segments=list(zip(cfg.z_c, [cfg.z_p[com.zp_sel], cfg.z_pB])),
        pad_capacitance=cfg.C_p[ix] / 1e9,
        is_rx=is_rx,
    )


def _raw_channel_differential(path: Path) -> skrf.Network:
    """Converts a raw 4-port Touchstone file to its differential two-port
    — the same renumber + se2gmm + subnetwork recipe
    channel.SParameterChannel applies internally (port order (TX+, RX+,
    TX-, RX-), same as this project's other Touchstone-loading tests),
    reproduced here rather than reused: SParameterChannel doesn't expose
    this step on its own, and package/ + this test are meant to stay
    decoupled from it (see differential_pair's own docstring for the
    same reasoning applied to a different case).
    """
    network = skrf.Network(str(path)).copy()
    network.renumber([0, 1, 2, 3], [0, 2, 1, 3])
    network.se2gmm(p=2)
    return network.subnetwork([0, 1])


def test_matches_pychopmargs_own_add_pkg_h21(com_and_channel_path: tuple[COM, Path]) -> None:
    com, path = com_and_channel_path
    freqs = com.freqs

    channel = _raw_channel_differential(path)
    tx = _package(com, is_rx=False)
    rx = _package(com, is_rx=True)
    cascaded = cascade_channel(channel, tx, rx, freqs)
    packaged_channel = SParameterChannel(cascaded, gamma1=com.gamma1_Tx, gamma2=com.gamma2_Rx)

    actual = packaged_channel.transfer_function(freqs)

    expected = com.chnls[0][1]  # THRU channel's H21, package included — always computed, not debug-only
    np.testing.assert_allclose(actual, expected, atol=1e-12)
