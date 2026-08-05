"""End-to-end golden test: Package.network() against a real COM
instance's own sPkgTx/sPkgRx properties.

Distinct from test_package.py's own golden test, which compares against
this project's *transcription* of PyChOpMarg's sPkgTx/sPkgRx formula
(hand-copied from com.py, same as test_die.py/test_transmission_line.py
do for their own methods) — a transcription mistake shared between that
test's expected-value helper and this project's implementation could
hide behind an equally-wrong point of comparison. Here, the actual COM
class computes its own answer from a real, fully-constructed instance,
so there's nothing to transcribe and nothing to get subtly wrong on both
sides at once.
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

from serdeskit.package import Package


@pytest.fixture(scope="module")
def com() -> COM:
    """A real COM instance. Channel content is irrelevant to sPkgTx/
    sPkgRx (they don't depend on it at all), so this is a minimal
    synthetic THRU rather than tests/com/test_com.py's own — this
    fixture only exists to get a genuine sPkgTx/sPkgRx out of PyChOpMarg
    itself.

    Applies the full-precision PI/TWOPI patch itself, around
    construction, rather than via a function-scoped fixture some test
    requests: sPkgTx/sPkgRx are computed once in __init__ and cached, and
    pytest sets up higher-scoped fixtures (this one) before
    function-scoped ones — so patching after this fixture has already
    built the COM instance would be too late (same reasoning
    tests/com/test_com.py's own `reference` fixture documents).
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

        return COM(cfg, {"THRU": [path], "FEXT": [], "NEXT": []}, debug=True)


def _package(com: COM, is_rx: bool) -> Package:
    cfg = com.com_params
    ix = 1 if is_rx else 0
    return Package(
        r0=cfg.R_0,
        # COMParams types C_d/L_s as list[float], but they're actually
        # reshaped to list[NDArray] (one array per side) below — same
        # mismatch tests/link/test_pulse_response_vs_pychopmarg.py's
        # _com_params() docstring documents.
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


def test_tx_matches_a_real_com_instances_own_sPkgTx(com: COM) -> None:
    actual = _package(com, is_rx=False).network(com.freqs)

    expected = com.sPkgTx
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-12)


@pytest.mark.skip(
    reason="RX package's tline_segments now reversed to match MATLAB (631f2f5), "
    "deliberately diverging from PyChOpMarg's own sPkgRx, which doesn't reverse "
    "them -- see docs/known-issues.md"
)
@pytest.mark.rx_tline_segments
def test_rx_matches_a_real_com_instances_own_sPkgRx(com: COM) -> None:
    actual = _package(com, is_rx=True).network(com.freqs)

    expected = com.sPkgRx
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-12)
