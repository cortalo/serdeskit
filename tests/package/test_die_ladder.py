import numpy as np
import pychopmarg.utility.sparams
import pytest
from pychopmarg.utility.sparams import sDieLadderSegment

from serdeskit.package import die_ladder_segment


@pytest.fixture
def exact_pi(monkeypatch: pytest.MonkeyPatch) -> None:
    """See tests/package/test_passive.py's own exact_pi — same reasoning,
    same module needing the patch (sDieLadderSegment calls straight
    through to sCshunt/sLseries, both defined in pychopmarg.utility.sparams).
    """
    monkeypatch.setattr(pychopmarg.utility.sparams, "PI", np.pi)
    monkeypatch.setattr(pychopmarg.utility.sparams, "TWOPI", 2 * np.pi)


@pytest.mark.usefixtures("exact_pi")
def test_matches_pychopmarg_golden_reference() -> None:
    """One rung of the on-die parasitic ladder: a shunt capacitor cascaded
    with a series inductor, per sDieLadderSegment(freqs, (R0, Cd, Ls)).
    """
    freqs = np.linspace(0, 50e9, 501)
    r0 = 50.0
    capacitance = 3e-14
    inductance = 1e-10

    actual = die_ladder_segment(freqs, capacitance, inductance, r0=r0)

    expected = sDieLadderSegment(freqs, (r0, capacitance, inductance))
    np.testing.assert_allclose(actual.s, expected.s, atol=1e-14)


def test_default_r0_is_50() -> None:
    freqs = np.linspace(0, 50e9, 11)
    capacitance = 3e-14
    inductance = 1e-10

    default = die_ladder_segment(freqs, capacitance, inductance)
    explicit = die_ladder_segment(freqs, capacitance, inductance, r0=50.0)

    np.testing.assert_allclose(default.s, explicit.s)


def test_is_reciprocal_but_not_symmetric() -> None:
    """Cascading two reciprocal two-ports still gives a reciprocal one
    (S12=S21 always holds for a passive cascade) — checked independently
    of the golden reference.

    S11=S22 does *not* hold here, unlike shunt_capacitor/series_inductor
    individually: a capacitor-then-inductor cascade isn't front-back
    symmetric — looking in from the C side sees a different reflection
    than looking in from the L side — even though each piece alone is
    symmetric on its own. Asserted here too, so this asymmetry stays a
    documented, checked fact rather than something a future refactor
    could silently break in either direction.
    """
    freqs = np.linspace(0, 50e9, 51)

    segment = die_ladder_segment(freqs, capacitance=3e-14, inductance=1e-10, r0=50.0)

    np.testing.assert_allclose(segment.s[:, 0, 1], segment.s[:, 1, 0])
    assert not np.allclose(segment.s[1:, 0, 0], segment.s[1:, 1, 1])


def test_dc_behavior_hand_verified() -> None:
    """At f=0: the capacitor is open (no shunt path), the inductor is a
    short (no series impedance) — so the whole segment is just a
    straight-through matched line: S21=1, S11=0.
    """
    freqs = np.array([0.0])

    segment = die_ladder_segment(freqs, capacitance=3e-14, inductance=1e-10, r0=50.0)

    np.testing.assert_allclose(segment.s[0, 0, 0], 0.0, atol=1e-15)  # S11
    np.testing.assert_allclose(segment.s[0, 1, 0], 1.0, atol=1e-15)  # S21
