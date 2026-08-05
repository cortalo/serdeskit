import numpy as np
import pychopmarg.utility.filter
import pychopmarg.utility.sparams
import pytest


@pytest.fixture
def exact_pi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give PyChOpMarg full-precision PI/TWOPI for the duration of a test.

    `pychopmarg.common` defines `PI = 3.14159` — six significant figures,
    not double precision — and `pychopmarg.utility.filter` binds it (and
    TWOPI) at import time, so `calc_Hffe` and `raised_cosine` carry that
    truncation into every result. Against our own np.pi-based code that
    shows up as a ~1e-6 relative discrepancy, which is large enough to
    mask a real logic error.

    Requesting this fixture removes that one known difference, letting a
    golden-reference comparison assert exact agreement (~1e-17, i.e.
    floating-point noise) instead of hiding behind a loose tolerance.

    Deliberately opt-in rather than autouse: it mutates a third-party
    module's state, which should be visible at each call site that
    depends on it.
    """
    monkeypatch.setattr(pychopmarg.utility.filter, "PI", np.pi)
    monkeypatch.setattr(pychopmarg.utility.filter, "TWOPI", 2 * np.pi)


@pytest.fixture
def exact_pi_sparams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same reasoning as `exact_pi`, but for pychopmarg.utility.sparams —
    a separate PI/TWOPI binding from pychopmarg.common, imported at
    module load time independently of pychopmarg.utility.filter's own
    copy. Every S-parameter formula in tests/package/ (sCshunt, sLseries,
    sPkgTline, ...) lives in this module, hence its own fixture rather
    than folding into `exact_pi` — a single test could plausibly need
    both if it ever spans formulas from both modules.
    """
    monkeypatch.setattr(pychopmarg.utility.sparams, "PI", np.pi)
    monkeypatch.setattr(pychopmarg.utility.sparams, "TWOPI", 2 * np.pi)


@pytest.fixture
def no_raised_cosine_taper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip PyChOpMarg's own raised-cosine taper out of calc_H21 for the
    duration of a test.

    `calc_H21` (pychopmarg.utility.filter) applies `raised_cosine` to
    S12/S21 across the *entire* queried band, not just any extrapolated
    tail beyond the network's measured range. serdeskit's own
    SParameterChannel.transfer_function() used to transcribe this
    faithfully, but MATLAB COM3.70 — this project's authoritative
    reference, not PyChOpMarg (see docs/known-issues.md) — never applies
    anything like it at the H21 level, confirmed against MATLAB's own
    channel+package S21 for a real channel
    (tests/channel/test_transfer_function_vs_matlab.py). serdeskit
    dropped the taper to match; this fixture makes PyChOpMarg's own
    golden-reference computations comparable again by dropping it there
    too, for tests that need PyChOpMarg's machinery (a full COM instance,
    opt_eq, calc_fom, ...) rather than calc_H21 in isolation.

    `raised_cosine` is patched at its own defining module
    (pychopmarg.utility.filter), not wherever calc_H21 happens to be
    imported into (pychopmarg.com, a test file, ...) — calc_H21
    references it as a bare name resolved against its own module's
    globals at call time, so patching it there is a single point that
    covers every caller uniformly.

    Deliberately opt-in, same reasoning as `exact_pi`/`exact_pi_sparams`.
    """
    monkeypatch.setattr(pychopmarg.utility.filter, "raised_cosine", lambda x: x)
