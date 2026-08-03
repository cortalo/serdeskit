import numpy as np
import pychopmarg.utility.filter
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
