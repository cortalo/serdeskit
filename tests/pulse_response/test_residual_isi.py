import copy
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import skrf
from pychopmarg.com import COM
from pychopmarg.common import OptMode
from pychopmarg.config.ieee_8023dj import IEEE_8023dj

from serdeskit.pulse_response import PulseResponse


@dataclass(frozen=True)
class IsiCase:
    """One `calc_noise` run's ISI extraction: its input and its output."""

    pulse_response: npt.NDArray[np.float64]
    cursor_ix: int
    h_isi: npt.NDArray[np.float64]
    dfe_min: npt.NDArray[np.float64]
    dfe_max: npt.NDArray[np.float64]
    samples_per_ui: int
    baud_rate: float


@pytest.fixture(scope="module")
def pychopmarg_isi() -> IsiCase:
    """Run PyChOpMarg's own `calc_noise` and hand back the ISI extraction's
    input and output together, so the comparison below feeds *the same*
    pulse-response array into both sides.

    That isolates what's under test: any difference is in the extraction
    itself, not in how the pulse response was generated (PyChOpMarg's
    includes a package model this project doesn't implement — irrelevant
    here, since we take its output as our input).

    The COM setup mirrors tests/link/test_pulse_response_vs_pychopmarg.py:
    a synthetic channel written to a temp Touchstone file, and package
    fields reshaped per-end to work around IEEE_8023dj shipping them flat
    (see that file for the full explanation). `nRxTaps = 0` and PRZF mode
    keep calc_noise on the paths this project has: no Rx FFE, and the
    closed-form (93A-30) Tx-noise term rather than MMSE's NoiseCalc.
    """
    freq = np.linspace(1e8, 40e9, 400)
    loss = np.exp(-np.sqrt(freq / 1e9) * 0.08) * np.exp(-1j * 2 * np.pi * freq * 3e-10)
    refl = 0.03 * np.exp(-1j * 2 * np.pi * freq * 1e-10)
    s = np.zeros((len(freq), 4, 4), dtype=complex)
    for a, b in [(0, 1), (2, 3)]:
        s[:, a, a] = refl
        s[:, b, b] = refl
        s[:, b, a] = loss
        s[:, a, b] = loss
    path = Path(tempfile.mkdtemp()) / "synthetic_thru.s4p"
    skrf.Network(f=freq, s=s, z0=50, f_unit="Hz").write_touchstone(str(path))

    cfg = copy.deepcopy(IEEE_8023dj)
    cfg.fstep = 0.1
    cfg.R_d = np.array([50.0, 50.0])
    cfg.C_d = [np.array([4e-05, 9e-05, 0.00011])] * 2  # type: ignore[list-item]
    cfg.L_s = [np.array([0.13, 0.15, 0.14])] * 2  # type: ignore[list-item]
    cfg.C_p = [4e-05, 4e-05]
    cfg.C_b = [3e-05, 3e-05]

    com = COM(cfg, {"THRU": [path], "FEXT": [], "NEXT": []}, debug=True)
    com.gDC, com.gDC2 = -6.0, -2.0
    com.tx_ix = 5
    com.nRxTaps = 0
    com.rx_taps = np.array([])
    com.dfe_taps = np.array([])
    com.opt_mode = OptMode.PRZF
    com.calc_noise()

    return IsiCase(
        pulse_response=np.asarray(com.com_rslts["pulse_resps"][0], dtype=np.float64),
        cursor_ix=int(com.com_rslts["cursor_ix"]),
        h_isi=np.asarray(com.com_rslts["hISI"], dtype=np.float64),
        dfe_min=np.asarray(cfg.dfe_min, dtype=np.float64),
        dfe_max=np.asarray(cfg.dfe_max, dtype=np.float64),
        samples_per_ui=int(cfg.M),
        baud_rate=float(cfg.fb) * 1e9,
    )


def test_matches_pychopmarg_golden_reference(pychopmarg_isi: IsiCase) -> None:
    case = pychopmarg_isi
    ui = 1.0 / case.baud_rate
    fs = case.samples_per_ui / ui
    pr = PulseResponse(
        samples=case.pulse_response, fs=fs, cursor_time=case.cursor_ix / fs, ui=ui
    )

    actual = pr.residual_isi(case.dfe_min, case.dfe_max)

    np.testing.assert_allclose(actual, case.h_isi, rtol=0, atol=1e-15)


def test_hand_verifiable_extraction() -> None:
    """A ramp, 1 sample per UI, worked out by hand — independent of
    PyChOpMarg.

    samples = [0..9], cursor at index 4 (value 4.0), 1 DFE tap limited to
    [-1, 1] (so it *will* clip here).

    n_pre = min(5, 4) = 4, so the window starts at UI 0 and the cursor
    lands at index 4; n_post_ui exceeds what's available, so it runs to
    the end.

    Window        : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Cursor zeroed :  index 4 -> 0
    DFE tap       :  wants samples[5]/cursor = 5/4 = 1.25, clipped to 1.0,
                     so index 5 becomes 5 - 1.0*4 = 1.0 (not 0 — the clip
                     is exactly what leaves residue)
    Expected      : [0, 1, 2, 3, 0, 1, 6, 7, 8, 9]
    """
    samples = np.arange(10, dtype=np.float64)
    pr = PulseResponse(samples=samples, fs=1.0, cursor_time=4.0, ui=1.0)

    actual = pr.residual_isi(dfe_min=np.array([-1.0]), dfe_max=np.array([1.0]))

    expected = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 8.0, 9.0])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_unclipped_dfe_tap_fully_cancels_its_ui() -> None:
    """When the needed tap weight falls inside [dfe_min, dfe_max], the DFE
    cancels that UI exactly — the complement of the clipped case above,
    pinning down that the residue there comes from the clip and not from
    the subtraction being wrong.
    """
    samples = np.arange(10, dtype=np.float64)
    pr = PulseResponse(samples=samples, fs=1.0, cursor_time=4.0, ui=1.0)

    actual = pr.residual_isi(dfe_min=np.array([-2.0]), dfe_max=np.array([2.0]))

    assert actual[5] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("n_min", "n_max"),
    [
        # len(dfe_min) alone sets the DFE's span, so a longer dfe_min than
        # dfe_max silently broadcasts — every tap sharing one upper limit,
        # no error, wrong answer.
        (3, 1),
        # The reverse does raise, but as an opaque numpy broadcast error
        # about "non-broadcastable output operand" rather than anything
        # naming the real problem.
        (1, 3),
    ],
)
def test_mismatched_dfe_limit_lengths_raise(n_min: int, n_max: int) -> None:
    pr = PulseResponse(samples=np.arange(20, dtype=np.float64), fs=1.0, cursor_time=8.0, ui=1.0)

    with pytest.raises(ValueError, match="same length"):
        pr.residual_isi(dfe_min=np.full(n_min, -1.0), dfe_max=np.full(n_max, 1.0))
