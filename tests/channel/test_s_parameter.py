import numpy as np
import pytest
import skrf
from scipy.integrate import cumulative_trapezoid

from serdeskit.channel import SParameterChannel
from serdeskit.channel.s_parameter import _trim_impulse
from serdeskit.common.types import Signal


def _thru_trace_block(r: complex, t: complex) -> np.ndarray:
    return np.array([[r, t], [t, r]], dtype=complex)


def test_s21_2port_returns_interpolated_values() -> None:
    freq = np.array([1e9, 2e9, 3e9, 4e9, 5e9])
    s21_expected = np.array([0.9, 0.8, 0.6, 0.3, 0.1], dtype=complex)
    s = np.zeros((len(freq), 2, 2), dtype=complex)
    s[:, 1, 0] = s21_expected
    s[:, 0, 1] = s21_expected
    network = skrf.Network(f=freq, s=s, z0=50, f_unit="Hz")

    channel = SParameterChannel(network)
    result = channel.s21(freq)

    np.testing.assert_allclose(result, s21_expected, atol=1e-9)


def test_s21_4port_peters_convention_matches_single_trace() -> None:
    """Two identical, uncoupled thru traces in (TX+, RX+, TX-, RX-) port
    order should differentially transmit exactly like one trace alone —
    this is what pins down the renumber permutation in SParameterChannel.
    """
    freq = np.array([1e9, 2e9, 3e9, 4e9, 5e9])
    r, t = 0.1 + 0.0j, 0.5 + 0.0j
    trace = _thru_trace_block(r, t)

    s = np.zeros((len(freq), 4, 4), dtype=complex)
    for i in range(len(freq)):
        s[i, 0:2, 0:2] = trace  # ports 0,1 = TX+, RX+
        s[i, 2:4, 2:4] = trace  # ports 2,3 = TX-, RX-
    network = skrf.Network(f=freq, s=s, z0=50, f_unit="Hz")

    channel = SParameterChannel(network)
    result = channel.s21(freq)

    np.testing.assert_allclose(result, np.full(len(freq), t), atol=1e-9)


def test_impulse_response_step_settles_to_dc_s21() -> None:
    """Integrating the impulse response (the same cumulative_trapezoid step
    scikit-rf's own step_response() uses) should settle to S21(0 Hz) — the
    physical check that validated extrapolate_to_dc()+bandpass=False against
    the real B12 backplane file before this was written.
    """
    freq = np.linspace(0.0, 5e9, 64)  # already starts at DC
    dc_s21 = 0.5 + 0.0j
    s = np.zeros((len(freq), 2, 2), dtype=complex)
    s[:, 1, 0] = dc_s21
    s[:, 0, 1] = dc_s21
    network = skrf.Network(f=freq, s=s, z0=50, f_unit="Hz")

    channel = SParameterChannel(network)
    _, h = channel.impulse_response()

    # No `x=t`: scikit-rf's own step_response() integrates with implicit
    # unit spacing too — h is already scaled such that a unit-spacing
    # cumulative sum reconstructs the step response correctly.
    step = cumulative_trapezoid(h, initial=0)
    assert step[-1] == pytest.approx(dc_s21.real, abs=1e-2)


def test_impulse_response_dt_controls_time_step() -> None:
    """`dt` zero-pads the spectrum (same technique `xfr_fn_to_imp.m` uses,
    padding out to `fmax=1/Ts`) to hit a target time step, independent of
    whatever native resolution the frequency data implies.

    Note: this fixture's S21 is frequency-independent, so its "impulse
    response" is a scaled delta function — all its energy sits in a single
    sample regardless of dt, which is why peak amplitude doesn't shrink with
    dt here (unlike a real, dispersive channel — see impulse_response()'s
    docstring, confirmed against the real B12 file: dt=1ps/no-window peak
    matched MATLAB's ~5.3 mV, vs. ~0.15 at the coarser native ~33ps step,
    both unwindowed).
    The step-response check below is dt-independent either way.
    """
    freq = np.linspace(0.0, 5e9, 64)
    dc_s21 = 0.5 + 0.0j
    s = np.zeros((len(freq), 2, 2), dtype=complex)
    s[:, 1, 0] = dc_s21
    s[:, 0, 1] = dc_s21
    network = skrf.Network(f=freq, s=s, z0=50, f_unit="Hz")
    channel = SParameterChannel(network)

    for dt in (2e-10, 1e-10):
        t, h = channel.impulse_response(dt=dt)
        assert t[1] - t[0] == pytest.approx(dt, rel=1e-6)
        step = cumulative_trapezoid(h, initial=0)
        assert step[-1] == pytest.approx(dc_s21.real, abs=1e-2)


def test_trim_impulse_locates_main_lobe_and_reports_its_delay() -> None:
    """h simulates what impulse_response() returns: fftshift-centered (its
    own t=0 is at index n//2), with a main lobe some distance away from
    center — exactly like the real B12 channel's ~4ns delay showing up well
    away from the center of a -50ns..+50ns array.
    """
    n = 1000
    h = np.zeros(n)
    peak_ix = n // 2 + 50  # simulates a channel delay of +50 samples
    h[peak_ix - 2 : peak_ix + 3] = [0.1, 0.5, 1.0, 0.5, 0.1]

    trimmed, delay = _trim_impulse(h, kept_energy=0.999)

    assert len(trimmed) < n  # near-zero tails discarded
    # delay is trimmed[0]'s offset from h's own center (n//2); reconstructing
    # the main lobe's absolute position from it should land back on peak_ix.
    assert delay + n // 2 + np.argmax(np.abs(trimmed)) == peak_ix


def test_process_convolves_through_trimmed_impulse_response() -> None:
    """A single-sample [1.0] Signal is convolution's identity element, so
    process()'s output should reproduce impulse_response()+_trim_impulse()
    exactly — this pins down that process() derives dt from sig.fs
    correctly and gets the t0/delay bookkeeping right, on top of the
    already-validated physics of impulse_response() itself.
    """
    freq = np.linspace(0.0, 5e9, 256)
    fc = 5e8  # single-pole lowpass: genuine dispersion, not a delta function
    s21 = 1.0 / (1.0 + 1j * freq / fc)
    s = np.zeros((len(freq), 2, 2), dtype=complex)
    s[:, 1, 0] = s21
    s[:, 0, 1] = s21
    network = skrf.Network(f=freq, s=s, z0=50, f_unit="Hz")
    channel = SParameterChannel(network)

    fs = 2e10
    sig = Signal(samples=np.array([1.0]), fs=fs, t0=0.0)

    out = channel.process(sig)

    _, h = channel.impulse_response(dt=1.0 / fs)
    h_trimmed, delay_samples = _trim_impulse(h)

    assert out.fs == fs
    # A length-1 signal has no partial-overlap region to speak of, so
    # mode="valid" reproduces h_trimmed exactly here too.
    np.testing.assert_allclose(out.samples, h_trimmed)
    expected_t0 = sig.t0 + (delay_samples + len(h_trimmed) - 1) / fs
    assert out.t0 == pytest.approx(expected_t0)


def test_process_drops_partial_overlap_region_at_both_ends() -> None:
    """mode="valid" means output length is N-M+1, not the N+M-1 a 'full'
    convolution would give — this is what keeps eye extraction from slicing
    the first few UI worth of traces out of the warm-up transient, where
    the kernel doesn't yet fully overlap real input (a spotted-by-eye bug:
    a handful of faint, straight-ish outlier traces at the start of an eye
    diagram, distinct from the properly-settled bundle).
    """
    freq = np.linspace(0.0, 5e9, 256)
    fc = 5e8
    s21 = 1.0 / (1.0 + 1j * freq / fc)
    s = np.zeros((len(freq), 2, 2), dtype=complex)
    s[:, 1, 0] = s21
    s[:, 0, 1] = s21
    network = skrf.Network(f=freq, s=s, z0=50, f_unit="Hz")
    channel = SParameterChannel(network)

    fs = 2e10
    sig = Signal(samples=np.ones(2000), fs=fs, t0=0.0)

    out = channel.process(sig)

    _, h = channel.impulse_response(dt=1.0 / fs)
    h_trimmed, _ = _trim_impulse(h)

    assert len(out.samples) == len(sig.samples) - len(h_trimmed) + 1
