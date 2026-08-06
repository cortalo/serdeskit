import numpy as np

from serdeskit.pulse_response import PulseResponse


def test_hand_verifiable_two_samples_per_ui() -> None:
    """A simple triangular pulse, two samples per UI, hand-worked by
    central difference (93A-28) -- independent of any golden reference.

    samples = [0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0], cursor at index
    6 (peak), nspui=2.

    Matches MATLAB COM3.70's own h_J for OP.LIMIT_JITTER_CONTRIB_TO_
    DFE_SPAN=0: spans the *entire* array (both pre- and post-cursor), no
    amplitude filtering -- see PulseResponse.local_slopes()'s own
    docstring. cursor_i (1-indexed) = 7; sampling_offset = mod(7, 2) = 1,
    bumped to 3 (<=1 rule); early/late sampled at (0-indexed) 1::2 / 3::2.

    late  = samples[3::2]  = [2, 4, 4, 2, 0, 0]  (indices 3,5,7,9,11,13)
    early = samples[1::2]  = [0, 2, 4, 4, 2, 0, 0][:6] (indices 1,3,5,7,9,11)
    slope = late - early (nspui=2, so (late-early)/2*2 == late-early):
      = [2-0, 4-2, 4-4, 2-4, 0-2, 0-0] = [2, 2, 0, -2, -2, 0]
    """
    fs = 2.0  # nspui samples/UI at 1 UI/s
    nspui = 2
    ui = nspui / fs
    cursor_ix = 6
    cursor_time = cursor_ix / fs

    samples = np.array([0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0], dtype=np.float64)

    pr = PulseResponse(samples=samples, fs=fs, cursor_time=cursor_time, ui=ui)

    actual = pr.local_slopes()

    expected = np.array([2.0, 2.0, 0.0, -2.0, -2.0, 0.0])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
