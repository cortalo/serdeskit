import numpy as np

from serdeskit.crosstalk import worst_case_phase_samples


def test_hand_verifiable_worst_case_phase() -> None:
    """3 UI, 3 samples each. Phase 1 (index 1, 4, 7) has by far the largest
    energy (5^2 + 4^2 + 6^2 = 77) against phase 0 (1+1+1=3) and phase 2
    (4+1+1=6) — worked out by hand so the expected answer doesn't share any
    machinery with the implementation under test.
    """
    samples_per_ui = 3
    pulse_response = np.array(
        [
            1.0, 5.0, 2.0,  # UI 0
            1.0, 4.0, 1.0,  # UI 1
            1.0, 6.0, 1.0,  # UI 2
        ]
    )

    result = worst_case_phase_samples(pulse_response, samples_per_ui)

    np.testing.assert_allclose(result, [5.0, 4.0, 6.0])


def test_matches_a_direct_per_phase_energy_scan() -> None:
    """Cross-checks against an independently-written (not shared-code)
    argmax-of-energy scan over random data, rather than another hand
    example — catches an off-by-one in which phase or which slice wins
    that a single small example could miss.
    """
    rng = np.random.default_rng(0)
    samples_per_ui = 4
    pulse_response = rng.normal(size=4 * 37)

    result = worst_case_phase_samples(pulse_response, samples_per_ui)

    expected = pulse_response[0::samples_per_ui]
    best_energy = float((expected**2).sum())
    for m in range(1, samples_per_ui):
        phase_samples = pulse_response[m::samples_per_ui]
        energy = float((phase_samples**2).sum())
        if energy > best_energy:
            best_energy = energy
            expected = phase_samples

    np.testing.assert_allclose(result, expected)


def test_length_not_a_multiple_of_samples_per_ui() -> None:
    """8 samples, samples_per_ui=3: phase 2 (indices 2, 5) only gets 2
    samples, one short of phases 0/1's 3 (indices 0,3,6 and 1,4,7) — index
    8 doesn't exist. Phase 2 still wins on energy, so the winning slice
    itself is the shorter one: a trailing partial UI is simply dropped,
    same as PyChOpMarg's `pulse_resp[i::M]` — not a case this function
    needs to guard against, just pinned down so the behavior doesn't
    silently change later.
    """
    samples_per_ui = 3
    pulse_response = np.array([0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = worst_case_phase_samples(pulse_response, samples_per_ui)

    np.testing.assert_allclose(result, [9.0, 0.0])
