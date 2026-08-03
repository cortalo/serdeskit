"""PulseResponse: a Signal that is specifically a link's pulse response,
with its cursor and UI identified — enough to derive per-UI features
(local slope, ISI, ...) without any caller manually converting between
absolute time and array indices (same rationale as Signal.t0 itself).

`cursor_time`/`ui` are in seconds, on the same absolute time basis as
`Signal.t0` — kept in the time domain (rather than raw sample indices) so
a PulseResponse can be constructed without knowing `fs` in advance, and so
its meaning doesn't silently break if resampled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from serdeskit.common.types import Signal


@dataclass(frozen=True, slots=True)
class PulseResponse(Signal):
    cursor_time: float = field(kw_only=True)  # seconds, absolute, same basis as t0
    ui: float = field(kw_only=True)  # seconds, one unit interval's duration

    @property
    def cursor_index(self) -> int:
        return round((self.cursor_time - self.t0) * self.fs)

    @property
    def samples_per_ui(self) -> int:
        return round(self.ui * self.fs)

    @classmethod
    def from_signal(
        cls,
        signal: Signal,
        ui: float,
        dfe1_max: float,
        dfe1_min: float,
        max_range: int = 1,
        eps: float = 0.001,
    ) -> PulseResponse:
        """(93A-25)/(93A-26), the Muller-Mueller criterion: locate the
        cursor within `max_range` UI of `signal`'s peak sample, anticipating
        the first DFE tap's clipped value at each candidate, and return a
        PulseResponse with that cursor identified.

        Args:
            signal: The pulse response to locate a cursor within.
            ui: Seconds, one unit interval's duration.
            dfe1_max: Maximum allowed value for the first DFE tap.
            dfe1_min: Minimum allowed value for the first DFE tap.
            max_range: Search radius, in UI, from `signal`'s peak sample.
                Default: 1.
            eps: Threshold below which a Muller-Mueller residual is
                considered an exact solution. Default: 0.001.

        Returns:
            A PulseResponse wrapping `signal`, with `cursor_time` set to
            the located cursor and `ui` as given.
        """
        samples = signal.samples
        nspui = round(ui * signal.fs)
        peak_ix = int(np.argmax(samples))

        ix_best = peak_ix
        res_min = 1e6
        zero_res_ixs: list[int] = []
        for ix in range(max(0, peak_ix - nspui * max_range), min(len(samples), peak_ix + nspui * max_range)):
            # Anticipate the first DFE tap's clipped value: (93A-26)
            b1 = min(dfe1_max, max(dfe1_min, samples[ix + nspui] / samples[ix]))
            # Muller-Mueller residual: (93A-25)
            res = abs(samples[ix - nspui] - (samples[ix + nspui] - b1 * samples[ix]))
            if res < eps:
                zero_res_ixs.append(ix)
            elif res < res_min:
                ix_best = ix
                res_min = res

        if zero_res_ixs:
            pre_peak_ixs = [ix for ix in zero_res_ixs if ix <= peak_ix]
            cursor_ix = pre_peak_ixs[-1] if pre_peak_ixs else zero_res_ixs[0]
        else:
            cursor_ix = ix_best

        cursor_time = signal.t0 + cursor_ix / signal.fs
        return cls(samples=samples, fs=signal.fs, t0=signal.t0, cursor_time=cursor_time, ui=ui)

    def signal_amplitude(self, rlm: float, levels: int) -> float:
        """As (93A.1.6.c): the signal amplitude, volts — derived from this
        pulse response's cursor value.

        Args:
            rlm: Relative level mismatch (1.0 for NRZ; a modulation-format
                parameter, otherwise).
            levels: Number of modulation levels (2 for NRZ, 4 for PAM4).

        Returns:
            As, volts.
        """
        cursor_value = self.samples[self.cursor_index]
        return float(rlm * cursor_value / (levels - 1))

    def residual_isi(
        self,
        dfe_min: npt.NDArray[np.float64],
        dfe_max: npt.NDArray[np.float64],
        n_pre_ui: int = 5,
        n_post_ui: int = 2048,
    ) -> npt.NDArray[np.float64]:
        """(93A-26)/(93A-27): the intersymbol interference left over after
        the DFE cancels what it can — one value per UI, ready to hand to
        `pmf.delta_pmf` as its `h_samples`.

        Three things happen here. The pulse response is sampled once per
        UI (on the cursor's own sub-UI phase, so the cursor sample is
        included exactly); the cursor's own UI is zeroed, since that's
        the signal being detected rather than interference; and over the
        UI the DFE spans, each tap's value is set to what it would need
        to cancel that UI's ISI, clipped to [dfe_min, dfe_max] (93A-26),
        and subtracted (93A-27) — leaving only what a real, range-limited
        DFE can't remove.

        Args:
            dfe_min: Per-tap lower limits on DFE tap weight, normalized
                to the cursor value. Note this argument carries two
                things: the limits themselves, and — via its length —
                how many post-cursor UI the DFE spans. The tap weights
                are not passed in at all; they're derived here, per
                (93A-26).
            dfe_max: Per-tap upper limits; must be the same length as
                `dfe_min`.
            n_pre_ui: How many pre-cursor UI to include, clamped to how
                many actually precede the cursor. Default: 5.
            n_post_ui: How many post-cursor UI to include. Default: 2048.

        Returns:
            Residual ISI, volts, one entry per included UI, ordered
            earliest first (so the cursor's own zeroed entry sits at
            index `min(n_pre_ui, cursor UI index)`).

        Raises:
            ValueError: `dfe_min` and `dfe_max` differ in length. Only
                `dfe_min`'s length sets the DFE's span, so a longer
                `dfe_min` would otherwise broadcast silently — every tap
                sharing one upper limit, no error, wrong answer.
        """
        if len(dfe_min) != len(dfe_max):
            raise ValueError(
                f"dfe_min and dfe_max must be the same length (one entry per DFE "
                f"tap), got {len(dfe_min)} and {len(dfe_max)}."
            )

        nspui = self.samples_per_ui
        cursor_ui, cursor_phase = divmod(self.cursor_index, nspui)
        cursor_value = self.samples[self.cursor_index]

        # Sample on the cursor's own sub-UI phase, so its sample is hit
        # exactly rather than approached.
        per_ui = self.samples[cursor_phase::nspui]

        n_pre = min(n_pre_ui, cursor_ui)
        # TODO: this window (n_pre <= 5 pre-cursor UI, n_post_ui = 2048
        # post-cursor) follows PyChOpMarg, whose own comment flags it as
        # "Inconsistent w/ IEEE 802.3-22, but consistent w/ v2.60 of MATLAB
        # code." Kept for golden-test comparability; revisit alongside the
        # other documented PyChOpMarg-vs-standard divergences if this
        # project ever needs to match the published text instead.
        h_isi = per_ui[cursor_ui - n_pre : cursor_ui + n_post_ui].copy()

        h_isi[n_pre] = 0.0  # the cursor is signal, not interference

        dfe = slice(n_pre + 1, n_pre + 1 + len(dfe_min))
        tap_weights = np.clip(h_isi[dfe] / cursor_value, dfe_min, dfe_max)  # (93A-26)
        h_isi[dfe] -= tap_weights * cursor_value  # (93A-27)

        return np.asarray(h_isi, dtype=np.float64)

    def local_slopes(
        self, signal_amplitude: float, threshold: float = 0.001
    ) -> npt.NDArray[np.float64]:
        """(93A-28): central-difference slope (volts/UI) at each valid
        UI-spaced sample from this pulse response's cursor onward.

        Args:
            signal_amplitude: As, the signal amplitude (volts) — sets the
                (relative) threshold below which a sample is excluded.
            threshold: Samples with |amplitude| below `threshold *
                signal_amplitude` are excluded. Default: 0.001 (0.1%, per
                NOTE 2 of 93A.1.7.1).

        Returns:
            The slope (volts/UI) at each included UI sample, in order.
        """
        thresh = signal_amplitude * threshold
        nspui = self.samples_per_ui

        candidate_ixs = np.arange(self.cursor_index, len(self.samples) - 1, nspui)
        # TODO: filtering candidates by *sample amplitude* here matches
        # PyChOpMarg's calc_hJ exactly (our golden reference), but a small
        # sample amplitude doesn't imply a small slope there (e.g. near a
        # zero crossing) — this filter can silently drop a UI with a
        # genuinely large slope. The official IEEE 802.3 MATLAB COM tool
        # (com_ieee8023_93a_370.m, both h_J computations, ~line 6253/7308)
        # applies no amplitude filtering at all for this quantity: it either
        # takes the full available range unconditionally, or truncates by
        # *tap position* (OP.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN, limited to
        # the DFE span) — never by amplitude. Left as-is for now, matching
        # the golden reference; revisit if this project ever needs to match
        # the official tool instead of PyChOpMarg bit-for-bit.
        valid_ixs = candidate_ixs[np.abs(self.samples[candidate_ixs]) >= thresh]

        m1s = self.samples[valid_ixs - 1]
        p1s = self.samples[valid_ixs + 1]
        return np.asarray((p1s - m1s) / (2.0 / nspui), dtype=np.float64)
