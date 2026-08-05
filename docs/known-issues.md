# Known issues

Running list of confirmed, non-obvious problems discovered while exercising
this project against real data — kept here rather than as a code comment
because each one took real investigation to pin down and the "why" is worth
preserving for whoever picks it up next.

## `pmf.noise_margin` silently saturates at the voltage grid's edge

Status: confirmed root cause, not yet fixed. Reproduced via
`examples/compute_com_kr_backplane.py` (IEEE 802.3ck "KR" config, PAM4,
real `Std_BP_12inch_Meg7` backplane data, 5 FEXT + 3 NEXT aggressors).

**Symptom**: `Com.compute().com_db` was found to be essentially invariant
to CTLE gain and Tx FFE tap changes for this scenario — bit-identical
(`-0.8278537031645011`) across wildly different equalization settings,
including MATLAB COM3.70's own found-optimal point for this exact
channel/config plugged in directly (no search involved). A -15 dB CTLE
DC-gain swing moved the result only in the 13th significant digit.

**Root cause**: `pmf/margin.py::noise_margin`

```python
cdf = np.cumsum(pmf)
cdf /= cdf[-1]
ix = int(np.where(cdf >= der0)[0][0])
return float(-y[ix])
```

reads the DER0 crossing off a voltage grid `y` that `pmf/grid.py::voltage_grid`
builds spanning only `±1.1 * signal_amplitude` (93A.1.7.1 Note 1). When the
combined noise/ISI/crosstalk PMF's cumulative probability already exceeds
`der0` at the grid's very *first* sample (`ix == 0`), the function returns
`-y[0]`, i.e. exactly `1.1 * signal_amplitude` — the grid's own edge, not a
genuine computed crossing. Since `com_db = 20*log10(signal_amplitude /
noise_amplitude)`, this saturation forces `com_db` to the fixed constant
`20*log10(1/1.1) = -0.8278537031645016 dB` **regardless of the actual
interference distribution** — which is exactly the number observed.

**Why it saturates for the KR/PAM4/real-backplane case specifically**: not
because serdeskit's DFE deviates from the standard — 93A-27 itself has two
cases, confirmed directly from MATLAB's own source comments
(`com_ieee8023_93a_370.m:6130`/`6157`): `1 <= n <= N_b` (a fixed set of N_b
consecutive taps, clipped to `[dfe_min, dfe_max]` and subtracted) and
`n > N_b`, "otherwise". `pulse_response.residual_isi()` implements exactly
the first case, correctly — every existing golden test in this project
(against PyChOpMarg, against IEEE_8023dj-based configs) only exercises that
branch, since those configs' own `N_bg` is 0/unset, at which point the
"otherwise" case correctly collapses to "leave it uncancelled" (what
`residual_isi()` already does for UI beyond `len(dfe_min)`). The KR config
is one of the PMD types that turns the extension on (`N_bg=3`): its
"Floating Tap Control" section (`N_bg`=3 groups × `N_bf`=3 taps, reaching
out to `N_f`=40 UI beyond the 12 fixed taps, `bmaxg`/`B_float_RSS_MAX`
governing tap strength/tail limits) is MATLAB's `floatingDFE()` implementing
that "otherwise" branch — a real, standard-referenced, per-PMD-config-gated
extension, not a MATLAB-only addition. PyChOpMarg (the reference this whole
project golden-tests against) doesn't implement it either (`grep -i floating`
across its source turns up nothing but an unrelated "floating point" match)
— so serdeskit's gap here is shared with the reference implementation this
project trusts, not a unique oversight.

Confirmed the consequence by direct comparison against MATLAB's own run for
the identical channel/config/EQ point: serdeskit's `sigma_isi` ≈ 15–21 mV vs
MATLAB's own `sgm_isi` = 0.0018 mV — over 1000x larger, since KR's config
expects that extra reach to be there and none of serdeskit's fixed 12 taps
can substitute for it. That leftover ISI dwarfs the ±1.1×signal_amplitude
grid, so `noise_margin` never finds a real interior crossing; it hits the
wall on every candidate the search tries. What's new here is the
*mechanism* by which the gap manifests — not a gracefully degraded (very
negative) COM, but a numerically meaningless constant that happens to look
like a plausible-ish small-negative dB number.

**Consequence**: `examples/compute_com_kr_backplane.py`'s reported
`COM = -0.828 dB` is not a real measurement of this channel — it's the
saturation artifact. The equalization search itself (`EqualizationSearch`,
driven by `figure_of_merit`) is *not* affected, since `figure_of_merit`
never calls `noise_margin` — it sums variances directly (93A-36) rather than
building/reading a PMF. So the search still picks a real (if grid-limited)
optimum; only the final `Com.compute()` pass that scores it is broken for
this scenario.

**Not yet decided**: whether the fix is (a) a loud guard in `noise_margin`
that raises when `ix == 0` (so this fails visibly instead of returning a
fake number — was in progress when deprioritized in favor of writing this
doc down first), (b) implementing the floating-tap DFE extension so real
residual ISI stays small enough that saturation stops happening for
scenarios like this, or (c) something else. (a) is cheap and makes the
failure mode legible immediately; (b) is the real fix but nontrivial scope
(`pulse_response.residual_isi`, `com.Com`, `ComParams` would all need a
floating-tap notion). Revisit before trusting any `com_db` result from a
config with DFE capability much weaker than the real receiver it's modeling.

## Search grid coarsening picks a different (but locally correct) optimum than MATLAB

Status: understood, not a bug — noted here for context since it was
investigated alongside the issue above and could otherwise look like part
of the same problem.

For the same KR/backplane scenario, `EqualizationSearch`'s shrunk grid
(`examples/compute_com_kr_backplane.py`'s `DC_GAIN_CANDIDATES`/
`SHELF_GAIN_CANDIDATES`/`TX_TAPS_BOUNDS`, coarsened from the real config's
step sizes the same way `tests/evaluate/test_evaluate.py` already
establishes as precedent) found `dc_gain=0 dB, shelf_gain=-3 dB,
taps=[0, 0, -0.34, 0]`, versus MATLAB's own found-optimal
`dc_gain=0 dB, shelf_gain=-2 dB, taps=[-0.02, 0.06, -0.26, -0.02]` (read
from `KR_eval_..._case1_results.csv`, package case 1, COM=3.608 dB PASS).
DC gain matches exactly; shelf gain is the nearest reachable point on the
coarsened `{-6,-3,0}` grid; the taps differ more since the coarsened grid
doesn't contain MATLAB's exact multi-tap point.

Checked this wasn't a search bug: evaluating `figure_of_merit` at both
points directly, serdeskit's own search winner scores *better*
(FOM=-15.84) than MATLAB's exact point (FOM=-17.55) under serdeskit's own
objective — the search is correctly maximizing FOM over the grid it's
given; the grid just doesn't contain MATLAB's answer. Not investigated
further: whether `figure_of_merit`'s formula differences from MATLAB's own
`calc_fom` (documented in `optimize/figure_of_merit.py`'s own docstring)
would still prefer serdeskit's point even on an unrestricted grid.

## Project policy: MATLAB, not PyChOpMarg, is the authoritative reference

Status: settled (as of the RX `tline_segments` fix below), noted here since
every earlier entry in this file — and most of this project's existing test
suite — was written against the older assumption.

PyChOpMarg was adopted early on as the golden-test reference purely for
development convenience (a Python implementation is far easier to
cross-check against than the official MATLAB COM3.70 tool). It was never
meant to be the actual target of correctness — that's always been "does
this match the official IEEE 802.3 methodology," which the MATLAB tool is
the closest available proxy for. Where the two disagree, MATLAB wins.

First concrete case: `Package.network()`'s RX side didn't reverse
`tline_segments` order (only `die_model`'s own `flip` reversed the die
ladder) — matching PyChOpMarg's own `sPkgRx` exactly, but diverging from
MATLAB's `make_full_pkg` by ~0.09 max error on a real 2-segment package
trace. Fixed in `src/serdeskit/package/package.py` (commit `631f2f5`) to
reverse segments for RX, matching MATLAB; this broke 7 tests that
golden-tested the old (PyChOpMarg-matching) RX behavior, all now
`@pytest.mark.skip`-ed with a reason pointing back here rather than
updated to assert the divergence (asserting "not equal" would verify
nothing).

Second case, found the same way immediately after: `SParameterChannel.
transfer_function()` applies a raised-cosine taper across the *entire*
queried frequency band (not just any extrapolated tail beyond the
channel's measured range) before returning H21 — traced to PyChOpMarg's
own `calc_H21` (`pychopmarg/utility/filter.py`), which does exactly this;
serdeskit's version is a faithful transcription, not an independent bug.
MATLAB never applies anything like this at the H21/S-parameter level —
it handles extrapolation/causality differently, via time-domain
alternating-projections causality correction
(`com_ieee8023_93a_370.m::s21_to_impulse_DC`), not frequency-domain
windowing. Confirmed by direct comparison against MATLAB's own
channel+package S21 for the real C2C thru channel
(`matlab_golden/data/channel_plus_package_c2c_thru.csv`,
`tests/channel/test_transfer_function_vs_matlab.py`, currently failing on
purpose): exact match near DC, growing divergence with frequency, and at
the high-frequency end serdeskit's `transfer_function()` decays toward
zero (the taper's own shape) while MATLAB's own H21 stays at a real,
non-trivial value there. Not yet fixed — this one's a bigger question
than the RX fix, since MATLAB's own causality-correction approach is a
genuinely different technique (time-domain, iterative), not a one-line
formula swap. Whether serdeskit needs the equivalent, or whether the
raised-cosine taper can just be dropped and DC-extrapolation/edge-padding
alone are good enough, is still open.

`tests/channel/test_channel_transfer_function.py`'s own PyChOpMarg golden
test still passes and should stay — it's still correct proof that
serdeskit faithfully transcribes PyChOpMarg's formula, which remains
useful (e.g. for catching accidental transcription bugs), even though
PyChOpMarg itself is no longer the thing serdeskit is trying to match.
