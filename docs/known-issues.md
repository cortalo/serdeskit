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
transfer_function()` applied a raised-cosine taper across the *entire*
queried frequency band (not just any extrapolated tail beyond the
channel's measured range) before returning H21 — traced to PyChOpMarg's
own `calc_H21` (`pychopmarg/utility/filter.py`), which does exactly this;
serdeskit's version was a faithful transcription, not an independent bug.
MATLAB never applies anything like this at the H21/S-parameter level. Its
own extrapolation/causality handling is a genuinely different technique
(time-domain alternating-projections causality correction,
`com_ieee8023_93a_370.m::s21_to_impulse_DC`) — but that turns out not to
matter here: `OP.ENFORCE_CAUSALITY` defaults to 0 ("Not recommended",
line 8621) and every config this project has exercised leaves it there
(confirmed by the "Causality correction = ... dB **(not applied)**"
message MATLAB itself prints on every run so far), and
`s21_to_impulse_DC` explicitly discards its own corrected result and
falls back to the plain, uncorrected one whenever it's off. So MATLAB's
actual real-world behavior at this stage is simply "no taper, no
correction" — confirmed by direct comparison against MATLAB's own
channel+package S21 for the real C2C thru channel
(`matlab_golden/data/channel_plus_package_c2c_thru.csv`,
`tests/channel/test_transfer_function_vs_matlab.py`): dropping the taper
brought serdeskit from a growing divergence with frequency to a float-
precision match (~1e-10).

**Fixed** by dropping the taper entirely (`src/serdeskit/channel/
s_parameter.py`) — a smaller change than it first looked, once the
causality-correction rabbit hole turned out to be a dead end. This is a
higher-blast-radius fix than the RX one, though: `transfer_function()` is
what every stage of the real pipeline calls, so it broke 11 tests, not 7.
Two different resolutions, depending on what each test actually needed:

- Ten were golden-reference comparisons against PyChOpMarg's own
  `calc_H21`/`COM` machinery (directly, or transitively through a real
  `COM` instance) — fixed by adding `tests/conftest.py`'s own
  `no_raised_cosine_taper` fixture, which monkeypatches
  `pychopmarg.utility.filter.raised_cosine` to identity for a test's
  duration. This works cleanly (unlike the RX case, where PyChOpMarg's
  own `sPkgTx`/`sPkgRx` assembly is inline in `COM.__init__` with no
  patchable seam): `calc_H21` references `raised_cosine` as a bare name
  resolved against its own defining module's globals at call time, so
  patching it there covers every caller uniformly, regardless of which
  module imported `calc_H21` itself.
- Four (`tests/com/test_com.py`) could *not* be fixed this way — even
  with the taper stripped from both sides, PyChOpMarg's own `calc_noise`
  crashes (`ValueError`, a negative `voltage_grid` sample count) building
  the shared `reference` fixture. Root cause: that fixture's synthetic
  channel (a simple `exp(-sqrt(f)*loss)` model, evaluated to 40 GHz)
  doesn't roll off toward its own band edge the way a real measured
  channel does — the taper had been silently masking a numerical
  instability specific to that idealized shape, on both sides of the
  comparison, not resolving a real formula disagreement. Not a policy
  mismatch a monkeypatch can fix; `@pytest.mark.skip`-ed with a reason,
  needs its own separate numerical-robustness investigation (either a
  better-behaved synthetic channel, or hardening cursor detection /
  `voltage_grid` against this shape).
- One (`tests/evaluate/test_pychopmarg_example2.py`) doesn't compare
  against PyChOpMarg at all — its own docstring already says so, it's a
  regression test against hardcoded values captured from serdeskit's own
  prior output. Its six expected values were simply stale after the
  taper removal; recomputed and updated directly (no fixture needed).

`tests/channel/test_channel_transfer_function.py`'s own PyChOpMarg golden
test now requests `no_raised_cosine_taper` too, and still passes — it's
still correct proof that serdeskit's `transfer_function()` matches
PyChOpMarg's formula once PyChOpMarg's own taper is set aside, which
remains useful (e.g. for catching accidental transcription bugs in
everything *except* the taper), even though PyChOpMarg itself is no
longer the thing serdeskit is trying to match.

Third case: `TapWeightFfe.transfer_function()` (93A-21) referenced delay 0
at the *first* (most-precursor) tap. MATLAB's own `FFE`
(`matlab_golden/lib/FFE.m`, a time-domain circshift-based tap-delay sum)
references delay 0 at the *cursor* tap instead — precursor taps get
negative delay, postcursor taps positive. Confirmed via
`matlab_golden/generate/gen_ffe.m` + `tests/ffe/test_tap_weight_vs_matlab.py`
(FFT of MATLAB's own impulse response against serdeskit's analytic DTFT,
exact agreement once serdeskit switched convention). **Fixed** in
`src/serdeskit/ffe/tap_weight.py` by referencing delays to the cursor tap
(`delays = np.arange(len(taps)) - n_pre`), matching MATLAB.

This is a pure linear-phase difference from PyChOpMarg's own `calc_Hffe`
(still first-tap-referenced) — `exp(-j*2*pi*f*T*n_pre)` — so most affected
PyChOpMarg-golden tests were fixed by compensating for that known,
derived factor directly (`tests/ffe/test_transfer_function.py`,
`tests/link/test_pulse_response_vs_pychopmarg.py`, `tests/optimize/
test_figure_of_merit.py`), the same pattern the MATLAB FFE test itself
used before the fix made it unnecessary there.

`tests/optimize/test_search.py::test_matches_pychopmargs_own_opt_eq`
could *not* be fixed this way, and is `@pytest.mark.skip`-ed +
`@pytest.mark.ffe_cursor_referenced_delay`. Unlike `test_figure_of_merit.
py`'s version of this same comparison (a test-side helper, where
compensating the pulse response array before handing it to
`PulseResponse.from_signal` is a legitimate, test-only fix), this test
exercises `EqualizationSearch`'s real, unmodified pulse-response
composition. With the new convention, that composition places the
victim's cursor at sample index ~28 of ~34000 — close enough to the array
start that `PulseResponse.from_signal`'s Muller-Mueller search window
(`range(max(0, peak_ix - nspui), ...)`, not circular) clips precursor
samples instead of wrapping to the end of the periodic IFFT'd array,
rather than genuinely disagreeing with PyChOpMarg on a formula. Confirmed
by rolling both the victim's and every aggressor's pulse response by the
known `n_pre * samples_per_ui` offset before cursor detection: FOM then
matches PyChOpMarg bit-for-bit (`-24.1650564425157` both sides), versus
`-23.834...` without it.

This is **not** a MATLAB-alignment issue — it's a pre-existing
architectural gap (`PulseResponse.from_signal`'s window search isn't
circular, even though `SystemGrid.pulse_response`'s IFFT output genuinely
is periodic) that the old FFE convention happened to paper over, by
coincidentally keeping the cursor far enough from index 0. Any config
whose net delay places the cursor near the array edge could hit this
regardless of the FFE convention change. Left unresolved for now — needs
its own decision (build a delay margin into `Link`/`SystemGrid`, or make
`PulseResponse.from_signal`'s window wrap circularly) before re-enabling
this test.

## `TapWeightRxFfe` likely needs the same cursor-referenced fix as Tx FFE

Status: found while investigating what's next to verify against MATLAB
after the Tx FFE fix above; not yet confirmed with a `matlab_golden` test,
not yet fixed.

`TapWeightRxFfe.transfer_function()` (`src/serdeskit/rx_ffe/tap_weight.py`)
is first-tap-referenced (`delays = np.arange(len(tap_weights))`) — by its
own docstring, deliberately built to match PyChOpMarg's `Hffe_Rx`, not
independently checked against MATLAB.

MATLAB's real Rx FFE *signal-path* application reuses the exact same
`FFE.m` primitive as the Tx side (`Vfiltered = FFE(Cmod, param.RxFFE_cmx,
spui, V)`, line 3850) — the same function `matlab_golden/lib/FFE.m` was
extracted from for the Tx FFE golden test above, and thus the same
cursor-referenced convention (delay 0 at the cursor tap, `RxFFE_cmx`
precursor taps at negative delay). If so, `TapWeightRxFfe` has the exact
same bug `TapWeightFfe` did before this file's Tx FFE section — just
never surfaced, because every config this project has exercised uses Rx
FFE as a single unity tap, where the convention is moot (a 1-tap array
has no reference point to disagree about).

Separately, MATLAB also computes a frequency-domain `H_Rx_FFE` used only
for the receiver-noise term (`sigma_N`, around 93A-35) via `exp(-1j*2*pi*
(ii+1)*f/param.fb)` — an apparent extra 1-UI offset (`ii+1`, not `ii`)
relative to `FFE.m`'s own convention. Not yet investigated whether this
is a genuine second, distinct convention (e.g. a real pipeline-latency
term specific to the noise calc) or an artifact of how `phase_memory`'s
columns are laid out; needs its own read of `phase_memory`'s construction
and `RxFFE_cmx`/`RxFFE_cpx`/`force()`'s `t_s` sampling-point logic before
concluding anything.

**Next step, if picked up**: same recipe as the Tx FFE fix — extend
`matlab_golden/generate/gen_ffe.m` (or add a sibling script) to call
`FFE.m` with `cmx > 0`, write a golden CSV, add `tests/rx_ffe/
test_tap_weight_vs_matlab.py` asserting direct agreement (TDD: should
fail first against the current first-tap-referenced implementation),
then fix `TapWeightRxFfe.transfer_function()`'s `delays` the same way
`TapWeightFfe.transfer_function()`'s were fixed.

## Full composed pulse response vs. MATLAB's `best_sbr`: root cause found and fixed (missing Tx risetime filter), ~1-2% residual remains

Status: root cause confirmed and fixed. A small residual gap remains,
not yet root-caused.

**Root cause**: `Link.ffe_channel_ctle_pulse_response()` composed
channel + CTLE + Tx FFE + Rx AFE + Rx FFE, but omitted the transmitter's
finite-risetime filter entirely — the method's own docstring used to
claim "(93A-19) says it has no role in the pulse response". That
assumption was wrong. MATLAB COM3.70's own `s21_pkg_tester`
(`com_ieee8023_93a_370.m:9839-9856`) bakes a risetime filter `H_t` into
every `chdata(i).sdd21` whenever `OP.FORCE_TR` is set (its own comment:
"should be set to 1 in most later config sheets" — not a rare case), and
this project's real C2C config does set it. Confirmed two ways: (1)
dividing a live run's captured `chdata(1).sdd21`
(`matlab_golden/data/uneq_h_c2c_thru.csv`) by the already-verified
channel+package+Butterworth product matched the theoretical `H_t`
formula to 5.3e-10; (2) a proper standalone `matlab_golden` extraction
(`matlab_golden/lib/tx_transition_time_filter.m`, verbatim from
`s21_pkg_tester`, no live COM run needed) golden-tests it directly
(`tests/tx_filter/test_risetime_vs_matlab.py`).

**Fixed**: `src/serdeskit/tx_filter/risetime.py`'s `TxRisetimeFilter`
(pre-existing but never wired into `Link`, and itself missing a term)
now implements the real formula —

```
H(f) = exp(-2*(pi*f*risetime/1.6832)^2) * exp(-1j*2*pi*f*risetime*3)
```

— the same Gaussian magnitude as the textbook 93A-46, plus a
`exp(-1j*2*pi*f*risetime*3)` phase (a pure `3*risetime` delay) the old
implementation didn't have. `Link` gained a `tx_filter` field (new
`TxFilter` Protocol, `src/serdeskit/link/link.py`) and now includes it
in `h`; `Com._next_links()`/`_fext_links()` propagate `self.link.
tx_filter` to aggressor Links the same way they already do for
`ctle`/`rx_afe`/`rx_ffe`. Every existing `Link(...)` call site needed a
`tx_filter=...` update: this project's own real C2C config uses
`TxRisetimeFilter(risetime=0.0075e-9)` (7.5 ps, this config's own `T_r`);
every PyChOpMarg-comparison test/example uses `TxRisetimeFilter(risetime=0.0)`
(the identity) — confirmed correct by checking PyChOpMarg's own `COM.H()`
(`Htx * H21 * Hr * Hctf * ...`), which has no risetime factor at all, so
matching it means applying none on this side either.

Before the fix: cursor 70.63 mV (serdeskit) vs. 63.30 mV (MATLAB), a
+11.6% divergence, with 267/321 samples in a ±5 UI window exceeding
`rtol=1e-3, atol=1e-4` (worst point 15% of cursor, RMS 4.7% of cursor).
After: cursor 63.68 mV vs. 63.30 mV, **+0.6%**, 111/321 samples still
outside tolerance (worst point ~2.0% of cursor, RMS 0.65% of cursor) —
`tests/link/test_pulse_response_vs_matlab_c2c.py`'s two tests are left
failing at their original tight tolerance rather than loosened, since
this residual is real and not yet explained (see below), not something
to paper over.

**Update: root cause of the "uneq" (pre-CTLE/FFE) portion of the residual confirmed and fixed.**
`Link.uneq_pulse_response()` (channel+tx_filter+rx_afe, no CTLE/FFE) vs.
MATLAB's `chdata(1).uneq_pulse_response` initially showed a real shape
gap (172/321 samples outside `rtol=1e-3, atol=1e-4` in a ±5 UI window,
worst point ~1.4% of peak) even though the *frequency-domain* product
feeding the IFFT already matched MATLAB to floating-point precision
(`test_uneq_h_vs_matlab_c2c.py`) — meaning the gap was in the IFFT/pulse-
shaping step itself, not in any stage's own formula.

Root cause: `SystemGrid.pulse_response()` does one IFFT over the whole
periodic record (`x_sinc(f) * H(f)`, then IFFT). MATLAB's
`s21_to_impulse_DC` (`com_ieee8023_93a_370.m:9895-9964`) does something
structurally different: IFFT to an impulse response, **truncate** once
its magnitude decays below `OP.impulse_response_truncation_threshold`
(default `1e-3` of peak, confirmed this config doesn't override it), and
only *then* box-car-integrate into a pulse response — via `filter()`, a
linear (causal, non-circular) FIR, not a circular convolution, since the
truncated array is no longer the periodic record `pulse_response()`
assumes.

**Fixed** by adding `SystemGrid.truncated_pulse_response()`
(`src/serdeskit/link/system_grid.py`) — replicates MATLAB's IFFT-then-
truncate-then-linear-box-car exactly — and switching
`Link.uneq_pulse_response()` to use it instead of `pulse_response()`.
Deliberately a *new* method, not a change to `pulse_response()` itself:
`pulse_response()` is depended on by most of this project's test suite,
including several bit-exact (`atol=1e-15`) PyChOpMarg comparisons that
don't do this truncation, so changing it in place would have risked
regressing already-verified behavior for no reason — this is scoped to
exactly the one caller currently being investigated.

Result: `tests/link/test_uneq_pulse_response_vs_matlab_c2c.py`'s shape
test now passes at **0/321 mismatched**, peak indices align exactly
(0 samples apart, vs. 15 before), peak value within 0.0004%, worst point
in the window within 0.0006% of peak — floating-point-level agreement.

**Update: full composed pulse response also closed, via a new time-domain path.**
`ffe_channel_ctle_pulse_response()` (frequency-domain composition, one
IFFT) still uses the untouched `pulse_response()` and still fails at its
original ~1-2% residual — left as-is, still documenting a real,
unexplained gap in that specific method.

But the residual isn't actually about CTLE/FFE's *own* formulas — it's
that MATLAB never composes this in the frequency domain at all past the
"uneq" stage. `TwoStageCtle.process()` and `TapWeightFfe.process()` are
now real implementations (previously `NotImplementedError` stubs),
TDD-verified against MATLAB's own `TD_CTLE` and `FFE.m` directly, each to
floating-point precision (`tests/ctle/test_two_stage_process_vs_matlab.py`,
`tests/ffe/test_tap_weight_process_vs_matlab.py`). `Link.
sbr_pulse_response()` chains them exactly the way MATLAB's real pipeline
does — `truncated_impulse_response()` → `ctle.process()` (twice, CL120d's
two stages) → `box_car_integrate()` → `ffe.process()` — and matches
MATLAB's `best_sbr` end to end: peak value within 0.0002%, 0/321 samples
outside tolerance in a ±5 UI window (`tests/link/
test_sbr_pulse_response_vs_matlab_c2c.py`). The "S-parameter
extrapolation" and "time- vs frequency-domain composition" hypotheses
above are superseded by this — composition method was the whole story.

`sbr_pulse_response()` is a new, additional method, not a replacement for
`ffe_channel_ctle_pulse_response()` — the latter stays as-is (still
useful, still frequency-domain, still has its own known residual) rather
than being rewritten in place.
