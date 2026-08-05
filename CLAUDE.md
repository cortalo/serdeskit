# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

System-level (behavioral, not SPICE) SerDes link modeling: channel (S-parameter)
+ CTLE + FFE + DFE + ADC, composed into an end-to-end link, producing eye
diagrams / bathtub curves / BER. A personal learning project accompanying
Texas A&M's ECEN 720 "High-Speed Links" course materials — the goal is both
to learn SerDes system-level modeling and to build a well-architected tool
while doing it, not to move fast and leave a pile of throwaway scripts (see
`~/Documents/test/serdespy-eval/` for what that looks like — many one-off
`N_description.py` exploration scripts, valuable for the exploration but not
something to build on top of).

## Architecture: clean architecture, Go-flavored

The explicit model for this project is **`~/Documents/VirtuosoKit`**
(`autorouter/` and `placer/`, both Go) — read `autorouter/session/session.go`
and `autorouter/common/types.go` before writing new code here if unfamiliar
with the pattern. Key properties to preserve:

- **Packages, not flat modules.** Every concept that might grow multiple
  files (implementation variants, focused unit tests) gets its own directory
  under `src/serdeskit/`, e.g. `link/`, `common/`, and (as they're added)
  `channel/`, `ctle/`, `ffe/`, `dfe/`, `adc/`. Don't add a bare `foo.py`
  sibling to `link/` — a new concept gets a new directory.
- **Interfaces defined where consumed, satisfied implicitly.** Use
  `typing.Protocol`, not `abc.ABC` — a concrete stage class should never
  need to import or subclass anything from the package that consumes it
  (mirrors "infra/postgres.Store satisfies it implicitly" from the author's
  other project, `~/Documents/digitalgarden-backend`). Each stage kind gets
  its own Protocol (`Channel`, and — as they're added — `Ctle`, `Ffe`),
  defined in `link/link.py` where `Link` (the consumer) needs them —
  deliberately **not** a single generic `Stage` protocol shared across all
  of them. They'd currently be structurally identical
  (`process(Signal) -> Signal`), but DFE won't fit that shape at all (it's
  a feedback structure, not a plain filter), so a uniform `Stage` type was
  never going to cover the whole pipeline anyway — separate names keep
  each stage's role self-documenting in `Link`'s own type signature
  instead.
- **Domain data is plain and presentation-free.** `Link.simulate()` returns
  `LinkResult` — a `@dataclass(frozen=True)` of arrays/floats — never a
  matplotlib figure. Plotting is a separate, outer layer (not yet built)
  that consumes `LinkResult`; the domain layer must never `import
  matplotlib`. This is what keeps `Link` unit-testable by direct assertion.
- **Constructor injection over globals/singletons.** A `Link` is built from
  an explicit list of `Stage`s passed to its constructor, the same way
  `Session` in VirtuosoKit takes `canvas`/`router` as constructor args.

`mypy --strict` is the enforcement mechanism for all of the above — it is
the substitute for what Go's compiler guarantees for free. **Treat a
`mypy --strict` failure as equivalent to a Go build failure: it blocks
committing, not just a lint warning to clean up later.**

## Why Python, not Go

Considered and rejected writing this in Go (the author's preferred language,
and what VirtuosoKit is written in) specifically for this project: the
numerical ecosystem gap (NumPy/SciPy's global optimizers, FFT edge-case
correctness, `scikit-rf`'s Touchstone/S-parameter handling) is large and
reimplementing it would cost real time for zero architectural or
domain-learning benefit. Unlike VirtuosoKit's router (where Go's
compiled-speed + goroutines were a genuine performance requirement for
grid-search/geometry work), this project's core computation — convolution,
occasional optimizer runs — doesn't need that. The one real gap versus Go
is that Python's structural typing (`Protocol`) is opt-in (checked by
`mypy`, not the interpreter) rather than compiler-enforced — hence the hard
`mypy --strict` gate above as the deliberate substitute.

## Current goal: aligning with MATLAB COM3.70, not PyChOpMarg

**Project policy (settled):** the official MATLAB COM3.70 tool
(`reference/matlab/COM3.70/com_ieee8023_93a_370.m`, gitignored — the user
has it locally) is the authoritative reference this project is trying to
match. `PyChOpMarg` (a Python reimplementation) was only ever adopted
early on as a development convenience — easier to cross-check against
than MATLAB. Where the two disagree, **MATLAB wins**. Most of the existing
test suite still golden-tests against PyChOpMarg; that remains useful
(catches transcription bugs) but is no longer the correctness target.
Full writeup: `docs/known-issues.md`'s "Project policy" section.

**Methodology — never guess MATLAB's behavior from reading its source.**
Always execute a small, git-tracked MATLAB script against a real MATLAB
install and compare its actual output. The established pattern (see
`matlab_golden/README.md`):

1. Extract the exact MATLAB function(s) verbatim into `matlab_golden/lib/*.m`
   (never modified).
2. Call them from a small script in `matlab_golden/generate/*.m` with
   fixed, explicit inputs.
3. Commit the output to `matlab_golden/data/*.csv` (not gitignored — this
   is ground truth, not build output).
4. Write a pytest in `tests/*/test_*_vs_matlab.py` that asserts against
   that CSV directly, never a hand-copied number.

When a divergence from MATLAB is found and fixed, and it breaks existing
PyChOpMarg-golden tests as a result, resolve each broken test one of three
ways (all with precedent in `docs/known-issues.md`): (a) if the difference
is a provable, derivable factor (e.g. a linear phase), compensate for it
directly in the test; (b) if PyChOpMarg's own code crashes or has no
patchable seam, `@pytest.mark.skip` with a reason and a dedicated pytest
marker (registered in `pyproject.toml`, e.g. `rx_tline_segments`,
`h21_taper_removal`, `ffe_cursor_referenced_delay`) so the batch stays
`grep`/`pytest -m`-discoverable; (c) if the test doesn't actually compare
against PyChOpMarg (just hardcodes prior output), recompute the expected
values.

**Verified against MATLAB so far** (each via the recipe above — commits
in parentheses): package S-parameters incl. the RX `tline_segments` order
bug (`631f2f5`), `SParameterChannel.transfer_function()`'s erroneous
raised-cosine taper, now removed (`59fe9ba`), `TwoStageCtle` (`be754bf`,
no source change needed — already correct), `TapWeightFfe` including a
real cursor- vs first-tap delay-convention bug, now fixed (`f85c59b`,
`687129e`), `RxAfeButterworth` (`db5f812`, no source change needed),
`TxRisetimeFilter` — previously unwired into `Link` entirely and missing
a phase term; now implements MATLAB's real `H_t` formula (Gaussian
magnitude + a `3*risetime` delay phase) and is wired into
`Link.ffe_channel_ctle_pulse_response()` via a new `tx_filter` field.

**Known open issues** (full detail in `docs/known-issues.md`):

- `TapWeightRxFfe` likely has the same delay-convention bug `TapWeightFfe`
  had — MATLAB's real Rx FFE signal path reuses the same cursor-referenced
  `FFE.m` primitive as Tx. Not yet verified with a `matlab_golden` test or
  fixed. Zero practical impact today (every config exercised so far uses
  Rx FFE as a single unity tap), but a real, unverified divergence.
- `PulseResponse.from_signal`'s Muller-Mueller search window is bounded
  (`range(max(0, peak_ix - nspui), ...)`), not circular, even though
  `SystemGrid.pulse_response()`'s IFFT output is genuinely periodic. If a
  config's net delay places the cursor close to sample index 0, precursor
  samples get clipped instead of wrapping — found via the Tx FFE fix
  above (which removed an incidental delay margin), confirmed real via
  `tests/optimize/test_search.py::test_matches_pychopmargs_own_opt_eq`
  (currently `@pytest.mark.skip`-ed). Unresolved: needs either a delay
  margin built into `Link`/`SystemGrid`, or a circular window in
  `PulseResponse.from_signal`.
- `pmf.noise_margin` silently saturates at the voltage grid's edge —
  confirmed root cause, not yet fixed.
- The full composed pulse response (channel + tx_filter + CTLE + FFE +
  RxAFE, IFFT'd) was missing a Tx risetime filter entirely — this file
  used to claim (93A-19) gives it "no role in the pulse response", which
  was wrong. **Root cause found and fixed**: `TxRisetimeFilter`
  (`src/serdeskit/tx_filter/risetime.py`, golden-tested,
  `tests/tx_filter/test_risetime_vs_matlab.py`) is now wired into `Link`
  via a new `tx_filter` field, closing most of the gap (cursor magnitude
  +11.6% off before the fix, +0.6% after). A smaller residual (~1-2%,
  111/321 samples in a ±5 UI window still outside `rtol=1e-3,
  atol=1e-4`) remains, not yet root-caused — matches an earlier informal
  "~2% off" estimate, plausibly the same unexplained gap. Leading
  untested candidates: the S-parameter extrapolation-range/method
  difference (serdeskit edge-pads past 60 GHz measured data; MATLAB's
  `interp_Sparam` uses `'linear_trend_to_DC'`/
  `'extrap_cubic_to_dc_linear_to_inf'`), or MATLAB composing this in the
  time domain (IFFT once, then `TD_CTLE`'s IIR filter, then `FFE.m`'s
  circular-shift-sum) rather than serdeskit's single frequency-domain
  multiply + one IFFT. Full detail and next step (isolate by stage,
  post-CTLE this time) in `docs/known-issues.md`.
- **Explicitly out of scope for now**: search/optimization-level alignment
  (`EqualizationSearch`, `figure_of_merit`) against MATLAB's own
  `opt_eq`/`calc_fom` — hasn't been started. Everything verified so far is
  per-stage transfer functions and pulse-response composition, not the
  tap-combination search itself.

No DFE or ADC implementation exists yet. Bathtub-curve/BER extraction
(PDA vs. statistical/Gaussian-tail methodology) is still an open design
question; `pmf/`'s delta-PMF/convolution machinery is the statistical
route taking shape, deliberately not PDA (see the "PDA skipped" note if
resuming that decision).

## Commands

```bash
source .venv/bin/activate   # Python 3.12 — not the system default python3
                             # (3.14), which some deps in this ecosystem
                             # don't yet support cleanly
pip install -e ".[dev]"
mypy                         # strict mode, see pyproject.toml — must pass
pytest -q
ruff check .
```
