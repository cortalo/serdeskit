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
the substitute for what Go's compiler guarantees for free (Python's
`Protocol` is opt-in, checked by `mypy` rather than the compiler). **Treat
a `mypy --strict` failure as equivalent to a Go build failure: it blocks
committing, not just a lint warning to clean up later.**

## Comments

Keep them short. A comment that restates what the code already shows is
wasted effort — reading the code directly is usually faster, including
for an AI picking this up later. Write one only for a genuinely
non-obvious *why* (a MATLAB cross-reference, a subtle invariant, a
rejected-alternative reason), and keep it to a sentence or two. Detailed
investigation narratives (numbers, hypotheses, what was ruled out) belong
in `docs/known-issues.md`, not inline.

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

**Verified against MATLAB so far** (each via the recipe above): package
S-parameters incl. the RX `tline_segments` order bug, `SParameterChannel.
transfer_function()`'s erroneous raised-cosine taper (removed),
`TwoStageCtle`, `TapWeightFfe` (a real cursor- vs first-tap delay-
convention bug, fixed), `RxAfeButterworth`, `TxRisetimeFilter` (missing a
phase term, fixed, now wired into `Link` via a `tx_filter` field). Commit
history has the details; `docs/known-issues.md` has the investigation
writeups.

**Known open issues** (full detail, numbers, and next steps in
`docs/known-issues.md` — don't restate them here):

- `TapWeightRxFfe` likely has the same delay-convention bug `TapWeightFfe`
  had. Not yet verified or fixed; zero practical impact so far (every
  config exercised uses Rx FFE as a single unity tap).
- `PulseResponse.from_signal`'s Muller-Mueller search window isn't
  circular, even though the pulse response it searches genuinely is — can
  clip precursor samples if a config's cursor lands near array index 0.
- `pmf.noise_margin` silently saturates at the voltage grid's edge —
  confirmed root cause, not yet fixed.
- ~~The full composed pulse response was missing a Tx risetime filter~~ —
  fixed, then a ~1-2% residual, also fixed: MATLAB composes CTLE/FFE in
  the time domain, not frequency-domain multiplication like this
  project's `ffe_channel_ctle_pulse_response()`. `Ctle.process()`/
  `Ffe.process()`/`RxFfe.process()` are now real (TDD-verified against
  `TD_CTLE`/`FFE.m`/`force()`), and `Link.sbr_pulse_response()` chains
  them the way MATLAB really does — matches MATLAB's real `eq_pulse_
  response` (not `fom_result.sbr`, which never applies Rx FFE — a
  separate gap found and fixed the same way) to floating-point precision.
  `ffe_channel_ctle_pulse_response()` itself is untouched and still has
  the residual; `sbr_pulse_response()` is the new, separate method to
  use instead. Full detail in `docs/known-issues.md`.
- **Out of scope for now**: search/optimization-level alignment
  (`EqualizationSearch`, `figure_of_merit`) against MATLAB's own
  `opt_eq`/`calc_fom`.

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
