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
  other project, `~/Documents/digitalgarden-backend`). See
  `src/serdeskit/common/types.py`'s `Stage` protocol.
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

## State of things (as of this file's writing)

- `common/types.py`: `Signal` (samples + `fs` + `t0` — `t0` exists
  specifically to prevent the class of index-misalignment bug that silently
  corrupted a BER estimate during earlier `serdespy` exploration: a stage
  must never require callers to track array-index-to-absolute-time offsets
  by hand) and the `Stage` protocol (`process(Signal) -> Signal`).
- `link/link.py`: `Link` (constructor: `stages: list[Stage]`,
  `symbol_rate: float`) and the result types `LinkResult`/`EyeData`/
  `BathtubCurve`. `Link.simulate()` runs stages in order, then currently
  raises `NotImplementedError` at the eye/bathtub extraction step —
  **deliberately unimplemented**, not a bug: whether to use PDA
  (worst-case, deterministic) or statistical (Gaussian-tail-extrapolated)
  BER methodology is an open decision, to be made after working through
  the ECEN 720 labs (`~/Documents/test/ecen720-materials/`), not guessed at
  here. Don't fill this in without that context.
- No concrete `Stage` implementations exist yet (no channel/CTLE/FFE/DFE/ADC
  packages). `tests/link/test_link.py` uses an inline no-op `PassThrough`
  stage as a stand-in.

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
