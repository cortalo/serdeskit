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

## State of things (as of this file's writing)

- `common/types.py`: `Signal` (samples + `fs` + `t0` — `t0` exists
  specifically to prevent the class of index-misalignment bug that silently
  corrupted a BER estimate during earlier `serdespy` exploration: a stage
  must never require callers to track array-index-to-absolute-time offsets
  by hand). No shared `Stage` protocol — see the architecture note above.
- `link/link.py`: the `Channel` protocol (`process(Signal) -> Signal`),
  `Link` (constructor: `channel: Channel`, single stage so far), and the
  result types `LinkResult`/`EyeData`. `Link.simulate(bits, fs,
  symbol_rate)` upsamples `bits` into a waveform, runs it through the
  channel, and slices out an eye (2-UI sliding window, 1-UI step). No
  bathtub-curve/BER extraction yet — not started, not merely unimplemented:
  whether to use PDA (worst-case, deterministic) or statistical
  (Gaussian-tail-extrapolated) BER methodology was an open decision;
  `pmf/`'s delta-PMF/convolution machinery (below) is the statistical route
  taking shape. `Ctle`/`Ffe` protocols are about to be added alongside
  `Channel`, one per stage kind (see architecture note above) — `Link`
  itself doesn't yet accept them.
- `channel/`: `PassThroughChannel` (no-op stand-in) and `SParameterChannel`
  (Touchstone-loaded, via `scikit-rf`) — both satisfy `Channel` implicitly.
- `pmf/`: `delta_pmf`/`gaussian_pmf`/`combine_pmfs`/`noise_margin` — COM's
  noise/interference PMF construction (IEEE 802.3-2022 Annex 93A),
  golden-tested against `PyChOpMarg`.
- `pulse_response/`: `PulseResponse` (a `Signal` subtype with cursor/UI
  identified — `is-a Signal`, purely additive fields), `from_signal`
  (Muller-Mueller cursor location), `local_slopes`, `signal_amplitude`.
  Not yet wired to a real multi-stage `Link` output — next up is teaching
  `Link` to produce a `PulseResponse` by running an impulse through its
  stages (channel, then CTLE/FFE once they exist), which is what motivates
  adding `Ctle`/`Ffe` protocols now.
- No CTLE/FFE/DFE/ADC implementations exist yet.

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
