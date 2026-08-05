# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

System-level (behavioral, not SPICE) SerDes link modeling: channel
(S-parameter) + CTLE + FFE + DFE + ADC, composed into an end-to-end link,
producing eye diagrams / bathtub curves / BER. A personal learning project
accompanying Texas A&M's ECEN 720 "High-Speed Links" — built as a
well-architected tool, not throwaway scripts.

## Architecture

Modeled on `~/Documents/VirtuosoKit` (Go). Preserve:

- **Packages, not flat modules** — every concept gets its own directory
  under `src/serdeskit/` (`link/`, `channel/`, `ctle/`, `ffe/`, ...).
- **`typing.Protocol`, not `abc.ABC`** — interfaces defined where consumed
  (e.g. in `link/link.py`), satisfied implicitly. One Protocol per stage
  kind, not a shared generic `Stage`.
- **Domain data is plain and presentation-free** — results are
  `@dataclass(frozen=True)`, never a plot; the domain layer never imports
  matplotlib.
- **Constructor injection over globals/singletons.**

`mypy --strict` enforces this — treat a failure like a Go build failure,
not a lint warning to clean up later.

## Comments

Keep them short — a comment restating the code is wasted effort. Write one
only for a genuinely non-obvious *why*. Investigation narratives belong in
commit messages, not inline or in long docs.

## Current policy: MATLAB COM3.70, not PyChOpMarg, is authoritative

`reference/matlab/COM3.70/com_ieee8023_93a_370.m` (gitignored, user has it
locally) is the correctness target. PyChOpMarg is still used for some
golden tests (catches transcription bugs) but isn't the target — where the
two disagree, MATLAB wins.

**Never guess MATLAB's behavior from reading its source.** Run a small,
git-tracked script against a real MATLAB install and compare actual output
(`matlab_golden/README.md` has the recipe: extract into `lib/*.m`, call
from `generate/*.m`, commit output to `data/*.csv`, assert against it in
`tests/*/test_*_vs_matlab.py`).

Open gaps: `docs/known-issues.md`.

## Commands

```bash
source .venv/bin/activate   # Python 3.12, not system python3
pip install -e ".[dev]"
mypy                         # strict — must pass before committing
pytest -q
ruff check .
```
