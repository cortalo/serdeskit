# Eye diagram data: design survey

Status: v1 implemented in `link/link.py::_extract_eye` — 1 UI step, 2 UI
window (conclusion 2 below), `EyeData.traces` shape (conclusion 1). Not yet
done: zero-crossing alignment (conclusion 3) and a transient-skip parameter
(conclusion 4) — v1 just starts windows at sample 0, which is only correct
because `PassThroughChannel` has zero delay and zero transient. Revisit both
before a real S-parameter channel replaces it.

## Sources compared

- MATLAB, `reference/ecen720/channel_data.m` (gitignored, downloaded from
  https://people.engr.tamu.edu/spalermo/ecen720.html — TAMU ECEN 720 course
  reference code)
- serdespy, `~/Documents/serdespy/serdespy/eye_diagram.py`
  (`simple_eye`, `rx_jitter_eye`)
- PyBERT, `~/Documents/PyBERT/src/pybert/utility/sigproc.py` (`calc_eye`),
  driven from `~/Documents/PyBERT/src/pybert/models/bert.py`

## Comparison

| Dimension | MATLAB (`channel_data.m`) | serdespy (`simple_eye`) | PyBERT (`calc_eye`) |
|---|---|---|---|
| Data/plot separation | Mixed script, but `eye_data` itself is a clean 2D array before it gets plotted | Not separated — plots inline with matplotlib, never returns the traces | Actually separated — `calc_eye` is a pure data function (no matplotlib import), plotting happens elsewhere |
| Output shape | Per-trace raw waveform, one column per trace | Same idea, row-major via `np.split` | Pre-rasterized (height × width) heat-map/histogram (pixel hit counts) |
| Window step | **1 UI** step, 2 UI width, sliding/overlapping traces | **2 UI** step (`np.split`, non-overlapping) — only catches every other transition | **1 UI** step, matches MATLAB |
| Alignment/trigger | Hardcoded constant `offset=144`, same for every trace | None — assumes signal starts exactly at phase 0 | Uses `clock_times` from CDR if given; else falls back to zero-crossing detection on the signal itself |
| Transient handling | Hardcoded: skip first 55 / last 500 symbols | None | Caller's responsibility (`ignore_samps`, computed in `bert.py` before calling `calc_eye`) |

## Conclusions (for whenever eye extraction gets implemented)

1. **Data shape**: keep the already-sketched `EyeData.traces` —
   `(samples_per_2ui, n_traces)`. This matches what MATLAB and serdespy
   actually compute under the hood. Don't adopt PyBERT's pre-rasterized
   heat-map — baking `height`/`y_max` (a rendering resolution choice) into
   domain data violates "domain data is plain and presentation-free"
   (CLAUDE.md). Rasterizing into a heat-map, if wanted, is a plotting-layer
   concern for later.

2. **Window step**: use 1 UI step / 2 UI width (MATLAB and PyBERT agree).
   Reject serdespy's non-overlapping 2-UI `np.split` — it silently drops
   half the UI-to-UI transitions.

3. **Alignment**: prefer PyBERT's zero-crossing-detection fallback over
   MATLAB's hardcoded `offset`. MATLAB's constant is implicitly compensating
   for that specific channel's group delay — it breaks the moment the
   channel changes. Zero-crossing detection re-derives alignment from the
   actual output signal every time, so it keeps working once a real
   S-parameter channel (with its own delay) replaces `PassThroughChannel`,
   with no per-channel recalibration and no dependency on a CDR stage that
   doesn't exist yet.

4. **Transient skip**: make it a parameter instead of MATLAB's hardcoded
   55/500 symbol counts. `PassThroughChannel` has no transient (0 is fine);
   revisit the right default once a real S-parameter channel exists.

Open before implementing: exact zero-crossing detection method (single
first-crossing + fixed UI stride, like PyBERT's fallback branch, vs.
re-detecting per transition), and how `n_traces`/skip-symbol-count get
threaded through `Link.simulate()`'s signature.
