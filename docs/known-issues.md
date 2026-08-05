# Known issues

Open, unresolved problems only. Fixed issues live in commit history, not
here.

## `pmf.noise_margin` silently saturates at the voltage grid's edge

`pmf/margin.py::noise_margin` reads the DER0 crossing off a grid spanning
only ±1.1×signal_amplitude. If the combined PMF's CDF already exceeds der0
at the grid's first sample, it returns the grid edge itself, not a real
crossing — `com_db` collapses to the fixed constant `20*log10(1/1.1) ≈
-0.828 dB` regardless of the actual interference distribution. Confirmed
via `examples/compute_com_kr_backplane.py` (KR/PAM4, real backplane,
floating-tap DFE extension not implemented → residual ISI far exceeds the
grid). Not yet fixed.

## `TapWeightRxFfe.transfer_function()` may have the same cursor-convention bug `TapWeightFfe` had

Unverified against MATLAB (only checked against PyChOpMarg's `Hffe_Rx`,
first-tap-referenced). MATLAB's real Rx FFE signal path reuses the same
cursor-referenced `FFE.m` primitive the Tx side does — likely needs the
same fix `TapWeightFfe` already got. Zero practical impact so far: every
config exercised uses Rx FFE as a single unity tap, where the convention
is moot.

## `PulseResponse.from_signal`'s Muller-Mueller search window isn't circular

The pulse response it searches genuinely is periodic (IFFT of a periodic
record), but the search window (`range(max(0, peak_ix - nspui), ...)`)
isn't — can clip precursor samples if a config's cursor lands near array
index 0.

## Search grid coarsening can pick a different (but locally correct) optimum than MATLAB

Not a bug — a coarsened search grid may not contain MATLAB's exact
found-optimal point. Confirmed serdeskit's own search winner scores better
than MATLAB's point under serdeskit's own `figure_of_merit`, given the
grid it's searching.

## Out of scope for now

Search/optimization-level alignment (`search()`, `figure_of_merit`)
against MATLAB's own `opt_eq`/`calc_fom`. No DFE or ADC implementation
yet.
