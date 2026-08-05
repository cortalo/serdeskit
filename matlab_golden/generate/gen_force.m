% Calls MATLAB COM3.70's own force() (lib/force.m, copied verbatim) --
% the function that actually applies Rx FFE to the signal path
% (com_ieee8023_93a_370.m:732, inside Apply_EQ, called with fom_result's
% own already-chosen RxFFE coefficients as `C` -- confirmed by reading
% Apply_EQ's own call site, not guessed). Depends on FFE.m (already
% extracted) and FFE_Fast.m (also extracted here).
%
% cpx=0 (no post-cursor taps) is deliberate: force()'s own "DNH" backoff
% loop (lib/force.m, the "for i=0:min(param.ffe_backoff,cpx)" block)
% adaptively re-picks Cmod by testing whether zeroing trailing
% post-cursor taps improves a figure of merit -- a search, not just
% filter application, and this project's own architecture keeps searches
% (EqualizationSearch) separate from applying already-chosen taps
% (TapWeightFfe.process() etc.), explicitly out of scope for now (see
% CLAUDE.md). With cpx=0, min(param.ffe_backoff, cpx)=0 regardless of
% param.ffe_backoff's own value (default 4), so the loop runs exactly
% once at i=0 and Cmod is provably left unmodified -- isolating this
% golden fixture to "does the FFE application match", the same scope
% test_tap_weight_process_vs_matlab.py already covers for Tx FFE.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

spui = 8;
n = 80;  % 10 UI
peak_idx = 33;  % 1-indexed -- UI index 4 (0-indexed) within the 10 UI span
V = zeros(1, n);
for k = 1:n
    V(k) = exp(-abs(k - peak_idx) / 10);
end

param.RxFFE_cmx = 2;
param.RxFFE_cpx = 0;
param.RxFFE_stepz = 0;  % unused (C is given -- see force.m's own "if ~exist('C','var')" branch)
param.ndfe = 0;          % unused, same reason
param.ffe_backoff = 4;   % this config's own real default (com_ieee8023_93a_370.m:8365) -- included
                         % for fidelity even though cpx=0 makes it a no-op here (see comment above)
param.samples_per_ui = spui;
OP = struct();

C = [0.05, 0.15, 1.0];  % [pre2, pre1, cursor] -- genuinely non-trivial, multi-tap Rx FFE
                         % (unlike this project's own C2C config, whose Rx FFE is a single unity tap)

[Vfiltered, Cmod] = force(V, param, OP, peak_idx, C);

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'force_rx_ffe.csv'), 'w');
fprintf(fid, 'sample,response\n');
for i = 1:length(Vfiltered)
    fprintf(fid, '%d,%.10g\n', i-1, Vfiltered(i));
end
fclose(fid);
fprintf('wrote force_rx_ffe.csv (%d points, Cmod unchanged from input C: %d)\n', ...
    length(Vfiltered), isequal(Cmod(:), C(:)));
