% Calls MATLAB COM3.70's own Apply_EQ (lib/Apply_EQ.m, copied verbatim)
% end to end -- CTLE (twice, CL120d), box-car, Tx FFE, Rx FFE -- the same
% chain com_ieee8023_93a_370.m:426 runs to build chdata(i).eq_pulse_response,
% what get_pdf() (the real COM/ISI/noise calculation) actually reads.
% Confirmed by reading Apply_EQ's own call site, not guessed.
%
% Deliberately NOT the real C2C channel: this project's own C2C config's
% Rx FFE is a single unity tap (see docs/known-issues.md's "full composed
% pulse response" entry for why that made an earlier comparison misleading
% -- fom_result.sbr and the real eq_pulse_response coincide when Rx FFE is
% trivial). A synthetic uneq_imp_response plus explicit, genuinely
% multi-tap CTLE/Tx FFE/Rx FFE parameters isolates whether
% Link.sbr_pulse_response() -- which has no Rx FFE step at all -- diverges
% from the real eq_pulse_response, which is the actual point of this
% fixture.
%
% cpx=0 for Rx FFE, same reason as gen_force.m: keeps force()'s own
% adaptive backoff search out of scope.
%
% Ground truth for tests/link/test_sbr_pulse_response_missing_rx_ffe.py.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

baud_rate = 53.125e9;
spui = 32;
% n=1214, not a round number: chosen to exactly match what
% SystemGrid.truncated_impulse_response() (threshold=1e-3 default)
% naturally produces for this exact formula on Link's own real grid
% (FREQ_STEP=10e6) -- confirmed empirically in Python, not guessed, so
% tests/link/test_sbr_pulse_response_missing_rx_ffe.py can call
% Link.sbr_pulse_response() directly with a synthetic channel whose
% transfer_function() is this array's own FFT, and land on identical
% starting data on both sides without a Python-to-MATLAB data handoff.
n = 1214;
peak_idx = 801;  % 1-indexed
uneq_imp_response = zeros(1, n);
for k = 1:n
    uneq_imp_response(k) = exp(-abs(k - peak_idx) / 60) * cos((k - peak_idx) * 0.03);
end

param.fb = baud_rate;
param.CTLE_type = 'CL120d';
param.CTLE_fz = 21.25e9;
param.CTLE_fp1 = 21.25e9;
param.CTLE_fp2 = 53.125e9;
param.ctle_gdc_values = -3.0;
param.f_HP = 0.6640625e9;
param.g_DC_HP_values = -2.0;
param.f_HP_Z = [];
param.f_HP_P = [];
param.samples_per_ui = spui;
param.number_of_s4p_files = 1;
param.RxFFE_cmx = 2;
param.RxFFE_cpx = 0;
param.RxFFE_stepz = 0;
param.ndfe = 0;
param.ffe_backoff = 4;

fom_result.ctle = 1;
fom_result.best_G_high_pass = 1;
fom_result.sbr = zeros(1, n);  % length <= uneq_imp_response, so Apply_EQ's own padding branch never triggers
fom_result.txffe = [0.1, 0.85, -0.05];  % [pre1, cursor, post1] -- cursor = 1-(0.1+0.05), matching
                                          % TapWeightFfe(tap_weights=[0.1,-0.05], n_post=1)'s own
                                          % normalization (93A-21), so the Python side's derived
                                          % cursor and this explicit one agree
fom_result.cur = 2;  % 1-indexed cursor position in txffe => cmx = cur-1 = 1 precursor tap
fom_result.RxFFE = [0.05, 0.15, 1.0];  % [pre2, pre1, cursor] -- genuinely non-trivial, unlike this project's real C2C config

chdata(1).uneq_imp_response = uneq_imp_response;
chdata(1).uneq_pulse_response = zeros(1, n);  % only touched by the padding branch, unreachable here
chdata(1).t = (0:n-1) / (baud_rate * spui);
chdata(1).type = 'THRU';

OP.INCLUDE_CTLE = 1;

% Pass 1: no Rx FFE, to find the (deterministic) cursor sample point
% Apply_EQ's own caller (com_ieee8023_93a_370.m:426, via optimize_fom)
% would have already resolved via search -- fixed here explicitly rather
% than re-deriving that search, out of scope the same way EqualizationSearch
% is (see CLAUDE.md).
OP.RxFFE = 0;
chdata_no_rxffe = Apply_EQ(param, fom_result, chdata, OP);
eq_pulse_no_rxffe = chdata_no_rxffe(1).eq_pulse_response;
[~, peak_i] = max(abs(eq_pulse_no_rxffe));
fom_result.t_s = peak_i;

% Pass 2: with Rx FFE, at that same cursor point.
OP.RxFFE = 1;
chdata_with_rxffe = Apply_EQ(param, fom_result, chdata, OP);
eq_pulse_with_rxffe = chdata_with_rxffe(1).eq_pulse_response;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'apply_eq_no_rx_ffe.csv'), 'w');
fprintf(fid, 'sample,response\n');
for i = 1:n
    fprintf(fid, '%d,%.10g\n', i-1, eq_pulse_no_rxffe(i));
end
fclose(fid);

fid = fopen(fullfile(out_dir, 'apply_eq_with_rx_ffe.csv'), 'w');
fprintf(fid, 'sample,response\n');
for i = 1:n
    fprintf(fid, '%d,%.10g\n', i-1, eq_pulse_with_rxffe(i));
end
fclose(fid);
fprintf('wrote apply_eq_no_rx_ffe.csv and apply_eq_with_rx_ffe.csv (%d points, cursor t_s=%d)\n', n, fom_result.t_s);
