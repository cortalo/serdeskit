% Calls MATLAB COM3.70's own TD_CTLE (lib/TD_CTLE.m, copied verbatim)
% twice in sequence -- the CL120d two-stage structure com_ieee8023_93a_370.m
% actually uses (com_ieee8023_93a_370.m:5925,5930, confirmed by reading
% the call sites, not guessed): main stage (zero/pole1/pole2/dc_gain),
% then shelf stage (same freq for zero+pole1, pole2=100e100 so its
% bilinear-transformed pole lands at z=-1 and cancels TD_CTLE's own
% always-present z=-1 zero, collapsing it to a true single-pole/zero
% stage -- shelf_gain).
%
% Applied to a unit impulse -- TD_CTLE is LTI (a bilinear-transformed IIR
% filter), so its output on a unit impulse *is* its own impulse response,
% the cleanest possible test signal.
%
% Uses this project's own real C2C config CTLE point (already verified
% in the frequency domain, tests/ctle/test_two_stage_vs_matlab.py):
% zero=pole1=21.25 GHz, pole2=53.125 GHz, dc_gain=-3dB; shelf=0.6640625
% GHz, shelf_gain=-2dB. baud_rate=53.125 GBd, samples_per_ui=32 (=
% TD_CTLE's own `oversampling` arg, com_ieee8023_93a_370.m:5925).
%
% Ground truth for tests/ctle/test_two_stage_process_vs_matlab.py.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

baud_rate = 53.125e9;
samples_per_ui = 32;

n = 2000;  % samples -- plenty: pole/zero freqs (~21-53 GHz) decay in
           % ~tens of samples at this sample rate (baud_rate*samples_per_ui
           % =~1.7 THz)
ir_in = zeros(1, n);
ir_in(1) = 1.0;

stage1 = TD_CTLE(ir_in, baud_rate, 21.25e9, 21.25e9, 53.125e9, -3.0, samples_per_ui);
stage2 = TD_CTLE(stage1, baud_rate, 0.6640625e9, 0.6640625e9, 100e100, -2.0, samples_per_ui);

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'td_ctle_impulse_response.csv'), 'w');
fprintf(fid, 'sample,response\n');
for i = 1:n
    fprintf(fid, '%d,%.10g\n', i-1, stage2(i));
end
fclose(fid);
fprintf('wrote td_ctle_impulse_response.csv (%d points)\n', n);
