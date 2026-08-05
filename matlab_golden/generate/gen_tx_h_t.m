% Calls MATLAB COM3.70's own H_t (transmitter transition-time filter)
% computation (lib/tx_transition_time_filter.m, copied verbatim from
% s21_pkg_tester) at a handful of test frequencies -- ground truth for
% tests/tx_filter/test_risetime_vs_matlab.py.
%
% OP.T_r_filter_type=1, OP.T_r_meas_point=0: the branch OP.FORCE_TR
% forces every config this project has exercised into (its own comment:
% "should be set to 1 in most later config sheets") -- confirmed for the
% real C2C config via matlab_golden/generate/gen_sbr_c2c.m's own live
% run (docs/known-issues.md's "full composed pulse response" entry).
% OP.IDEAL_TX_TERM=false, matching that same config.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

f = [1e7, 1e9, 5e9, 1.001e10, 2.001e10, 2.651e10, 3.001e10, 4.001e10, 5e10];

OP.IDEAL_TX_TERM = false;
OP.T_r_filter_type = 1;
OP.T_r_meas_point = 0;
OP.transmitter_transition_time = 0.0075;  % ns -- this project's C2C config's own T_r

H = tx_transition_time_filter(f, OP);

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'tx_h_t.csv'), 'w');
fprintf(fid, 'freq,h_r,h_i\n');
for i = 1:length(f)
    fprintf(fid, '%.10g,%.10g,%.10g\n', f(i), real(H(i)), imag(H(i)));
end
fclose(fid);
disp('wrote tx_h_t.csv');
