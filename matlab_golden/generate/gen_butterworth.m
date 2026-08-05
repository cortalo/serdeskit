% Calls MATLAB COM3.70's own Butterworth_Filter (lib/Butterworth_Filter.m,
% copied verbatim) at a handful of test frequencies -- ground truth for
% tests/rx_afe/test_butterworth_vs_matlab.py.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

f = [1e7, 1e9, 5e9, 1.001e10, 2.001e10, 2.651e10, 3.001e10, 4.001e10, 5e10];

param.fb = 53.125;   % GBd
param.fb_BW_cutoff = 0.75;  % *fb -- matches this project's own examples

H = Butterworth_Filter(param, f / 1e9, true);  % MATLAB's own f is in GHz here (fb is in GBd)

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'butterworth.csv'), 'w');
fprintf(fid, 'freq,h_r,h_i\n');
for i = 1:length(f)
    fprintf(fid, '%.10g,%.10g,%.10g\n', f(i), real(H(i)), imag(H(i)));
end
fclose(fid);
disp('wrote butterworth.csv');
