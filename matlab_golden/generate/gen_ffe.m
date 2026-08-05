% Applies MATLAB COM3.70's own FFE (lib/ffe.m, copied verbatim -- a
% time-domain circshift-based tap-delay sum, cursor-referenced: shift 0
% for the cursor tap, negative for pre-cursor, positive for post-cursor)
% to a delta impulse, then FFTs the result to get its frequency response
% at the DFT's own discrete bins -- ground truth for
% tests/ffe/test_tap_weight_vs_matlab.py, which evaluates serdeskit's own
% analytic (first-tap-referenced) DTFT formula at those same bins.
%
% Simple, arbitrary test taps (not a real config's own) since this is
% purely checking the delay/phase convention, already-verified
% separately (matlab_golden's own KR/C2C investigation) from the actual
% tap *values* a real search would pick.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

N = 64;
spui = 8;  % samples per UI -- matches serdeskit's tap_delay = UI, sampled at N/spui per tap spacing... see py side
delta = zeros(1, N);
delta(1) = 1;

C = [0.1, 0.85, -0.05];  % [pre1, cursor, post1]
cmx = 1;  % 1 precursor tap

V0 = FFE(C, cmx, spui, delta);

H = fft(V0);
freqs = (0:N-1) / N;  % cycles/sample

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'ffe_fft.csv'), 'w');
fprintf(fid, 'cycles_per_sample,samples_per_ui,h_r,h_i\n');
for i = 1:N
    fprintf(fid, '%.10g,%d,%.10g,%.10g\n', freqs(i), spui, real(H(i)), imag(H(i)));
end
fclose(fid);
disp('wrote ffe_fft.csv');
