% Replicates MATLAB COM3.70's own CL120d two-stage CTLE formula exactly
% (com_ieee8023_93a_370.m's optimize_fom, ~line 5896-5916 -- inline in
% the search loop, not a standalone extractable function, so this is a
% direct transcription rather than a lib/ call):
%
%   ctle_gain = (10^(dc_gain_db/20) + jf/f_z) / ((1+jf/f_p1)(1+jf/f_p2))
%   H_low     = (10^(shelf_gain_db/20) + jf/f_HP) / (1 + jf/f_HP)
%   H_ctf     = H_low .* ctle_gain
%
% using the C2C config's own CTLE frequencies (f_z=f_p1=21.25GHz,
% f_p2=53.125GHz, f_HP=0.6640625GHz) and MATLAB's own found-optimal gains
% for that config (dc_gain=-3dB, shelf_gain=-2dB, case 1).
%
% Ground truth for tests/ctle/test_two_stage_vs_matlab.py.
f = [1e7, 1e9, 5e9, 1.001e10, 2.001e10, 2.651e10, 3.001e10, 4.001e10, 5e10];

f_z = 21.25e9; f_p1 = 21.25e9; f_p2 = 53.125e9; f_HP = 0.6640625e9;
dc_gain_db = -3.0; shelf_gain_db = -2.0;

kacdc = 10^(dc_gain_db/20);
ctle_gain = (kacdc + 1i*f/f_z) ./ ((1+1i*f/f_p1) .* (1+1i*f/f_p2));

kacde_DC_low = 10^(shelf_gain_db/20);
H_low = (kacde_DC_low + 1i*f/f_HP) ./ (1 + 1i*f/f_HP);

H_ctf = H_low .* ctle_gain;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'ctle_two_stage.csv'), 'w');
fprintf(fid, 'freq,h_r,h_i\n');
for i = 1:length(f)
    fprintf(fid, '%.10g,%.10g,%.10g\n', f(i), real(H_ctf(i)), imag(H_ctf(i)));
end
fclose(fid);
disp('wrote ctle_two_stage.csv');
