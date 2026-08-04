% Calls MATLAB COM3.70's own synth_tline (lib/synth_tline.m, copied
% verbatim) with the C2C config's own package transmission-line
% parameters, at a handful of test frequencies -- ground truth for
% tests/package/test_synth_tline_vs_matlab.py.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

f = [1e7, 1e9, 5e9, 1.001e10, 2.001e10, 2.651e10, 3.001e10, 4.001e10, 5e10];
Z_0 = 50;
gamma_coeff = [0, 0.0009909, 0.0002772];  % [gamma0, a1, a2]
tau = 0.006141;

% Segment 1: Zc=87.5 Ohm, length=13mm (the C2C TX package's first segment)
[s11_a, s12_a, s21_a, s22_a] = synth_tline(f, 87.5, Z_0, gamma_coeff, tau, 13.0);
% Segment 2: Zc=92.5 Ohm, length=1.8mm
[s11_b, s12_b, s21_b, s22_b] = synth_tline(f, 92.5, Z_0, gamma_coeff, tau, 1.8);
% Zero-length segment (d=0) -- exercises synth_tline's own d==0 special case
[s11_c, s12_c, s21_c, s22_c] = synth_tline(f, 100.0, Z_0, gamma_coeff, tau, 0.0);

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'synth_tline.csv'), 'w');
fprintf(fid, 'freq,zc,len,s11_r,s11_i,s21_r,s21_i\n');
for i = 1:length(f)
    fprintf(fid, '%.10g,87.5,13.0,%.10g,%.10g,%.10g,%.10g\n', f(i), real(s11_a(i)), imag(s11_a(i)), real(s21_a(i)), imag(s21_a(i)));
end
for i = 1:length(f)
    fprintf(fid, '%.10g,92.5,1.8,%.10g,%.10g,%.10g,%.10g\n', f(i), real(s11_b(i)), imag(s11_b(i)), real(s21_b(i)), imag(s21_b(i)));
end
for i = 1:length(f)
    fprintf(fid, '%.10g,100.0,0.0,%.10g,%.10g,%.10g,%.10g\n', f(i), real(s11_c(i)), imag(s11_c(i)), real(s21_c(i)), imag(s21_c(i)));
end
fclose(fid);
disp('wrote synth_tline.csv');
