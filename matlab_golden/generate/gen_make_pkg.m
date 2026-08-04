% Calls MATLAB COM3.70's own make_pkg (lib/make_pkg.m, copied verbatim --
% needs lib/combines4p.m and lib/synth_tline.m on the path too) with the
% C2C TX package's own "block 1" parameters (die capacitance, compensating
% inductor, bump capacitance, then the first tline segment -- Cball=0,
% i.e. no board-side shunt cap on this block, matching real
% make_full_pkg's own per-block split). Ground truth for
% tests/package/test_make_pkg_vs_matlab.py.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

f = [1e7, 1e9, 5e9, 1.001e10, 2.001e10, 2.651e10, 3.001e10, 4.001e10, 5e10];

param.pkg_tau = 0.006141;
param.pkg_gamma0_a1_a2 = [0, 0.0009909, 0.0002772];
param.Z0 = 50;

pkg_len = 13.0;
cpad = 1.2e-13;   % C_d = 1.2e-4 nF
cball = 0.0;      % this block has no board-side cap
pkg_z = 87.5;
lcomp = 1.2e-10;  % L_s = 0.12 nH
cbump = 3e-14;    % C_b = 0.3e-4 nF

[s11out, s12out, s21out, s22out] = make_pkg(f, pkg_len, cpad, cball, pkg_z, param, lcomp, cbump);

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'make_pkg.csv'), 'w');
fprintf(fid, 'freq,s11_r,s11_i,s21_r,s21_i\n');
for i = 1:length(f)
    fprintf(fid, '%.10g,%.10g,%.10g,%.10g,%.10g\n', f(i), real(s11out(i)), imag(s11out(i)), real(s21out(i)), imag(s21out(i)));
end
fclose(fid);
disp('wrote make_pkg.csv');
