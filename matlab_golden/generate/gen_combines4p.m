% Calls MATLAB COM3.70's own combines4p (lib/combines4p.m, copied
% verbatim from com_ieee8023_93a_370.m) with fixed, arbitrary 2-port
% S-parameter pairs and saves the result to CSV -- ground truth for
% tests/package/test_combines4p_vs_matlab.py, which builds the same two
% networks with skrf and cascades them with `**`.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

% Six arbitrary (index, not frequency) test points -- non-reciprocal
% (S12 != S21) so the test doesn't accidentally pass by symmetry.
s11in1 = [0.10+0.05i, -0.20+0.10i, 0.05-0.15i, 0.30+0.00i, -0.10-0.10i, 0.00+0.20i];
s12in1 = [0.02-0.01i,  0.03+0.02i, 0.01+0.01i, 0.04-0.02i,  0.02+0.00i, 0.01-0.01i];
s21in1 = [0.90+0.10i,  0.80-0.05i, 0.85+0.15i, 0.70+0.20i,  0.95-0.10i, 0.60+0.30i];
s22in1 = [0.15-0.05i,  0.10+0.10i, 0.20-0.10i, 0.05+0.05i,  0.25+0.00i, 0.10-0.20i];

s11in2 = [0.20-0.10i,  0.05+0.05i, -0.10+0.20i, 0.15-0.15i,  0.00+0.10i, 0.30-0.05i];
s12in2 = [0.85+0.05i,  0.75-0.10i, 0.80+0.10i, 0.65+0.05i,  0.90-0.05i, 0.55+0.15i];
s21in2 = [0.05+0.02i,  0.04-0.01i, 0.03+0.03i, 0.06-0.02i,  0.02+0.01i, 0.08-0.03i];
s22in2 = [0.30+0.10i, -0.05-0.05i, 0.10+0.15i, 0.20+0.00i, -0.10+0.10i, 0.15-0.10i];

[s11out, s12out, s21out, s22out] = combines4p(s11in1, s12in1, s21in1, s22in1, s11in2, s12in2, s21in2, s22in2);

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'combines4p.csv'), 'w');
fprintf(fid, 'idx,s11in1_r,s11in1_i,s12in1_r,s12in1_i,s21in1_r,s21in1_i,s22in1_r,s22in1_i,s11in2_r,s11in2_i,s12in2_r,s12in2_i,s21in2_r,s21in2_i,s22in2_r,s22in2_i,s11out_r,s11out_i,s12out_r,s12out_i,s21out_r,s21out_i,s22out_r,s22out_i\n');
for i = 1:length(s11in1)
    fprintf(fid, '%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n', ...
        i-1, ...
        real(s11in1(i)), imag(s11in1(i)), real(s12in1(i)), imag(s12in1(i)), real(s21in1(i)), imag(s21in1(i)), real(s22in1(i)), imag(s22in1(i)), ...
        real(s11in2(i)), imag(s11in2(i)), real(s12in2(i)), imag(s12in2(i)), real(s21in2(i)), imag(s21in2(i)), real(s22in2(i)), imag(s22in2(i)), ...
        real(s11out(i)), imag(s11out(i)), real(s12out(i)), imag(s12out(i)), real(s21out(i)), imag(s21out(i)), real(s22out(i)), imag(s22out(i)));
end
fclose(fid);
disp('wrote combines4p.csv');
