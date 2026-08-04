% Replicates make_full_pkg's own 4-block loop (case 'otherwise' branch)
% exactly, calling the real lib/make_pkg.m + lib/combines4p.m for each
% block and cascading them in sequence -- using the C2C package's actual
% resolved parameters (mele=4; two real segments, two zero-length/
% zero-value no-op blocks -- see matlab_golden/README.md for how these
% were read off a live run's own workspace, not guessed).
%
% Cross-checked against com_ieee8023_93a_370.m's own make_full_pkg,
% called directly with the exact `param` struct captured from a live,
% case-tagged run (package_testcase_i explicitly recorded) -- this
% matters: the tool runs TWO package test cases (13mm/31mm) in one
% process, and an earlier, less careful capture that didn't tag which
% case was active produced numbers that looked like a real package-model
% bug but were actually just ambiguous about which case they reflected.
% Tagging by package_testcase_i and reproducing with this small,
% self-contained script (no full config/param struct needed) resolved
% it: this script's output matches the live capture exactly.
%
% Ground truth for tests/package/test_make_full_pkg_vs_matlab.py.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

f = [1e7, 1e9, 5e9, 1.001e10, 2.001e10, 2.651e10, 3.001e10, 4.001e10, 5e10];

param.pkg_tau = 0.006141;
param.pkg_gamma0_a1_a2 = [0, 0.0009909, 0.0002772];
param.Z0 = 50;

Cpad = [1.2e-13, 0.0, 0.0, 0.0];   % C_d = 1.2e-4 nF, shared TX/RX
Lcomp = [1.2e-10, 0.0, 0.0, 0.0];  % L_s = 0.12 nH, shared TX/RX
Cbump = [3e-14, 0.0, 0.0, 0.0];    % C_b = 0.3e-4 nF, shared TX/RX
Cball = [0.0, 0.0, 0.0, 8.7e-14];  % C_p = 0.87e-4 nF, shared TX/RX
Zpkg = [87.5, 92.5, 100.0, 100.0]; % package_Z_c, shared TX/RX

sides = struct('name', {'tx', 'rx'}, 'len', {[13.0, 1.8, 0.0, 0.0], [11.0, 1.8, 0.0, 0.0]});

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
for si = 1:numel(sides)
    Len = sides(si).len;
    s11out = []; s12out = []; s21out = []; s22out = [];
    for j = 1:4
        [spkg11, spkg12, spkg21, spkg22] = make_pkg(f, Len(j), Cpad(j), Cball(j), Zpkg(j), param, Lcomp(j), Cbump(j));
        if j == 1
            s11out = spkg11; s12out = spkg12; s21out = spkg21; s22out = spkg22;
        else
            [s11out, s12out, s21out, s22out] = combines4p(s11out, s12out, s21out, s22out, spkg11, spkg12, spkg21, spkg22);
        end
    end

    fname = fullfile(out_dir, sprintf('make_full_pkg_%s.csv', sides(si).name));
    fid = fopen(fname, 'w');
    fprintf(fid, 'freq,s11_r,s11_i,s21_r,s21_i\n');
    for i = 1:length(f)
        fprintf(fid, '%.10g,%.10g,%.10g,%.10g,%.10g\n', f(i), real(s11out(i)), imag(s11out(i)), real(s21out(i)), imag(s21out(i)));
    end
    fclose(fid);
    fprintf('wrote %s\n', fname);
end
