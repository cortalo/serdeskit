% Computes channel + package (TX pkg -> channel -> RX pkg, no CTLE yet)
% for the real C2C thru channel, replicating s21_pkg_tester's own
% cascade formula exactly:
%   [s11,...] = combines4p(TX_pkg, channel)
%   [s11,...] = combines4p(result, RX_pkg reversed: s22in,s21in,s12in,s11in)
% (the RX reversal is the port-swap s21_pkg_tester itself applies before
% cascading -- see com_ieee8023_93a_370.m's own s21_pkg_tester, "s22 is
% ball side of package").
%
% Self-contained: reads the raw channel via lib/read_Nport_touchstone.m
% and builds TX/RX packages via lib/make_pkg.m + lib/combines4p.m, using
% the same resolved block parameters gen_make_full_pkg.m already uses
% (captured from a live run and cross-checked -- see matlab_golden/README.md).
% No live com_ieee8023_93a_370 run needed for this script itself.
%
% Requires the real (gitignored) channel file -- see
% examples/compute_com_c2c.py's own docstring for the download link.
%
% Ground truth for tests/package/test_cascade_vs_matlab_c2c.py.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

thru_file = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'reference', 'ck_channels', 'c2c_pcb', 'C2C_PCB_SYSVIA_12dB_thru.s4p');
if ~isfile(thru_file)
    error('Channel file not found: %s -- see examples/compute_com_c2c.py for the download link.', thru_file);
end

port_order = [1, 3, 2, 4];
[sch, faxis] = read_Nport_touchstone(thru_file, port_order);

n = size(sch, 1);
s11ch = zeros(1, n); s12ch = zeros(1, n); s21ch = zeros(1, n); s22ch = zeros(1, n);
T = [1 1 0 0 ; 1 -1 0 0 ; 0 0 1 1 ; 0 0 1 -1];
for i = 1:n
    S(:,:) = sch(i,:,:);
    W = T * (S / T);
    s11ch(i) = W(2,2); s12ch(i) = W(2,4); s21ch(i) = W(4,2); s22ch(i) = W(4,4);
end

param.pkg_tau = 0.006141;
param.pkg_gamma0_a1_a2 = [0, 0.0009909, 0.0002772];
param.Z0 = 50;

Cpad = [1.2e-13, 0.0, 0.0, 0.0];
Lcomp = [1.2e-10, 0.0, 0.0, 0.0];
Cbump = [3e-14, 0.0, 0.0, 0.0];
Cball = [0.0, 0.0, 0.0, 8.7e-14];
Zpkg = [87.5, 92.5, 100.0, 100.0];
Len_tx = [13.0, 1.8, 0.0, 0.0];
Len_rx = [11.0, 1.8, 0.0, 0.0];

function [s11, s12, s21, s22] = full_pkg(f, Len, Cpad, Lcomp, Cbump, Cball, Zpkg, param)
    s11 = []; s12 = []; s21 = []; s22 = [];
    for j = 1:4
        [b11, b12, b21, b22] = make_pkg(f, Len(j), Cpad(j), Cball(j), Zpkg(j), param, Lcomp(j), Cbump(j));
        if j == 1
            s11 = b11; s12 = b12; s21 = b21; s22 = b22;
        else
            [s11, s12, s21, s22] = combines4p(s11, s12, s21, s22, b11, b12, b21, b22);
        end
    end
end

[s11tx, s12tx, s21tx, s22tx] = full_pkg(faxis, Len_tx, Cpad, Lcomp, Cbump, Cball, Zpkg, param);
[s11rx, s12rx, s21rx, s22rx] = full_pkg(faxis, Len_rx, Cpad, Lcomp, Cbump, Cball, Zpkg, param);

% TX package -> channel
[s11, s12, s21, s22] = combines4p(s11tx, s12tx, s21tx, s22tx, s11ch, s12ch, s21ch, s22ch);
% -> RX package, port-reversed (s21_pkg_tester's own "s22 is ball side of package")
[s11, s12, s21, s22] = combines4p(s11, s12, s21, s22, s22rx, s21rx, s12rx, s11rx);

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'channel_plus_package_c2c_thru.csv'), 'w');
fprintf(fid, 'freq,s11_r,s11_i,s21_r,s21_i\n');
for i = 1:n
    fprintf(fid, '%.10g,%.10g,%.10g,%.10g,%.10g\n', faxis(i), real(s11(i)), imag(s11(i)), real(s21(i)), imag(s21(i)));
end
fclose(fid);
fprintf('wrote channel_plus_package_c2c_thru.csv (%d points)\n', n);
