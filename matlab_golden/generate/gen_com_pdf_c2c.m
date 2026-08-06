% Captures MATLAB COM3.70's own combined interference+noise PDF (Equation
% 93A-45) for the real IEEE 802.3ck "C2C" thru channel, package case 1 --
% the same live run gen_sbr_c2c.m already does, extended one step further.
%
% com_ieee8023_93a_370.m's top-level output_args never carries this PDF
% (Output_Arg_Fill only copies a handful of scalar summary fields out of
% it, e.g. Noise_Struct.sigma_N) -- unlike fom_result.sbr, it isn't
% reachable through BREAD_CRUMBS at all. It's built by Create_Noise_PDF
% (com_ieee8023_93a_370.m:1173-1285), a local function of that file, so
% not directly callable from outside it either. Rather than patch the
% (gitignored, "never edit to fix anything") reference file, this script
% extracts Create_Noise_PDF itself -- plus everything *it* calls
% (conv_fct, d_cpdf, get_pdf_from_sampled_signal, Init_PDF_Fast,
% normal_dist) -- verbatim into lib/, and re-runs it standalone against
% the fom_result/chdata/param/OP a live run already exposes via
% BREAD_CRUMBS (same as gen_sbr_c2c.m captures for r.sbr).
%
% get_sigma_noise (only reachable when OP.RX_CALIBRATION is true) was
% deliberately NOT extracted -- this config runs with RX_CALIBRATION=0
% (same as gen_sbr_c2c.m's own config), so that branch never executes.
%
% Ground truth for a future tests/pmf/test_*_vs_matlab.py checking
% serdeskit's own combine_pmfs()/noise_margin() (com.py's p_total/y)
% against MATLAB's real Equation 93A-45 output, not just the individual
% formulas each already have their own golden tests.
%
% Requires MATLAB + reference/matlab/COM3.70/ + reference/ck_channels/
% c2c_pcb/ locally (both gitignored -- see examples/compute_com_c2c.py's
% docstring for the channel download link).

matlab_root = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'reference', 'matlab', 'COM3.70');
addpath(matlab_root);
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

% See gen_sbr_c2c.m's own comment for why .mat, not .xlsx, and why cd
% into config_sheets_dir with a bare filename.
config_sheets_dir = fullfile(matlab_root, 'config_sheets_3p1');
config_name = 'config_com_ieee8023_93a=3ck_d3p1_120F_C2C_11_30_21.mat';
if ~isfile(fullfile(config_sheets_dir, config_name))
    error('Config cache not found: %s -- see gen_sbr_c2c.m''s own comment.', fullfile(config_sheets_dir, config_name));
end
prev_dir = pwd;
restore_dir = onCleanup(@() cd(prev_dir));
cd(config_sheets_dir);
config_file = config_name;

chan_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'reference', 'ck_channels', 'c2c_pcb');
thru_file = fullfile(chan_dir, 'C2C_PCB_SYSVIA_12dB_thru.s4p');
if ~isfile(thru_file)
    error('Channel file not found: %s -- see examples/compute_com_c2c.py for the download link.', thru_file);
end
fext_files = arrayfun(@(n) fullfile(chan_dir, sprintf('C2C_PCB_SYSVIA_12dB_fext%d.s4p', n)), 1:6, 'UniformOutput', false);
next_files = arrayfun(@(n) fullfile(chan_dir, sprintf('C2C_PCB_SYSVIA_12dB_next%d.s4p', n)), 1:4, 'UniformOutput', false);

results = com_ieee8023_93a_370(config_file, 6, 4, thru_file, fext_files{:}, next_files{:});

% Package case 1 (13mm TX / 11mm RX) -- same case every other C2C golden
% fixture in this project uses.
r = results{1};
fom_result = r.fom_result;
chdata = r.chdata;
param = r.param;
OP = r.OP;

% com_ieee8023_93a_370.m:404 -- A_s is always abs(fom_result.A_s) at the
% point Create_Noise_PDF gets called.
A_s = abs(fom_result.A_s);

[PDF, CDF, NS] = Create_Noise_PDF(A_s, param, fom_result, chdata, OP);

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'com_pdf_c2c_thru.csv'), 'w');
fprintf(fid, 'y,pdf,cdf\n');
for i = 1:length(PDF.y)
    fprintf(fid, '%.10g,%.10g,%.10g\n', PDF.x(i), PDF.y(i), CDF(i));
end
fclose(fid);
fprintf('wrote com_pdf_c2c_thru.csv (%d points, sigma_G=%.6g V)\n', length(PDF.y), NS.sigma_G);
