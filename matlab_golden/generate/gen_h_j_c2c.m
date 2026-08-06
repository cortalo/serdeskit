% Captures MATLAB COM3.70's own h_J -- (93A-28)'s per-UI slope samples,
% used both to build p_DD (93A.1.7.1, param.A_DD*h_J) and NS.sigma_rjit
% (param.sigma_RJ*param.sigma_X*norm(h_J), 93A-31) -- for the real IEEE
% 802.3ck "C2C" thru channel, package case 1. Same live run gen_sbr_c2c.m
% already does.
%
% fom_result.h_J is already exposed via BREAD_CRUMBS (result.h_J is set
% at com_ieee8023_93a_370.m:6588/:7647, inside optimize_fom/
% optimize_fom_Dynamic_txffe, and fom_result *is* that same struct) --
% no lib/ extraction needed, unlike Create_Noise_PDF/plot_bathtub_curves.
%
% Ground truth for comparing against PulseResponse.local_slopes(): a
% ~20% gap in sigma_Jitter (matlab_golden vs serdeskit, see
% examples/compute_com_c2c_matlab_coeffs.py's printed comparison) was
% traced to this term specifically -- everything else (signal_amplitude,
% sigma_Tx, sigma_Noise, sigma_ISI, sigma_Crosstalk) already agrees to
% within ~3%.
%
% Requires MATLAB + reference/matlab/COM3.70/ + reference/ck_channels/
% c2c_pcb/ locally (both gitignored -- see examples/compute_com_c2c.py's
% docstring for the channel download link).

matlab_root = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'reference', 'matlab', 'COM3.70');
addpath(matlab_root);

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
fom_result = results{1}.fom_result;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'h_j_c2c_thru.csv'), 'w');
fprintf(fid, 'index,h_j\n');
for i = 1:length(fom_result.h_J)
    fprintf(fid, '%d,%.10g\n', i - 1, fom_result.h_J(i));
end
fclose(fid);
fprintf('wrote h_j_c2c_thru.csv (%d points, norm(h_J)=%.6g, LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=%d)\n', ...
    length(fom_result.h_J), norm(fom_result.h_J), results{1}.OP.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN);
