% Captures MATLAB COM3.70's own "Voltage bathtub curves" plot (the
% ISI/Xtalk/ISI+Xtalk/Jitter+SNR_TX+RLM+eta0-noise/Jitter-noise/total-
% noise-left/total-noise-right family, com_ieee8023_93a_370.m's
% Bathtub_Contribution_Wrapper -> plot_bathtub_curves, :736-809) for the
% real IEEE 802.3ck "C2C" thru channel, package case 1 -- the same live
% run gen_com_pdf_c2c.m already does, one step further.
%
% plot_bathtub_curves (lib/plot_bathtub_curves.m, copied verbatim) only
% depends on conv_fct/d_cpdf, both already extracted for gen_com_pdf_c2c.m
% -- no new dependencies. It's a pure plotting function (draws directly
% into an Axes, returns nothing), so rather than re-deriving its
% cumsum/conv_fct math by hand here (risking a transcription bug), this
% script calls it against an invisible figure and reads back the exact
% (XData, YData) MATLAB itself plotted off each Line object -- guaranteed
% identical to what you'd see on screen.
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

[PDF, ~, NS] = Create_Noise_PDF(A_s, param, fom_result, chdata, OP);

fig = figure('Visible', 'off');
hax = axes('Parent', fig);
plot_bathtub_curves(hax, A_s, NS.sci_pdf, NS.cci_pdf, NS.isi_and_xtalk_pdf, NS.noise_pdf, NS.jitt_pdf, PDF, param.delta_y);

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'bathtub_c2c_thru.csv'), 'w');
fprintf(fid, 'curve,x,y\n');
lines = findobj(hax, 'Type', 'Line');
for i = 1:length(lines)
    name = get(lines(i), 'DisplayName');
    if isempty(name)
        continue  % skip the dashed BER-threshold guide box (no DisplayName set)
    end
    x = get(lines(i), 'XData');
    y = get(lines(i), 'YData');
    for k = 1:length(x)
        fprintf(fid, '"%s",%.10g,%.10g\n', name, x(k), y(k));
    end
end
fclose(fid);
close(fig);
fprintf('wrote bathtub_c2c_thru.csv (%d curves)\n', length(lines) - 1);
