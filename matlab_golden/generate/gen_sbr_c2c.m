% Captures two things from MATLAB COM3.70, for the real IEEE 802.3ck
% "C2C" thru channel, package case 1 (13mm TX / 11mm RX --
% OP.pkg_len_select(1)):
%   1. its composed pulse response ("sbr" -- channel + package + CTLE +
%      Tx FFE + Rx AFE, IFFT'd, scaled to the victim's real launch
%      amplitude) -> data/sbr_c2c_thru.csv
%   2. its pre-CTLE/FFE channel+package+RxAFE frequency response
%      (chdata(1).sdd21) -> data/uneq_h_c2c_thru.csv, an isolation step
%      for narrowing down (1)'s divergence -- see the block below.
%
% Ground truth for tests/link/test_pulse_response_vs_matlab_c2c.py, which
% checks Link.ffe_channel_ctle_pulse_response() (composed in the frequency
% domain, one IFFT) against this. See CLAUDE.md's "full composed pulse
% response" known-issue entry: an earlier informal comparison found
% cursor magnitude ~2% off and a persisting "narrower/sharper near
% cursor" shape pattern, but never as a committed matlab_golden test.
%
% Confirmed the amplitude convention directly from source (not guessed --
% project policy, see CLAUDE.md): result.sbr is already in absolute
% volts, not a unit-normalized pulse. com_ieee8023_93a_370.m:919
% (`chdata(i).uneq_imp_response=chdata(i).uneq_imp_response*chdata(i).A`)
% bakes the victim launch amplitude (A_v, this config's 'A_v' = 0.413 V)
% in before CTLE/FFE ever run. So this is directly comparable to
% serdeskit's `link.ffe_channel_ctle_pulse_response(grid).scale(p.
% victim_amplitude)` with no extra scale factor needed.
%
% This is a live full-tool run, not a lib/ extraction (matlab_golden/
% README.md's "if a script needs live values MATLAB itself computed
% internally... capture them from an actual run" fallback): sbr depends
% on ~900 lines of setup (S-parameter loading, package cascading,
% TD_CTLE's own pole/zero search) that isn't practically re-derivable
% standalone the way FFE.m or Butterworth_Filter.m were.
%
% Requires MATLAB + reference/matlab/COM3.70/ + reference/ck_channels/
% c2c_pcb/ locally (both gitignored -- see examples/compute_com_c2c.py's
% docstring for the channel download link). Runs interactively (whatever
% OP.DISPLAY_WINDOW the config sheet itself specifies) -- any plot
% windows that pop up don't affect the captured result.

matlab_root = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'reference', 'matlab', 'COM3.70');
addpath(matlab_root);

% .mat, not .xlsx: on this MATLAB release, read_ParamConfigFile's own
% xlsfinfo/xlsread path (com_ieee8023_93a_370.m:8257/:8261) fails to see
% the 'COM_Settings' sheet at all (confirmed directly -- xlsfinfo returns
% an empty sheet list, xlsread errors "Worksheet not found" -- even
% though the modern `sheetnames()` sees it fine), so the .xlsx path never
% works here. read_ParamConfigFile has its own .mat branch
% (`case upper('.mat'): load(matcongfile)`, :8253) for exactly this: a
% pre-parsed cache of the same 'parameter' cell array xlsread would
% produce, written by its own success-path `save(matcongfile,'parameter')`
% (:8711) the one time xlsread did work. That cache already exists on
% this machine (reference/matlab/COM3.70/config_sheets_3p1/config_com_
% ieee8023_93a=3ck_d3p1_120F_C2C_11_30_21.mat) from an earlier run.
%
% com_ieee8023_93a's own top-level return value (`output_args`, built by
% Output_Arg_Fill) never includes the raw sbr/t waveform -- only the
% internal `fom_result` struct does (`result.sbr=best_sbr` etc., set deep
% inside optimize_fom/optimize_fom_Dynamic_txffe). The tool has a
% documented hook for exactly this: `if OP.BREAD_CRUMBS, output_args.
% fom_result = fom_result; ... end` (:613-618). This config's own sheet
% already sets BREAD_CRUMBS=1 (confirmed directly: `parameter{17,7}`),
% so no patching needed -- results{i}.fom_result is populated as-is.
config_sheets_dir = fullfile(matlab_root, 'config_sheets_3p1');
config_name = 'config_com_ieee8023_93a=3ck_d3p1_120F_C2C_11_30_21.mat';
if ~isfile(fullfile(config_sheets_dir, config_name))
    error('Config cache not found: %s -- re-derive it (see this script''s own comment above) or point config_file back at the .xlsx if xlsread works on your MATLAB release.', fullfile(config_sheets_dir, config_name));
end

% read_ParamConfigFile (com_ieee8023_93a_370.m:8246-8250) reconstructs its
% own load path as `[fileparts(paramFile) '\' name '.mat']` -- a literal
% backslash, Windows-only, that breaks path resolution for any config_file
% with a directory component on macOS/Linux (confirmed directly: passing
% a full path here reliably fails with "Unable to find file or
% directory"). Its `if ~isempty(filepath)` guard is skipped entirely when
% paramFile has no directory component at all, so cd into the config
% directory and pass a bare filename instead of working around the bug
% any other way.
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

% Legacy-mode call: com_ieee8023_93a_370(config_file, num_fext, num_next,
% thru_file, <fext files>, <next files>) -- argument order confirmed from
% get_s4p_files (com_ieee8023_93a_370.m:4959): thru first, then num_fext
% FEXT files, then num_next NEXT files, matching examples/compute_com_c2c.py's
% own fext1..6/next1..4 file layout.
results = com_ieee8023_93a_370(config_file, 6, 4, thru_file, fext_files{:}, next_files{:});

% Package case 1 (13mm TX / 11mm RX) -- OP.pkg_len_select(1), matching
% every other C2C golden fixture in this project (matlab_golden/data/
% channel_plus_package_c2c_thru.csv, tests/package/test_cascade_vs_matlab_c2c.py).
r = results{1}.fom_result;

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
fid = fopen(fullfile(out_dir, 'sbr_c2c_thru.csv'), 'w');
fprintf(fid, 't,sbr\n');
for i = 1:length(r.sbr)
    fprintf(fid, '%.10g,%.10g\n', r.t(i), r.sbr(i));
end
fclose(fid);
fprintf('wrote sbr_c2c_thru.csv (%d points, cursor index t_s=%d, A_s=%.6g V)\n', length(r.sbr), r.t_s, r.A_s);

% Isolation step (docs/known-issues.md's "full composed pulse response"
% entry, "next step" section): channel+package+RxAFE's own frequency
% response, chdata(1).sdd21 (thru file, index 1) -- computed exactly once
% per run in COM_FD_to_TD (com_ieee8023_93a_370.m:904-910), *before* any
% CTLE/FFE candidate search starts, so unlike chdata(k).sdd21ctf (built
% fresh on every CTLE candidate inside the search loop -- not safe to
% read back out after the loop without checking which candidate it was
% last computed for) this one is unambiguous: whatever candidates the
% search tries afterward, chdata(1).sdd21 itself is never touched again.
% chdata(i).sdd21 = (raw channel+package S21) .* Butterworth .* (Bessel-
% Thomson, off for this config -- confirmed: 'Bessel_Thomson' isn't even
% a row in this config's own parameter sheet, so xls_parameter's default
% (false) applies) -- i.e. directly comparable to serdeskit's
% `channel.transfer_function(f) * rx_afe.transfer_function(f)`, no CTLE/
% FFE/RxFFE factors yet.
c = results{1}.chdata(1);
fid = fopen(fullfile(out_dir, 'uneq_h_c2c_thru.csv'), 'w');
fprintf(fid, 'freq,h_r,h_i\n');
for i = 1:length(c.sdd21)
    fprintf(fid, '%.10g,%.10g,%.10g\n', c.faxis(i), real(c.sdd21(i)), imag(c.sdd21(i)));
end
fclose(fid);
fprintf('wrote uneq_h_c2c_thru.csv (%d points)\n', length(c.sdd21));
