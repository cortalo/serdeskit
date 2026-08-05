function H_t = tx_transition_time_filter(faxis, OP)
% Not its own function in com_ieee8023_93a_370.m -- this wraps, verbatim,
% the H_t computation inlined in s21_pkg_tester (com_ieee8023_93a_370.m:
% 9839-9856), the only change being the function signature itself so it
% can be called standalone. Body copied exactly, all branches intact
% (only the T_r_filter_type==1/T_r_meas_point==0 branch is exercised by
% gen_tx_h_t.m below, since that's the one every config this project has
% exercised actually hits via OP.FORCE_TR -- see that script's own
% comment).
H_t=ones(1,length(faxis)); % .3bj compatibility
if OP.IDEAL_TX_TERM ||  OP.T_r_filter_type == 1
    % for RITT testing with good termination as in some instruments
    % and tx filter when required
    if OP.T_r_filter_type==0
        H_t = exp(-(pi*faxis/1e9*OP.transmitter_transition_time/1.6832).^2); %% Equation 93A-46 %%
    else
        tr=OP.transmitter_transition_time;
        f9=faxis/1e9;
        if OP.T_r_meas_point == 1
            k=1.9466+7.12*sqrt(1-6.51e-3/tr);
            H_t=105./(f9.^4*(k*tr)^4 - f9.^3*(k*tr)^3*10i - 45*f9.^2*(k*tr)^2 + f9*(k*tr)*105i + 105);
        else
            H_t = exp( -2*(pi*f9*tr/1.6832).^2 ).*exp(-1j*2*pi*f9*tr*3);
        end
    end
end
