function [ Vfiltered, Cmod ] = force( V ,param, OP , ix, C)
% Vfilter is vector forced filtered sbr
% Cmod is the ffe tap co-efficient vector
% if C is passed, just process V with C else compute C
% cmx=param.rx_cmx; number of pre cursor taps
% cpx=param.rx_cps; number of post cursor taps
% V=sbr; pass pulse response
% ix the sample point in the passed pulse response
% the sample point is recomputed by optimize_fom
% OP not used for now
% test with load('SBR_FIR_resp.mydata','-mat')
%% set up parameters ------------------------------------------------------
if ~exist('cursor_gain','var'), cursor_gain=0; end
csm = @(b,n) [ b((length(b)-n)+1:length(b)) b(1:(length(b)-n))]; % minus circshift
csp = @(b,n) [ b(n:length(b)) b(1:n-1) ]; %  plus circshift
db = @(x) 20*log10(abs(x));
if ~exist('ix','var'),ix=find(V==max(V),1,'first');end
cmx=param.RxFFE_cmx;
cpx=param.RxFFE_cpx;
num_taps=cmx+cpx+1;
cstep=param.RxFFE_stepz;
ndfe=param.ndfe;
spui=param.samples_per_ui;
%% resample to align with sample poont ------------------------------------
%% only consider sample aligned with sample point
% if ix < length(V);
%     if isrow(V)
%         vsampled_raw = [V(mod(ix,spui)+1:spui:(mod(ix,spui)+spui*(floor(ix/spui)-1)))'; V(ix:spui:end)'];
%     else
%         vsampled_raw = [V(mod(ix,spui)+1:spui:(mod(ix,spui)+spui*(floor(ix/spui)-1))); V(ix:spui:end)];
%     end
% else
%     if isrow(V)
%         vsampled_raw = V(mod(ix,spui)+1:spui:end)';
%     else
%         vsampled_raw = V(mod(ix,spui)+1:spui:end) ;
%     end
% end
if ix < length(V);
    if isrow(V)
        %         vsampled_raw = [V(mod(ix,spui):spui:(mod(ix,spui)+spui*(floor(ix/spui)-1)))'; V(ix:spui:end)'];
        if mod(ix,spui) == 0
            vsampled_raw = [V(spui+mod(ix,spui):spui:(mod(ix,spui)+spui*(floor(ix/spui)-1)))'; V(ix:spui:end)'];
        else
            vsampled_raw = [V(mod(ix,spui):spui:(mod(ix,spui)+spui*(floor(ix/spui)-1)))'; V(ix:spui:end)'];
        end

    else
        %         vsampled_raw = [V(mod(ix,spui):spui:(mod(ix,spui)+spui*(floor(ix/spui)-1))); V(ix:spui:end)];
        if mod(ix,spui) == 0
            vsampled_raw = [V(spui+mod(ix,spui):spui:(mod(ix,spui)+spui*(floor(ix/spui)-1))); V(ix:spui:end)];
        else
            vsampled_raw = [V(mod(ix,spui):spui:(mod(ix,spui)+spui*(floor(ix/spui)-1))); V(ix:spui:end)];
        end
    end
else
    if isrow(V)
        %        vsampled_raw = V(mod(ix,spui):spui:end)'; %Yasou Hidaka 11/16/2018
        if mod(ix,spui) == 0%Yasou Hidaka 11/16/2018
            vsampled_raw = V(spui+mod(ix,spui):spui:end)';%Yasou Hidaka 11/16/2018
        else
            vsampled_raw = V(mod(ix,spui):spui:end)';%Yasou Hidaka 11/16/2018
        end
    else
        %        vsampled_raw = V(mod(ix,spui):spui:end) ;%Yasou Hidaka 11/16/2018
        if mod(ix,spui) == 0%Yasou Hidaka 11/16/2018
            vsampled_raw = V(spui+mod(ix,spui):spui:end) ;
        else
            vsampled_raw = V(mod(ix,spui):spui:end) ;%Yasou Hidaka 11/16/2018
        end
    end


end

vsampled=[zeros(1,cmx) vsampled_raw' zeros(1,cpx)];% pad for pre and post cursor prior to shifting

%% find the index for the sample point but in the UI resample vector, vsampled
% ivs=find(vsampled==V(ix),1,'first');% ivs is the sample point for V
% Upen Kareti suggested fix for indexing 11/04/18
if ix < length(V)
    ivs=find(vsampled==V(ix),1,'first');% ivs is the sample point for V
else
    ivs=find(vsampled == max(vsampled),1,'first');
end


%% create VV matrix of shifted UI spaced sample of the pulse response
% ishift=cmx+1;
% for i=1:cmx
%     ishift=ishift-1;
%     VV(i,:)=circshift(vsampled,[0,-ishift]);
% end
% VV(cmx+1,:)=vsampled;
% ishift=0;
% for i=1:cpx
%     ishift=ishift+1;
%     VV(i+cmx+1,:)=circshift(vsampled,[0,ishift]);
% end

% only consider the VV matrix that correstonds to the FFE taps
% can bypass the slow circshift of the entire vector (commented out above) by just grabbing the proper columns
VV=zeros(num_taps,num_taps);
for i=1:num_taps
    start_idx=ivs+i-1;
    end_idx=start_idx-num_taps+1;
    VV(:,i)=vsampled(start_idx:-1:end_idx);
end

%% Apply RXFFE
if ~exist('C','var')
    % cmx+1 is the cursor or sample point
    %VV=VV(:,ivs-cmx:ivs+cpx); % only consider the VV matrix that correstonds to the FFE taps
    FV=zeros(1,cmx+cpx+1); % zero the forceing veotor, FV first
    FV(cmx+1)=vsampled(ivs)*10^(param.current_ffegain/20); % force the voltage at sample point
    if param.ndfe~=0 && (cpx > 0) % Yasuo Hidaka suggest fix for no postC 11/11/18
        %          FV(cmx+2)=param.bmax(1)*FV(cmx+1);   % force the post cursor to bmax if dfe exists
        FV(cmx+2)=min(param.bmax(1)*FV(cmx+1),abs(vsampled(ivs+1)))*sign(vsampled(ivs+1));
    end
    C=((VV'*VV)^-1*VV')'*FV'; % sikve for FFE taps, C
    if cstep ~= 0
        %         C=C/sum(C); % r240 constrain taps to sum of taps = 1
        C=C/C(cmx+1); % r241 make cursor tap 1
        Cmod=floor(abs(C/cstep)).*sign(C)*cstep;% r250 quantize with floor ad sign(C)
    else
        %         C=C/sum(C); % r240
        Cmod=C/C(cmx+1); % r241 constrain taps to sum of taps = 1
    end
    Cmod=Cmod(1:cmx+1+cpx);
else
    Cmod=C;%just us the FFE taps, C, passed for filtering
end
%% The DNH routing (do no hard) for long FFE i.e. see if zeroing taps helps
response_shift=zeros(length(vsampled),num_taps);
for k=1:num_taps
    %pre compute the circshift vectors to save time (can call FFE_Fast instead of FFE
    %in a loop when taps change & the sbr is unchanged)
    response_shift(:,k)=circshift(vsampled(:),[(k-1-cmx) 0]);
end
fom_ffe=0;
Cmod1=Cmod;
% for i=0:param.ffe_backoff
for i=0:min(param.ffe_backoff,cpx)
    Cmodtest=[ Cmod(1:1+cpx+cmx-i).' zeros( 1,i) ].';
    %Vtest=FFE(Cmodtest , param.RxFFE_cmx,1, vsampled );
    Vtest= FFE_Fast(Cmodtest,response_shift);
    Vs=Vtest(ivs);
    Vtest(ivs)=0;
    fom_ffe_test= db( Vs/norm(Vtest));
    if abs(fom_ffe_test) > abs(fom_ffe)
        fom_ffe=fom_ffe_test;
        Cmod1=Cmodtest; % modify Cmod1
    end
end
Cmod=Cmod1;
%%
%% filter the pulse response with the solved FFE
Vfiltered=FFE( Cmod , param.RxFFE_cmx,spui, V );
