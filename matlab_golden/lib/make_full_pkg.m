function [ s11out, s12out, s21out, s22out]=make_full_pkg(type,faxis,param,channel_type,mode)

%This function makes the TX or RX package.  The type input must be
%'TX' or 'RX'
%If the mode argument is omitted, mode='dd' is assumed.  Currently
%mode='dc' is only used when making the TX package for AC CM noise
%inclusion.  The Rx package for 'dc' mode is still generated using
%the same parameters as 'dd' mode
%channel_type should be 'THRU' 'FEXT' or 'NEXT'
%
%One instance of package block looks like this (if no elements are set to 0):
%-------------Lcomp----------Tline---------------
%   |                   |               |
%   Cpad                Cbump           Cball
%   |                   |               |
%------------------------------------------------

if nargin<5
    mode='dd';
end

C_diepad = param.C_diepad;
C_pkg_board = param.C_pkg_board;
% [ahealey] Unpack optional compensating L and "bump" C model parameters.
L_comp = param.L_comp;
C_bump = param.C_bump;
% [ahealey] End of modifications.
% generate TX package according to channel type.
[ncases, mele]=size(param.z_p_next_cases);

%Syntax update for C_diepad and L_comp
%Allow a chain of values to be entered as a matrix:
%[L_Tx1 L_Tx2 L_Tx3 ; L_Rx1 L_Rx2 L_Rx3]
if isvector(C_diepad)
    Cd_Tx=C_diepad(1);
    Cd_Rx=C_diepad(2);
    L_comp_Tx=L_comp(1);
    L_comp_Rx=L_comp(2);
    num_blocks=mele;
else
    Cd_Tx=C_diepad(1,:);
    Cd_Rx=C_diepad(2,:);
    L_comp_Tx=L_comp(1,:);
    L_comp_Rx=L_comp(2,:);
    num_blocks=mele+length(Cd_Tx)-1;
end
extra_LC=length(Cd_Tx)-1;
%note:  "insert_zeros" is empty if length(Cd_Tx) = 1
insert_zeros=zeros([1 extra_LC]);

%Updated technique of building Tx/Rx packages
%each index corresponds to the package segment
switch type
    case 'TX'
        switch mele
            case 1
                Cpad=Cd_Tx;
                Lcomp=L_comp_Tx;
                Cbump=C_bump(1);
                Cball=C_pkg_board(1);
                Zpkg=param.pkg_Z_c(1);
            case 4
                Cpad=[Cd_Tx 0 0 0];
                Lcomp=[L_comp_Tx 0 0 0];
                Cbump=[C_bump(1) 0 0 0];
                Cball=[0 0 param.C_v(1) C_pkg_board(1)];
                Zpkg=param.pkg_Z_c(1,:);
            otherwise
                error('package syntax error')
        end
        switch upper(channel_type)
            case 'THRU'
                Len=param.Pkg_len_TX;
            case 'NEXT'
                Len=param.Pkg_len_NEXT;
            case 'FEXT'
                Len=param.Pkg_len_FEXT;
        end
    case 'RX'
        switch mele
            case 1
                Cpad=Cd_Rx;
                Lcomp=L_comp_Rx;
                Cbump=C_bump(2);
                Cball=C_pkg_board(2);
                Zpkg=param.pkg_Z_c(2);
            case 4
                Cpad=[Cd_Rx 0 0 0];
                Lcomp=[L_comp_Rx 0 0 0];
                Cbump=[C_bump(2) 0 0 0];
                Cball=[0 0 param.C_v(2) C_pkg_board(2)];
                Zpkg=param.pkg_Z_c(2,:);
            otherwise
                error('package syntax error')
        end
        switch upper(channel_type)
            case 'THRU'
                Len=param.Pkg_len_RX;
            case 'NEXT'
                Len=param.Pkg_len_RX;
            case 'FEXT'
                Len=param.Pkg_len_RX;
        end
end

%Insert the extra 0 at the front end of Cball, Cbump, Len, and Zpkg
Cball=[insert_zeros Cball];
Cbump=[insert_zeros Cbump];
Len=[insert_zeros Len];
Zpkg=[insert_zeros Zpkg];

% debug_string='';
% for j=1:length(Zpkg)
%     if Cpad(j)~=0
%         debug_string=[debug_string sprintf(', Cd=%0.4g',Cpad(j))];
%     end
%     if Lcomp(j)~=0
%         debug_string=[debug_string sprintf(', Ls=%0.4g',Lcomp(j))];
%     end
%     if Cbump(j)~=0
%         debug_string=[debug_string sprintf(', Cb=%0.4g',Cbump(j))];
%     end
%     if Len(j)~=0
%         debug_string=[debug_string sprintf(', Len=%0.4g Zc=%0.3g',Len(j),Zpkg(j))];
%     end
%     if Cball(j)~=0
%         debug_string=[debug_string sprintf(', Cp=%0.4g',Cball(j))];
%     end
% end
% if length(debug_string)>2
%     debug_string=debug_string(3:end);
% end

% tx package
pkg_param=param;
if strcmpi(mode,'dc')
    % change tx package to CC mode
    pkg_param.Z0=pkg_param.Z0/2;
    Cpad=Cpad*2;
    Cball=Cball*2;
    Zpkg=Zpkg*2;
    Lcomp=Lcomp/2;
    Cbump=Cbump*2;
end
switch num_blocks
    case 1
        [ s11out, s12out, s21out, s22out ]= make_pkg(faxis, Len(1), Cpad(1), Cball(1),Zpkg(1), pkg_param, Lcomp(1), Cbump(1));
    otherwise
        for j=1:num_blocks
            [spkg11,spkg12,spkg21,spkg22]=make_pkg(faxis, Len(j),  Cpad(j),Cball(j) ,Zpkg(j), pkg_param, Lcomp(j),Cbump(j));
            if j==1
                s11out=spkg11; s12out=spkg12; s21out=spkg21; s22out=spkg22;
            else
                [ s11out, s12out, s21out, s22out ]=combines4p(  s11out, s12out, s21out, s22out, spkg11,spkg12,spkg21,spkg22   );
            end
        end
end
