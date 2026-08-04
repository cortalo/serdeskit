function [ s11out, s12out, s21out, s22out ] = make_pkg(f, pkg_len, cpad, cball, pkg_z, param, varargin)
f(f<eps)=eps;
tau = param.pkg_tau; gamma0_a1_a2=param.pkg_gamma0_a1_a2; Lenscale=pkg_len; zref=param.Z0;
%% Equation 93A-8
s11pad= -1i*2*pi.*f*cpad*zref./(2+1i*2*pi.*f*cpad*zref);
s21pad= 2./(2+1i*2*pi.*f*cpad*zref);

% [ahealey] Add compensating L and shunt C (bump) when requested.
s12pad = s21pad;
s22pad = s11pad;
if nargin > 6
    lcomp = varargin{1};
    if lcomp>0
        s11comp = (1i*2*pi*f*lcomp/zref)./(2+1i*2*pi*f*lcomp/zref);
        s21comp = 2./(2+1i*2*pi*f*lcomp/zref);
        [s11pad, s12pad, s21pad, s22pad] = combines4p( ...
            s11pad, s12pad, s21pad, s22pad, ...
            s11comp, s21comp, s21comp, s11comp);
    end
end
if nargin > 7
    cbump = varargin{2};
    if cbump>0
        s11bump = -1i*2*pi.*f*cbump*zref./(2+1i*2*pi.*f*cbump*zref);
        s21bump = 2./(2+1i*2*pi.*f*cbump*zref);
        [s11pad, s12pad, s21pad, s22pad] = combines4p( ...
            s11pad, s12pad, s21pad, s22pad, ...
            s11bump, s21bump, s21bump, s11bump);
    end
end
% [ahealey] End of modifications.

[ S11, S12, S21, S22 ] = synth_tline(f, pkg_z, zref, gamma0_a1_a2, tau, Lenscale); %#ok<NASGU,ASGLU>
% [ahealey] Symmetry cannot be assumed with more complex termination models.
% [ s11out1, s12out1, s21out1, s22out1 ]= ...
%     combines4p(  s11pad, s21pad, s21pad, s11pad, S11, S21, S21, S11 ); % first part of equation 93A-15
[s11out1, s12out1, s21out1, s22out1] = combines4p( ...
    s11pad, s12pad, s21pad, s22pad, ...
    S11, S21, S21, S11);
% [ahealey] End of modifications.

%% Equation 93A-8
s11ball= -1i*2*pi.*f*cball*zref./(2+1i*2*pi.*f*cball*zref);
s21ball= 2./(2+1i*2*pi.*f*cball*zref);
[ s11out, s12out, s21out, s22out ]= ...
    combines4p( s11out1, s12out1, s21out1, s22out1, s11ball, s21ball, s21ball, s11ball );% second part of equation 93A-15
