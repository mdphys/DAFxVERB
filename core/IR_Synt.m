function out = IR_Synt(aTot,bTot,fTot,sigTot,fs,Tsim,exactScheme,OnlyReal)

T              = 1/fs ;
Ts             = floor(Tsim/T) ;


Nmod           = length(aTot) ;

omTot           = 2*pi*fTot ;
omdTot          = sqrt(omTot.^2-sigTot.^2) ;
cTot            = 2*aTot.*sigTot - 2*bTot.*omdTot  ;
dTot            = 2*aTot ;

if OnlyReal == 1
    dTot             = 0*dTot ;
end

%sigTot = movmean(sigTot,1) ;

omrefk = omdTot*T ;
G1     = zeros(Nmod,1) ;
G2     = zeros(Nmod,1) ;
P0     = zeros(Nmod,1) ;
Pm     = zeros(Nmod,1) ;
Pp     = zeros(Nmod,1) ;

if exactScheme == 1
    for m = 1 : Nmod
        sig        = sigTot(m);
        omdk       = omrefk(m) ;
        den        = 1 + 2*exp(-sig*T)*cos(omdk)+exp(-2*sig*T) ;
        omtildasq  = 4/T^2 * (1 - 2*exp(-sig*T)*cos(omdk) + exp(-2*sig*T)) / den ;
        sigtilda   = 4/T * (1 - exp(-sig*T)) / den ;
        G0         = (1 + 0.25*T^2*omtildasq + sigtilda*T) ;
        G1(m)      = (2 - 0.5*T^2*omtildasq) / G0 ;
        G2(m)      = - (1 + 0.25*T^2*omtildasq - sigtilda*T) / G0 ;
        P0(m)      = T^2*cTot(m) / G0 ;
        Pm(m)      = -0.5*T*dTot(m) / G0 ;
        Pp(m)      = 0.5*T*dTot(m) / G0 ;
    end
else
    for m = 1 : Nmod
        sig        = sigTot(m) ;
        om0        = omTot(m) ;
        G0         = (1 + sig*T) ;
        G1(m)      = (2 - T^2*om0^2) / G0 ;
        G2(m)      = - (1 - sig*T) / G0 ;
        P0(m)      = T^2*cTot(m) / G0 ;
        Pm(m)      = -0.5*T*dTot(m) / G0 ;
        Pp(m)      = 0.5*T*dTot(m) / G0 ;
    end
end


q2    = 0*dTot ;
q1    = 0*exp(-sigTot*T).*((cTot - sigTot.*dTot)./omdTot.*sin(omdTot*T) + dTot.*exp(-sigTot*T).*cos(omdTot*T));

% -------------------------------------------------------------------------
% Main time loop

out   = zeros(Ts,1) ;
fin   = zeros(Ts,1) ;
fin(11) = 1 ;

for n = 2:Ts-1


    % Compute displacement
    q =  G1.*q1 + G2.*q2 + P0.*fin(n) + (Pm*fin(n-1) + Pp*fin(n+1)) ;

    out(n) = sum(q) ;

    % Swap states
    q2 = q1; q1 = q;


end


%out = out / max(abs(out)) ;
