function [value,isterm,dir] = reaching_steadyA(t,y,startTime)
% This function is the event function for ODE
%
% INPUTS:
%    t:                     Time point at which ODE is simulated
%    y:                     current solution
%    startTime:             time when ODE solver was started
%
% OUTPUT:
%    value:                 An event occurs when value is equal to zero
%    isterminal:            isterminal = 1 if the integration is to terminate
%    dir:                   default value = 0

global Gs_VEL
if Gs_VEL(end,1)>5*60 % 5 minutes simulation time
    closestime=Gs_VEL(end,1)-5*60; 
    diff=abs(Gs_VEL(:,1)-closestime);
    [~,ind] = min(diff);
    % display(Gs_VEL(ind(1)))
    % steady state for at least 5 minutes
    value      = abs((Gs_VEL(end,2)-Gs_VEL(ind(1),2))/(Gs_VEL(end,1)-Gs_VEL(ind(1),1)))<1e-3;
else
    value=false;
end

isterm = 1;
dir = 0;

% steady state A is not reached but maybe the time limit is already
% exceeded
if ~value
    value = timeLimit(t, y, startTime);
    if value
        warning('Time limit exceeded')
    end
end

end