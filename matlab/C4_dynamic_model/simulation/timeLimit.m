function [value,isterminal,direction] = timeLimit(t,y,startTime)
%TIMELIMIT Event function to stop after some global time limit
%
% Input
%   t:                     Time point at which ODE is simulated
%   y:                     current solution
%   startTime:             time when ODE solver was started
% 
% Output
%    value:                 An event occurs when value is equal to zero
%    isterminal:            isterminal = 1 if the integration is to terminate
%    dir:                   default value = 0

% global time limit (minutes)
tmax = 0.5;

% absoute time where simulation should be stopped
stopTime = startTime + minutes(tmax);

% determine if stop time as been reached
if datetime('now') >= stopTime
    value = true;
else
    value = false;
end

isterminal = 1;
direction = 0;

end