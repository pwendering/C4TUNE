function idx = filterPhotRespCurves(A)
%FILTERPHOTRESPCURVES Filter out unrealistic photsynthesis simulations
% Input
%   double A:       matrix of net CO2 assimilation rate simulations (rows),
%                   ordered by experimental parameter (CO2 / light)
%                   (columns)
% Output
%   logical idx:    indices of rows, which should be removed
% 
% Criteria
% * unrealisticly high values (>70)
% * curves with more than one zero value for Anet
% * numbers with imaginary parts greater than 1e-9 (if present)
% * curves with throughout negative numbers
% * unrealisticly low numbers (<=-10)
% * more than one sign change in the differences between ordered
%   measurements

A_max = 70;
A_min = -10;

A_real = real(A);

idx = ...
    any(A_real>A_max, 2) | ...
    sum(A_real==0, 2)>1 | ...
    any(imag(A)>1e-9, 2) | ...
    all(A_real<0, 2) | ...
    any(A_real<A_min, 2) | ...
    sum(diff(sign(diff(A_real, 1, 2)), 1, 2)~=0, 2)>1;

end