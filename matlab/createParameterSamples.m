function P = createParameterSamples(N, X, stype)
%CREATEPARAMETERSAMPLES Create a matrix of parameter samples
% Input
%   double N:       number of samples
%   double X:       matrix of feasible parameter points
%   char stype:     sampling type
%                   * lognorm
%                   * cholesky
% Output
%   double P:       parameter matrix

switch stype

    case 'lognorm'
        
        % mean and standard deviation of log-transformed parameters in X
        p_mean = mean(log(X))';
        p_sd = std(log(X))';
        % create parameter samples
        P = exp(cell2mat(arrayfun(@(i)normrnd(p_mean, p_sd), 1:N, 'un', 0)))';

    case 'cholesky'
        
        % covariance matrix of log-transformed parameter points
        SIGMA = cov(log(X));

        % Cholesky decomposition
        [L, flag] = chol(SIGMA, 'lower');

        if flag
            error(['Cholesky decomposition returned code ' num2str(flag)])
        end
        
        % Get correlated random parameter samples
        Z = normrnd(0, 1, [size(X, 2), N]);
        P = exp(L*Z+log(mean(X)'))';

end