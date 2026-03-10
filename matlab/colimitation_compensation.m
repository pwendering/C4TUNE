% Co-limitation and compensation of model parameters

%% Load sampling results
sampling_workspace = fullfile(configC4Sim('result_dir'), 'sampling', ...
    'lognorm_chol', 'workspace_lognorm_chol.mat');
sampling_results = load(sampling_workspace, 'aci_samples_chol', ...
    'aq_samples_chol', 'p_samples_chol');

%% Filter out irrelevant curves

% get sorted CO2 steps
ca = configC4Sim('Ca_t');
ca(6) = [];
[~, ca_order] = sort(ca, 'ascend');

% get sorted light intensities
q = configC4Sim('Q_t');
[~, q_order] = sort(q, 'ascend');

% get indices of invalid curves
f_idx = filterPhotRespCurves( ...
    sampling_results.aci_samples_chol(:, ca_order)) | ...
    filterPhotRespCurves(sampling_results.aq_samples_chol(:, q_order));

% remove invalid curves
sampling_results.aci_samples_chol = ...
    real(sampling_results.aci_samples_chol(~f_idx, ca_order));
sampling_results.aq_samples_chol = ...
    real(sampling_results.aq_samples_chol(~f_idx, q_order));
sampling_results.p_samples_chol = ...
    real(sampling_results.p_samples_chol(~f_idx, :));

n_params = size(sampling_results.p_samples_chol, 2);
param_names = getParameterDescriptions;

clear aci_dist aq_dist rand_idx

load(fullfile(configC4Sim('result_dir'), 'parameter_identifiability', ...
    'workspace_parameter_redundancy'))

%% Compensation

% Determine correlations between sampled parameters and simulates Anet
% values in A/CO2 and A/light curves
if ~exist('corr_mat_params_aci', 'var')
    corr_mat_params_aci = corr(sampling_results.p_samples_chol, ...
        sampling_results.aci_samples_chol, 'Type', 'Spearman');
    corr_mat_params_aq = corr(sampling_results.p_samples_chol, ...
        sampling_results.aq_samples_chol, 'Type', 'Spearman');
end

% Miniumum number of identical curves per seed
min_num_sim = 10;

% Initialize result matrices
dist_mat = nan(n_params);

pw_corr = zeros(n_params);

n_hc_param = zeros(n_params);

corr_balance_mat = zeros(n_params);

% Function that defines valid combinations between pairwise parameter
% correlation and correlations to Anet
sign_check_fun = @(rp1a, rp2a, rp1p2) ...
    (sign(rp1a)==sign(rp2a))&(rp1p2<0) | ...
    (sign(rp1a)~=sign(rp2a))&(rp1p2>0) & ...
    all(~isnan([rp1a, rp2a, rp1p2]), 2);

% Threshold for parameter correlation to Anet across the entire synthetic
% dataset
t_r_anet = 0.1;

% A/CO2 curves
for c = 1:numel(ca)

    % Set correlations below threshold to NaN
    corr_mat_params_anet = corr_mat_params_aci(:, c);
    corr_mat_params_anet(abs(corr_mat_params_anet)<t_r_anet) = NaN;

    for i = find(n_sim_aci_aq>=min_num_sim)'

        % distances for the two curve types
        d_aci = mean(cbd_fun(sampling_results.aci_samples_chol, ...
            sampling_results.aci_samples_chol(rand_idx(i), :)), 2);
        d_aq = mean(cbd_fun(sampling_results.aq_samples_chol, ...
            sampling_results.aq_samples_chol(rand_idx(i), :)), 2);
        sim_idx = mean([d_aci d_aq], 2) < t_aci_aq;

        % Find correlated parameters
        corr_mat_params = triu(corr(sampling_results.p_samples_chol(sim_idx, :), ...
            'Type', 'Spearman')-eye(n_params));
        pw_corr = pw_corr + corr_mat_params;

        % Filter for high pairwise correlation between parameters
        hc_idx_linear = find(abs(corr_mat_params)>0.8);
        [hc_row_idx, hc_col_idx] = ind2sub(size(corr_mat_params), hc_idx_linear);
        
        % Find parameter pairs that comply with the correlation rules for
        % compensation
        sign_check = sign_check_fun( ...
            corr_mat_params_anet(hc_row_idx), ...
            corr_mat_params_anet(hc_col_idx), ...
            corr_mat_params(hc_idx_linear));
        
        hc_idx_linear = hc_idx_linear(sign_check);
        hc_row_idx = hc_row_idx(sign_check);
        hc_col_idx = hc_col_idx(sign_check);
        
        % Increase counter for high-correlation pairs in current condition
        n_hc_param(hc_idx_linear) = n_hc_param(hc_idx_linear) + 1;
        
        % Calculate distance score
        tmp_anet_corr = abs([ ...
            corr_mat_params_anet(hc_row_idx), ...
            corr_mat_params_anet(hc_col_idx) ...
        ]);

        tmp_corr_bal = max(tmp_anet_corr, [], 2) ./ min(tmp_anet_corr, [], 2) - 1;
        corr_balance_mat(hc_idx_linear) = corr_balance_mat(hc_idx_linear) + tmp_corr_bal;

        d_corr = tmp_corr_bal ./ mean(abs(tmp_anet_corr), 2) ./ abs(corr_mat_params(hc_idx_linear));

        dist_mat(hc_idx_linear) = sum([dist_mat(hc_idx_linear), d_corr], 2, 'omitmissing');

    end
end

% A/light curves
for l = 1:numel(q)
    
    % Set correlations below threshold to NaN
    corr_mat_params_anet = corr_mat_params_aq(:, l);
    corr_mat_params_anet(abs(corr_mat_params_anet)<t_r_anet) = NaN;

    for i = find(n_sim_aci_aq>=min_num_sim)'

        % distances for the two curve types
        d_aci = mean(cbd_fun(sampling_results.aci_samples_chol, ...
            sampling_results.aci_samples_chol(rand_idx(i), :)), 2);
        d_aq = mean(cbd_fun(sampling_results.aq_samples_chol, ...
            sampling_results.aq_samples_chol(rand_idx(i), :)), 2);
        sim_idx = mean([d_aci d_aq], 2) < t_aci_aq;

        % Find correlated parameters
        corr_mat_params = triu(corr(sampling_results.p_samples_chol(sim_idx, :), ...
            'Type', 'Spearman')-eye(n_params));
        pw_corr = pw_corr + corr_mat_params;
        
        % Filter for high pairwise correlation between parameters
        hc_idx_linear = find(abs(corr_mat_params)>0.8);
        [hc_row_idx, hc_col_idx] = ind2sub(size(corr_mat_params), hc_idx_linear);
        
        % Find parameter pairs that comply with the correlation rules for
        % compensation
        sign_check = sign_check_fun( ...
            corr_mat_params_anet(hc_row_idx), ...
            corr_mat_params_anet(hc_col_idx), ...
            corr_mat_params(hc_idx_linear));
        
        hc_idx_linear = hc_idx_linear(sign_check);
        hc_row_idx = hc_row_idx(sign_check);
        hc_col_idx = hc_col_idx(sign_check);

        % Increase counter for high-correlation pairs in current condition
        n_hc_param(hc_idx_linear) = n_hc_param(hc_idx_linear) + 1;
        
        % Calculate distance score
        tmp_anet_corr = abs([ ...
            corr_mat_params_anet(hc_row_idx), ...
            corr_mat_params_anet(hc_col_idx) ...
        ]);
        
        tmp_corr_bal = max(tmp_anet_corr, [], 2) ./ min(tmp_anet_corr, [], 2) - 1;
        corr_balance_mat(hc_idx_linear) = corr_balance_mat(hc_idx_linear) + tmp_corr_bal;

        d_corr = tmp_corr_bal ./ mean(abs(tmp_anet_corr), 2) ./ abs(corr_mat_params(hc_idx_linear));

        dist_mat(hc_idx_linear) = sum([dist_mat(hc_idx_linear), d_corr], 2, 'omitmissing');

    end
end

% Average pairwise correlations across all curves and sets of identical
% curves
pw_corr = pw_corr / sum(n_sim_aci_aq>=min_num_sim) / (numel(q) + numel(ca));

% Average of correlation balances for parameters to Anet
corr_balance_mat = corr_balance_mat ./ n_hc_param;

% Find compensating parameter pairs
score_mat_av = dist_mat ./ n_hc_param;
comp_idx_linear = find(~isnan(score_mat_av));
[comp_row_idx, comp_col_idx] = ind2sub(size(score_mat_av), comp_idx_linear);

% Compile result table
compensation_tab = table( ...
    param_names(comp_row_idx)', ...
    param_names(comp_col_idx)', ...
    score_mat_av(comp_idx_linear), ...
    pw_corr(comp_idx_linear), ...
    corr_balance_mat(comp_idx_linear) + 1, ...
    n_hc_param(comp_idx_linear), ...
    'VariableNames', {'P1', 'P2', 'dist_score', 'r_P_pw_av', 'corr_balance_av', 'occurrences'});
compensation_tab = sortrows(compensation_tab, 'dist_score', 'ascend');

writetable(compensation_tab, ...
    fullfile(configC4Sim('result_dir'), 'compensation_colimitation', ...
    'compensation.xlsx'))

%% Co-limitation

% Load A/CO2 and A/light curves measurements for 2022
a_co2_2022 = readtable(fullfile(configC4Sim('input_dir'), 'a_co2_maize_2022'), ...
    'ReadRowNames', true, 'ReadVariableNames', true);
a_co2_2022 = table2array(a_co2_2022);
a_light_2022 = readtable(fullfile(configC4Sim('input_dir'), 'a_light_maize_2022'), ...
    'ReadRowNames', true, 'ReadVariableNames', true);
a_light_2022 = table2array(a_light_2022);

% Load predicted parameter values for 2022
params_2022 = readtable(fullfile(configC4Sim('result_dir'), ...
    'parameter_prediction', 'params_2022'), ...
    'ReadRowNames', true, 'ReadVariableNames', true);
params_2022 = table2array(params_2022);

% Independent variables: standardized parameter values
X = zscore(params_2022);
% Dependent variables: logathmized Anet at highest irradiance
y = log(a_light_2022(:, end));
% Variance with one DOF for R2 calculation
var_y = var(y, 1);

p = gcp('nocreate');
if isempty(p)
    parpool(6);
end

% Send X to parallel workers
X_PAR = parallel.pool.Constant(X);

% Initialize R2 values and log-likelihood  values of one-parameter GP models
r2_ind = zeros(n_params, 1);
loglikelihood_ind = zeros(n_params, 1);

disp('Individual Models')
parfor i = 1:n_params

    % Fit GP model 
    mdl_i = fitrgp( ...
        X_PAR.Value(:, i), y, ...
        'KernelFunction', 'ardsquaredexponential');
    
    % Get model stats
    loglikelihood_ind(i) = mdl_i.LogLikelihood;
    cv_i  = crossval(mdl_i, 'KFold', 5);
    mse_i = kfoldLoss(cv_i);
    r2_ind(i) = 1 - mse_i / var_y;
end

% Define parameter pairs
pairs = nchoosek(1:n_params, 2);
pairs = pairs(randsample(1:size(pairs, 1), size(pairs, 1), false), :);

% Initialize results for two-parameter GP models
r2 = zeros(size(pairs, 1), 1);
lengthscale_ratio = zeros(size(pairs, 1), 1);
loglikelihood_pw = zeros(size(pairs, 1), 1);

disp('Pairwise Models')
parfor i = 1:size(pairs, 1)
    
    mdl_ij = fitrgp( ...
        X_PAR.Value(:, pairs(i, :)), y, ...
        'KernelFunction', 'ardsquaredexponential');

    % Get model stats
    loglikelihood_pw(i) = mdl_ij.LogLikelihood;

    cv_ij  = crossval(mdl_ij,'KFold', 5);
    mse_ij = kfoldLoss(cv_ij);

    r2(i) = 1 - mse_ij / var_y;
    
    l = 1./mdl_ij.KernelInformation.KernelParameters(1:2).^2;
    
    lengthscale_ratio(i) = min(l) / max(l);
end

% R2 differences
delta_r2_ind = r2 - max(r2_ind(pairs), [], 2);

% log-likelihood ratio test
ll_ratios = 2 * (loglikelihood_pw - max(loglikelihood_ind(pairs), [], 2));
p = 1 - chi2cdf(ll_ratios, 1);
[~, ~, ~, p_corr] = fdr_bh(p);

% Bayesian information criterion
bic_pw = 4 * log(size(y, 1)) - 2*loglikelihood_pw;
bic_ind = 3 * log(size(y, 1)) - 2*loglikelihood_ind;
delta_bic = min(bic_ind(pairs), [], 2) - bic_pw;

idx_co = p_corr < 0.05 & delta_r2_ind > prctile(delta_r2_ind, 99);

co_tab = table(...
    param_names(pairs(idx_co, 1))', ...
    param_names(pairs(idx_co, 2))', ...
    lengthscale_ratio(idx_co), ...
    r2(idx_co), ...
    r2_ind(pairs(idx_co, :)), ...
    delta_r2_ind(idx_co), ...
    delta_bic(idx_co), ...
    ll_ratios(idx_co), ...
    p_corr(idx_co), ...
    'VariableNames', {'Parameter 1', 'Parameter 2', 'Lr', ...
    'R2_pw', 'R2_ind', 'Delta_R2', 'Delta_BIC', 'LogLikelihood_ratio', ...
    'p_LL'});

co_tab = sortrows(co_tab, 'p_LL', 'ascend');
co_tab = sortrows(co_tab, 'LogLikelihood_ratio', 'descend');

save('colimitation_a_sat_2022', 'co_tab')
writetable(co_tab, fullfile(configC4Sim('result_dir'), 'compensation_colimitation', ...
    'colimitation_2022.xlsx'))
