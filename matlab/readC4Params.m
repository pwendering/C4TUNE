function P = readC4Params(ke_type)
%READC4PARAMS read and combine estimated C4 photosynthesis parameters
% Input
%   char ke_type:       indicated whether the original equilibrium
%                       constants from Wang et al. 2021 should be used or
%                       the calculated ones from eQuilibrator 3.0 
% Output
%   double P:           parameter matrix with observations in rows
%                       (accessions x years (Vmax))

% parameter names
param_names = load_parameter_name;

% path to estimated parameters
param_dir = fullfile(configC4Sim('input_dir'), 'mcmc_params');
param_file = fullfile(param_dir, ['optimized_parameters_' ke_type '.csv']);

% create matrix containing optimal parameters from each accession
param_tab = readtable(param_file,...
    'ReadRowNames', 1, 'ReadVariableNames', 1);
varnames = param_tab.Properties.VariableNames;

% indices of parameters, which are the same in 2022 and 2023
common_idx = ~startsWith(varnames, {'Vm', 'Jmax', 'Vtp'});

% 2022
idx_22 = endsWith(varnames, 'y22');
idx_22_full = find(common_idx|idx_22);
params_2022 = table2array(param_tab(:,...
    idx_22_full(cellfun(@(x)find(ismember(erase(varnames(idx_22_full), 'y22'), x)),...
    param_names))));

% 2023
idx_23 = endsWith(varnames, 'y23');
idx_23_full = find(common_idx|idx_23);
params_2023 = table2array(param_tab(:, ...
    idx_23_full(cellfun(@(x)find(ismember(erase(varnames(idx_23_full), 'y23'), x)),...
    param_names))));

% combined set of parameter points
P = [params_2022; params_2023];

end