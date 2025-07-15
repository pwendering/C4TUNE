function writeTrainingData(work_space, out_folder, param_prefix, aci_prefix, aq_prefix)
%WRITETRAININGDATA Write parameters, A/Ci-, and A/Q-curves to text files
% Ci and Q values
% Input
%   char work_space:        path to matlab workspace (.mat)
%   char out_folder:        path to output directory
%                           parameters, A/Ci- and A/Q-curves will be
%                           written to text files with the following names:
%                           * params.csv
%                           * aci.csv
%                           * aq.csv
%   char param_prefix:      (optional) prefix for parameter variables
%                           default: 'p_samples'
%   char aci_prefix:        (optional) prefix of A/Ci-curve variables
%                           default: 'aci_samples'
%   char aq_prefix:         (optional) prefix of A/Q-curve variables
%                           default: 'aq_samples'

if ~exist('param_prefix', 'var') || isempty(param_prefix)
    param_prefix = 'p_samples';
end

if ~exist('aci_prefix', 'var') || isempty(aci_prefix)
    aci_prefix = 'aci_samples';
end

if ~exist('aq_prefix', 'var') || isempty(aq_prefix)
    aq_prefix = 'aq_samples';
end

prefix_list =  {param_prefix aci_prefix aq_prefix};

% create MatFile object from workspace file
if exist(work_space, 'file')
    matObj = matfile(work_space, 'Writable', true);
else
    error('Workspace does not exist.')
end

% If there are multiple variables with the same prefix, combine them
var_list = properties(matObj);
mat_array = cell(1, numel(prefix_list));
for i = 1:numel(prefix_list)
    var_idx = find(startsWith(var_list, prefix_list(i)));
    if any(var_idx)
        [~, abc_order] = sort(var_list(var_idx));
        for j = 1:numel(var_idx)
            mat_array{i} = vertcat(mat_array{i}, matObj.(var_list{var_idx(abc_order(j))}));
        end
    end
end

% get sorted CO2 steps
ca = configC4Sim('Ca_t');
ca(6) = [];
[~, ca_order] = sort(ca, 'ascend');

% get sorted light intensities
q = configC4Sim('Q_t');
[~, q_order] = sort(q, 'ascend');

% Get indices of invalid curves
f_idx = filterPhotRespCurves(mat_array{2}(:, ca_order)) | ...
    filterPhotRespCurves(mat_array{3}(:, q_order));

% Write parameters
param_names = load_parameter_name;
writetable(...
    array2table(real(mat_array{1}(~f_idx, :)), 'VariableNames', cellstr(param_names)), ...
    fullfile(out_folder, 'params.csv'))
mat_array{1} = {};

% Write A/Ci curves
writetable( ...
    array2table(real(mat_array{2}(~f_idx, ca_order)),...
    'VariableNames', arrayfun(@(i)sprintf('%i', i), ca(ca_order), 'un', 0)), ...
    fullfile(out_folder, 'a_co2.csv'))

% Write A/Q curves
writetable( ...
    array2table(real(mat_array{3}(~f_idx, q_order)), ...
    'VariableNames', arrayfun(@(i)sprintf('%i', i), q(q_order), 'un', 0)), ...
    fullfile(out_folder, 'a_light.csv'))

end