function p_names = getParameterDescriptions
%GETPARAMETERDESCRIPTIONS Create Parameter descriptions 
% Output
%   cellstr p_names:            descriptions for model parameters
% Read Excel file with parameter descriptions
param_info = readtable(fullfile(configC4Sim('input_dir'), 'parameter_info', 'parameter_info.xlsx'));

p_names = arrayfun(@(i)[ ...
    param_info.EnzymeShort{i}, ' ', ...
    param_info.Type{i}, ' (', ...
    param_info.Specificity{i}, ')'], 1:size(param_info, 1), 'un', 0);

empty_idx = cellfun(@isempty, param_info.EnzymeShort);
p_names(empty_idx) = param_info.Description(empty_idx);

empty_idx = cellfun(@isempty, p_names);
p_names(empty_idx) = param_info.Description(empty_idx);

empty_idx = cellfun(@isempty, p_names);
p_names(empty_idx) = param_info.ID(empty_idx);

p_names = erase(p_names, '()');

end