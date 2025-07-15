% Assess the identifiability of model parameters

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

clear aci_dist aq_dist rand_idx

%% Find curves with distances below threshold

rng(37)

% random indices for pariwise comparison and calculation of CV
n_curves = 1e4;
rand_idx = randsample(1:sum(~f_idx), n_curves, false);

% threshold for pairwise distance between curves
t_aci = 0.01;
t_aq = 0.01;
t_aci_aq = 0.01;

% function to calculate the coeffcient of variation
cv_fun = @(X)std(X)./mean(X);
% function to calculate the Canberra distance
cbd_fun = @(X,Y)abs(X-Y)./(abs(X)+abs(Y));

% intialize result arrays
param_cv_aci = zeros(n_curves, size(sampling_results.p_samples_chol, 2));
n_sim_aci = zeros(n_curves, 1);
param_cv_aq = zeros(n_curves, size(sampling_results.p_samples_chol, 2));
n_sim_aq = zeros(n_curves, 1);
param_cv_aci_aq = zeros(n_curves, size(sampling_results.p_samples_chol, 2));
n_sim_aci_aq = zeros(n_curves, 1);

fprintf('Identifying identical curves for %i seeds.\n', n_curves)
for i = 1:n_curves
    
    if mod(i, 1000) == 0
        fprintf('Done with %i curves.\n', i)
    end

    % A/CO2
    d_aci = mean(cbd_fun(sampling_results.aci_samples_chol, ...
        sampling_results.aci_samples_chol(rand_idx(i), :)), 2);
    
    sim_idx_aci = d_aci < t_aci;

    param_cv_aci(i, :) = cv_fun(sampling_results.p_samples_chol(sim_idx_aci, :));
    n_sim_aci(i) = sum(sim_idx_aci);

    % A/light
    d_aq = mean(cbd_fun(sampling_results.aq_samples_chol, ...
        sampling_results.aq_samples_chol(rand_idx(i), :)), 2);

    sim_idx_aq = d_aq < t_aq;

    param_cv_aq(i, :) = cv_fun(sampling_results.p_samples_chol(sim_idx_aq, :));
    n_sim_aq(i) = sum(sim_idx_aq);

    % A/CO2 and A/light    
    sim_idx_comb = mean([d_aci, d_aq], 2) < t_aci_aq;

    param_cv_aci_aq(i, :) = cv_fun(sampling_results.p_samples_chol(sim_idx_comb, :));
    n_sim_aci_aq(i) = sum(sim_idx_comb);

end

save('workspace_parameter_redundancy', 'param_cv_aci', 'n_sim_aci', ...
    'param_cv_aq', 'n_sim_aq', 'param_cv_aci_aq', 'n_sim_aci_aq', ...
    'cbd_fun', 'rand_idx', 't_aci', 't_aq', 't_aci_aq')

%% Plot results

load('workspace_parameter_redundancy.mat')

rng(73)

close all

colors = [[255, 127, 0]; [55,  126, 184]; [152, 78,  163]]/255;

min_num_sim = 5;

% Example plots of similar curves with distance below threshold
i = randsample(find(n_sim_aci_aq>=min_num_sim), 1);

% distances for the two curve types
d_aci = mean(cbd_fun(sampling_results.aci_samples_chol, ...
        sampling_results.aci_samples_chol(rand_idx(i), :)), 2);
d_aq = mean(cbd_fun(sampling_results.aq_samples_chol, ...
        sampling_results.aq_samples_chol(rand_idx(i), :)), 2);
sim_idx = mean([d_aci d_aq], 2) < t_aci_aq;

% data for plotting
aci_plot_data = sampling_results.aci_samples_chol(sim_idx, :)';
aq_plot_data = sampling_results.aq_samples_chol(sim_idx, :)';

y_limits = [...
    min(min(aci_plot_data, [], 'all'), min(aq_plot_data, [], 'all')), ...
    max(max(aci_plot_data, [], 'all'), max(aq_plot_data, [], 'all'))];

% Plot example of highly-similar A/CO2 and A/light curves 
fig1 = figure;

t = tiledlayout(1, 2, 'TileSpacing', 'compact');

nexttile(t)
plot(repmat(ca(ca_order), size(aci_plot_data, 2), 1)', aci_plot_data)
xlabel('$C_a\ (\mu bar)$', 'Interpreter', 'latex')
xtickangle(30)
ylabel('$A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$', 'Interpreter', 'latex')
set(gca, ...
    'FontSize', 14, ...
    'YLim', y_limits)

nexttile(t)
plot(repmat(q(q_order), size(aq_plot_data, 2), 1)', aq_plot_data)
xlabel('$I\ (\mu mol\ m^{-2}\ s^{-2})$', 'Interpreter', 'latex')
xtickangle(30)
set(gca, ...
    'FontSize', 14, ...
    'YLim', y_limits)
set(fig1, 'OuterPosition', [121.6667  323.5000  660.3333  384.1667])

exportgraphics(fig1, fullfile(configC4Sim('result_dir'), ...
    'anet_identical_curves_t_0.01.png'), ...
    'Resolution', 300)

%% Plot distribution of CV 
% CV of parameters associated to highly-similar curves
param_names = getParameterDescriptions;

fig2 = figure('OuterPosition', 1000*[-1.2655    0.4455    1.2165    0.6090], ...
    'Renderer', 'painters');

ax1 = subplot('Position', [0.1428    0.5    0.2006    0.4]);
plot_idx_aci_aq = n_sim_aci_aq>=min_num_sim;
med_cv_aci_aq = median(param_cv_aci_aq(plot_idx_aci_aq, :)./cv_all);
[~, med_sort_idx_aq] = sort(med_cv_aci_aq, 'ascend');
box_colors = repmat(colors(2, :), numel(param_names), 1);
box_colors(1:10, :) = repmat(colors(1, :), 10, 1);
box_colors(end-9:end, :) = repmat(colors(3, :), 10, 1);
boxplot(ax1, param_cv_aci_aq(plot_idx_aci_aq, med_sort_idx_aq)./cv_all(med_sort_idx_aq), ...
    'PlotStyle', 'compact', ...
    'Labels', repmat({''}, size(param_names)),...
    'Colors', box_colors)
xticks(ax1, [])
xticklabels(ax1, [])
xlabel(ax1, 'model parameters')
ylabel(ax1, 'coefficient of variation')
text(ax1, 0.05, 0.92, 'D', 'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized')
set(ax1, 'FontSize', 10)

ax2 = subplot('Position', [0.4    0.33    0.2    0.57]);
boxplot(ax2, param_cv_aci_aq(plot_idx_aci_aq, med_sort_idx_aq(1:10))./cv_all(med_sort_idx_aq(1:10)), ...
    'PlotStyle', 'compact', ...
    'Labels', param_names(med_sort_idx_aq(1:10)), ...
    'Colors', colors(1, :))
set(gca, 'FontSize', 10, 'YLim', get(ax1, 'YLim'))
ax2_xticklabels = findobj(ax2.Children, 'Type', 'Text');
ax2_tick_pos = vertcat(ax2_xticklabels.Position);
ax2_tick_pos(:, 2) = -7;
text(ax2, 0.1, 0.92, 'E', 'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized')

ax3 = subplot('Position', [0.67    0.31    0.2    0.59]);
boxplot(ax3, param_cv_aci_aq(plot_idx_aci_aq, med_sort_idx_aq(end-9:end))./cv_all(med_sort_idx_aq(end-9:end)), ...
    'PlotStyle', 'compact', ...
    'Labels', param_names(med_sort_idx_aq(end-9:end)), ...
    'Colors', colors(3, :))
set(ax3, 'FontSize', 10, 'YLim', get(ax1, 'YLim'))
ax3_xticklabels = findobj(ax3.Children, 'Type', 'Text');
ax3_tick_pos = vertcat(ax3_xticklabels.Position);
ax3_tick_pos(:, 2) = -7;
text(ax3, 0.1, 0.92, 'F', 'FontSize', 14, 'FontWeight', 'bold', 'Units', 'normalized')

pause(1)

for i = 1:size(ax2_tick_pos, 1)
    set(ax2_xticklabels(i), 'Rotation', 45, 'HorizontalAlignment', 'right', ...
    'FontSize', 10, 'Position', ax2_tick_pos(i, :))
end

for i = 1:size(ax3_tick_pos, 1)
    set(ax3_xticklabels(i), 'Rotation', 45, 'HorizontalAlignment', 'right', ...
    'FontSize', 10, 'Position', ax3_tick_pos(i, :))
end

exportgraphics(fig2, fullfile(configC4Sim('result_dir'), ...
    'cv_params_identical_curves_t_0.01.png'), ...
    'Resolution', 300)
saveas(fig2, fullfile(configC4Sim('result_dir'), ...
    'cv_params_identical_curves_t_0.01.svg'))
