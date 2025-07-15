% Perform clustering of simulated Anet response curves

%% Load and filter simulated curves and sampled parameters
load(fullfile(configC4Sim('result_dir'), 'sampling', 'lognorm_chol', ...
    'workspace_lognorm_chol.mat'))

ca = configC4Sim('Ca_t');
ca(6) = [];
[ca_sort, ca_order] = sort(ca, 'ascend');

q = configC4Sim('Q_t');
[q_sort, q_order] = sort(q, 'ascend');

p_all = [p_samples_lognorm; p_samples_chol];
aci_all = [aci_samples_lognorm; aci_samples_chol];
aq_all = [aq_samples_lognorm; aq_samples_chol];

filter_idx = ...
    filterPhotRespCurves(aci_all(:, ca_order)) | ...
    filterPhotRespCurves(aq_all(:, q_order));

parameter_mat = real(p_all(~filter_idx, :)');
a_ci_sim = real(aci_all(~filter_idx, :));
a_q_sim = real(aq_all(~filter_idx, :));

% sample a subset of simulated curves for clustering
rng(1)
rand_idx = randsample(size(parameter_mat, 2), 1e4);

%% === Clustering of A/Ci curves ===
rng(42)

% perform spectral clustering of non-linear embedding
Y = tsne(a_ci_sim(rand_idx, ca_order));
[C_aci, best_settings_aci, sh_idx] = clusterCurves(...
    Y, ...
    'test_flag', false, 'K', 12, 'transformation', 'spectral', ...
    'dist_method', 'euclidean', 'algorithm', 'kmedoids', ...
    'kmed_alg', 'large');
disp(sh_idx)

% Plot the individual clusters
fig_aco2_clust = plotClustering(a_ci_sim(rand_idx, ca_order), C_aci, ca_sort, ...
    '$C_{a} \left(\mu bar\right)$',...
    '$A \left(\mu mol\;m^{-2}\;s^{-1}\right)$');
saveas(fig_aco2_clust, fullfile(configC4Sim('result_dir'), ...
    'a_co2_clust_kmedoids_euclidean_spectral_tsne_n1e4.png'), ...
    'Resolution', 300)

set(fig_aco2_clust, 'OuterPosition', [91.0000  501.5000  641.5000  457.0000], ...
    'Renderer', 'painters')
saveas(fig_aco2_clust, fullfile(configC4Sim('result_dir'), ...'
    'a_co2_clust_kmedoids_euclidean_spectral_tsne_n1e4_small.svg'))

% Plot the clustering
fig_a_co2_tsne = figure;
colors = [[215,76,84]; [83,132,51]; [133,42,95]; [96,137,212]; [165,182,65];...
    [146,57,27]; [211,83,135]; [189,128,212]; [85,52,131]; [184,73,164];...
    [203,133,48]; [71,187,138]; [51,212,209]; [166,148,70]; [178,69,85];...
    [102,198,115]; [122,128,234]; [219,124,87]; [215,123,183]; [86,86,186]]/255;
colormap(fig_a_co2_tsne, colors)
ax = axes(fig_a_co2_tsne);
scatter(ax, Y(:, 1), Y(:, 2), 'filled', 'CData', colors(C_aci, :))
xlabel(ax, 't-SNE 1')
ylabel(ax, 't-SNE 2')
set(ax, 'FontSize', 18, 'XLim', [-150, 150], 'YLim', [-150, 150])
set(fig_a_co2_tsne, 'OuterPosition', 1000*[1.2510    0.4760    0.6660    0.5895])
exportgraphics(fig_a_co2_tsne, fullfile(configC4Sim('result_dir'), ...
    'a_co2_tsne_n1e4.png'), ...
    'Resolution', 300)
saveas(fig_a_co2_tsne, fullfile(configC4Sim('result_dir'), ...
    'a_co2_tsne_n1e4.svg'))

%% === Clustering of A/Q curves ===

% perform spectral clustering of non-linear embedding
rng(42)
Y = tsne(a_q_sim(rand_idx, q_order));
[C_aq, best_settings_aq, sh_idx] = clusterCurves(...
    Y, ...
    'test_flag', false, 'K', 7, 'transformation', 'spectral', ...
    'dist_method', 'euclidean', 'algorithm', 'kmedoids', ...
    'kmed_alg', 'large');
disp(sh_idx)

% Plot the individual clusters
fig_alight_clust = plotClustering(a_q_sim(rand_idx, q_order), C_aq, q_sort, '$I \left(\mu mol\;m^{-2}\;s^{-1}\right)$',...
    '$A \left(\mu mol\;m^{-2}\;s^{-1}\right)$');
exportgraphics(fig_alight_clust, fullfile(configC4Sim('result_dir'), ...
    'a_light_clust_kmedoids_euclidean_spectral_tsne_n1e4.png'), ...
    'Resolution', 300)

set(fig_alight_clust, 'OuterPosition', [91.0000  501.5000  641.5000  457.0000], ...
    'Renderer', 'painters')
saveas(fig_alight_clust, fullfile(configC4Sim('result_dir'), ...'
    'a_light_clust_kmedoids_euclidean_spectral_tsne_n1e4_small.svg'))

% Plot the clustering
fig_a_light_tsne = figure;
colors = [[96,137,212]; [83,132,51]; [133,42,95]; [215,76,84]; [211,83,135]; ...
    [146,57,27]; [165,182,65]; [189,128,212]; [85,52,131]; [184,73,164];...
    [203,133,48]; [71,187,138]; [51,212,209]; [166,148,70]; [178,69,85];...
    [102,198,115]; [122,128,234]; [219,124,87]; [215,123,183]; [86,86,186]]/255;
colormap(fig_a_light_tsne, colors)
ax = axes(fig_a_light_tsne);
scatter(ax, Y(:, 1), Y(:, 2), 'filled', 'CData', colors(C_aq, :))
xlabel(ax, 't-SNE 1')
ylabel(ax, 't-SNE 2')
set(ax, 'FontSize', 18, 'XLim', [-150, 150], 'YLim', [-150, 150])
set(fig_a_light_tsne, 'OuterPosition', 1000*[1.2510    0.4760    0.6660    0.5895])
exportgraphics(fig_a_light_tsne, fullfile(configC4Sim('result_dir'), ...
    'a_light_tsne_n1e4.png'), ...
    'Resolution', 300)
saveas(fig_a_light_tsne, fullfile(configC4Sim('result_dir'), ...
    'a_light_tsne_n1e4.svg'))