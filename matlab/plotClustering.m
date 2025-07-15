function [f, t] = plotClustering(Y, C, xdata, x_label, y_label)
%PLOTCLUSTERING Plot clustered curves as separate panels
% Input
%   double Y:               data on which row-wise clustering was performed 
%   double C:               vector pf cluster IDs
%   double xdata:           (optional) x-axis data; default: 1:size(Y, 2)
%   char/cellstr xlabel:    (optional) X-axis label
%   char/cellstr ylabel:    (optional) Y-axis label
% 
% Output
%   figure handle f:        output figure
%   tiledlayout t:          tiledlayout that holds the panels

if ~exist('xdata', 'var')
    xdata = 1:size(Y, 2);
end

if ~exist('x_label', 'var')
    x_label = [];
end

if ~exist('y_label', 'var')
    y_label = [];
end

colors = [[215,76,84]; [83,132,51]; [133,42,95]; [96,137,212]; [165,182,65];...
    [146,57,27]; [211,83,135]; [189,128,212]; [85,52,131]; [184,73,164];...
    [203,133,48]; [71,187,138]; [51,212,209]; [166,148,70]; [178,69,85];...
    [102,198,115]; [122,128,234]; [219,124,87]; [215,123,183]; [86,86,186]]/255;
n_colors = size(colors, 1);

% cluster IDs
K = unique(C);

% repeat color array if necessary
colors = repmat(colors, ceil(numel(K)/n_colors), 1);

% for a high number of observations, make lines transparent
lalpha = 0.2;
if numel(C)>=1000
    colors = [colors lalpha*ones(size(colors, 1), 1)];
end

% sort clusters by number of curves
n_per_clust = arrayfun(@(i)sum(C==i), K);
[~, c_order] = sort(n_per_clust, 'descend');

% X- and Y-axis limits
ymin = min(Y, [], 'all');
ymax = 1.4*max(Y, [], 'all');
y_limits = [ymin ymax];
x_limits = [min(xdata) max(xdata)];

f = figure;

t = tiledlayout(f, 'flow');

% initialize graphics objects array
ax = gobjects(1, numel(K));

for i = K'

    ax(i) = nexttile(t);
    hold on

    % plot curves in current cluster
    X_C = Y(C == i, :);
    if size(X_C, 1) == size(X_C, 2)
        X_C = X_C';
    end
    plot(ax(i), xdata,...
        X_C,...
        'LineWidth', 1.3,...
        'Color', colors(i,:));

    % set x- and y-limits to global values
    set(ax(i), 'XLim', x_limits)
    set(ax(i), 'YLim', y_limits);
    
    % add rectangle for cluster title
    rect_y = y_limits(2)-0.2*range(y_limits);
    rectangle(ax(i), ...
        'Position', [x_limits(1) rect_y range(x_limits) y_limits(2)-rect_y],...
        'FaceColor', [.8 .8 .8],...
        'EdgeColor', 'k',...
        'LineWidth', 1.3)

    % cluster title
    text(ax(i), 0.5, 0.9, sprintf('Cluster %d (n=%d)', i, sum(C == i)),...
        'Units', 'normalized',...
        'FontName', 'Arial',...
        'FontSize', 8,...
        'FontWeight', 'bold',...
        'HorizontalAlignment', 'center')
    
    set(ax(i),...
        'Box', 'on',...
        'FontName', 'Arial',...
        'LineWidth', 1.3)
end

% add global X- and Y-axes titles
if startsWith(x_label, '$') && endsWith(x_label, '$')
    interpreter_x = 'latex';
else
    interpreter_x = 'tex';
end
t.XLabel.String = x_label;
t.XLabel.FontSize = 14;
t.XLabel.Interpreter = interpreter_x;

if startsWith(y_label, '$') && endsWith(y_label, '$')
    interpreter_y = 'latex';
else
    interpreter_y = 'tex';
end
t.YLabel.String = y_label;
t.YLabel.FontSize = 14;
t.YLabel.Interpreter = interpreter_y;

set(f, 'OuterPosition', 1000*[0.1977    0.1203    1.0513    0.5787])

end