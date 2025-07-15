function [C, best_settings, best_silh, silh_array, test_settings] = clusterCurves(data, varargin)
%CLUSTERCURVES cluster photosynthesis response curves
% Input
%   double data:                matrix with obervations in rows
%   logical test_flag:          (optional) whether to optimize clustering
%                               settings by grid search; default: true
%                               Will be set to false if any clustering
%                               settings are specified.
%   integer n_replicates:       (optional) number of replicates for K-means
%                               or K-medoids
%   char algorithm:             (optional) clustering algorithm
%                               [kmedoids|kmeans]; default: kmedoids
%   char kmed_alg:              (optional) algorithm choice for kmedoids
%                               [pam|small|clara|large];
%                               default: [] (let kmedoids decide)
%   char transformation:        (optional) data transformation
%                               [lin|log|tsne|spectral|zscore|minmax|pca]; default: log
%   integer K:                  (optional) number of clusters; default: 5
%   char dist_method:           (optional) distance method [euclidean|
%                               mahalanobis|cosine|correlation|spearman];
%                               default: mahalanobis
%   
% Output
%   double C:
%   table best_settings:
%   double best_silh:
%   double silh_array:
%   table test_settings:

[data, test_flag, best_settings, n_replicates] = parseInput(data, varargin);

if test_flag

    test_settings = getTestSettings;

    silh_array = nan(size(test_settings, 1), 1);

    % try all combinations
    for i = 1:size(test_settings, 1)
        
        try
            [C_test, X_test] = runClusteringWithSettings(test_settings(i, :), data,...
                n_replicates);
        catch ME
            fprintf('Error: %s\n', ME.message)
        end

        silh = assessClusterQuality(C_test, X_test, test_settings.dist_methods{i});
        silh_array(i) = median(silh);
    end

    % identify best settings and rerun clustering
    [~, silh_order] = max(silh_array);
    best_settings = test_settings(silh_order(1), :);
end

[C, X] = runClusteringWithSettings(best_settings, data, n_replicates);
best_silh = median(assessClusterQuality(C, X, best_settings.dist_methods{1}));

    function [data, test_flag, best_settings, n_replicates] = parseInput(data, args)
        
        p = inputParser;
        p.addRequired('data', @(x)isnumeric(x)&size(data, 2)>1)
        p.addParameter('test_flag', true, @(x)islogical(x)||ismember(x, [0 1]))
        p.addParameter('n_replicates', 10, @(x)isnumeric(x)&&isscalar(x)&&round(x)==x)
        p.addParameter('algorithm', 'kmedoids', @(x)ischar(x)&&ismember(x, ...
            {'kmedoids', 'kmeans'}))
        p.addParameter('kmed_alg', [], @(x)ischar(x)&&ismember(x, ...
            {'pam', 'small', 'clara', 'large'}))
        p.addParameter('transformation', 'log', @(x)ischar(x)&&ismember(x, ...
            {'lin', 'log', 'tsne', 'spectral', 'zscore', 'minmax', 'pca'}))
        p.addParameter('K', 5, @(x)isnumeric(x)&&isscalar(x)&&round(x)==x)
        p.addParameter('dist_method', 'mahalanobis', @(x)ischar(x)&&ismember(x,...
            {'euclidean', 'mahalanobis', 'cosine', 'correlation', 'spearman', ...
            'sqeuclidean', 'cityblock'}))
        
        parse(p, data, args{:})

        data = p.Results.data;
        test_flag = p.Results.test_flag;

        if ~all(ismember({'algorithm', 'kmed_alg', 'transformation',...
                'K', 'dist_method'}, p.UsingDefaults))
            test_flag = false;
        end

        if ~test_flag
            best_settings = cell2table({p.Results.algorithm, p.Results.kmed_alg,...
                p.Results.transformation, p.Results.K, p.Results.dist_method},...
                'VariableNames', {'cluster_alg', 'kmed_alg', 'transformations',...
                'K_array', 'dist_methods'});
        else
            best_settings = table;
        end

        n_replicates = p.Results.n_replicates;
    end

    function test_settings = getTestSettings
        % Creates a table to test settings for curve clustering, including
        % * clustering algorithm
        % * algoirthm for K-medoids
        % * data transformation
        % * number of clusters
        % * distance method
        
        cluster_alg = {'kmedoids'};
        % cluster_alg = {...
        %     'kmedoids', ...
        %     'kmeans'...
        %     };
        
        kmed_alg = {'large'};
        % kmed_alg = {...
        %     'pam', ...
        %     'small', ...
        %     'clara', ...
        %     'large'...
        %     };
        
        transformations = {'spectral'};
        % transformations = {...
        %     'lin', ...
        %     'log', ...
        %     'tsne', ...
        %     'spectral'
        %     };

        K_array = 2:15;
        
        dist_methods = {'euclidean'};
        % dist_methods = {...
        %     'euclidean', ...
        %     'mahalanobis', ...
        %     'cosine', ...
        %     'correlation', ...
        %     'spearman'...
        %     };

        test_settings = combinations(cluster_alg, kmed_alg, dist_methods, K_array,...
            transformations);

    end

    function [C, X] = runClusteringWithSettings(settings, data, n_rep)
        switch settings.transformations{:}
            case 'log'
                X = log10(log(1+exp(data)));
            case 'tsne'
                X = tsne(data);
            case 'spectral'
                [~, X] = spectralcluster(data, 5);
            case 'lin'
                X = data;
            case 'zscore'
                X = zscore(data,0,1);  % column-wise
            case 'minmax'
                X = (data - min(data)) ./ range(data);
            case 'pca'
                [~, X] = pca(zscore(data));
        end

        switch settings.cluster_alg{:}
            case 'kmedoids'
                if isempty(settings.kmed_alg{:})
                    C = kmedoids(X, settings.K_array,...
                        "Distance", settings.dist_methods{:},...
                        "Replicates", n_rep);
                else
                    C = kmedoids(X, settings.K_array,...
                        "Algorithm", settings.kmed_alg{:},...
                        "Distance", settings.dist_methods{:},...
                        "Replicates", n_rep);
                end
            case 'kmeans'
                if ismember(settings.dist_methods{:}, {'sqeuclidean',...
                        'cityblock', 'cosine', 'correlation', 'hamming'})
                    C = kmeans(X, settings.K_array,...
                        "Distance", settings.dist_methods{:},...
                        "Replicates", n_rep);
                else
                    warning('%s distance is not implemented in kmeans.', ...
                        settings.dist_methods{:})
                    C = ones(size(X, 1), 1);
                end
        end
    end

    function q = assessClusterQuality(C, X, dist_method)
        if ismember(dist_method, {'Euclidean', 'Jaccard', ...
                'cityblock', 'cosine', 'correlation', 'Hamming'})
            q = silhouette(X, C, dist_method);
        else
            q = silhouette(X, C, "Euclidean");
        end

    end

end
