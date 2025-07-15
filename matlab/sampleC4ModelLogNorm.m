function sampleC4ModelLogNorm
%
% Generate photosynthesis rate (A) response curves with sampled model
% parameters.
%   - model parameters are sampled from a log-normal distribution with mean and
%     standard deviation estimated across parameters from all accessions
%   - the first set of (feasible) sampled points is used to generate
%     additional samples while respecting the covariance structure of
%     previous samples (Cholesky decomposition)
%

%% Add simulation code to path
addpath(fullfile(configC4Sim('model_dir')))
addpath(fullfile(configC4Sim('model_dir'), 'simulation'))

%% Settings and paths

ncpu = configC4Sim('ncpu');

result_dir = fullfile(configC4Sim('result_dir'), 'sampling', 'lognorm_chol');
if ~exist(result_dir, 'dir')
    mkdir(result_dir)
end

work_space = fullfile(result_dir, 'workspace_lognorm_chol.mat');

% number of parameter points to simulate in step 1
n_samples_1 = 1000;
% number of parameter points to simulate in step 2
n_samples_2 = configC4Sim('n_samples');

% batch size
batch_size = 500;

% Which equilibrium constants should be used
ke_flag = configC4Sim('ke_flag');

%% Load data

% number of experimental p(CO2) steps
ca_steps = configC4Sim('Ca_t');
ca_steps(6) = [];
[~, ca_order] = sort(ca_steps, 'ascend');
n_ci = numel(ca_steps);

% number of experimenal PAR steps
light_steps = configC4Sim('Q_t');
[~, light_order] = sort(light_steps, 'ascend');
n_q = numel(light_steps);

%% ================ Sampling step 1 ================
% Sample random parameters points from normal distribution across
% accessions and years.

if exist(work_space, 'file')
    load(work_space, 'batch_lognorm')

    matObj = matfile(work_space, 'Writable', true);
    n_prev = size(matObj.p_samples_lognorm, 1);

    if n_samples_1 > n_prev
        % check if the number of requested samples is sill the same
        % and if yes, extend the parameter and result matrices

        rng(42)
        rand(1, n_prev);

        % create matrix containing optimal parameters from each accession
        all_params = readC4Params(ke_flag);
        p_samples_lognorm  = [matObj.p_samples_lognorm; ...
            createParameterSamples(n_samples_1-n_prev, all_params, 'lognorm')];
        clear all_params

        aci_samples_lognorm = [matObj.aci_samples_lognorm; ...
            complex(zeros(n_samples_1-n_prev, n_ci))];
        aq_samples_lognorm = [matObj.aq_samples_lognorm; ...
            complex(zeros(n_samples_1-n_prev, n_q))];

        % save rng status
        s = rng;
        
        batch_size = matObj.batch_size;

        % overwrite workspace intentionally
        save(work_space, ...
            'p_samples_lognorm', ...
            'aci_samples_lognorm', ...
            'aq_samples_lognorm', ...
            's', ...
            'batch_size', ...
            '-v7.3')

        clear p_samples_lognorm aci_samples_lognorm aq_samples_lognorm

    end

    if matObj.batch_size ~= batch_size
        % Check if batch size has been updated
        n_processed = matObj.batch_lognorm*matObj.batch_size;
        % start with the batch that is clostest to, but smaller than the
        % number of processed samples
        batch_lognorm = floor(n_processed/batch_size);
        if n_processed == n_samples_1
            batch_lognorm = batch_lognorm+1;
        end

        save(work_space, 'batch_lognorm', 'batch_size', '-append')
    end
    clear matObj
else
    % First time performing step 1

    % Ensure reproducibility of random numbers
    rng(42)

    % create matrix containing optimal parameters from each accession
    all_params = readC4Params(ke_flag);

    % create random parameter points
    p_samples_lognorm  = createParameterSamples(n_samples_1, all_params, 'lognorm');
    clear all_params

    % initialize simulated response curves
    aci_samples_lognorm = complex(zeros(n_samples_1, n_ci));
    aq_samples_lognorm = complex(zeros(n_samples_1, n_q));

    batch_lognorm = 0;

    % save rng status
    s = rng;

    save(work_space, ...
        'p_samples_lognorm', ...
        'aci_samples_lognorm', ...
        'aq_samples_lognorm', ...
        'batch_lognorm', ...
        'batch_size', ...
        's',...
        '-v7.3')
    clear p_samples_lognorm aci_samples_lognorm aq_samples_lognorm
end

% batch index to resume
bstart = batch_lognorm + 1;

% set up parallel pool
p = gcp('nocreate');
if isempty(p) || p.NumWorkers ~= ncpu
    delete(p);
    parpool(ncpu);
end

% simulate response curves
fprintf(['====================== STEP 1 ======================\n' ...
    'Simulating %i A/Ci and A/Q curves using parameters\n' ...
    'from log-normal distribution without considering covariance\n'],...
    n_samples_1)

runSimulationInBatches(bstart, batch_size, n_samples_1, ncpu, work_space, ...
    '_lognorm', ke_flag);

%% ================ Sampling step 2 ================
% Generate additional random parameters points, which consider the
% covariance structure of the feasible parameter points obtained in step 1.
% For this, Cholesky decomposition is applied to the covariance matrix of
% the logarithmized parameter points obtained in step 1.

% Load results from step 1
load(work_space, ...
    'batch_chol', ...
    's')

if exist('batch_chol', 'var')

    matObj = matfile(work_space, 'Writable', true);
    n_prev = size(matObj.p_samples_chol, 1);

    if n_samples_2 > n_prev
        % check if the number of requested samples is sill the same
        % and if yes, extend the parameter and result matrices

        rng(s)

        % create additional parameter samples
        matObj = matfile(work_space);

        aci = matObj.aci_samples_lognorm;
        aq = matObj.aq_samples_lognorm;
        f_idx = filterPhotRespCurves(aci(:, ca_order)) | filterPhotRespCurves(aq(:, light_order));
        clear aci aq

        p_samples_lognorm = matObj.p_samples_lognorm;
        matObj.p_samples_chol = [matObj.p_samples_chol;...
            createParameterSamples(n_samples_2-n_prev,...
            p_samples_lognorm(~f_idx, :), 'cholesky')];
        clear p_samples_lognorm

        matObj.aci_samples_chol = [matObj.aci_samples_chol; ...
            complex(zeros(n_samples_1-n_prev, n_ci))];
        matObj.aq_samples_chol = [matObj.aq_samples_chol; ...
            complex(zeros(n_samples_1-n_prev, n_q))];

        % save rng status
        matObj.s = rng;

    end

    if matObj.batch_size ~= batch_size
        n_processed = matObj.batch_chol*matObj.batch_size;
        % start with the batch that is clostest to, but smaller than the
        % number of processed samples
        batch_chol = floor(n_processed/batch_size);
        if n_processed == n_samples_1
            batch_chol = batch_chol+1;
        end
        save(work_space, 'batch_chol', 'batch_size', -append')
    end
    clear matObj
else
    % First time performing step 2

    batch_chol = 0;

    % Ensure reproducibility of random numbers
    rng(s)

    % create parameter samples
    matObj = matfile(work_space);
    
    aci = matObj.aci_samples_lognorm;
    aq = matObj.aq_samples_lognorm;
    f_idx = filterPhotRespCurves(aci(:, ca_order)) | filterPhotRespCurves(aq(:, light_order));
    clear aci aq

    p_samples_lognorm = matObj.p_samples_lognorm;
    p_samples_chol = createParameterSamples(n_samples_2,...
        p_samples_lognorm(~f_idx, :), 'cholesky');
    clear p_samples_lognorm matObj

    % initialize simulated response curves
    aci_samples_chol = complex(zeros(n_samples_2, n_ci));
    aq_samples_chol = complex(zeros(n_samples_2, n_q));

    % save rng status
    s = rng;

    save(work_space, ...
        'p_samples_chol', ...
        'aci_samples_chol', ...
        'aq_samples_chol', ...
        'batch_chol', ...
        's', ...
        '-append')
    clear p_samples_chol aci_samples_chol aq_samples_chol

end

% batch index to resume
bstart = batch_chol + 1;

% set up parallel pool
p = gcp('nocreate');
if isempty(p) || p.NumWorkers ~= ncpu
    delete(p);
    parpool(ncpu);
end

% simulate response curves
fprintf(['====================== STEP 2 ======================\n' ...
    'Simulating %i A/Ci and A/Q curves using parameters\n' ...
    'from log-normal distribution considering covariance\n'],...
    n_samples_2)

runSimulationInBatches(bstart, batch_size, n_samples_2, ncpu, work_space,...
    '_chol', ke_flag);

end