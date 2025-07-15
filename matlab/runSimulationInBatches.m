function [a_ci, a_q, tend] = runSimulationInBatches(bstart, batch_size,...
    n_samples, ncpu, work_space, suffix, ke_flag)
%RUNSIMULATIONINBATCHES Simulate A/Ci and A/Q curves in batches
% Input
%   double p_samples:           matrix of parameter points where the number
%                               columns corresponds to the number of 
%                               parameters
%   double a_ci:                matrix of A/Ci curves, where the row
%                               dimension correponds to the number of rows
%                               in p_samples
%   double a_q:                 matrix of A/Q curves, where the row
%                               dimension correponds to the number of rows
%                               in p_samples
%   integer b_start:            index of the batch to start with
%   integer batch_size:         
%   integer n_samples:          total number of samples
%   integer ncpu:               number of parallel workers
%   char suffix:                suffix for batch, A/Ci-, and A/Q curve 
%                               variables
%   char ke_flag:               
% Output
%   double a_ci:                simulated A/Ci curves
%   double a_q:                 simulated A/Q curves
%   double tend:                total runtime for all simulations

% set up parallel pool if necessary
p = gcp('nocreate');
if isempty(p) || p.NumWorkers ~= ncpu
    delete(p);
    parpool(ncpu);
end

% create MatFile object from workspace file
if exist(work_space, 'file')
    matObj = matfile(work_space, 'Writable', true);
else
    error('Workspace does not exist.')
end

expected_vars = strcat({'p_samples' 'aci_samples' 'aq_samples'}, suffix);
if ~all(cellfun(@(x)isprop(matObj, x), expected_vars))
    error('Workspace does not contain the expected variables.')
else
    ncol_aci = size(matObj.(expected_vars{2}), 2);
    ncol_aq = size(matObj.(expected_vars{3}), 2);
end

% number of batches
n_batches = ceil(n_samples/batch_size);

% initialize total runtime
t_start = tic;

for batch = bstart:n_batches
    
    % row indices of current batch
    b_idx_s = (batch-1)*batch_size+1;
    b_idx_e = min(batch*batch_size, n_samples);
    sim_idx =  b_idx_s:b_idx_e;
    
    % transfer parameter points once to each worker
    P = parallel.pool.Constant(matObj.(expected_vars{1})(sim_idx, :));
    K = parallel.pool.Constant(ke_flag);

    % initialize batch processing time
    tb_start = tic;
    
    % initialize A/Ci and A/Q curve matrices for current batch
    % initialized as complex bacause sometimes the ODE solver returns
    % complex values
    a_ci = complex(zeros(numel(sim_idx), ncol_aci));
    a_q = complex(zeros(numel(sim_idx), ncol_aq));

    parfor i = 1:numel(sim_idx)
        
        % suppress warnings from ODE solver
        warning off

        % simulate A-Ci curve
        a_ci(i, :) = simulate_ACI(P.Value(i, :)', K.Value);

        % simulate A-Q curve
        a_q(i, :) = simulate_AQ(P.Value(i, :)', K.Value);
        
    end
    
    % report progress
    tb_end = seconds(toc(tb_start));
    tb_end.Format = 'hh:mm:ss';
    t_tmp = seconds(toc(t_start));
    t_tmp.Format = 'hh:mm';
    fprintf('[%d/%d] t = %d h %d min %d s (%d h %d min)\n', ...
        b_idx_e, n_samples, ...
        cellfun(@str2double, strsplit(char(tb_end), ':')), ...
        cellfun(@str2double, strsplit(char(t_tmp), ':')))
    
    % create checkpoint
    matObj.(expected_vars{2})(sim_idx, :) = a_ci;
    matObj.(expected_vars{3})(sim_idx, :) = a_q;
    matObj.(['batch' suffix]) = batch;
    
end

% report total runtime
tend = seconds(toc(t_start));
tend.Format = 'hh:mm:ss';
fprintf('Total runtime: %d h %d min %d s\n', ...
    cellfun(@str2double, strsplit(char(tend), ':')))

end