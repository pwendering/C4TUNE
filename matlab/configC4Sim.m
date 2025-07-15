function param = configC4Sim(param_name)
% configure global settings for C4 model simulations

top_dir = fullfile('');
data_dir = top_dir;

switch param_name

    case 'result_dir'
        % path to directory where results are stored
        param = fullfile(data_dir, 'results', 'matlab_output');
    case 'model_dir'
        % path to top level directory with model files
        param = fullfile(top_dir, 'matlab', 'C4_dynamic_model');
    case 'input_dir'
        % path to directory where input data are stored
        param = fullfile(top_dir, 'data');
    case 'n_samples'
        % number of simulated curves to simulate
        param = 2e6;
    case 'Ca_t'
        % ambient p(CO2) used to generate A-Ci curves (in measurement
        % order)
        param = [400, 600, 800, 1000, 1250, 400, 300, 250, 200, 100, 75, 25];
    case 'Q_t'
        % PAR used to generate A-Q curves (in measurement order)
        param = [1800, 1100, 500, 300, 150, 50];
    case 'ncpu'
        % number of workers for parallel pool
        param = 6;
    case 'use_cluster'
        % whether we are working on a cluster
        param = false;
    case 'ke_flag'
        % indicates hich equilibrium constants should be used
        % either 'equilibrator' or 'original'
        param = 'equilibrator';
end

end

