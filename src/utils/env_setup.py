
import torch
from omegaconf import OmegaConf


def set_training_environment(config):
    '''
    
    Sets up the training environment:
        - number of threads
        - random number seeds
        - deterministic behavior

    Parameters
    ----------
    config : OmegaConf
        model-specific options.


    '''
    
    # set number of threads
    if hasattr(config.hpc, "n_threads_torch"):
        torch.set_num_threads(config.hpc.n_threads_torch)
    elif hasattr(config.settings, "n_threads_torch"):
        torch.set_num_threads(config.settings.n_threads_torch)
    
    # set random number seeds
    if hasattr(config.training, "seed"):
        seed = config.training.seed
    else:
        seed = 42
    
    torch.manual_seed(seed)
    
    # deterministic behavior
    torch.use_deterministic_algorithms(True)

def get_config(base_config_file, model_config_file):
    base_config = OmegaConf.load(base_config_file)
    model_config = OmegaConf.load(model_config_file)
    return OmegaConf.merge(base_config, model_config)