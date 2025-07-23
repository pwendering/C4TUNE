# do no move this file

import os
from pathlib import Path


# Set root directory (relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def resolve_config_paths(config, base_dir=PROJECT_ROOT):
    """
    Resolve relative paths in config file with the absolute path of the 
    root directory.

    Parameters
    ----------
    config : OmegaConf
        Model configuration file.
    base_dir : Path, optional
        Absolute path to C4TUNE root directory. The default is PROJECT_ROOT.

    Returns
    -------
    config : TYPE
        DESCRIPTION.

    """
    for key, val in config.get("paths", {}).items():
        
        if isinstance(val, str):
            path = Path(val)
            
            config["paths"][key] = path if path.is_absolute() else os.path.join(base_dir, path)
            
    return config