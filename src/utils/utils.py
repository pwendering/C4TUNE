# -*- coding: utf-8 -*-
"""

Utility functions and general parameters for neural network traning and evaluation.

"""

import torch
import pandas as pd
import numpy as np
import os


# https://gist.github.com/thriveth/8560036#file-cbcolors-py
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']

def load_and_freeze_model(model_class, config, checkpoint, device):
    """
    Creates an instance of the model class with input configuration.
    The weights are read from the checkpoint file and the model is frozen.

    Parameters
    ----------
    model_class : nn.Module
        Neural network.
    config : OmegaConfig
        model-specific configurations.
    checkpoint : str
        path to checkpoint.
    device : str
        model device.

    Returns
    -------
    model : nn.Module
        frozen model

    """
    
    model = model_class(config).to(device)
    cp_dir = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(cp_dir['model_state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model.eval()

def load_param_names(config):
    """
    
    Returns
    -------
    param_names : list
        Read the parameter information from an Excel spreadsheet and generates
        a name that combines
        * the enzyme short name,
        * the parameter type, and
        * the specificity of the parameter.
        
        If there is no enzyme short name associated with the parameter, the 
        Description column will be used as the name. If that is empty too, the 
        parameter ID will be returned.

    """
    
    
    param_info = pd.read_excel(os.path.join(config.paths.base_dir, "data", "parameter_info", "parameter_info.xlsx"))

    param_names = [str(E)+" "+str(T)+" ("+str(S)+")" for E, T, S in 
                   zip(param_info.loc[:, "Enzyme short"],
                       param_info.loc[:, "Type"],
                       param_info.loc[:, "Specificity"]
                       )]

    for i in range(0, len(param_names)):
        
        if type(param_info.loc[i, "Enzyme short"])!=str and np.isnan(param_info.loc[i, "Enzyme short"]):
            param_names[i] = str(param_info.loc[i, "Description"])
            
        if param_names[i]=='nan':
            param_names[i] = str(param_info.loc[i, "Description"])
            
        if param_names[i]=='nan':
            param_names[i] = str(param_info.loc[i, "ID"])
    param_names = [x.removesuffix(" (nan)") for x in param_names]
    
    return param_names

