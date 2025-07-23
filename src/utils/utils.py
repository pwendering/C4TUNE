# -*- coding: utf-8 -*-
"""

Utility functions and general parameters for neural network traning and evaluation.

"""

import torch
import pandas as pd
import numpy as np


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

def load_param_names():
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
    
    
    param_info = pd.read_excel("C:\\Users\\pw543\\OneDrive - University of Cambridge\\LearnPhotParams\\c4_model_simulation\\data\\parameter_info.xlsx")

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

def print_tree(tree):
    """
    
    Copied from sklearn documentation
    
    Prints decision tree to console
    
    """
    
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value
    
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
    
        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    
    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with value={value}.".format(
                    space=node_depth[i] * "\t", node=i, value=np.around(values[i], 3)
                )
            )
        else:
            print(
                "{space}node={node} is a split node with value={value}: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                    value=np.around(values[i], 3),
                )
            )