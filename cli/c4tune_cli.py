import os
import sys
from pathlib import Path
import torch
from torch import FloatTensor
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import json

sys.path.append(str(Path().resolve().parents[0]))

from src.models.model_c4tune import ParameterPredictionModel
from src.prediction.c4tune_predictor import C4tunePredictor
from src.utils.env_setup import set_training_environment, get_config
from src.utils.utils import load_param_names
from src.utils.paths import PROJECT_ROOT
from src.data.data import PhotResponseDataset


def process_input(a_co2_file, a_light_file, cache_file):
    
    a_co2 = pd.read_csv(a_co2_file, index_col=0)
    a_light = pd.read_csv(a_light_file, index_col=0)
    
    # check if columns are in order
    co2_steps = a_co2.columns.to_numpy(dtype='int')
    light_steps = a_light.columns.to_numpy(dtype='int')
    
    if ~np.all(np.sort(co2_steps) == co2_steps):
        print("Reordering the columns in the A/Ci curves.")
        new_order = np.argsort(co2_steps, axis=0)
        print("New order: " + ", ".join([str(x) for x in co2_steps[new_order]]))
        a_co2 = a_co2.iloc[:, new_order]
        
    if ~np.all(np.sort(light_steps) == light_steps):
        print("Reordering the columns in the A/Q curves.")
        new_order = np.argsort(light_steps, axis=0)
        print("New order: " + ", ".join([str(x) for x in light_steps[new_order]]))
        a_light = a_light.iloc[:, new_order]
    
    # check if steps correspond to cache
    if Path(cache_file).exists():
        with open(cache_file) as f:
            parameters = json.load(f)
            
            # Check the number of CO2 steps
            assert len(parameters['env_input']['co2_steps'])==len(co2_steps), \
                   f"The number of CO2 steps in the input file (n={len(co2_steps)}) " \
                   "does not correspond to the expected number of CO2 steps " \
                   f"(n={len(parameters['env_input']['co2_steps'])}). " \
                   "\nIf the number of CO2 steps is correct, rerun with \"--cache none\"" \
                   " or specify a cache file name to generate new inputs."

            # Check the number of light steps            
            assert len(parameters['env_input']['light_steps'])==len(light_steps), \
                   f"The number of light steps in the input file (n={len(light_steps)}) " \
                   "does not correspond to the expected number of light steps " \
                   f"(n={len(parameters['env_input']['light_steps'])}). " \
                   "\nIf the number of light steps is correct, rerun with \"--cache none\"" \
                   " or specify a cache file name to generate new inputs."
    
    return a_co2, a_light

def write_parameters(params, index, columns, outfile, out_fmt):
    
    if out_fmt == 'csv':
        if '.csv' not in outfile:
            outfile = outfile + ".csv"
        pd.DataFrame(params, index=index, columns=columns).to_csv(outfile)
    elif out_fmt == 'xlsx':
        if '.xlsx' not in outfile:
            outfile = outfile + ".xlsx"
        pd.DataFrame(params, index=index, columns=columns).to_excel(outfile)

def main(a_co2_file, a_light_file, outfile, input_cache, out_fmt):
    
    
    # Load Anet measurements from a CSV file
    print("Processing inputs...")  
    a_co2, a_light = process_input(a_co2_file, a_light_file, input_cache)
    
    # ===== Load model and data
    print("Loading the model...")
    # get C4TUNE and surrogate model configurations
    base_config_file = os.path.join(PROJECT_ROOT, "config/base.yaml")
    c4tune_config_file = os.path.join(PROJECT_ROOT, "config/c4tune.yaml")
    config_c4tune = get_config(base_config_file, c4tune_config_file)
    
    # numpy random seed 
    np.random.seed(config_c4tune.training.rng_seed)
    
    # model weights after training
    c4tune_checkpoint = os.path.join(config_c4tune.paths.run_dir, "2025-03-21", "c4tune-epoch-60.pth")
    
    # Load Cholesky decomposition matrix and change the model's property
    L = np.loadtxt(config_c4tune.paths.cholesky_test, delimiter=',')
    
    # create C4TUNE and surrogate model predictors
    set_training_environment(config_c4tune)
    device = torch.device(config_c4tune.training.device if torch.cuda.is_available() else "cpu")
    c4tune_model = ParameterPredictionModel(config_c4tune.model, L=FloatTensor(L))
    c4tune = C4tunePredictor(c4tune_model, c4tune_checkpoint, device, config_c4tune)
    
    if Path(input_cache).exists():
        env_input = None
    else:
        # Load the training dataset 
        dataset = PhotResponseDataset(config_c4tune.paths.datasets)
    
        # ===== Predict parameters
        print("Predicting parameters...")
        env_input = {
            "co2_steps": dataset.co2_steps,
            "light_a_co2": dataset.light_a_co2,
            "light_steps": dataset.light_steps,
            "co2_a_light": dataset.co2_a_light,
            }        
    
    curve_input = {
        "a_co2": a_co2.to_numpy(),
        "a_light": a_light.to_numpy()
        }
    
    # Predict model parameters
    pred_params = c4tune.predict(curve_input, env_input)
    
    param_names = load_param_names()
    write_parameters(pred_params, a_co2.index, param_names, outfile, out_fmt)
    
    print(f"Saved predicted parameters under {outfile}.")

if __name__ == "__main__":
    
    config = OmegaConf.load(os.path.join(PROJECT_ROOT, "config/base.yaml"))
    
    # create default output path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_file = os.path.join(PROJECT_ROOT, "outputs", "predictions",
                                       "parameters-" + timestamp)
    # default parameter cache JSON file
    default_cache_file = os.path.join(config.paths.cache_dir, "c4tune.json")

    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='c4tune',
        description='Predition of parameters for a C4 photosynthesis model.')

    parser.add_argument('a_co2_file', type=str, help='CSV file with A/Ci curves')
    parser.add_argument('a_light_file', type=str, help='CSV file with A/Q curves')
    
    parser.add_argument('-o', '--out', type=str, help='output file name',
                        default=default_output_file)
    parser.add_argument('-f', '--out_fmt', type=str, help='output format',
                        default='csv', choices=['csv', 'xlsx'])
    parser.add_argument('-c', '--cache', type=str, help='cached model input file name (JSON)',
                        default=default_cache_file)
    
    args = parser.parse_args()
    
    main(args.a_co2_file, args.a_light_file, args.out, args.cache, args.out_fmt)