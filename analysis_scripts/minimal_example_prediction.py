
import os
import sys
from pathlib import Path
import torch
from torch import FloatTensor
import numpy as np
import pandas as pd

sys.path.append(str(Path().resolve().parents[0]))

from src.models.model_c4tune import ParameterPredictionModel
from src.prediction.c4tune_predictor import C4tunePredictor
from src.utils.env_setup import set_training_environment, get_config
from src.data.data import PhotResponseDataset
from src.utils.paths import PROJECT_ROOT

# ===== Load model and data

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

# Load the training dataset 
dataset = PhotResponseDataset(config_c4tune.paths.datasets)

# Load Anet measurements from a CSV file
a_co2_file = "a_co2_measurements.csv"  # example file name
a_light_file = "a_light_measurements.csv"  # example file name

a_co2 = pd.read_csv(a_co2_file)
a_light = pd.read_csv(a_light_file)

# ===== Predict parameters for random subset of the test set 

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


