"""
Predict model parameters for genotypes of a MAGIC maize population
"""

import os
import time
import pandas as pd
import numpy as np
import torch
from torch import FloatTensor
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from c4tune.models.model_c4tune import ParameterPredictionModel
from c4tune.prediction.c4tune_predictor import C4tunePredictor
from c4tune.models.model_surrogate import SurrogateModel
from c4tune.prediction.surrogate_predictor import SurrogatePredictor
from c4tune.utils.env_setup import set_training_environment, get_config
from c4tune.utils.utils import load_param_names
from c4tune.data.data import PhotResponseDataset
from c4tune.c4_kinetic_model.c4model import C4DynamicModel
from c4tune.utils.paths import PROJECT_ROOT, resolve_config_paths


np.random.seed(123)
torch.manual_seed(321)

plt.rc('font', size=14)
plt.rc('legend', fontsize=10)

def filter_response_curves(anet):
    a_max = 70
    a_min = -10
    
    remove_bool = np.any(anet>a_max, axis=1) | np.sum(anet==0, axis=1)>1 | \
        np.all(anet<0, axis=1) | np.any(anet<a_min, axis=1) | \
        (np.sum(np.diff(np.sign(np.diff(anet, n=1, axis=1)), n=1, axis=1), axis=1)>0)
        
    return remove_bool
    
def coeff_var(x):
    return x.std()/x.mean()

def rvcoeff(x, y):
    return np.trace(x@x.T@y@y.T)/np.sqrt(np.trace(x@x.T@x@x.T)*np.trace(y@y.T@y@y.T))
    

# https://gist.github.com/thriveth/8560036#file-cbcolors-py
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']

# load base configuration
base_config_file = os.path.join(PROJECT_ROOT, "config/base.yaml")
base_config = resolve_config_paths(OmegaConf.load(base_config_file))

result_dir = os.path.join(PROJECT_ROOT, "results",
                         "parameter_prediction_maize_genotypes", "test")
data_dir = os.path.join(PROJECT_ROOT, "data",
                         "anet_measurements")

#%% Load experimental data

a_co2_2022 = pd.read_csv(os.path.join(data_dir, "a_co2_maize_2022.csv"), index_col=0)
a_co2_2023 = pd.read_csv(os.path.join(data_dir, "a_co2_maize_2023.csv"), index_col=0)
a_light_2022 = pd.read_csv(os.path.join(data_dir, "a_light_maize_2022.csv"), index_col=0)
a_light_2023 = pd.read_csv(os.path.join(data_dir, "a_light_maize_2023.csv"), index_col=0)

a_co2_2022_sd = pd.read_csv(os.path.join(data_dir, "a_co2_maize_2022_sd.csv"), index_col=0)
a_co2_2023_sd = pd.read_csv(os.path.join(data_dir, "a_co2_maize_2023_sd.csv"), index_col=0)
a_light_2022_sd = pd.read_csv(os.path.join(data_dir, "a_light_maize_2022_sd.csv"), index_col=0)
a_light_2023_sd = pd.read_csv(os.path.join(data_dir, "a_light_maize_2023_sd.csv"), index_col=0)

a_co2_2022_raw = pd.read_csv(os.path.join(data_dir, "a_co2_raw_2022.csv"), index_col=0)
a_co2_2023_raw = pd.read_csv(os.path.join(data_dir, "a_co2_raw_2023.csv"), index_col=0)
a_light_2022_raw = pd.read_csv(os.path.join(data_dir, "a_light_raw_2022.csv"), index_col=0)
a_light_2023_raw = pd.read_csv(os.path.join(data_dir, "a_light_raw_2023.csv"), index_col=0)

# number of accessions
n_acc = a_co2_2022.shape[0]

#%% Load artificial dataset
dataset = PhotResponseDataset(base_config.paths.datasets)
co2_steps = dataset.a_co2.columns.to_numpy(dtype='int')
light_steps = dataset.a_light.columns.to_numpy(dtype='int')
n_co2 = len(co2_steps)
n_light = len(light_steps)
n_params = dataset.params.shape[1]

#%% Create C4TUNE predictor

c4tune_config_file = os.path.join(PROJECT_ROOT, "config/c4tune.yaml")
config_c4tune = get_config(base_config_file, c4tune_config_file)

c4tune_checkpoint = os.path.join(config_c4tune.paths.run_dir, "2025-03-21", "c4tune-epoch-60.pth")

# Load Cholesky decomposition matrix and change the model's property
L = np.loadtxt(config_c4tune.paths.cholesky_test, delimiter=',')

# create C4TUNE and surrogate model predictors
set_training_environment(config_c4tune)
device = torch.device(config_c4tune.training.device if torch.cuda.is_available() else "cpu")

c4tune_model = ParameterPredictionModel(config_c4tune.model, L=FloatTensor(L))

c4tune = C4tunePredictor(c4tune_model, c4tune_checkpoint, device, config_c4tune)

#%% Create Surrogate model predictor

surrogate_config_file = os.path.join(PROJECT_ROOT, "config/surrogate.yaml")
surrogate_config = get_config(base_config_file, surrogate_config_file)

# numpy random seed 
np.random.seed(surrogate_config.training.rng_seed)

# model weights after training
surrogate_checkpoint = os.path.join(surrogate_config.paths.run_dir, "2025_02_21", "surrogate-epoch-60.pth")

# create surrogate model predictor
set_training_environment(surrogate_config)
surrogate_model = SurrogateModel(surrogate_config.model)
surrogate = SurrogatePredictor(surrogate_model, surrogate_checkpoint, device, surrogate_config)

#%% C4 kinetic model

# create Wrapper for C4 kinetic model simulation written in Matlab
c4model = C4DynamicModel(base_config)
co2_order = c4model.order_co2_steps
light_order = c4model.order_light_steps

#%% Predict parameters for all genotypes and years

env_input = {
    "co2_steps": dataset.co2_steps,
    "light_a_co2": dataset.light_a_co2,
    "light_steps": dataset.light_steps,
    "co2_a_light": dataset.co2_a_light,
    }

curve_input_2022 = {
    "a_co2": a_co2_2022.to_numpy(),
    "a_light": a_light_2022.to_numpy()
    }

curve_input_2023 = {
    "a_co2": a_co2_2023.to_numpy(),
    "a_light": a_light_2023.to_numpy()
    }

params_2022 = c4tune.predict(curve_input_2022, env_input)
params_2023 = c4tune.predict(curve_input_2023, env_input)

np.savetxt(os.path.join(result_dir, "params_2022.csv"), params_2022)
np.savetxt(os.path.join(result_dir, "params_2023.csv"), params_2023)

# parameter names
param_names = load_param_names()
n_params = len(param_names)

#%% Curve simulations

# ============ Surrogate model 
env_input = {
    "co2_steps": dataset.co2_steps,
    "light_a_co2": dataset.light_a_co2,
    "light_steps": dataset.light_steps,
    "co2_a_light": dataset.co2_a_light,
    }

curves_surrogate_2022 = surrogate.predict(params_2022, env_input)
curves_surrogate_2023 = surrogate.predict(params_2023, env_input)

# ============ ODE model
curves_ode_2022 = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]
curves_ode_2023 = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]

for i in range(0, n_acc):
    
    # simulate curves with predicted parameters using ODE model
    aci_tmp_2022, aq_tmp_2022 = c4model.simulate(params_2022[i].tolist())
    aci_tmp_2023, aq_tmp_2023 = c4model.simulate(params_2023[i].tolist())
    
    curves_ode_2022[0][i, :] = aci_tmp_2022.T
    curves_ode_2022[1][i, :] = aq_tmp_2022.T

    curves_ode_2023[0][i, :] = aci_tmp_2023.T
    curves_ode_2023[1][i, :] = aq_tmp_2023.T
    
#%% ============ Set failed ODE model simulations to nan

curves_ode_2022[0][np.any(curves_ode_2022[0]>=1e5, axis=1), :] = np.nan
curves_ode_2022[1][np.any(curves_ode_2022[1]>=1e5, axis=1), :] = np.nan

curves_ode_2023[0][np.any(curves_ode_2023[0]>=1e5, axis=1), :] = np.nan
curves_ode_2023[1][np.any(curves_ode_2023[1]>=1e5, axis=1), :] = np.nan

#%% ============ Calculate MSE

mse_surrogate_2022 = np.mean(np.concatenate(
    ((curves_surrogate_2022[0] - a_co2_2022.values)**2,
    (curves_surrogate_2022[1] - a_light_2022.values)**2), axis=1),
    axis=1)

mse_surrogate_2023 = np.mean(np.concatenate(
    ((curves_surrogate_2023[0] - a_co2_2023.values)**2,
    (curves_surrogate_2023[1] - a_light_2023.values)**2), axis=1),
    axis=1)

mse_ode_2022 = np.mean(np.concatenate(
    ((curves_ode_2022[0] - a_co2_2022.values)**2,
    (curves_ode_2022[1] - a_light_2022.values)**2), axis=1),
    axis=1)

mse_ode_2023 = np.mean(np.concatenate(
    ((curves_ode_2023[0] - a_co2_2023.values)**2,
    (curves_ode_2023[1] - a_light_2023.values)**2), axis=1),
    axis=1)

#%% ============ Plot MSE histograms

fig_mse_surrogate, axs_mse_surrogate = plt.subplots(1, 2, layout='tight')

nbins = 10
all_mse = [mse_surrogate_2022, mse_surrogate_2023,
           mse_ode_2022, mse_ode_2023]
bin_limits = [np.nanmin(all_mse), np.nanmax(all_mse)]

axs_mse_surrogate[0].hist(mse_surrogate_2022, bins=nbins, color=colors[0], range=bin_limits)
axs_mse_surrogate[0].set_xlabel("MSE (2022)")
axs_mse_surrogate[0].set_ylabel("Count")

axs_mse_surrogate[1].hist(mse_surrogate_2023, bins=nbins, color=colors[0], range=bin_limits)
axs_mse_surrogate[1].set_xlabel("MSE (2023)")

fig_mse_surrogate.savefig(os.path.join(result_dir, "mse_predicted_curves_surrogate.png"), dpi=300)

fig_mse_ode, axs_mse_ode = plt.subplots(1, 2, layout='tight')

axs_mse_ode[0].hist(mse_ode_2022, bins=nbins, color=colors[0], range=bin_limits)
axs_mse_ode[0].set_xlabel("MSE (2022)")
axs_mse_ode[0].set_ylabel("Count")

axs_mse_ode[1].hist(mse_ode_2023, bins=nbins, color=colors[0], range=bin_limits)
axs_mse_ode[1].set_xlabel("MSE (2023)")

fig_mse_ode.savefig(os.path.join(result_dir, "mse_predicted_curves_ode.png"), dpi=300)

#%% ============ Plot example curves

rand_idx = np.random.choice(range(0, n_acc), 1)[0]

fig_example_curves, axs_example_curves = plt.subplots(2, 2, layout='tight', figsize=(8, 6))

co2_x = np.array(a_co2_2022.columns, dtype='int')
light_x = np.array(a_light_2022.columns, dtype='int')

axs_example_curves[0, 0].plot(co2_x, a_co2_2022.iloc[rand_idx, :].values, color=colors[0])
axs_example_curves[0, 0].plot(co2_x, curves_surrogate_2022[0][rand_idx, :], color=colors[1])
axs_example_curves[0, 0].plot(co2_x, curves_ode_2022[0][rand_idx, :], color=colors[2])
axs_example_curves[0, 0].set_ylabel("2022"
                      "\n"
                      r"$A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$")

axs_example_curves[0, 1].plot(light_x, a_light_2022.iloc[rand_idx, :], color=colors[0])
axs_example_curves[0, 1].plot(light_x, curves_surrogate_2022[1][rand_idx, :], color=colors[1])
axs_example_curves[0, 1].plot(light_x, curves_ode_2022[1][rand_idx, :], color=colors[2])

axs_example_curves[1, 0].plot(co2_x, a_co2_2023.iloc[rand_idx, :], color=colors[0])
axs_example_curves[1, 0].plot(co2_x, curves_surrogate_2023[0][rand_idx, :], color=colors[1])
axs_example_curves[1, 0].plot(co2_x, curves_ode_2023[0][rand_idx, :], color=colors[2])
axs_example_curves[1, 0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")
axs_example_curves[1, 0].set_ylabel("2023"
                      "\n"
                      r"$A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$")

axs_example_curves[1, 1].plot(light_x, a_light_2023.iloc[rand_idx, :], color=colors[0])
axs_example_curves[1, 1].plot(light_x, curves_surrogate_2023[1][rand_idx, :], color=colors[1])
axs_example_curves[1, 1].plot(light_x, curves_ode_2023[1][rand_idx, :], color=colors[2])
axs_example_curves[1, 1].set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")

# unify y axes
ylim_all = [ax.get_ylim() for axs in axs_example_curves for ax in axs]
ylim = [np.min(ylim_all), np.max(ylim_all)]
[ax.set_ylim(ylim) for axs in axs_example_curves for ax in axs]

axs_example_curves[1, 1].legend(["measured", "surrogate", "ODE model"], frameon=False)

fig_example_curves.savefig(os.path.join(result_dir, "example_curves_predicted_params.png"), dpi=300)

#%% Parameter predictions within standard deviation using covariance matrix

n_samples_sd = 10;

params_2022_sd = [torch.Tensor]*n_samples_sd
params_2023_sd = [torch.Tensor]*n_samples_sd

a_co2_2022_samples = np.zeros((n_samples_sd, n_acc, n_co2))
a_light_2022_samples = np.zeros((n_samples_sd, n_acc, n_light))
a_co2_2023_samples = np.zeros((n_samples_sd, n_acc, n_co2))
a_light_2023_samples = np.zeros((n_samples_sd, n_acc, n_light))

for i in range(n_acc):

    # 2022
    acc_idx_co2 = a_co2_2022_raw.index==a_co2_2022.index[i]
    acc_idx_light = a_light_2022_raw.index==a_light_2022.index[i]
    
    n_rep_diff = np.sum(acc_idx_co2)-np.sum(acc_idx_light)
    if n_rep_diff > 0:
        padding = np.zeros((np.abs(n_rep_diff), n_light))
        padding[:] = np.nan
        tmp_anet = np.concatenate((a_co2_2022_raw.iloc[acc_idx_co2, :],
                                   np.concatenate((a_light_2022_raw.iloc[acc_idx_light, :],
                                                   padding), axis=0)), axis=1)
    elif  n_rep_diff < 0:
        padding = np.zeros((np.abs(n_rep_diff), n_co2))
        padding[:] = np.nan
        tmp_anet = np.concatenate((np.concatenate((a_co2_2022_raw.iloc[acc_idx_co2, :],
                                                   padding), axis=0),
                                   a_light_2022_raw.iloc[acc_idx_light, :]), axis=1)
    else:
        tmp_anet = np.concatenate((a_co2_2022_raw.iloc[acc_idx_co2, :],
                                   a_light_2022_raw.iloc[acc_idx_light, :]), axis=1)
    tmp_cov = np.ma.cov(np.ma.array((tmp_anet.T-tmp_anet.mean(axis=1)), mask=np.isnan(tmp_anet.T)))
    tmp_samples = np.random.multivariate_normal(np.concatenate((
        a_co2_2022_raw.iloc[acc_idx_co2, :].mean(),
        a_light_2022_raw.iloc[acc_idx_light, :].mean()), axis=0), 
        tmp_cov, size=n_samples_sd)
        
    a_co2_2022_samples[:, i, :] = tmp_samples[:, :n_co2][:, co2_order]
    a_light_2022_samples[:, i, :] = tmp_samples[:, n_co2:][:, light_order]
    
    # 2023
    acc_idx_co2 = a_co2_2023_raw.index==a_co2_2023.index[i]
    acc_idx_light = a_light_2023_raw.index==a_light_2023.index[i]
    
    n_rep_diff = np.sum(acc_idx_co2)-np.sum(acc_idx_light)
    if n_rep_diff > 0:
        padding = np.zeros((np.abs(n_rep_diff), n_light))
        padding[:] = np.nan
        tmp_anet = np.concatenate((a_co2_2023_raw.iloc[acc_idx_co2, :],
                                   np.concatenate((a_light_2023_raw.iloc[acc_idx_light, :],
                                                   padding), axis=0)), axis=1)
    elif  n_rep_diff < 0:
        padding = np.zeros((np.abs(n_rep_diff), n_co2))
        padding[:] = np.nan
        tmp_anet = np.concatenate((np.concatenate((a_co2_2023_raw.iloc[acc_idx_co2, :],
                                                   padding), axis=0),
                                   a_light_2023_raw.iloc[acc_idx_light, :]), axis=1)
    else:
        tmp_anet = np.concatenate((a_co2_2023_raw.iloc[acc_idx_co2, :],
                                   a_light_2023_raw.iloc[acc_idx_light, :]), axis=1)
    tmp_cov = np.ma.cov(np.ma.array((tmp_anet.T-tmp_anet.mean(axis=1)),
                                    mask=np.isnan(tmp_anet.T)))
    tmp_samples = np.random.multivariate_normal(np.concatenate((
        a_co2_2023_raw.iloc[acc_idx_co2, :].mean(),
        a_light_2023_raw.iloc[acc_idx_light, :].mean()), axis=0), 
        tmp_cov, size=n_samples_sd)
        
    a_co2_2023_samples[:, i, :] = tmp_samples[:, :n_co2][:, co2_order]
    a_light_2023_samples[:, i, :] = tmp_samples[:, n_co2:][:, light_order]
    
for i in range(0, n_samples_sd):
    
    
    curve_input_tmp = {
        "a_co2": a_co2_2022_samples[i],
        "a_light": a_light_2022_samples[i]
        }

    params_2022_sd[i] = c4tune.predict(curve_input_tmp, env_input)
    
    curve_input_tmp = {
        "a_co2": a_co2_2023_samples[i],
        "a_light": a_light_2023_samples[i]
        }

    params_2023_sd[i] = c4tune.predict(curve_input_tmp, env_input)

#%% Curve simulations within standard deviation

# ============ Surrogate model 

curves_surrogate_2022_sd = [[]]*n_samples_sd
curves_surrogate_2023_sd = [[]]*n_samples_sd

for i in range(0, n_samples_sd):
    
    curves_surrogate_2022_sd[i] = surrogate.predict(params_2022_sd[i], env_input)
    curves_surrogate_2023_sd[i] = surrogate.predict(params_2023_sd[i], env_input)

# ============ ODE model
curves_ode_2022_sd = [[None, None] for _ in range(0, n_samples_sd)]
curves_ode_2023_sd = [[None, None] for _ in range(0, n_samples_sd)]

for i in range(0, n_samples_sd):
    curves_ode_2022_sd[i][0] = np.zeros((n_acc, n_co2))
    curves_ode_2022_sd[i][1] = np.zeros((n_acc, n_light))
    curves_ode_2023_sd[i][0] = np.zeros((n_acc, n_co2))
    curves_ode_2023_sd[i][1] = np.zeros((n_acc, n_light))

for i in range(0, n_samples_sd):
    
    for j in range(0, n_acc):
        
        # simulate curves with predicted parameters using ODE model
        aci_tmp_2022, aq_tmp_2022 = c4model.simulate(params_2022_sd[i][j].tolist())
        aci_tmp_2023, aq_tmp_2023 = c4model.simulate(params_2023_sd[i][j].tolist())
        
        curves_ode_2022_sd[i][0][j, :] = aci_tmp_2022.T.copy()
        curves_ode_2022_sd[i][1][j, :] = aq_tmp_2022.T.copy()
        
        curves_ode_2023_sd[i][0][j, :] = aci_tmp_2023.T.copy()
        curves_ode_2023_sd[i][1][j, :] = aq_tmp_2023.T.copy()

#%% ============ Set failed ODE model simulations to nan

for i in range(0, n_samples_sd):
    curves_ode_2022_sd[i][0][np.any(curves_ode_2022_sd[i][0]>=1e5, axis=1), :] = np.nan
    curves_ode_2022_sd[i][1][np.any(curves_ode_2022_sd[i][1]>=1e5, axis=1), :] = np.nan
    
    curves_ode_2023_sd[i][0][np.any(curves_ode_2023_sd[i][0]>=1e5, axis=1), :] = np.nan
    curves_ode_2023_sd[i][1][np.any(curves_ode_2023_sd[i][1]>=1e5, axis=1), :] = np.nan

#%% Calculate means and standard deviations of simulated curves

curves_surrogate_2022_sd_mean = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]
curves_surrogate_2022_sd_sd = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]
curves_surrogate_2023_sd_mean = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]
curves_surrogate_2023_sd_sd = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]

for i in range(0, n_acc):
    
    tmp_a_co2 = [curves_surrogate_2022_sd[j][0][i] for j in range(0, n_samples_sd)]
    curves_surrogate_2022_sd_mean[0][i] = np.mean(tmp_a_co2, axis=0)
    curves_surrogate_2022_sd_sd[0][i] = np.std(tmp_a_co2, axis=0)
        
    tmp_a_light = [curves_surrogate_2022_sd[j][1][i] for j in range(0, n_samples_sd)]
    curves_surrogate_2022_sd_mean[1][i] = np.mean(tmp_a_light, axis=0)
    curves_surrogate_2022_sd_sd[1][i] = np.std(tmp_a_light, axis=0)
    
    tmp_a_co2 = [curves_surrogate_2023_sd[j][0][i] for j in range(0, n_samples_sd)]
    curves_surrogate_2023_sd_mean[0][i] = np.mean(tmp_a_co2, axis=0)
    curves_surrogate_2023_sd_sd[0][i] = np.std(tmp_a_co2, axis=0)
        
    tmp_a_light = [curves_surrogate_2023_sd[j][1][i] for j in range(0, n_samples_sd)]
    curves_surrogate_2023_sd_mean[1][i] = np.mean(tmp_a_light, axis=0)
    curves_surrogate_2023_sd_sd[1][i] = np.std(tmp_a_light, axis=0)

curves_ode_2022_sd_mean = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]
curves_ode_2022_sd_sd = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]
curves_ode_2023_sd_mean = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]
curves_ode_2023_sd_sd = [np.zeros((n_acc, n_co2)), np.zeros((n_acc, n_light))]

for i in range(0, n_acc):
    
    tmp_a_co2 = [curves_ode_2022_sd[j][0][i] for j in range(0, n_samples_sd)]
    curves_ode_2022_sd_mean[0][i] = np.nanmean(tmp_a_co2, axis=0)
    curves_ode_2022_sd_sd[0][i] = np.nanstd(tmp_a_co2, axis=0)
        
    tmp_a_light = [curves_ode_2022_sd[j][1][i] for j in range(0, n_samples_sd)]
    curves_ode_2022_sd_mean[1][i] = np.nanmean(tmp_a_light, axis=0)
    curves_ode_2022_sd_sd[1][i] = np.nanstd(tmp_a_light, axis=0)
    
    tmp_a_co2 = [curves_ode_2023_sd[j][0][i] for j in range(0, n_samples_sd)]
    curves_ode_2023_sd_mean[0][i] = np.nanmean(tmp_a_co2, axis=0)
    curves_ode_2023_sd_sd[0][i] = np.nanstd(tmp_a_co2, axis=0)
        
    tmp_a_light = [curves_ode_2023_sd[j][1][i] for j in range(0, n_samples_sd)]
    curves_ode_2023_sd_mean[1][i] = np.nanmean(tmp_a_light, axis=0)
    curves_ode_2023_sd_sd[1][i] = np.nanstd(tmp_a_light, axis=0)


#%% ============ Calculate MSE

mse_surrogate_2022_sd = [np.mean(np.concatenate(
    ((curves_surrogate_2022_sd[i][0] - a_co2_2022_samples[i])**2,
     (curves_surrogate_2022_sd[i][1] - a_light_2022_samples[i])**2), axis=1),
    axis=1)
    for i in range(0, n_samples_sd)
]

mse_surrogate_2023_sd = [np.mean(np.concatenate(
    ((curves_surrogate_2023_sd[i][0] - a_co2_2023_samples[i])**2,
     (curves_surrogate_2023_sd[i][1] - a_light_2023_samples[i])**2), axis=1),
    axis=1)
    for i in range(0, n_samples_sd)
]

mse_ode_2022_sd = [np.mean(np.concatenate(
    ((curves_ode_2022_sd[i][0] - a_co2_2022_samples[i])**2,
     (curves_ode_2022_sd[i][1] - a_light_2022_samples[i])**2), axis=1),
    axis=1)
    for i in range(0, n_samples_sd)
]
            
mse_ode_2023_sd = [np.mean(np.concatenate(
    ((curves_ode_2023_sd[i][0] - a_co2_2023_samples[i])**2,
     (curves_ode_2023_sd[i][1] - a_light_2023_samples[i])**2), axis=1),
    axis=1)
    for i in range(0, n_samples_sd)
]

#%% ============ Calculate R2

r2_surrogate_2022_sd = [1 - np.sum(np.concatenate(
    ((curves_surrogate_2022_sd[i][0] - a_co2_2022_samples[i])**2,
    (curves_surrogate_2022_sd[i][1] - a_light_2022_samples[i])**2), axis=1),
    axis=1) \
    / np.sum(np.concatenate(
        ((a_co2_2022_samples[i]- a_co2_2022_samples[i].mean())**2,
        (a_light_2022_samples[i] - a_light_2022_samples[i].mean())**2), axis=1),
        axis=1)
    for i in range(0, n_samples_sd)
    ]
    
r2_surrogate_2023_sd = [1 - np.sum(np.concatenate(
    ((curves_surrogate_2023_sd[i][0] - a_co2_2023_samples[i])**2,
    (curves_surrogate_2023_sd[i][1] - a_light_2023_samples[i])**2), axis=1),
    axis=1) \
    / np.sum(np.concatenate(
        ((a_co2_2023_samples[i]- a_co2_2023_samples[i].mean())**2,
        (a_light_2023_samples[i] - a_light_2023_samples[i].mean())**2), axis=1),
        axis=1)
    for i in range(0, n_samples_sd)
    ]
    
r2_ode_2022_sd = [1 - np.sum(np.concatenate(
    ((curves_ode_2022_sd[i][0] - a_co2_2022_samples[i])**2,
    (curves_ode_2022_sd[i][1] - a_light_2022_samples[i])**2), axis=1),
    axis=1) \
    / np.sum(np.concatenate(
        ((a_co2_2022_samples[i]- a_co2_2022_samples[i].mean())**2,
        (a_light_2022_samples[i] - a_light_2022_samples[i].mean())**2), axis=1),
        axis=1)
    for i in range(0, n_samples_sd)
    ]
    
r2_ode_2023_sd = [1 - np.sum(np.concatenate(
    ((curves_ode_2023_sd[i][0] - a_co2_2023_samples[i])**2,
    (curves_ode_2023_sd[i][1] - a_light_2023_samples[i])**2), axis=1),
    axis=1) \
    / np.sum(np.concatenate(
        ((a_co2_2023_samples[i]- a_co2_2023_samples[i].mean())**2,
        (a_light_2023_samples[i] - a_light_2023_samples[i].mean())**2), axis=1),
        axis=1)
    for i in range(0, n_samples_sd)
    ]

#%% ============ Plot MSE histograms

fig_mse_surrogate_sd, axs_mse_surrogate_sd = plt.subplots(1, 2, layout='tight')

nbins = 100
all_mse_sd = [mse_surrogate_2022_sd, mse_surrogate_2023_sd,
              mse_ode_2022_sd, mse_ode_2023_sd]
bin_limits = [np.nanmin(all_mse_sd), np.nanmax(all_mse_sd)]

axs_mse_surrogate_sd[0].hist(np.array(mse_surrogate_2022_sd).ravel(), bins=nbins,
             color=colors[0], range=bin_limits)
axs_mse_surrogate_sd[0].set_xlabel("MSE (2022)")
axs_mse_surrogate_sd[0].set_ylabel("Count")

axs_mse_surrogate_sd[1].hist(np.array(mse_surrogate_2023_sd).ravel(), bins=nbins,
             color=colors[0], range=bin_limits)
axs_mse_surrogate_sd[1].set_xlabel("MSE (2023)")

fig_mse_surrogate_sd.savefig(
    os.path.join(result_dir, "mse_predicted_curves_surrogate_within_sd.png"),
    dpi=300)


fig_mse_ode_sd, axs_mse_ode_sd = plt.subplots(1, 2, layout='tight')

axs_mse_ode_sd[0].hist(np.array(mse_ode_2022_sd).ravel(), bins=nbins,
             color=colors[0], range=bin_limits)
axs_mse_ode_sd[0].set_xlabel("MSE (2022)")
axs_mse_ode_sd[0].set_ylabel("Count")

axs_mse_ode_sd[1].hist(np.array(mse_ode_2023_sd).ravel(), bins=nbins,
             color=colors[0], range=bin_limits)
axs_mse_ode_sd[1].set_xlabel("MSE (2023)")

fig_mse_ode_sd.savefig(
    os.path.join(result_dir, "mse_predicted_curves_ode_within_sd.png"),
    dpi=300)

#%% ============ Plot example curves

rand_idx = np.random.choice(range(0, n_acc), 1)[0]

fig_example_curves_sd, axs_example_curves_sd = plt.subplots(2, 2, layout='tight', figsize=(8, 6))

co2_x = np.array(a_co2_2022.columns, dtype='int')
light_x = np.array(a_light_2022.columns, dtype='int')

fill_alpha = 0.2

# A/CO2 2022
axs_example_curves_sd[0, 0].plot(co2_x, a_co2_2022.iloc[rand_idx, :].values, color=colors[0])
axs_example_curves_sd[0, 0].fill_between(
    co2_x,
    a_co2_2022.iloc[rand_idx, :].values-a_co2_2022_sd.iloc[rand_idx, :].values,
    a_co2_2022.iloc[rand_idx, :].values+a_co2_2022_sd.iloc[rand_idx, :].values,
    alpha=fill_alpha, color=colors[0])
axs_example_curves_sd[0, 0].plot(co2_x, curves_surrogate_2022_sd_mean[0][rand_idx, :], color=colors[1])
axs_example_curves_sd[0, 0].fill_between(
    co2_x,
    curves_surrogate_2022_sd_mean[0][rand_idx, :]-curves_surrogate_2022_sd_sd[0][rand_idx, :],
    curves_surrogate_2022_sd_mean[0][rand_idx, :]+curves_surrogate_2022_sd_sd[0][rand_idx, :],
    alpha=fill_alpha, color=colors[1])
axs_example_curves_sd[0, 0].plot(co2_x, curves_ode_2022_sd_mean[0][rand_idx, :], color=colors[2])
axs_example_curves_sd[0, 0].fill_between(
    co2_x,
    curves_ode_2022_sd_mean[0][rand_idx, :]-curves_ode_2022_sd_sd[0][rand_idx, :],
    curves_ode_2022_sd_mean[0][rand_idx, :]+curves_ode_2022_sd_sd[0][rand_idx, :],
    alpha=fill_alpha, color=colors[2])
axs_example_curves_sd[0, 0].set_ylabel("2022"
                      "\n"
                      r"$A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$")

# A/light 2022
axs_example_curves_sd[0, 1].plot(light_x, a_light_2022.iloc[rand_idx, :].values, color=colors[0])
axs_example_curves_sd[0, 1].fill_between(
    light_x,
    a_light_2022.iloc[rand_idx, :].values-a_light_2022_sd.iloc[rand_idx, :].values,
    a_light_2022.iloc[rand_idx, :].values+a_light_2022_sd.iloc[rand_idx, :].values,
    alpha=fill_alpha, color=colors[0])
axs_example_curves_sd[0, 1].plot(light_x, curves_surrogate_2022_sd_mean[1][rand_idx, :], color=colors[1])
axs_example_curves_sd[0, 1].fill_between(
    light_x,
    curves_surrogate_2022_sd_mean[1][rand_idx, :]-curves_surrogate_2022_sd_sd[1][rand_idx, :],
    curves_surrogate_2022_sd_mean[1][rand_idx, :]+curves_surrogate_2022_sd_sd[1][rand_idx, :],
    alpha=fill_alpha, color=colors[1])
axs_example_curves_sd[0, 1].plot(light_x, curves_ode_2022_sd_mean[1][rand_idx, :], color=colors[2])
axs_example_curves_sd[0, 1].fill_between(
    light_x,
    curves_ode_2022_sd_mean[1][rand_idx, :]-curves_ode_2022_sd_sd[1][rand_idx, :],
    curves_ode_2022_sd_mean[1][rand_idx, :]+curves_ode_2022_sd_sd[1][rand_idx, :],
    alpha=fill_alpha, color=colors[2])

# A/CO2 2023
axs_example_curves_sd[1, 0].plot(co2_x, a_co2_2023.iloc[rand_idx, :].values, color=colors[0])
axs_example_curves_sd[1, 0].fill_between(
    co2_x,
    a_co2_2023.iloc[rand_idx, :].values-a_co2_2023_sd.iloc[rand_idx, :].values,
    a_co2_2023.iloc[rand_idx, :].values+a_co2_2023_sd.iloc[rand_idx, :].values,
    alpha=fill_alpha, label='_nolegend_', color=colors[0])
axs_example_curves_sd[1, 0].plot(co2_x, curves_surrogate_2023_sd_mean[0][rand_idx, :], color=colors[1])
axs_example_curves_sd[1, 0].fill_between(
    co2_x,
    curves_surrogate_2023_sd_mean[0][rand_idx, :]-curves_surrogate_2023_sd_sd[0][rand_idx, :],
    curves_surrogate_2023_sd_mean[0][rand_idx, :]+curves_surrogate_2023_sd_sd[0][rand_idx, :],
    alpha=fill_alpha, label='_nolegend_', color=colors[1])
axs_example_curves_sd[1, 0].plot(co2_x, curves_ode_2023_sd_mean[0][rand_idx, :], color=colors[2])
axs_example_curves_sd[1, 0].fill_between(
    co2_x,
    curves_ode_2023_sd_mean[0][rand_idx, :]-curves_ode_2023_sd_sd[0][rand_idx, :],
    curves_ode_2023_sd_mean[0][rand_idx, :]+curves_ode_2023_sd_sd[0][rand_idx, :],
    alpha=fill_alpha, label='_nolegend_', color=colors[2])
axs_example_curves_sd[1, 0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")
axs_example_curves_sd[1, 0].set_ylabel("2023"
                      "\n"
                      r"$A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$")

# A/light 2023
axs_example_curves_sd[1, 1].plot(light_x, a_light_2023.iloc[rand_idx, :].values, color=colors[0])
axs_example_curves_sd[1, 1].fill_between(
    light_x,
    a_light_2023.iloc[rand_idx, :].values-a_light_2023_sd.iloc[rand_idx, :].values,
    a_light_2023.iloc[rand_idx, :].values+a_light_2023_sd.iloc[rand_idx, :].values,
    alpha=fill_alpha, label='_nolegend_', color=colors[0])
axs_example_curves_sd[1, 1].plot(light_x, curves_surrogate_2023_sd_mean[1][rand_idx, :], color=colors[1])
axs_example_curves_sd[1, 1].fill_between(
    light_x,
    curves_surrogate_2023_sd_mean[1][rand_idx, :]-curves_surrogate_2023_sd_sd[1][rand_idx, :],
    curves_surrogate_2023_sd_mean[1][rand_idx, :]+curves_surrogate_2023_sd_sd[1][rand_idx, :],
    alpha=fill_alpha, label='_nolegend_', color=colors[1])
axs_example_curves_sd[1, 1].plot(light_x, curves_ode_2023_sd_mean[1][rand_idx, :], color=colors[2])
axs_example_curves_sd[1, 1].fill_between(
    light_x,
    curves_ode_2023_sd_mean[1][rand_idx, :]-curves_ode_2023_sd_sd[1][rand_idx, :],
    curves_ode_2023_sd_mean[1][rand_idx, :]+curves_ode_2023_sd_sd[1][rand_idx, :],
    alpha=fill_alpha, label='_nolegend_', color=colors[2])
axs_example_curves_sd[1, 1].set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")

# unify y axes
ylim_all = [ax.get_ylim() for axs in axs_example_curves_sd for ax in axs]
ylim = [np.min(ylim_all), np.max(ylim_all)]
[ax.set_ylim(ylim) for axs in axs_example_curves_sd for ax in axs]

axs_example_curves_sd[1, 1].legend(["measured", "surrogate", "ODE model"], frameon=False)

fig_example_curves_sd.savefig(
    os.path.join(result_dir, "example_curves_predicted_params_within_sd.png"),
    dpi=300)

#%% CV of predicted parameters within standard deviation

param_cv_2022 = np.apply_along_axis(coeff_var, 0, np.array(params_2022_sd))
param_cv_2023 = np.apply_along_axis(coeff_var, 0, np.array(params_2023_sd))

fig_param_cv_sd, ax_param_cv_sd = plt.subplots(layout='constrained', figsize=(3, 4))
nbins = 100
ax_param_cv_sd.hist([np.log10(param_cv_2022.ravel()),
           np.log10(param_cv_2023.ravel())],
          bins=nbins, color=colors[:2], histtype='step', linewidth=2)
ax_param_cv_sd.set_ylabel("Count")
ax_param_cv_sd.set_xlabel(r"$log_{10}\ CV$")
ax_param_cv_sd.legend(["2022", "2023"], frameon=False)
fig_param_cv_sd.savefig(os.path.join(result_dir, "parameter_prediction_sd_cv_hist_2022_2023.png"),
                        dpi=300)

#%% Investigate variability of experimental measurements

# === A/CO2
cv_a_co2_2022 = a_co2_2022_sd / a_co2_2022
cv_a_co2_median_2022 = np.median(cv_a_co2_2022)
cv_a_co2_mad_2022 = np.median(np.abs(cv_a_co2_2022-cv_a_co2_median_2022))

cv_a_co2_2023 = a_co2_2023_sd / a_co2_2023
cv_a_co2_median_2023 = np.median(cv_a_co2_2023)
cv_a_co2_mad_2023 = np.median(np.abs(cv_a_co2_2023-cv_a_co2_median_2023))

print("Median CV of A/CO2 measurements overall:")
print(f"2022:\t{cv_a_co2_median_2022:.2f} +/- {cv_a_co2_mad_2022:.2f}")
print(f"2023:\t{cv_a_co2_median_2023:.2f} +/- {cv_a_co2_mad_2023:.2f}")
print("")
print("Median CV of A/CO2 measurements per step:")
print(pd.concat([cv_a_co2_2022.median(), cv_a_co2_2023.median()], axis=1,
                keys=["2022", "2023"]))

# === A/light
cv_a_light_2022 = a_light_2022_sd / a_light_2022
cv_a_light_median_2022 = np.median(cv_a_light_2022)
cv_a_light_mad_2022 = np.median(np.abs(cv_a_light_2022-cv_a_light_median_2022))

cv_a_light_2023 = a_light_2023_sd / a_light_2023
cv_a_light_median_2023 = np.median(cv_a_light_2023)
cv_a_light_mad_2023 = np.median(np.abs(cv_a_light_2023-cv_a_light_median_2023))

print("Median CV of A/light measurements overall:")
print(f"2022:\t{cv_a_light_median_2022:.2f} +/- {cv_a_light_mad_2022:.2f}")
print(f"2023:\t{cv_a_light_median_2023:.2f} +/- {cv_a_light_mad_2023:.2f}")
print("")
print("Median CV of A/light measurements per step:")
print(pd.concat([cv_a_light_2022.median(), cv_a_light_2023.median()], axis=1,
                keys=["2022", "2023"]))

#%% predict parameters for 100 samples to investigate distribution and time
n_samples_sd = 100;

params_2022_sd_100 = [torch.Tensor]*n_samples_sd
params_2023_sd_100 = [torch.Tensor]*n_samples_sd

a_co2_2022_samples_100 = np.random.normal(loc=a_co2_2022.values, scale=a_co2_2022_sd.values,
                                      size=(n_samples_sd, n_acc, n_co2))
a_co2_2023_samples_100 = np.random.normal(loc=a_co2_2023.values, scale=a_co2_2023_sd.values,
                                      size=(n_samples_sd, n_acc, n_co2))
a_light_2022_samples_100 = np.random.normal(loc=a_light_2022.values, scale=a_light_2022_sd.values,
                                        size=(n_samples_sd, n_acc, n_light))
a_light_2023_samples_100 = np.random.normal(loc=a_light_2023.values, scale=a_light_2023_sd.values,
                                        size=(n_samples_sd,  n_acc, n_light))


tstart = time.time()
for i in range(0, n_samples_sd):
    
    curve_input_tmp = {
        "a_co2": a_co2_2022_samples_100[i],
        "a_light": a_light_2022_samples_100[i]
        }
    params_2022_sd_100[i] = c4tune.predict(curve_input_tmp, env_input)
    
    curve_input_tmp = {
        "a_co2": a_co2_2023_samples_100[i],
        "a_light": a_light_2023_samples_100[i]
        }
    params_2023_sd_100[i] = c4tune.predict(curve_input_tmp, env_input)
    

tend = time.time()-tstart
print(f"Prediction time for {n_acc*n_samples_sd} parameter sets: {tend:.2f} s.")
print(f"On average {1000*tend/n_acc/n_samples_sd:.2f} ms per parameter set.")
print(f"Estimated time for 10^6 samples: {1e6*tend/n_acc/n_samples_sd:.2f} s.")

a_co2_random = pd.DataFrame(
    np.random.normal(loc=a_co2_2022.values[0, :], scale=a_co2_2022_sd.values[0, :],
                     size=(100000, n_co2)),
    columns=a_co2_2022.columns)
a_light_random = pd.DataFrame(
    np.random.normal(loc=a_light_2022.values[0, :], scale=a_light_2022_sd.values[0, :],
                     size=(100000, n_light)),
    columns=a_light_2022.columns)

curve_input_tmp = {
    "a_co2": a_co2_random.to_numpy(),
    "a_light": a_light_random.to_numpy()
    }

tstart = time.time()

for i in range(10):
    c4tune.predict(curve_input_tmp, env_input)
    
tend = time.time()-tstart
print(f"Prediction time for 10^5 parameter sets: {tend:.2f} s.")

#%% What is the correlation between predicted parameters in 2022 and 2023?

vmax_idx = [x.startswith('Vm') or "Vmax" in x for x in param_names]
n_vmax = np.sum(vmax_idx)
param_names_vmax = [param_names[i] for i in range(0, len(param_names)) if vmax_idx[i]]

# C4TUNE
corr_22_23 = np.diag(
    np.corrcoef(
        params_2022.T, params_2023.T)[:params_2022.shape[1], params_2022.shape[1]:])
corr_22_23_vmax = np.diag(
    np.corrcoef(
        params_2022[:, vmax_idx].T, params_2023[:, vmax_idx].T)
    [:n_vmax, n_vmax:])

print(f"Average correlation of predicted parameters in 2022 and 2023: {corr_22_23.mean():.2f} +- {corr_22_23.std():.2f}")
print(f"Only Vmax values: {corr_22_23_vmax.mean():.2f} +- {corr_22_23_vmax.std():.2f}")

corr_22_23_order = np.argsort(corr_22_23)
[param_names[i] for i in corr_22_23_order[:10]]  # lowest agreement
[param_names[i] for i in corr_22_23_order[-10:]]  # highest agreement

corr_22_23_vmax_order = np.argsort(corr_22_23_vmax)
[param_names_vmax[i] for i in corr_22_23_vmax_order[:10]]  # lowest agreement
[param_names_vmax[i] for i in corr_22_23_vmax_order[-10:]]  # highest agreement

