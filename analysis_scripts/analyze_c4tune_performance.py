# -*- coding: utf-8 -*-
"""

Assess the performance of the trained C4TUNE model.

"""

import os
import sys
from pathlib import Path
import torch
from torch import FloatTensor
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

sys.path.append(str(Path().resolve().parents[0]))

from src.models.model_surrogate import SurrogateModel
from src.models.model_c4tune import ParameterPredictionModel
from src.prediction.surrogate_predictor import SurrogatePredictor
from src.prediction.c4tune_predictor import C4tunePredictor
from src.utils.env_setup import set_training_environment, get_config
from src.utils.utils import load_param_names
from src.data.data import PhotResponseDataset
from src.c4_kinetic_model.c4model import C4DynamicModel

# font size for plotting
plt.rc('font', size=14)
plt.rc('legend', fontsize=10)

def calculate_r2(y, y_pred):
    return 1 - np.sum((y_pred - y)**2) / np.sum((y - y.mean())**2)

def calculate_mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def calculate_relative_error(y, y_pred):
    return np.abs(y_pred-y)/y

# =============================================================================
#%% Load model and data
# =============================================================================

# get C4TUNE and surrogate model configurations
base_config_file = "../config/base.yaml"
c4tune_config_file = "../config/c4tune.yaml"
config_c4tune = get_config(base_config_file, c4tune_config_file)

surrogate_config_file = "../config/surrogate.yaml"
config_surrogate = get_config(base_config_file, surrogate_config_file)

# numpy random seed 
np.random.seed(config_c4tune.training.rng_seed)

# model weights after training
c4tune_checkpoint = os.path.join(config_c4tune.paths.run_dir, "2025-03-21", "c4tune-epoch-60.pth")
surrogate_checkpoint = os.path.join(config_surrogate.paths.run_dir, "2025_02_21", "surrogate-epoch-60.pth")

# Load Cholesky decomposition matrix and change the model's property
L = np.loadtxt(config_c4tune.paths.cholesky_test, delimiter=',')

# create C4TUNE and surrogate model predictors
set_training_environment(config_c4tune)
device = torch.device(config_c4tune.training.device if torch.cuda.is_available() else "cpu")

c4tune_model = ParameterPredictionModel(config_c4tune.model, L=FloatTensor(L))
surrogate_model = SurrogateModel(config_surrogate.model)

c4tune = C4tunePredictor(c4tune_model, c4tune_checkpoint, device, config_c4tune)
surrogate = SurrogatePredictor(surrogate_model, surrogate_checkpoint, device, 
                               config_surrogate)

# read the dataset and indices of the test set
dataset = PhotResponseDataset(config_c4tune.paths.datasets)
idx_test = np.load(config_c4tune.paths.datasets.test_idx)
co2_steps = dataset.a_co2.columns.to_numpy(dtype='int')
light_steps = dataset.a_light.columns.to_numpy(dtype='int')
n_co2 = len(co2_steps)
n_light = len(light_steps)
n_params = dataset.params.shape[1]

# indices of dataset subsample for testing
n_samples = 5000
idx_samples = np.random.choice(idx_test,
                               np.min((n_samples, idx_test.shape[0]-1)), 
                               replace=False)

# data directory
data_dir = os.path.join(config_c4tune.paths.base_dir, "data")

# output directory
result_dir = os.path.join(config_c4tune.paths.base_dir, "results", "c4tune")

# =============================================================================
#%% Predict parameters for random subset of the test set 
# =============================================================================

env_input = {
    "co2_steps": dataset.co2_steps,
    "light_a_co2": dataset.light_a_co2,
    "light_steps": dataset.light_steps,
    "co2_a_light": dataset.co2_a_light,
    }

curve_input = {
    "a_co2": dataset.a_co2.iloc[idx_samples, :].to_numpy(),
    "a_light": dataset.a_light.iloc[idx_samples, :].to_numpy()
    }

pred_params = c4tune.predict(curve_input, env_input)

# =============================================================================
#%% Simulate A/CO2 and A/light curves using the kinetic model
# =============================================================================

# dataset curves for the subset
a_co2_ref = dataset.a_co2.iloc[idx_samples, :].to_numpy()
a_light_ref = dataset.a_light.iloc[idx_samples, :].to_numpy()

# create C4 model instance
c4model = C4DynamicModel(config_c4tune)

# initialize MSE and R2 arrays
mse = np.zeros((n_samples, 1))
mse[:] = np.nan
mse_a_co2 = mse.copy()
mse_a_light = mse.copy()
r2 = mse.copy()
r2_a_co2 = mse.copy()
r2_a_light = mse.copy()

print("Simulating curves using ODE model...")
for i in range(n_samples):
    
    # Call Matlab function to simulate curves using C4 dynamic model
    a_co2_ode, a_light_ode = c4model.simulate(pred_params[i].tolist())
    
    a_co2_ref_tmp = a_co2_ref[i, :].copy()
    a_co2_ref_tmp = np.expand_dims(a_co2_ref_tmp, 1).T
    a_light_ref_tmp = a_light_ref[i, :].copy
    a_light_ref_tmp = np.expand_dims(a_light_ref_tmp, 1).T
    
    # calculate MSE and R2 for feasible simulations
    if np.all(a_co2_ode<1e5) and np.all(a_light_ode<1e10):
        
        mse[i] = calculate_mse(np.concatenate((a_co2_ref_tmp, a_light_ref_tmp), axis=1),
                      np.concatenate((a_co2_ode, a_light_ode), axis=1))
        mse_a_co2[i] = calculate_mse(a_co2_ref_tmp, a_co2_ode)
        mse_a_light[i] = calculate_mse(a_light_ref_tmp, a_light_ode)
        
        r2[i] = calculate_r2(np.concatenate((a_co2_ode, a_light_ode), axis=1),
                             np.concatenate((a_co2_ref_tmp, a_light_ref_tmp), axis=1))
        r2_a_co2[i] = calculate_r2(a_co2_ode, a_co2_ref_tmp)
        r2_a_light[i] = calculate_r2(a_light_ode, a_light_ref_tmp)
    
    if i>0 and i%9==0:
        print(f"Done with {i+1} simulations.")

# save MSE and R2 values and corresponding parameter set indices
mse_df = pd.DataFrame(np.concatenate((mse, mse_a_co2, mse_a_light), axis=1),
                      index=idx_samples,
                      columns=["mse", "mse_a_co2", "mse_a_light"])
mse_df.to_csv(os.path.join(result_dir, "mse_simulation_test_set.csv"))

r2_df = pd.DataFrame(np.concatenate((r2, r2_a_co2, r2_a_light), axis=1),
                     index=idx_samples,
                     columns=["r2", "r2_a_co2", "r2_a_light"])
r2_df.to_csv(os.path.join(result_dir, "r2_simulation_test_set.csv"))

mse_median = np.nanmedian(mse)
mse_mad = np.nanmedian(np.abs(mse-mse_median))
mse_a_co2_median = np.nanmedian(mse_a_co2)
mse_a_co2_mad = np.nanmedian(np.abs(mse_a_co2-mse_a_co2_median))
mse_a_light_median = np.nanmedian(mse_a_light)
mse_a_light_mad = np.nanmedian(np.abs(mse_a_light-mse_a_light_median))

print(f"Median MSE ODE simulation (both curves): {mse_median:.2f} +- {mse_mad:.2f}")
print(f"Median MSE ODE simulation (A/CO2): {mse_a_co2_median:.2f} +- {mse_a_co2_mad:.2f}")
print(f"Median MSE ODE simulation (A/light): {mse_a_light_median:.2f} +- {mse_a_light_mad:.2f}")

r2_median = np.nanmedian(r2)
r2_mad = np.nanmedian(np.abs(r2-r2_median))
r2_a_co2_median = np.nanmedian(r2_a_co2)
r2_a_co2_mad = np.nanmedian(np.abs(r2_a_co2-r2_a_co2_median))
r2_a_light_median = np.nanmedian(r2_a_light)
r2_a_light_mad = np.nanmedian(np.abs(r2_a_light-r2_a_light_median))

print(f"Median R2 ODE simulation (both curves): {r2_median:.2f} +- {r2_mad:.2f}")
print(f"Median R2 ODE simulation (A/CO2): {r2_a_co2_median:.2f} +- {r2_a_co2_mad:.2f}")
print(f"Median R2 ODE simulation (A/light): {r2_a_light_median:.2f} +- {r2_a_light_mad:.2f}")

# =============================================================================
#%% Distribution of mean squared errors from curve simulations 
# =============================================================================

# read MSE results from ODE model simulations
# mse_df = pd.read_csv((os.path.join(result_dir, "mse_simulation_test_set.csv")))
# mse = mse_df.mse

nbins = 100

fig_mse_distr, ax_mse_distr = plt.subplots(figsize=(3, 4), layout='constrained')
N, bins, patches = ax_mse_distr.hist(np.log10(mse), color='#377eb8', bins=nbins)
ax_mse_distr.set_xlabel(r'$log_{10}\ MSE$')
ax_mse_distr.set_ylabel('Count')

fig_mse_distr.savefig(
    os.path.join(result_dir, "parameter_model_mse_ode_simulation.png"),
    dpi=300)


# =============================================================================
#%% Predict curves using the surrogate model using the predicted parameters
# =============================================================================

curves_surrogate = surrogate.predict(pred_params, env_input)

mse_surrogate = np.concatenate((
    (curves_surrogate[0] - a_co2_ref)**2,
    (curves_surrogate[1] - a_light_ref)**2),
    axis=1).mean(axis=1)

r2_surrogate = 1 - \
    np.sum(np.concatenate(
        ((curves_surrogate[0] - a_co2_ref)**2,
         (curves_surrogate[1] - a_light_ref)**2),
        axis=1), axis=1) \
    / np.sum(np.concatenate((
        (a_co2_ref - np.expand_dims(a_co2_ref.mean(axis=1), 1))**2,
        (a_light_ref - np.expand_dims(a_light_ref.mean(axis=1), 1))**2),
        axis=1), axis=1)

# =============================================================================
#%% Plot example curves from predicted parameters
# =============================================================================

# read MSE results from ODE model simulations
# mse_df = pd.read_csv((os.path.join(result_dir, "mse_simulation_test_set.csv")))
# mse = mse_df.mse

plot_idx = np.random.choice(np.arange(n_samples))

# ODE simulation with predicted parameters for selected index
a_co2_ode_plot, a_light_ode_plot = c4model.simulate(pred_params[plot_idx].tolist())

# Plot A/CO2 curve
fig_example_curve, axs_example_curve = plt.subplots(1, 2, figsize=(8, 4), layout='tight')
axs_example_curve[0].plot(co2_steps, a_co2_ref[plot_idx, :])  # sampling
axs_example_curve[0].plot(co2_steps, a_co2_ode_plot[0])  # ODE simulation with predicted parameters
axs_example_curve[0].plot(co2_steps, curves_surrogate[0][plot_idx])  # surrogate
axs_example_curve[0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")
axs_example_curve[0].set_ylabel(r"$A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$")
axs_example_curve[0].set_ylim([-5, axs_example_curve[0].get_ylim()[1]])

# Plot A/light curve
axs_example_curve[1].plot(light_steps, a_light_ref[plot_idx, :])  # sampling
axs_example_curve[1].plot(light_steps, a_light_ode_plot[0])  # ODE simulation with predicted parameters
axs_example_curve[1].plot(light_steps, curves_surrogate[1][plot_idx])  # surrogate
axs_example_curve[1].legend(["sampled", f"ODE model (MSE={mse[plot_idx][0]:.1f})", 
               f"surrogate (MSE={mse_surrogate[plot_idx]:.1f})"],
              loc='lower right', frameon=False)
axs_example_curve[1].set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")
axs_example_curve[1].set_ylim(axs_example_curve[0].get_ylim())

fig_example_curve.savefig(
    os.path.join(result_dir, "parameter_prediction_example_curves_random.png"),
    dpi=300)

# =============================================================================
#%% Compare true and predicted parameters 
# =============================================================================

# compute relative errors for parameter prediction
re = calculate_relative_error(dataset.params.iloc[idx_samples,:].to_numpy(),
                              pred_params)
re_median = np.expand_dims(np.median(re, axis=1), 1)
min_val = np.min((np.log10(pred_params), np.log10(dataset.params.iloc[idx_samples,:].to_numpy())))
max_val = np.max((np.log10(pred_params), np.log10(dataset.params.iloc[idx_samples,:].to_numpy())))

# create figure
fig_c4tune_performance, axs_c4tune_performance = plt.subplots(1, 2, layout='constrained')
axs_c4tune_performance[0].scatter(
    np.log10(dataset.params.iloc[idx_samples,:].to_numpy()),
    np.log10(pred_params),
    c=re_median.repeat(re.shape[1], axis=1))
axs_c4tune_performance[0].set_xlim([min_val, max_val])
axs_c4tune_performance[0].set_ylim([min_val, max_val])
axs_c4tune_performance[0].set_yticks([-5.0, -2.5, 0.0, 2.5])

nbins = 100

N, bins = np.histogram(re_median, bins=nbins, density=True)
xcenters = (bins[:-1] + bins[1:]) / 2
norm = colors.Normalize(re_median.min(), re_median.max())
bin_width = bins[1]-bins[0]
axs_c4tune_performance[1].barh(xcenters, 1.5*N.max()*np.ones(len(xcenters)),
        color=[plt.cm.viridis(norm(xc)) for xc in xcenters],
        height=bin_width)

axs_c4tune_performance[1].set_box_aspect(3)
axs_c4tune_performance[1].hist(re_median,
          bins=nbins,
          density=True,
          histtype='step',
          orientation='horizontal',
          color=[0, 0, 0])
axs_c4tune_performance[1].yaxis.tick_right()
axs_c4tune_performance[1].yaxis.set_label_position('right')

axs_c4tune_performance[1].set_ylabel('Median relative error')
axs_c4tune_performance[1].set_xlabel('density')
axs_c4tune_performance[1].set_ylim([xcenters[0], xcenters[-1]])
axs_c4tune_performance[1].set_xlim([N.min(), N.max()])
axs_c4tune_performance[1].yaxis.tick_right()
axs_c4tune_performance[1].yaxis.set_label_position('right')

axs_c4tune_performance[1].set_yticks(axs_c4tune_performance[1].get_yticks()[1:-1])
axs_c4tune_performance[1].set_yticklabels(axs_c4tune_performance[1].get_yticks().round(2))

axs_c4tune_performance[1].set_box_aspect(7)

fig_c4tune_performance.supxlabel(r"$log_{10}\ parameters\ (sampling)$")
fig_c4tune_performance.supylabel(r"$log_{10}\ parameters\ (prediction)$")

fig_c4tune_performance.savefig(
    os.path.join(result_dir, "parameter_model_mean_re_n1e4.png"),
    dpi=300)

# =============================================================================
#%% Relative error per parameter
# =============================================================================

param_names = load_param_names(config_c4tune)

min_val = re.min().min()
max_val = 1.1*re.max().max()

# sort relative errors in ascending order based on the median
re_order = np.argsort(np.median(re, axis=0))

# plot parameters with highest median relative errors
fig_re_distr, axs_re_distr = plt.subplots(1, 2, figsize=(8, 5), layout='constrained')
parts1 = axs_re_distr[0].violinplot(np.log10(re[:, re_order[:10]]),
                            showmedians=True)
for pc in parts1['bodies']:
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)
parts1['cbars'].set_color([.4, .4, .4])
parts1['cmins'].set_color([.4, .4, .4])
parts1['cmaxes'].set_color([.4, .4, .4])
parts1['cmedians'].set_color([.4, .4, .4])

axs_re_distr[0].set_xticks(np.arange(1, 11), [param_names[i] for i in re_order[:10]],
                   horizontalalignment='right')
axs_re_distr[0].tick_params(axis='x', rotation=45)
axs_re_distr[0].set_ylim(np.log10([min_val, max_val]))
axs_re_distr[0].set_xlim([-1, 11])
axs_re_distr[0].set_ylabel(r"$log_{10}\ relative\ error$")

# plot parameters with highest median relative errors
parts2 = axs_re_distr[1].violinplot(np.log10(re[:, re_order[-10:]]),
                            showmedians=True)
for pc in parts2['bodies']:
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)
parts2['cbars'].set_color([.4, .4, .4])
parts2['cmins'].set_color([.4, .4, .4])
parts2['cmaxes'].set_color([.4, .4, .4])
parts2['cmedians'].set_color([.4, .4, .4])

axs_re_distr[1].set_xticks(np.arange(1, 11), [param_names[i] for i in re_order[-10:]],
                   horizontalalignment='right')
axs_re_distr[1].set_yticks([])
axs_re_distr[1].set_xlim([-1, 11])
axs_re_distr[1].tick_params(axis='x', rotation=45)

fig_re_distr.savefig(
    os.path.join(result_dir, "parameter_model_relative_error_top_bottom_10.png"),
    dpi=300)

# =============================================================================
#%% Compare relative errors with coefficients of variation across identical curves
# =============================================================================

# parameters with overall highest/lowest CV
param_cv = pd.read_csv(os.path.join(data_dir, "parameter_cv", "parameter_cv.csv"))
n_sim = pd.read_csv(os.path.join(data_dir, "parameter_cv", "number_similiar_curves.txt"))

plot_idx = n_sim.values>=5
cv_order = np.argsort(-param_cv.iloc[plot_idx, :].median().values)

fig_cv_dist, axs_cv_dist = plt.subplots(2, 1, figsize=(5, 5), layout='constrained')
axs_cv_dist[0].boxplot(param_cv.iloc[plot_idx, cv_order[:10]],
            tick_labels=[param_names[i] for i in cv_order[:10]],
            vert=False)

# plot parameters with highest median CV
axs_cv_dist[1].boxplot(param_cv.iloc[plot_idx, cv_order[-9:]],
            tick_labels=[param_names[i] for i in cv_order[-9:]],
            vert=False)
axs_cv_dist[1].set_xlabel("coefficient of variation")
fig_cv_dist.savefig(os.path.join(result_dir, "parameter_cv_top_bottom_10.png"),
             dpi=300)

# parameters with overall highest/lowest relative errors
fig_re_distr, axs_re_distr = plt.subplots(2, 1, layout='constrained', figsize=(5, 5))
axs_re_distr[0].boxplot(re[:, re_order[:10]],
            tick_labels=[param_names[i] for i in re_order[:10]],
            vert=False)
axs_re_distr[0].yaxis.tick_right()

# plot parameters with highest median relative errors
axs_re_distr[1].boxplot(re[:, re_order[-9:]],
            tick_labels=[param_names[i] for i in re_order[-9:]],
            vert=False)
axs_re_distr[1].set_xlabel("relative error")
axs_re_distr[1].yaxis.tick_right()

fig_re_distr.savefig(os.path.join(result_dir, "parameter_re_top_bottom_10.png"),
             dpi=300)

# Pearson correlation beween CV and relative errors
r_p_cv_re = np.corrcoef(param_cv.iloc[plot_idx, :].mean(), re.mean(axis=0))[0, 1]
print(f"Pearson correlation between relative error and CV: {r_p_cv_re:.2f}.")

